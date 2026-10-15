// GPU batch evaluation of first-pass window edit distance. See ed_batch.h for the API
// and docs/gpu-plan.md (T1) for the design.
//
// Worker threads never call CUDA. Each worker fills a context (plain host buffers) and
// ha_gpu_ctx_run() hands it to a single service thread, which merges every context that
// is waiting into one kernel launch and scatters the results back. Small per-read batches
// from many streams do not overlap on the device (see ed_batchbench.cu), so aggregating
// them is what makes the GPU path faster than the CPU kernel.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <chrono>
#include <cuda_runtime.h>
#include "ed_batch.h"
#include "ed_semi64.cuh"

int ha_gpu_verify = 0;

#define CU(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "[ERROR::ha_gpu] %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return 0; } } while (0)

// ---------------------------------------------------------------- device-resident reads

typedef struct {
    uint64_t *seq;    // 2-bit packed bases, same layout as R_INF.read_sperate, each read padded to 8 bytes
    uint64_t *off;    // n_reads+1 byte offsets into seq (multiples of 8)
    uint32_t *len;    // read lengths
    uint64_t *noff;   // n_reads+1 offsets into npos
    uint32_t *npos;   // sorted N positions per read (from R_INF.N_site)
    uint8_t  *all_n;  // 1 if read_sperate[i] == NULL (recover_UC_Read gives all 'N')
    uint64_t n_reads, seq_bytes, n_npos;
} ha_gpu_reads_t;

static ha_gpu_reads_t R;
static int g_inited = 0;

// Allocate exactly what the current read set needs. The earlier power-of-two growth
// turned a 17-31 GB read set into a 32 GB request, which no 32 GB card can satisfy,
// and the sequence array dominates device memory so there is nothing to amortise.
static void dev_release(ha_gpu_reads_t *r)
{
    cudaFree(r->seq); cudaFree(r->off); cudaFree(r->len); cudaFree(r->noff); cudaFree(r->npos); cudaFree(r->all_n);
    memset(r, 0, sizeof(*r));
}

// Device bytes the resident copy of rs takes. all_n reads hold no sequence; every other
// read is (len+3)/4 bytes rounded up to 8. Also returns the counts the upload needs.
static uint64_t reads_footprint(const All_reads *rs, uint64_t *seq_bytes, uint64_t *n_npos)
{
    uint64_t n = rs->total_reads, i, sb = 0, np = 0;
    for (i = 0; i < n; i++) {
        if (rs->read_sperate[i]) sb += (((rs->read_length[i] + 3) >> 2) + 7) & ~(uint64_t)7;
        if (rs->N_site[i]) np += rs->N_site[i][0];
    }
    *seq_bytes = sb; *n_npos = np;
    return sb + 8 + (n + 1) * (8 + 4 + 8 + 1) + (np + 1) * 4;
}

// Compare the resident footprint of rs against free device memory (after releasing the
// copy of the previous round, which is replaced anyway). Prints both so a run that is
// about to fall back to the CPU says why.
static const uint64_t FIT_MARGIN = (uint64_t)256 << 20; // allocator granularity, kernel stacks

int ha_gpu_check_fit(const All_reads *rs, const char *what)
{
    if (!g_inited) return 0;
    uint64_t sb, np, need = reads_footprint(rs, &sb, &np);
    size_t mfree = 0, mtotal = 0;
    dev_release(&R);
    CU(cudaMemGetInfo(&mfree, &mtotal));
    const char *cap = getenv("HA_GPU_MEM_CAP"); // GB; pretend the card is smaller (testing the fallback)
    if (cap && atof(cap) > 0 && (uint64_t)(atof(cap) * 1073741824.0) < mfree) mfree = (size_t)(atof(cap) * 1073741824.0);
    int ok = need + FIT_MARGIN <= mfree;
    fprintf(stderr, "[%s::%s] %s: %lu reads, %.2f GB of 2-bit sequence, %.2f GB device memory needed, %.2f GB free of %.2f GB%s\n",
            ok ? "M" : "W", __func__, what, (unsigned long)rs->total_reads, sb / 1073741824.0, need / 1073741824.0,
            mfree / 1073741824.0, mtotal / 1073741824.0, ok ? "" : "; the reads do not fit on this GPU");
    return ok;
}

// Sequence accessor over the device-resident reads: operator[](i) returns the 2-bit
// code (4 = N) of base i of the window, honouring strand exactly like
// recover_UC_Read_sub_region (reverse strand = complement of read[rlen-1-(start+i)]).
// Sequential access reloads one 64-bit word every 32 bases.
struct ha_gpu_seq_t {
    const uint64_t *w; const uint32_t *ns;
    int32_t n_cnt, start, rlen;
    uint8_t strand, all_n;
    mutable int32_t cur_idx; mutable uint64_t cur;

    __device__ __forceinline__ uint8_t operator[](int32_t i) const {
        if (all_n) return 4;
        int32_t fp = strand ? (rlen - 1 - (start + i)) : (start + i);
        int32_t wi = fp >> 5;
        if (wi != cur_idx) { cur_idx = wi; cur = w[wi]; }
        uint32_t byte = (uint32_t)(cur >> (((fp >> 2) & 7) << 3)) & 0xFFu;
        uint32_t c = (byte >> (6 - ((fp & 3) << 1))) & 3u;
        if (strand) c = 3u - c;
        if (n_cnt) { // N bases are rare: binary search the read's N list
            int32_t lo = 0, hi = n_cnt - 1;
            while (lo <= hi) {
                int32_t mid = (lo + hi) >> 1; uint32_t v = ns[mid];
                if (v == (uint32_t)fp) { c = 4; break; }
                if (v < (uint32_t)fp) lo = mid + 1; else hi = mid - 1;
            }
        }
        return (uint8_t)c;
    }
};

__global__ void ha_gpu_ed_kernel(const uint64_t *__restrict__ seq, const uint64_t *__restrict__ off,
                                 const uint32_t *__restrict__ len, const uint64_t *__restrict__ noff,
                                 const uint32_t *__restrict__ npos, const uint8_t *__restrict__ all_n,
                                 const ha_gpu_ed_job_t *__restrict__ jobs, ha_gpu_ed_res_t *__restrict__ res, int32_t n)
{
    for (int32_t j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x) {
        ha_gpu_ed_job_t jb = jobs[j];
        ha_gpu_seq_t p, t;
        p.w = seq + (off[jb.y_id] >> 3); p.ns = npos + noff[jb.y_id]; p.n_cnt = (int32_t)(noff[jb.y_id + 1] - noff[jb.y_id]);
        p.start = jb.t_s; p.rlen = (int32_t)len[jb.y_id]; p.strand = jb.strand; p.all_n = all_n[jb.y_id]; p.cur_idx = -1; p.cur = 0;
        t.w = seq + (off[jb.x_id] >> 3); t.ns = npos + noff[jb.x_id]; t.n_cnt = (int32_t)(noff[jb.x_id + 1] - noff[jb.x_id]);
        t.start = jb.q_s; t.rlen = (int32_t)len[jb.x_id]; t.strand = 0; t.all_n = all_n[jb.x_id]; t.cur_idx = -1; t.cur = 0;
        int32_t err, pe;
        ha_ed_semi64_absent_diag_t(p, jb.t_l, t, jb.q_l, jb.thre, jb.aux_beg, &err, &pe);
        res[j].err = err; res[j].pe = pe;
    }
}

// ---------------------------------------------------------------- contexts and service thread

struct ha_gpu_ctx_s {
    ha_gpu_ed_job_t *jobs;   // plain host memory; the service thread copies into its pinned buffer
    ha_gpu_ed_res_t *res;
    int32_t n, cap;
    int done, ok;
};

static std::mutex g_lock;                       // guards everything below
static std::condition_variable g_cv_submit, g_cv_done;
static std::vector<ha_gpu_ctx_t *> g_pool_free, g_pool_all, g_queue;
static int g_stop = 0;
static std::thread g_server;
static uint64_t g_tot_jobs = 0, g_tot_batches = 0, g_max_batch = 0, g_wait_us = 0;
static double g_ms_kernel = 0, g_ms_batch = 0;

// service-thread state: pinned/device staging allocated once (pinned alloc/free is slow
// and fragments on some platforms); batches larger than the staging run in chunks.
static cudaStream_t g_stream;
static cudaEvent_t g_ev0, g_ev1;
static const size_t STAGE_JOBS = (size_t)1 << 20;
static ha_gpu_ed_job_t *g_hj = NULL, *g_dj = NULL; static ha_gpu_ed_res_t *g_hr = NULL, *g_dr = NULL;
static uint8_t *g_upload_stage = NULL;
static const uint64_t UPLOAD_CH = (uint64_t)64 << 20;

static int run_chunk(ha_gpu_ed_job_t *jobs_begin_unused, size_t total)
{
    (void)jobs_begin_unused;
    const int block = 128; int grid = (int)((total + block - 1) / block);
    CU(cudaMemcpyAsync(g_dj, g_hj, total * sizeof(*g_hj), cudaMemcpyHostToDevice, g_stream));
    CU(cudaEventRecord(g_ev0, g_stream));
    ha_gpu_ed_kernel<<<grid, block, 0, g_stream>>>(R.seq, R.off, R.len, R.noff, R.npos, R.all_n, g_dj, g_dr, (int32_t)total);
    CU(cudaGetLastError());
    CU(cudaEventRecord(g_ev1, g_stream));
    CU(cudaMemcpyAsync(g_hr, g_dr, total * sizeof(*g_hr), cudaMemcpyDeviceToHost, g_stream));
    CU(cudaStreamSynchronize(g_stream));
    float ms = 0; CU(cudaEventElapsedTime(&ms, g_ev0, g_ev1)); g_ms_kernel += ms;
    return 1;
}

static int run_batch(std::vector<ha_gpu_ctx_t *> &batch)
{
    auto t0 = std::chrono::steady_clock::now();
    size_t i = 0, n = batch.size(), total = 0;
    while (i < n) {
        // gather as many contexts as fit in the staging buffer, run, scatter back
        size_t j = i, cnt = 0;
        while (j < n && cnt + batch[j]->n <= STAGE_JOBS) { cnt += batch[j]->n; j++; }
        if (j == i) { fprintf(stderr, "[ERROR::ha_gpu] a single read has more than %zu windows\n", STAGE_JOBS); return 0; }
        size_t o = 0, k;
        for (k = i; k < j; o += batch[k]->n, k++) memcpy(g_hj + o, batch[k]->jobs, batch[k]->n * sizeof(*g_hj));
        if (!run_chunk(g_hj, cnt)) return 0;
        for (k = i, o = 0; k < j; o += batch[k]->n, k++) memcpy(batch[k]->res, g_hr + o, batch[k]->n * sizeof(*g_hr));
        g_tot_batches++; if (cnt > g_max_batch) g_max_batch = cnt;
        total += cnt; i = j;
    }
    g_tot_jobs += total;
    g_ms_batch += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return 1;
}

static void server_main(void)
{
    std::vector<ha_gpu_ctx_t *> batch;
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(g_lock);
            while (g_queue.empty() && !g_stop) g_cv_submit.wait(lk);
            if (g_queue.empty() && g_stop) return;
            batch.swap(g_queue); g_queue.clear();
        }
        int ok = run_batch(batch);
        {
            std::lock_guard<std::mutex> lk(g_lock);
            for (size_t i = 0; i < batch.size(); i++) { batch[i]->ok = ok; batch[i]->done = 1; }
        }
        g_cv_done.notify_all();
        batch.clear();
    }
}

ha_gpu_ctx_t *ha_gpu_acquire(void)
{
    if (!g_inited) return NULL;
    ha_gpu_ctx_t *c = NULL;
    std::lock_guard<std::mutex> lk(g_lock);
    if (!g_pool_free.empty()) { c = g_pool_free.back(); g_pool_free.pop_back(); }
    else {
        c = (ha_gpu_ctx_t *)calloc(1, sizeof(*c));
        c->cap = 4096;
        c->jobs = (ha_gpu_ed_job_t *)malloc(c->cap * sizeof(*c->jobs));
        c->res = (ha_gpu_ed_res_t *)malloc(c->cap * sizeof(*c->res));
        g_pool_all.push_back(c);
    }
    c->n = 0;
    return c;
}

void ha_gpu_release(ha_gpu_ctx_t *c)
{
    if (!c) return;
    std::lock_guard<std::mutex> lk(g_lock);
    g_pool_free.push_back(c);
}

ha_gpu_ed_job_t *ha_gpu_ctx_push(ha_gpu_ctx_t *c)
{
    if (c->n == c->cap) {
        c->cap <<= 1;
        c->jobs = (ha_gpu_ed_job_t *)realloc(c->jobs, c->cap * sizeof(*c->jobs));
        c->res = (ha_gpu_ed_res_t *)realloc(c->res, c->cap * sizeof(*c->res));
    }
    return &c->jobs[c->n++];
}

int32_t ha_gpu_ctx_n(const ha_gpu_ctx_t *c) { return c->n; }
const ha_gpu_ed_res_t *ha_gpu_ctx_res(const ha_gpu_ctx_t *c) { return c->res; }

int ha_gpu_ctx_run(ha_gpu_ctx_t *c)
{
    if (c->n == 0) return 1;
    auto t0 = std::chrono::steady_clock::now();
    {
        std::unique_lock<std::mutex> lk(g_lock);
        c->done = 0; c->ok = 0;
        g_queue.push_back(c);
        g_cv_submit.notify_one();
        while (!c->done) g_cv_done.wait(lk);
    }
    __sync_fetch_and_add(&g_wait_us, (uint64_t)std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    return c->ok;
}

// ---------------------------------------------------------------- lifecycle

int ha_gpu_available(void) { return 1; }

int ha_gpu_init(void)
{
    if (g_inited) return 1;
    auto t0 = std::chrono::steady_clock::now();
    CU(cudaFree(0));
    int dev = 0; cudaDeviceProp prop;
    CU(cudaGetDevice(&dev));
    CU(cudaGetDeviceProperties(&prop, dev));
    CU(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking));
    CU(cudaEventCreate(&g_ev0)); CU(cudaEventCreate(&g_ev1));
    CU(cudaHostAlloc((void **)&g_hj, STAGE_JOBS * sizeof(*g_hj), cudaHostAllocDefault));
    CU(cudaHostAlloc((void **)&g_hr, STAGE_JOBS * sizeof(*g_hr), cudaHostAllocDefault));
    CU(cudaMalloc((void **)&g_dj, STAGE_JOBS * sizeof(*g_dj)));
    CU(cudaMalloc((void **)&g_dr, STAGE_JOBS * sizeof(*g_dr)));
    CU(cudaHostAlloc((void **)&g_upload_stage, UPLOAD_CH, cudaHostAllocDefault));
    const char *verify_env = getenv("HA_GPU_VERIFY");
    ha_gpu_verify = (verify_env && *verify_env && strcmp(verify_env, "0") != 0) ? 1 : 0;
    fprintf(stderr, "[M::%s] using %s (sm_%d%d, %d SMs, %.1f GB)%s; init %.2f s\n", __func__, prop.name, prop.major, prop.minor,
            prop.multiProcessorCount, prop.totalGlobalMem / 1073741824.0, ha_gpu_verify ? ", verify mode" : "",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    memset(&R, 0, sizeof(R));
    g_stop = 0;
    g_server = std::thread(server_main);
    g_inited = 1;
    return 1;
}

static void print_stats(const char *who)
{
    size_t n_ctx, n_free;
    { std::lock_guard<std::mutex> lk(g_lock); n_ctx = g_pool_all.size(); n_free = g_pool_free.size(); }
    // contexts should stay near the thread count; a growing number means a caller is not releasing
    fprintf(stderr, "[M::%s] %lu windows in %lu launches (mean %.0f, max %lu); kernel %.2f s, service %.2f s, worker wait %.2f thread-s; %zu contexts (%zu free)\n",
            who, (unsigned long)g_tot_jobs, (unsigned long)g_tot_batches,
            g_tot_batches ? (double)g_tot_jobs / g_tot_batches : 0.0, (unsigned long)g_max_batch,
            g_ms_kernel / 1e3, g_ms_batch / 1e3, g_wait_us / 1e6, n_ctx, n_free);
    g_tot_jobs = g_tot_batches = g_max_batch = g_wait_us = 0; g_ms_kernel = g_ms_batch = 0;
}

int ha_gpu_upload_reads(const All_reads *rs)
{
    if (!g_inited) return 0;
    if (g_tot_batches) print_stats(__func__); // stats of the round that just finished
    auto t0 = std::chrono::steady_clock::now();
    if (!ha_gpu_check_fit(rs, "uploading reads")) return 0; // also frees the previous round's copy
    uint64_t n = rs->total_reads, i, k;
    std::vector<uint64_t> off(n + 1), noff(n + 1);
    std::vector<uint32_t> len(n), npos;
    std::vector<uint8_t> all_n(n);

    off[0] = noff[0] = 0;
    for (i = 0; i < n; i++) {
        uint64_t l = rs->read_length[i];
        len[i] = (uint32_t)l;
        all_n[i] = rs->read_sperate[i] ? 0 : 1;
        uint64_t bytes = all_n[i] ? 0 : ((l + 3) >> 2);
        off[i + 1] = off[i] + ((bytes + 7) & ~(uint64_t)7);
        if (rs->N_site[i]) for (k = 1; k <= rs->N_site[i][0]; k++) npos.push_back((uint32_t)rs->N_site[i][k]);
        noff[i + 1] = npos.size();
    }
    R.n_reads = n; R.seq_bytes = off[n]; R.n_npos = npos.size();

    CU(cudaMalloc((void **)&R.seq, (off[n] >> 3) * 8 + 8));
    CU(cudaMalloc((void **)&R.off, (n + 1) * sizeof(*R.off)));
    CU(cudaMalloc((void **)&R.len, (n + 1) * sizeof(*R.len)));
    CU(cudaMalloc((void **)&R.noff, (n + 1) * sizeof(*R.noff)));
    CU(cudaMalloc((void **)&R.all_n, n + 1));
    CU(cudaMalloc((void **)&R.npos, (npos.size() + 1) * sizeof(*R.npos)));

    CU(cudaMemcpy(R.off, off.data(), (n + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(R.len, len.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(R.noff, noff.data(), (n + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(R.all_n, all_n.data(), n, cudaMemcpyHostToDevice));
    if (!npos.empty()) CU(cudaMemcpy(R.npos, npos.data(), npos.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Sequences: stage through a pinned chunk so host memory stays bounded and the
    // number of transfers stays small.
    const uint64_t CH = UPLOAD_CH;
    uint8_t *stage = g_upload_stage, *dseq = (uint8_t *)R.seq;
    memset(stage, 0, CH);
    uint64_t base = 0, fill = 0;
    for (i = 0; i < n; i++) {
        uint64_t bytes = all_n[i] ? 0 : ((len[i] + 3) >> 2), pad = off[i + 1] - off[i];
        if (pad > CH) { fprintf(stderr, "[ERROR::ha_gpu] read %lu is longer than the staging chunk\n", (unsigned long)i); return 0; }
        if (off[i] + pad > base + CH) {
            CU(cudaMemcpy(dseq + base, stage, fill, cudaMemcpyHostToDevice));
            base = off[i]; fill = 0; memset(stage, 0, CH);
        }
        if (bytes) memcpy(stage + (off[i] - base), rs->read_sperate[i], bytes);
        fill = off[i + 1] - base;
    }
    if (fill) CU(cudaMemcpy(dseq + base, stage, fill, cudaMemcpyHostToDevice));
    CU(cudaDeviceSynchronize());

    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "[M::%s] %lu reads, %.1f MB of 2-bit sequence, %lu N sites, %.2f s\n", __func__,
            (unsigned long)n, off[n] / 1e6, (unsigned long)npos.size(), s);
    return 1;
}

void ha_gpu_destroy(void)
{
    if (!g_inited) return;
    { std::lock_guard<std::mutex> lk(g_lock); g_stop = 1; }
    g_cv_submit.notify_all();
    g_server.join();
    for (size_t i = 0; i < g_pool_all.size(); i++) { free(g_pool_all[i]->jobs); free(g_pool_all[i]->res); free(g_pool_all[i]); }
    g_pool_all.clear(); g_pool_free.clear();
    cudaFreeHost(g_hj); cudaFreeHost(g_hr); cudaFree(g_dj); cudaFree(g_dr); cudaFreeHost(g_upload_stage);
    g_hj = NULL; g_hr = NULL; g_dj = NULL; g_dr = NULL; g_upload_stage = NULL;
    cudaEventDestroy(g_ev0); cudaEventDestroy(g_ev1);
    cudaStreamDestroy(g_stream);
    dev_release(&R);
    print_stats(__func__);
    g_inited = 0;
}
