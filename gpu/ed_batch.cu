// GPU batch evaluation of first-pass window edit distance. See ed_batch.h for the API
// and docs/gpu-plan.md (T1) for the design.
//
// Worker threads never call CUDA. Each worker fills a context (plain host buffers: jobs
// plus the 2-bit words of the sequences they need) and ha_gpu_ctx_run() hands it to a
// single service thread, which merges every context that is waiting into one launch and
// scatters the results back. Small per-read batches from many streams do not overlap on
// the device (see ed_batchbench.cu), so aggregating them is what makes the GPU path
// faster than the CPU kernel. Two staging slots alternate so the host-side gather and
// the transfers of one chunk overlap the kernel of the previous one.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
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

// ---------------------------------------------------------------- kernel

// Sequence accessor over a staged slice of a read: operator[](i) returns the 2-bit code
// (4 = N) of base i of the window, honouring strand exactly like
// recover_UC_Read_sub_region (reverse strand = complement of read[rlen-1-(start+i)]).
// w holds the read's forward words from index wlo on, in the read_sperate byte layout.
// Sequential access reloads one 64-bit word every 32 bases.
struct ha_gpu_seq_t {
    const uint64_t *w; const uint32_t *ns;
    int32_t n_cnt, start, rlen, wlo;
    uint8_t strand, all_n;
    mutable int32_t cur_idx; mutable uint64_t cur;

    __device__ __forceinline__ uint8_t operator[](int32_t i) const {
        if (all_n) return 4;
        int32_t fp = strand ? (rlen - 1 - (start + i)) : (start + i);
        int32_t wi = fp >> 5;
        if (wi != cur_idx) { cur_idx = wi; cur = w[wi - wlo]; }
        uint32_t byte = (uint32_t)(cur >> (((fp >> 2) & 7) << 3)) & 0xFFu;
        uint32_t c = (byte >> (6 - ((fp & 3) << 1))) & 3u;
        if (strand) c = 3u - c;
        if (n_cnt) { // N bases are rare: binary search the slice's N list (forward positions)
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

__global__ void ha_gpu_ed_kernel(const uint64_t *__restrict__ seq, const uint32_t *__restrict__ npos,
                                 const ha_gpu_ed_job_t *__restrict__ jobs, ha_gpu_ed_res_t *__restrict__ res, int32_t n)
{
    for (int32_t j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x) {
        ha_gpu_ed_job_t jb = jobs[j];
        ha_gpu_seq_t p, t;
        p.w = seq + jb.t_wofs; p.ns = npos + jb.t_nofs; p.n_cnt = (int32_t)jb.t_ncnt; p.wlo = jb.t_wlo;
        p.start = jb.t_s; p.rlen = jb.t_rlen; p.strand = jb.strand; p.all_n = jb.t_all_n; p.cur_idx = -1; p.cur = 0;
        t.w = seq + jb.q_wofs; t.ns = npos + jb.q_nofs; t.n_cnt = (int32_t)jb.q_ncnt; t.wlo = 0;
        t.start = jb.q_s; t.rlen = 0; t.strand = 0; t.all_n = jb.q_all_n; t.cur_idx = -1; t.cur = 0;
        int32_t err, pe;
        ha_ed_semi64_absent_diag_t(p, jb.t_l, t, jb.q_l, jb.thre, jb.aux_beg, &err, &pe);
        res[j].err = err; res[j].pe = pe;
    }
}

// ---------------------------------------------------------------- contexts (host side)

struct ha_gpu_ctx_s {
    ha_gpu_ed_job_t *jobs;      // plain host memory; the service thread copies into its pinned buffers
    ha_gpu_ed_res_t *res;
    int32_t n, cap;
    uint64_t *seq; uint32_t seq_n, seq_cap;     // 2-bit words of the query and every target span
    uint32_t *np; uint32_t np_n, np_cap;        // forward N positions of those slices
    uint32_t q_wofs, q_nofs, q_ncnt; uint8_t q_all_n;
    int done, ok;
};

static std::mutex g_lock;                       // guards everything below
static std::condition_variable g_cv_submit, g_cv_done;
static std::vector<ha_gpu_ctx_t *> g_pool_free, g_pool_all, g_queue;
static int g_stop = 0, g_inited = 0;
static std::thread g_server;
static uint64_t g_tot_jobs = 0, g_tot_batches = 0, g_max_batch = 0, g_wait_us = 0, g_tot_words = 0;
static double g_ms_kernel = 0, g_ms_batch = 0;

template <class X>
static void host_reserve(X **p, uint32_t *cap, uint32_t need)
{
    if (need <= *cap) return;
    uint32_t n = *cap ? *cap : 1024;
    while (n < need) n <<= 1;
    *p = (X *)realloc(*p, (size_t)n * sizeof(X)); *cap = n;
}

// Copy forward words [wlo, whi) of read rid into the context; returns their word offset.
// Bytes past the end of the read are zero (never read: windows stay inside the read).
static uint32_t stage_words(ha_gpu_ctx_t *c, const All_reads *rs, uint32_t rid, int32_t wlo, int32_t whi)
{
    uint32_t nw = (uint32_t)(whi - wlo), wofs = c->seq_n;
    host_reserve(&c->seq, &c->seq_cap, c->seq_n + nw);
    uint64_t *dst = c->seq + wofs;
    memset(dst, 0, (size_t)nw << 3);
    uint64_t nbytes = (rs->read_length[rid] + 3) >> 2, b0 = (uint64_t)wlo << 3, b1 = (uint64_t)whi << 3;
    if (b1 > nbytes) b1 = nbytes;
    if (b1 > b0) memcpy(dst, rs->read_sperate[rid] + b0, b1 - b0);
    c->seq_n += nw;
    return wofs;
}

// Copy the N positions of read rid that fall in [lo, hi) (N_site is sorted); returns their offset.
static uint32_t stage_npos(ha_gpu_ctx_t *c, const All_reads *rs, uint32_t rid, int64_t lo, int64_t hi, uint32_t *cnt)
{
    uint32_t nofs = c->np_n, n = 0; uint64_t k;
    const uint64_t *ns = rs->N_site[rid];
    if (ns) for (k = 1; k <= ns[0]; k++) {
        if ((int64_t)ns[k] < lo || (int64_t)ns[k] >= hi) continue;
        host_reserve(&c->np, &c->np_cap, c->np_n + 1);
        c->np[c->np_n++] = (uint32_t)ns[k]; n++;
    }
    *cnt = n;
    return nofs;
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
    c->n = 0; c->seq_n = 0; c->np_n = 0;
    return c;
}

void ha_gpu_release(ha_gpu_ctx_t *c)
{
    if (!c) return;
    std::lock_guard<std::mutex> lk(g_lock);
    g_pool_free.push_back(c);
}

int ha_gpu_ctx_set_query(ha_gpu_ctx_t *c, const All_reads *rs, uint32_t rid)
{
    c->n = 0; c->seq_n = 0; c->np_n = 0;
    int64_t rlen = (int64_t)rs->read_length[rid];
    c->q_all_n = rs->read_sperate[rid] ? 0 : 1;
    c->q_wofs = c->q_all_n ? 0 : stage_words(c, rs, rid, 0, (int32_t)((rlen + 31) >> 5));
    c->q_nofs = stage_npos(c, rs, rid, 0, rlen, &c->q_ncnt);
    return 1;
}

ha_gpu_ed_job_t *ha_gpu_ctx_push(ha_gpu_ctx_t *c)
{
    if (c->n == c->cap) {
        c->cap <<= 1;
        c->jobs = (ha_gpu_ed_job_t *)realloc(c->jobs, c->cap * sizeof(*c->jobs));
        c->res = (ha_gpu_ed_res_t *)realloc(c->res, c->cap * sizeof(*c->res));
    }
    ha_gpu_ed_job_t *jb = &c->jobs[c->n++];
    jb->q_wofs = c->q_wofs; jb->q_nofs = c->q_nofs; jb->q_ncnt = c->q_ncnt; jb->q_all_n = c->q_all_n;
    return jb;
}

int ha_gpu_ctx_add_target(ha_gpu_ctx_t *c, const All_reads *rs, uint32_t rid, int32_t first)
{
    int32_t j; int64_t rlen = (int64_t)rs->read_length[rid], lo = INT64_MAX, hi = 0;
    if (first >= c->n) return 1; // no window survived init_waln
    // forward-strand span of all windows of this target
    for (j = first; j < c->n; j++) {
        ha_gpu_ed_job_t *jb = &c->jobs[j];
        int64_t s = jb->strand ? rlen - (jb->t_s + jb->t_l) : jb->t_s, e = s + jb->t_l;
        if (s < lo) lo = s;
        if (e > hi) hi = e;
    }
    if (lo < 0 || hi > rlen) return 0;
    uint8_t all_n = rs->read_sperate[rid] ? 0 : 1;
    int32_t wlo = (int32_t)(lo >> 5), whi = (int32_t)((hi + 31) >> 5);
    uint32_t wofs = all_n ? 0 : stage_words(c, rs, rid, wlo, whi), ncnt, nofs = stage_npos(c, rs, rid, lo, hi, &ncnt);
    for (j = first; j < c->n; j++) {
        ha_gpu_ed_job_t *jb = &c->jobs[j];
        jb->t_wofs = wofs; jb->t_wlo = wlo; jb->t_nofs = nofs; jb->t_ncnt = ncnt;
        jb->t_rlen = (int32_t)rlen; jb->t_all_n = all_n;
    }
    return 1;
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

// ---------------------------------------------------------------- service thread

// Pinned/device staging allocated once (pinned alloc/free is slow and fragments on some
// platforms). A batch larger than a slot runs in chunks; a single context larger than a
// slot goes back to the CPU.
static const size_t STAGE_JOBS = (size_t)1 << 18, STAGE_WORDS = (size_t)1 << 24, STAGE_NPOS = (size_t)1 << 21;

struct slot_t {
    cudaStream_t st; cudaEvent_t e0, e1;
    ha_gpu_ed_job_t *hj, *dj; ha_gpu_ed_res_t *hr, *dr; uint64_t *hw, *dw; uint32_t *hn, *dn;
    std::vector<ha_gpu_ctx_t *> pending;    // contexts whose chunk is in flight
    size_t n_jobs; int busy;
};
static slot_t g_slot[2];
static int g_cur = 0;

static void mark_done(ha_gpu_ctx_t **ctx, size_t n, int ok)
{
    { std::lock_guard<std::mutex> lk(g_lock); for (size_t k = 0; k < n; k++) { ctx[k]->ok = ok; ctx[k]->done = 1; } }
    g_cv_done.notify_all();
}

// Wait for the slot's chunk, scatter the results and wake its workers.
static int slot_complete(slot_t *s)
{
    if (!s->busy) return 1;
    auto t0 = std::chrono::steady_clock::now();
    int ok = cudaStreamSynchronize(s->st) == cudaSuccess;
    if (ok) {
        float ms = 0; if (cudaEventElapsedTime(&ms, s->e0, s->e1) == cudaSuccess) g_ms_kernel += ms;
        size_t o = 0;
        for (size_t k = 0; k < s->pending.size(); o += s->pending[k]->n, k++)
            memcpy(s->pending[k]->res, s->hr + o, s->pending[k]->n * sizeof(*s->hr));
    } else fprintf(stderr, "[ERROR::ha_gpu] %s while waiting for a batch\n", cudaGetErrorString(cudaGetLastError()));
    mark_done(s->pending.data(), s->pending.size(), ok);
    s->pending.clear(); s->busy = 0;
    g_ms_batch += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return ok;
}

// Gather contexts [0, n) into the slot (rebasing their buffer offsets) and launch.
static int slot_launch(slot_t *s, ha_gpu_ctx_t **ctx, size_t n)
{
    size_t oj = 0, ow = 0, on = 0, k; int32_t i;
    for (k = 0; k < n; k++) {
        ha_gpu_ctx_t *c = ctx[k];
        memcpy(s->hw + ow, c->seq, (size_t)c->seq_n << 3);
        if (c->np_n) memcpy(s->hn + on, c->np, (size_t)c->np_n << 2);
        for (i = 0; i < c->n; i++) {
            ha_gpu_ed_job_t jb = c->jobs[i];
            jb.q_wofs += (uint32_t)ow; jb.t_wofs += (uint32_t)ow; jb.q_nofs += (uint32_t)on; jb.t_nofs += (uint32_t)on;
            s->hj[oj + i] = jb;
        }
        oj += c->n; ow += c->seq_n; on += c->np_n;
    }
    const int block = 128; int grid = (int)((oj + block - 1) / block);
    CU(cudaMemcpyAsync(s->dj, s->hj, oj * sizeof(*s->hj), cudaMemcpyHostToDevice, s->st));
    CU(cudaMemcpyAsync(s->dw, s->hw, ow << 3, cudaMemcpyHostToDevice, s->st));
    if (on) CU(cudaMemcpyAsync(s->dn, s->hn, on << 2, cudaMemcpyHostToDevice, s->st));
    CU(cudaEventRecord(s->e0, s->st));
    ha_gpu_ed_kernel<<<grid, block, 0, s->st>>>(s->dw, s->dn, s->dj, s->dr, (int32_t)oj);
    CU(cudaGetLastError());
    CU(cudaEventRecord(s->e1, s->st));
    CU(cudaMemcpyAsync(s->hr, s->dr, oj * sizeof(*s->hr), cudaMemcpyDeviceToHost, s->st));
    s->pending.assign(ctx, ctx + n); s->n_jobs = oj; s->busy = 1;
    g_tot_batches++; g_tot_jobs += oj; g_tot_words += ow; if (oj > g_max_batch) g_max_batch = oj;
    return 1;
}

static void run_batch(std::vector<ha_gpu_ctx_t *> &batch)
{
    auto t0 = std::chrono::steady_clock::now();
    size_t i = 0, n = batch.size();
    while (i < n) {
        // as many contexts as fit in a slot
        size_t j = i, cj = 0, cw = 0, cn = 0;
        while (j < n && cj + batch[j]->n <= STAGE_JOBS && cw + batch[j]->seq_n <= STAGE_WORDS && cn + batch[j]->np_n <= STAGE_NPOS) {
            cj += batch[j]->n; cw += batch[j]->seq_n; cn += batch[j]->np_n; j++;
        }
        if (j == i) { // one read alone exceeds the staging: its worker takes the CPU path
            fprintf(stderr, "[W::ha_gpu] a read batch (%d windows, %u words, %u N) exceeds the staging buffers; using the CPU kernel for it\n",
                    batch[i]->n, batch[i]->seq_n, batch[i]->np_n);
            mark_done(&batch[i], 1, 0); i++; continue;
        }
        slot_t *s = &g_slot[g_cur];
        slot_complete(s); // the chunk launched two chunks ago; the other slot keeps the device busy meanwhile
        if (!slot_launch(s, &batch[i], j - i)) { mark_done(&batch[i], j - i, 0); s->pending.clear(); s->busy = 0; }
        g_cur ^= 1; i = j;
    }
    g_ms_batch += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

static void server_main(void)
{
    std::vector<ha_gpu_ctx_t *> batch;
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(g_lock);
            int busy = g_slot[0].busy || g_slot[1].busy;
            while (g_queue.empty() && !g_stop && !busy) g_cv_submit.wait(lk);
            batch.swap(g_queue); g_queue.clear();
        }
        if (!batch.empty()) { run_batch(batch); batch.clear(); continue; }
        // nothing new is waiting: finish what is in flight, oldest first
        slot_complete(&g_slot[g_cur ^ 1]); slot_complete(&g_slot[g_cur]);
        if (g_stop) { std::lock_guard<std::mutex> lk(g_lock); if (g_queue.empty()) return; }
    }
}

// ---------------------------------------------------------------- lifecycle

int ha_gpu_available(void) { return 1; }

static int slot_init(slot_t *s)
{
    s->hj = s->dj = NULL; s->hr = s->dr = NULL; s->hw = s->dw = NULL; s->hn = s->dn = NULL; s->n_jobs = 0; s->busy = 0;
    CU(cudaStreamCreateWithFlags(&s->st, cudaStreamNonBlocking));
    CU(cudaEventCreate(&s->e0)); CU(cudaEventCreate(&s->e1));
    CU(cudaHostAlloc((void **)&s->hj, STAGE_JOBS * sizeof(*s->hj), cudaHostAllocDefault));
    CU(cudaHostAlloc((void **)&s->hr, STAGE_JOBS * sizeof(*s->hr), cudaHostAllocDefault));
    CU(cudaHostAlloc((void **)&s->hw, STAGE_WORDS * sizeof(*s->hw), cudaHostAllocDefault));
    CU(cudaHostAlloc((void **)&s->hn, STAGE_NPOS * sizeof(*s->hn), cudaHostAllocDefault));
    CU(cudaMalloc((void **)&s->dj, STAGE_JOBS * sizeof(*s->dj)));
    CU(cudaMalloc((void **)&s->dr, STAGE_JOBS * sizeof(*s->dr)));
    CU(cudaMalloc((void **)&s->dw, STAGE_WORDS * sizeof(*s->dw)));
    CU(cudaMalloc((void **)&s->dn, STAGE_NPOS * sizeof(*s->dn)));
    return 1;
}

static void slot_destroy(slot_t *s)
{
    cudaFreeHost(s->hj); cudaFreeHost(s->hr); cudaFreeHost(s->hw); cudaFreeHost(s->hn);
    cudaFree(s->dj); cudaFree(s->dr); cudaFree(s->dw); cudaFree(s->dn);
    cudaEventDestroy(s->e0); cudaEventDestroy(s->e1); cudaStreamDestroy(s->st);
    s->hj = s->dj = NULL; s->hr = s->dr = NULL; s->hw = s->dw = NULL; s->hn = s->dn = NULL; s->busy = 0;
}

int ha_gpu_init(void)
{
    if (g_inited) return 1;
    auto t0 = std::chrono::steady_clock::now();
    CU(cudaFree(0));
    int dev = 0; cudaDeviceProp prop;
    CU(cudaGetDevice(&dev));
    CU(cudaGetDeviceProperties(&prop, dev));
    if (!slot_init(&g_slot[0]) || !slot_init(&g_slot[1])) return 0;
    const char *verify_env = getenv("HA_GPU_VERIFY");
    ha_gpu_verify = (verify_env && *verify_env && strcmp(verify_env, "0") != 0) ? 1 : 0;
    size_t slot_bytes = STAGE_JOBS * (sizeof(ha_gpu_ed_job_t) + sizeof(ha_gpu_ed_res_t)) + STAGE_WORDS * 8 + STAGE_NPOS * 4;
    fprintf(stderr, "[M::%s] using %s (sm_%d%d, %d SMs, %.1f GB); 2 x %.0f MB staging on host and device%s; init %.2f s\n", __func__,
            prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.totalGlobalMem / 1073741824.0,
            slot_bytes / 1048576.0, ha_gpu_verify ? "; verify mode" : "",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    g_stop = 0; g_cur = 0;
    g_server = std::thread(server_main);
    g_inited = 1;
    return 1;
}

static void print_stats(const char *who)
{
    size_t n_ctx, n_free;
    { std::lock_guard<std::mutex> lk(g_lock); n_ctx = g_pool_all.size(); n_free = g_pool_free.size(); }
    // contexts should stay near the thread count; a growing number means a caller is not releasing
    fprintf(stderr, "[M::%s] %lu windows in %lu launches (mean %.0f, max %lu); %.2f GB of sequence staged; kernel %.2f s, service %.2f s, worker wait %.2f thread-s; %zu contexts (%zu free)\n",
            who, (unsigned long)g_tot_jobs, (unsigned long)g_tot_batches,
            g_tot_batches ? (double)g_tot_jobs / g_tot_batches : 0.0, (unsigned long)g_max_batch,
            g_tot_words * 8 / 1073741824.0, g_ms_kernel / 1e3, g_ms_batch / 1e3, g_wait_us / 1e6, n_ctx, n_free);
    g_tot_jobs = g_tot_batches = g_max_batch = g_wait_us = g_tot_words = 0; g_ms_kernel = g_ms_batch = 0;
}

void ha_gpu_new_round(void)
{
    if (g_inited && g_tot_batches) print_stats(__func__); // stats of the round that just finished
}

void ha_gpu_destroy(void)
{
    if (!g_inited) return;
    { std::lock_guard<std::mutex> lk(g_lock); g_stop = 1; }
    g_cv_submit.notify_all();
    g_server.join();
    for (size_t i = 0; i < g_pool_all.size(); i++) {
        free(g_pool_all[i]->jobs); free(g_pool_all[i]->res); free(g_pool_all[i]->seq); free(g_pool_all[i]->np); free(g_pool_all[i]);
    }
    g_pool_all.clear(); g_pool_free.clear();
    slot_destroy(&g_slot[0]); slot_destroy(&g_slot[1]);
    print_stats(__func__);
    g_inited = 0;
}
