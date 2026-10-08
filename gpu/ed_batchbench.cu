// Replays a job dump (see ed_dump.h) the way hifiasm --gpu submits work: small batches,
// one per read, from many host threads each on its own stream. Measures whether batches
// from different streams overlap on the device and what one batch costs end to end.
//
//   ed_batchbench <dump.bin> <batch_size> <threads> [max_jobs]
//
// Compare threads=1 with threads=16: if jobs/s does not scale, batches are serialised.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <thread>
#include <chrono>
#include <cuda_runtime.h>
#include "ed_semi64.cuh"

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static std::vector<uint8_t> g_seq;
static std::vector<ha_ed_job_t> g_jobs;
static std::vector<ha_ed_res_t> g_ref;
static uint8_t *d_seq;
static int g_batch;
static long g_bad = 0;

static void load(const char *fn, long max_jobs)
{
    FILE *fp = fopen(fn, "rb"); if (!fp) { perror(fn); exit(1); }
    char magic[8]; if (fread(magic, 1, 8, fp) != 8 || memcmp(magic, "HAED0001", 8)) { fprintf(stderr, "bad magic\n"); exit(1); }
    std::vector<char> buf; int32_t h[6];
    while ((max_jobs <= 0 || (long)g_jobs.size() < max_jobs) && fread(h, 4, 6, fp) == 6) {
        int32_t pn = h[0], tn = h[1]; buf.resize(pn + tn);
        if (fread(buf.data(), 1, pn + tn, fp) != (size_t)(pn + tn)) break;
        ha_ed_job_t jb; jb.p_off = g_seq.size(); jb.pn = pn;
        for (int i = 0; i < pn; i++) g_seq.push_back(ha_nt4(buf[i]));
        jb.t_off = g_seq.size(); jb.tn = tn;
        for (int i = 0; i < tn; i++) g_seq.push_back(ha_nt4(buf[pn + i]));
        jb.thre = h[2]; jb.abs_diag = h[3]; g_jobs.push_back(jb);
        ha_ed_res_t r; r.err = h[4]; r.pe = h[5]; g_ref.push_back(r);
    }
    fclose(fp);
}

// Each worker owns a stream and pinned buffers, and replays batches [first, last) of size g_batch.
static void worker(int first, int last)
{
    cudaStream_t s; CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    ha_ed_job_t *hj, *dj; ha_ed_res_t *hr, *dr;
    CUDA_CHECK(cudaHostAlloc((void **)&hj, g_batch * sizeof(*hj), 0));
    CUDA_CHECK(cudaHostAlloc((void **)&hr, g_batch * sizeof(*hr), 0));
    CUDA_CHECK(cudaMalloc(&dj, g_batch * sizeof(*dj)));
    CUDA_CHECK(cudaMalloc(&dr, g_batch * sizeof(*dr)));
    long bad = 0;
    for (int b = first; b < last; b++) {
        int j0 = b * g_batch, n = (int)g_jobs.size() - j0; if (n > g_batch) n = g_batch;
        memcpy(hj, &g_jobs[j0], n * sizeof(*hj));
        CUDA_CHECK(cudaMemcpyAsync(dj, hj, n * sizeof(*hj), cudaMemcpyHostToDevice, s));
        ha_ed_semi64_kernel<<<(n + 127) / 128, 128, 0, s>>>(d_seq, dj, dr, n);
        CUDA_CHECK(cudaMemcpyAsync(hr, dr, n * sizeof(*hr), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
        for (int i = 0; i < n; i++) {
            const ha_ed_res_t &r = g_ref[j0 + i];
            if (hr[i].err != r.err || (r.err <= g_jobs[j0 + i].thre && hr[i].pe != r.pe)) bad++;
        }
    }
    __sync_fetch_and_add(&g_bad, bad);
    cudaFreeHost(hj); cudaFreeHost(hr); cudaFree(dj); cudaFree(dr); cudaStreamDestroy(s);
}

int main(int argc, char **argv)
{
    if (argc < 4) { fprintf(stderr, "usage: %s <dump.bin> <batch_size> <threads> [max_jobs]\n", argv[0]); return 1; }
    g_batch = atoi(argv[2]); int T = atoi(argv[3]); long max_jobs = argc > 4 ? atol(argv[4]) : 0;
    load(argv[1], max_jobs);
    int n = g_jobs.size(), nb = (n + g_batch - 1) / g_batch;
    CUDA_CHECK(cudaMalloc(&d_seq, g_seq.size()));
    CUDA_CHECK(cudaMemcpy(d_seq, g_seq.data(), g_seq.size(), cudaMemcpyHostToDevice));
    worker(0, 1); // warm-up
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> th;
    for (int t = 0; t < T; t++) th.emplace_back(worker, (int)((long)nb * t / T), (int)((long)nb * (t + 1) / T));
    for (auto &x : th) x.join();
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("jobs=%d batch=%d threads=%d: %.3f s, %.0f us/batch (per thread), %.2f M jobs/s, mismatches=%ld\n",
           n, g_batch, T, s, 1e6 * s * T / nb, n / s / 1e6, g_bad);
    return g_bad ? 2 : 0;
}
