// Replays ed_band_cal_semi_64_w_absent_diag jobs recorded by hifiasm (HA_ED_DUMP=file,
// see ed_dump.h) on the GPU and checks err/pe against the CPU results in the dump.
// Also times the same __host__ __device__ kernel single-threaded on the CPU.
//
//   ed_replay <dump.bin> [reps=10] [block=128]

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "ed_semi64.cuh"

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

typedef struct {
    std::vector<uint8_t> seq;
    std::vector<ha_ed_job_t> jobs;
    std::vector<ha_ed_res_t> ref;
    uint64_t columns; // sum of tn, the DP work
} dump_t;

static int load_dump(const char *fn, dump_t *d)
{
    FILE *fp = fopen(fn, "rb");
    if (!fp) { perror(fn); return 0; }
    char magic[8];
    if (fread(magic, 1, 8, fp) != 8 || memcmp(magic, "HAED0001", 8) != 0) {
        fprintf(stderr, "bad magic in %s\n", fn); fclose(fp); return 0;
    }
    std::vector<char> buf;
    int32_t h[6];
    d->columns = 0;
    while (fread(h, sizeof(int32_t), 6, fp) == 6) {
        int32_t pn = h[0], tn = h[1];
        if (pn < 0 || tn < 0 || pn > (1 << 20) || tn > (1 << 20)) { fprintf(stderr, "corrupt record\n"); break; }
        buf.resize(pn + tn);
        if (fread(buf.data(), 1, pn + tn, fp) != (size_t)(pn + tn)) break;
        ha_ed_job_t jb;
        jb.p_off = (int32_t)d->seq.size(); jb.pn = pn;
        for (int32_t i = 0; i < pn; i++) d->seq.push_back(ha_nt4(buf[i]));
        jb.t_off = (int32_t)d->seq.size(); jb.tn = tn;
        for (int32_t i = 0; i < tn; i++) d->seq.push_back(ha_nt4(buf[pn + i]));
        jb.thre = h[2]; jb.abs_diag = h[3];
        d->jobs.push_back(jb);
        ha_ed_res_t r; r.err = h[4]; r.pe = h[5];
        d->ref.push_back(r);
        d->columns += tn;
    }
    fclose(fp);
    return 1;
}

static int64_t compare(const char *tag, const std::vector<ha_ed_res_t> &got, const dump_t &d, int show)
{
    int64_t bad = 0;
    for (size_t j = 0; j < got.size(); j++) {
        const ha_ed_res_t &r = d.ref[j], &g = got[j];
        // when the CPU did not align (err > thre), pe is undefined; only err matters
        int aligned = r.err <= d.jobs[j].thre;
        if (g.err != r.err || (aligned && g.pe != r.pe)) {
            if (bad < show)
                fprintf(stderr, "[%s] job %zu mismatch: pn=%d tn=%d thre=%d abs_diag=%d ref(err=%d,pe=%d) got(err=%d,pe=%d)\n",
                        tag, j, d.jobs[j].pn, d.jobs[j].tn, d.jobs[j].thre, d.jobs[j].abs_diag, r.err, r.pe, g.err, g.pe);
            bad++;
        }
    }
    return bad;
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s <dump.bin> [reps=10] [block=128]\n", argv[0]); return 1; }
    int reps = argc > 2 ? atoi(argv[2]) : 10;
    int block = argc > 3 ? atoi(argv[3]) : 128;

    dump_t d;
    if (!load_dump(argv[1], &d)) return 1;
    int32_t n = (int32_t)d.jobs.size();
    int64_t aligned = 0, mx_pn = 0, mx_tn = 0;
    for (int32_t j = 0; j < n; j++) {
        aligned += d.ref[j].err <= d.jobs[j].thre;
        if (d.jobs[j].pn > mx_pn) mx_pn = d.jobs[j].pn;
        if (d.jobs[j].tn > mx_tn) mx_tn = d.jobs[j].tn;
    }
    fprintf(stderr, "[load] %d jobs, %.1f MB of sequence, %.2f G columns, %.1f%% aligned on CPU, max pn=%ld max tn=%ld\n",
            n, d.seq.size() / 1e6, d.columns / 1e9, 100.0 * aligned / n, (long)mx_pn, (long)mx_tn);

    // ---- CPU single-thread reference (same code as the device function) ----
    std::vector<ha_ed_res_t> cpu(n);
    {
        auto t0 = std::chrono::steady_clock::now();
        for (int32_t j = 0; j < n; j++) {
            const ha_ed_job_t &jb = d.jobs[j];
            ha_ed_semi64_absent_diag(d.seq.data() + jb.p_off, jb.pn, d.seq.data() + jb.t_off, jb.tn,
                                     jb.thre, jb.abs_diag, &cpu[j].err, &cpu[j].pe);
        }
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        int64_t bad = compare("cpu-port", cpu, d, 5);
        fprintf(stderr, "[cpu ] 1 thread: %.3f s, %.2f us/job, %.2f M jobs/s, %.2f G col/s, mismatches vs dump: %ld\n",
                s, 1e6 * s / n, n / s / 1e6, d.columns / s / 1e9, (long)bad);
    }

    // ---- GPU ----
    int dev = 0; cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    fprintf(stderr, "[gpu ] %s, sm_%d%d, %d SMs\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    uint8_t *d_seq; ha_ed_job_t *d_jobs; ha_ed_res_t *d_res;
    CUDA_CHECK(cudaMalloc(&d_seq, d.seq.size()));
    CUDA_CHECK(cudaMalloc(&d_jobs, n * sizeof(ha_ed_job_t)));
    CUDA_CHECK(cudaMalloc(&d_res, n * sizeof(ha_ed_res_t)));

    int grid = (n + block - 1) / block;
    if (grid > 65535 * 4) grid = 65535 * 4;

    cudaEvent_t e0, e1, e2, e3;
    CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&e2)); CUDA_CHECK(cudaEventCreate(&e3));

    // warm-up + correctness
    CUDA_CHECK(cudaMemcpy(d_seq, d.seq.data(), d.seq.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_jobs, d.jobs.data(), n * sizeof(ha_ed_job_t), cudaMemcpyHostToDevice));
    ha_ed_semi64_kernel<<<grid, block>>>(d_seq, d_jobs, d_res, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<ha_ed_res_t> gpu(n);
    CUDA_CHECK(cudaMemcpy(gpu.data(), d_res, n * sizeof(ha_ed_res_t), cudaMemcpyDeviceToHost));
    int64_t bad = compare("gpu", gpu, d, 10);
    fprintf(stderr, "[gpu ] mismatches vs dump: %ld / %d\n", (long)bad, n);

    // timing: end-to-end (H2D + kernel + D2H) and kernel-only, averaged over reps
    float ms_total = 0, ms_kernel = 0;
    for (int r = 0; r < reps; r++) {
        CUDA_CHECK(cudaEventRecord(e0));
        CUDA_CHECK(cudaMemcpyAsync(d_seq, d.seq.data(), d.seq.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_jobs, d.jobs.data(), n * sizeof(ha_ed_job_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(e1));
        ha_ed_semi64_kernel<<<grid, block>>>(d_seq, d_jobs, d_res, n);
        CUDA_CHECK(cudaEventRecord(e2));
        CUDA_CHECK(cudaMemcpyAsync(gpu.data(), d_res, n * sizeof(ha_ed_res_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(e3));
        CUDA_CHECK(cudaEventSynchronize(e3));
        float a, b; CUDA_CHECK(cudaEventElapsedTime(&a, e0, e3)); CUDA_CHECK(cudaEventElapsedTime(&b, e1, e2));
        ms_total += a; ms_kernel += b;
    }
    ms_total /= reps; ms_kernel /= reps;
    fprintf(stderr, "[gpu ] block=%d grid=%d: kernel %.3f ms (%.3f us/job, %.1f M jobs/s, %.1f G col/s); "
                    "with transfers %.3f ms (%.1f M jobs/s)\n",
            block, grid, ms_kernel, 1e3 * ms_kernel / n, n / ms_kernel / 1e3, d.columns / ms_kernel / 1e6,
            ms_total, n / ms_total / 1e3);

    CUDA_CHECK(cudaFree(d_seq)); CUDA_CHECK(cudaFree(d_jobs)); CUDA_CHECK(cudaFree(d_res));
    return bad ? 2 : 0;
}
