#ifndef HA_GPU_ED_BATCH_H
#define HA_GPU_ED_BATCH_H

// GPU batch evaluation of the first-pass window edit distance
// (ed_band_cal_semi_64_w_absent_diag) used by gen_hc_r_alin / align_hc_ed_post_extz.
//
// Lifecycle:
//   ha_gpu_init()                 once, before error correction (needs --gpu)
//   ha_gpu_upload_reads(&R_INF)   at the start of every EC round (reads change between rounds);
//                                 on failure the caller drops to the CPU kernel
//   per read, from a worker thread:
//       gc = ha_gpu_acquire(); push jobs with ha_gpu_ctx_push(); ha_gpu_ctx_run(gc);
//       read results with ha_gpu_ctx_res(); ha_gpu_release(gc);
//   ha_gpu_destroy()              at the end
//
// Every function returns 0 / NULL on failure so callers can fall back to the CPU path.
// Implemented in ed_batch.cu (make GPU=1) or ed_batch_stub.cpp (default build), where
// ha_gpu_available() is 0 and everything else fails.

#include <stdint.h>
#include "../All_reads.h"

typedef struct {
    uint32_t x_id, y_id;        // query read (text t) and target read (pattern p)
    int32_t q_s, q_l;           // window on the query, forward strand
    int32_t t_s, t_l;           // slice of the target in y_pos_strand orientation
    int32_t thre, aux_beg;      // band and absent-diagonal offset (init_waln)
    uint8_t strand;             // z->y_pos_strand
} ha_gpu_ed_job_t;

typedef struct {
    int32_t err, pe;            // what the CPU kernel leaves in exz->err / exz->pe
} ha_gpu_ed_res_t;

typedef struct ha_gpu_ctx_s ha_gpu_ctx_t;

extern int ha_gpu_verify;       // HA_GPU_VERIFY=1: also run the CPU kernel and abort on mismatch

int ha_gpu_available(void);
int ha_gpu_init(void);
int ha_gpu_upload_reads(const All_reads *rs);
// 1 if the resident copy of rs fits in free device memory; prints need vs free either
// way. Cheap: call it as soon as the reads are known so a run that cannot use the GPU
// says so before the index build, not an hour later.
int ha_gpu_check_fit(const All_reads *rs, const char *what);
void ha_gpu_destroy(void);

ha_gpu_ctx_t *ha_gpu_acquire(void);
void ha_gpu_release(ha_gpu_ctx_t *ctx);
// Returns the next job slot; fill it before the next push (the buffer may move).
ha_gpu_ed_job_t *ha_gpu_ctx_push(ha_gpu_ctx_t *ctx);
int32_t ha_gpu_ctx_n(const ha_gpu_ctx_t *ctx);
int ha_gpu_ctx_run(ha_gpu_ctx_t *ctx);  // synchronous
const ha_gpu_ed_res_t *ha_gpu_ctx_res(const ha_gpu_ctx_t *ctx);

#endif // HA_GPU_ED_BATCH_H
