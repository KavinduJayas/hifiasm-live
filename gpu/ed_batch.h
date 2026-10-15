#ifndef HA_GPU_ED_BATCH_H
#define HA_GPU_ED_BATCH_H

// GPU batch evaluation of the first-pass window edit distance
// (ed_band_cal_semi_64_w_absent_diag) used by gen_hc_r_alin / align_hc_ed_post_extz.
//
// Sequences are streamed with the jobs: every context carries the 2-bit words of its
// query read and of each target span it aligns to, so the device holds nothing between
// batches and the read set can be any size (an earlier version kept all reads resident
// on the device, which does not fit 40x human on a 32 GB card).
//
// Lifecycle:
//   ha_gpu_init()                 once, before error correction (needs --gpu)
//   ha_gpu_new_round()            at the start of every EC round (prints/resets the stats)
//   per read, from a worker thread:
//       gc = ha_gpu_acquire(); ha_gpu_ctx_set_query(gc, &R_INF, rid);
//       per overlap: first = ha_gpu_ctx_n(gc); push one job per window with
//           ha_gpu_ctx_push() (fill q_s/q_l/t_s/t_l/thre/aux_beg/strand), then
//           ha_gpu_ctx_add_target(gc, &R_INF, y_id, first);
//       ha_gpu_ctx_run(gc); read results with ha_gpu_ctx_res(); ha_gpu_release(gc);
//   ha_gpu_destroy()              at the end
//
// Every function returns 0 / NULL on failure so callers can fall back to the CPU path.
// Implemented in ed_batch.cu (make GPU=1) or ed_batch_stub.cpp (default build), where
// ha_gpu_available() is 0 and everything else fails.

#include <stdint.h>
#include "../All_reads.h"

typedef struct {
    // filled by the caller
    int32_t q_s, q_l;           // window on the query (text t), forward strand
    int32_t t_s, t_l;           // slice of the target (pattern p) in y_pos_strand orientation
    int32_t thre, aux_beg;      // band and absent-diagonal offset (init_waln)
    uint8_t strand;             // z->y_pos_strand
    // filled by ha_gpu_ctx_set_query / ha_gpu_ctx_add_target: where the sequences sit in
    // the batch buffers (word offsets into the 2-bit buffer, offsets into the N list)
    uint8_t q_all_n, t_all_n;   // read_sperate[] == NULL (recover_UC_Read gives all 'N')
    int32_t t_wlo;              // forward word index of the target read held at t_wofs
    int32_t t_rlen;             // target read length (reverse strand maps through it)
    uint32_t q_wofs, t_wofs;
    uint32_t q_nofs, t_nofs, q_ncnt, t_ncnt;
} ha_gpu_ed_job_t;

typedef struct {
    int32_t err, pe;            // what the CPU kernel leaves in exz->err / exz->pe
} ha_gpu_ed_res_t;

typedef struct ha_gpu_ctx_s ha_gpu_ctx_t;

extern int ha_gpu_verify;       // HA_GPU_VERIFY=1: also run the CPU kernel and abort on mismatch

int ha_gpu_available(void);
int ha_gpu_init(void);
void ha_gpu_new_round(void);
void ha_gpu_destroy(void);

ha_gpu_ctx_t *ha_gpu_acquire(void);
void ha_gpu_release(ha_gpu_ctx_t *ctx);
// Empties the context and stages the whole query read rid.
int ha_gpu_ctx_set_query(ha_gpu_ctx_t *ctx, const All_reads *rs, uint32_t rid);
// Returns the next job slot; fill it before the next push (the buffer may move).
ha_gpu_ed_job_t *ha_gpu_ctx_push(ha_gpu_ctx_t *ctx);
// Stages the span of read rid covering jobs [first, n) and fills their t_* fields.
int ha_gpu_ctx_add_target(ha_gpu_ctx_t *ctx, const All_reads *rs, uint32_t rid, int32_t first);
int32_t ha_gpu_ctx_n(const ha_gpu_ctx_t *ctx);
int ha_gpu_ctx_run(ha_gpu_ctx_t *ctx);  // synchronous
const ha_gpu_ed_res_t *ha_gpu_ctx_res(const ha_gpu_ctx_t *ctx);

#endif // HA_GPU_ED_BATCH_H
