// Linked when the build has no CUDA (default `make`). --gpu is reported as unavailable
// and every call fails so callers take the CPU path.
#include "ed_batch.h"

int ha_gpu_verify = 0;
int ha_gpu_available(void) { return 0; }
int ha_gpu_init(void) { return 0; }
int ha_gpu_upload_reads(const All_reads *rs) { (void)rs; return 0; }
void ha_gpu_destroy(void) {}
ha_gpu_ctx_t *ha_gpu_acquire(void) { return 0; }
void ha_gpu_release(ha_gpu_ctx_t *ctx) { (void)ctx; }
ha_gpu_ed_job_t *ha_gpu_ctx_push(ha_gpu_ctx_t *ctx) { (void)ctx; return 0; }
int32_t ha_gpu_ctx_n(const ha_gpu_ctx_t *ctx) { (void)ctx; return 0; }
int ha_gpu_ctx_run(ha_gpu_ctx_t *ctx) { (void)ctx; return 0; }
const ha_gpu_ed_res_t *ha_gpu_ctx_res(const ha_gpu_ctx_t *ctx) { (void)ctx; return 0; }
