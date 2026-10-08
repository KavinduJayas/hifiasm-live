#ifndef HA_ED_DUMP_H
#define HA_ED_DUMP_H

// Records the inputs and results of ed_band_cal_semi_64_w_absent_diag calls so the
// GPU port in gpu/ can replay them (gpu/ed_replay.cu). Enabled at runtime by setting
// HA_ED_DUMP=<file>; HA_ED_DUMP_MAX caps the number of records (default 1,000,000).
// Off by default: one predictable branch per call.
//
// File format: 8-byte magic "HAED0001", then per record six int32
// (pn, tn, thre, abs_diag, err, pe) followed by pn bytes of pstr and tn bytes of tstr.
//
// Implementation is static; include from exactly one translation unit (Correct.cpp).

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>

typedef struct {
    FILE *fp;
    pthread_mutex_t lock;
    long max, n;
    int state; // -1 unset, 0 disabled, 1 recording, 2 cap reached
} ha_ed_dump_t;

static ha_ed_dump_t ha_ed_dump_g = {NULL, PTHREAD_MUTEX_INITIALIZER, 0, 0, -1};

static inline void ha_ed_dump_semi64(const char *p, int32_t pn, const char *t, int32_t tn,
                                     int32_t thre, int32_t abs_diag, int32_t err, int32_t pe)
{
    ha_ed_dump_t *d = &ha_ed_dump_g;
    if (d->state < 0) {
        pthread_mutex_lock(&d->lock);
        if (d->state < 0) {
            const char *fn = getenv("HA_ED_DUMP"), *mx = getenv("HA_ED_DUMP_MAX");
            d->max = mx ? atol(mx) : 1000000;
            d->fp = fn ? fopen(fn, "wb") : NULL;
            if (d->fp) fwrite("HAED0001", 1, 8, d->fp);
            d->state = d->fp ? 1 : 0;
        }
        pthread_mutex_unlock(&d->lock);
    }
    if (d->state != 1) return;
    pthread_mutex_lock(&d->lock);
    if (d->state == 1) {
        int32_t h[6] = {pn, tn, thre, abs_diag, err, pe};
        fwrite(h, sizeof(int32_t), 6, d->fp);
        fwrite(p, 1, pn, d->fp);
        fwrite(t, 1, tn, d->fp);
        if (++d->n >= d->max) { fclose(d->fp); d->fp = NULL; d->state = 2; }
    }
    pthread_mutex_unlock(&d->lock);
}

#endif // HA_ED_DUMP_H
