#ifndef HA_GPU_ED_SEMI64_CUH
#define HA_GPU_ED_SEMI64_CUH

// Port of ed_band_cal_semi_64_w_absent_diag (Levenshtein_distance.h): Myers/Hyyro
// bit-vector banded edit distance, semi-global (text t fully aligned, pattern p ends
// free), band 2*thre+1 <= 64 bits, no traceback.
//
// Differences from the CPU version:
//   - sequences are 2-bit codes (0..3, 4 = N) rather than ASCII, so there is no
//     seq_nt4_table lookup. Convert with ha_nt4() on the host.
//   - the algorithm is a template over the sequence accessors P/T: anything with
//     operator[](int32_t) -> uint8_t. ed_replay.cu uses plain pointers; ed_batch.cu
//     decodes straight from the GPU-resident 2-bit reads.
//   - results are returned through out params instead of bit_extz_t. The CPU version
//     also sets ps=-1, ts=0, te=tn-1 and thre; those are constant per call.
//   - __host__ __device__ so the harness can run the same code on the CPU for timing.

#include <stdint.h>
#include <limits.h>

#ifdef __CUDACC__
#define HA_HD __host__ __device__ __forceinline__
#else
#define HA_HD static inline
#endif

#define ha_ed_core_64(Peq, VP, VN, X, D0, HN, HP, c) {  \
    (X) = (Peq)[(c)] | (VN);                            \
    (D0) = (((VP) + ((X) & (VP))) ^ (VP)) | (X);        \
    (HN) = (VP) & (D0);                                 \
    (HP) = (VN) | ~((VP) | (D0));                       \
    (X) = (D0) >> 1;                                    \
    (VN) = (X) & (HP);                                  \
    (VP) = (HN) | ~((X) | (HP));                        \
}

// Accessors are taken by value on purpose: a by-reference struct with mutable state
// forces it into local memory on the GPU, a local copy can live in registers.
template <class P, class T>
HA_HD void ha_ed_semi64_absent_diag_t(P p, int32_t pn, T t, int32_t tn,
                                      int32_t thre, int32_t abs_diag, int32_t *o_err, int32_t *o_pe)
{
    int32_t res_err = INT32_MAX, res_pe = -1;
    uint64_t c, Peq[5] = {0, 0, 0, 0, 0}, VP = 0, VN, X, D0, HN, HP, mm;
    int32_t bd, i, err = abs_diag, i_bd, last_high = (thre << 1), tn0 = tn - 1, cut = thre + last_high;
    int32_t site, ai, uge = INT32_MAX;

    if ((pn > tn + cut) || (tn > pn + cut)) goto done;

    bd = ((thre << 1) + 1) - abs_diag; bd = ((bd <= pn) ? bd : pn); i_bd = abs_diag;
    for (i = 0, mm = (((uint64_t)1) << i_bd); i < bd; i++) {
        Peq[p[i]] |= mm; mm <<= 1;
    }
    i_bd = (thre << 1) - abs_diag; VN = (((uint64_t)1) << abs_diag) - 1;

    i = 0; Peq[4] = 0; mm = ((uint64_t)1 << (thre << 1));
    while (i < tn0) {
        ha_ed_core_64(Peq, VP, VN, X, D0, HN, HP, t[i]);
        if (!(D0 & 1ULL)) {
            ++err; if (err > cut) goto done;
        }
        Peq[0] >>= 1; Peq[1] >>= 1; Peq[2] >>= 1; Peq[3] >>= 1;
        ++i; ++i_bd; c = 4;
        if (i_bd < pn) c = p[i_bd];
        if (c < 4) Peq[c] |= mm;
    }
    ha_ed_core_64(Peq, VP, VN, X, D0, HN, HP, t[i]);
    if (!(D0 & 1ULL)) {
        ++err; if (err > cut) goto done;
    }

    site = tn - 1 - abs_diag;
    ai = pn - tn + abs_diag; i = 0;
    for (i = 0; site < 0 && i < ai; i++, site++) {
        err += ((VP >> i) & 1ULL); err -= ((VN >> i) & 1ULL);
    }
    if ((err <= thre) && (err <= res_err)) {
        res_err = err; res_pe = site;
    }
    site -= i;
    while (i < ai) {
        err += ((VP >> i) & 1ULL); err -= ((VN >> i) & 1ULL); ++i;
        if ((err <= thre) && (err <= res_err)) {
            res_err = err; res_pe = site + i;
        }
        if (i == thre) uge = err;
    }
    if ((uge <= thre) && (uge == res_err)) res_pe = site + thre;

done:
    *o_err = res_err; *o_pe = res_pe;
}

// Plain-pointer instantiation (used by ed_replay.cu).
HA_HD void ha_ed_semi64_absent_diag(const uint8_t *p, int32_t pn, const uint8_t *t, int32_t tn,
                                    int32_t thre, int32_t abs_diag, int32_t *o_err, int32_t *o_pe)
{
    ha_ed_semi64_absent_diag_t<const uint8_t *, const uint8_t *>(p, pn, t, tn, thre, abs_diag, o_err, o_pe);
}

// Same mapping as seq_nt4_table in htab.cpp: ACGT/acgt -> 0..3, U/u -> 3, else 4.
HA_HD uint8_t ha_nt4(char ch)
{
    switch (ch) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': case 'U': case 'u': return 3;
        default: return (uint8_t)(ch >= 0 && ch < 4 ? ch : 4);
    }
}

// One job = one (pattern, text) window pair. Sequences live in one concatenated
// 2-bit-code buffer; jobs hold offsets into it.
typedef struct {
    int32_t p_off, pn, t_off, tn, thre, abs_diag;
} ha_ed_job_t;

typedef struct {
    int32_t err, pe;
} ha_ed_res_t;

#ifdef __CUDACC__
__global__ void ha_ed_semi64_kernel(const uint8_t *__restrict__ seq, const ha_ed_job_t *__restrict__ jobs,
                                    ha_ed_res_t *__restrict__ res, int32_t n_jobs)
{
    for (int32_t j = blockIdx.x * blockDim.x + threadIdx.x; j < n_jobs; j += gridDim.x * blockDim.x) {
        ha_ed_job_t jb = jobs[j];
        int32_t err, pe;
        ha_ed_semi64_absent_diag(seq + jb.p_off, jb.pn, seq + jb.t_off, jb.tn, jb.thre, jb.abs_diag, &err, &pe);
        res[j].err = err; res[j].pe = pe;
    }
}
#endif

#endif // HA_GPU_ED_SEMI64_CUH
