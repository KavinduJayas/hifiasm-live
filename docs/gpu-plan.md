# GPU acceleration plan for hifiasm (KavinduJayas fork, `dev`)

Status: analysis complete, kernel port in progress. Measurements below are from a
single-threaded gprof run on the upstream `chr11-2M.fa.gz` test set (120 s wall), plus a
16-thread run (23.7 s wall) for stage timings. Line numbers refer to the `dev` branch at
commit `35f36c1`.

## 1. Where the time goes

| Share of CPU | What | Where |
|---|---|---|
| ~46% | Base-level alignment. `ed_band_cal_semi_64_w_absent_diag` alone is 37.7% (11.0M calls); `ed_band_cal_global_64_w_trace` 5.4%; `gen_trace`, cigar extraction, error estimation and per-window target decode make up the rest | `Correct.cpp:12977`, `Correct.cpp:16178` |
| ~18% | Minimizer sketching: `mz1_ha_sketch` 14.6% + `ha_ft_cnt` 3.3% (706M filter-table lookups) | `sketch.cpp:454` |
| ~9% | Per-read anchor radix sort `rs_sort_ha_an1` | `anchor.cpp:1046` |
| ~4% | Chaining DP `lchain_qdp_mcopy_fast` | `Hash_Table.cpp:2097` |
| rest | Phasing, consensus, string graph — serial, irregular | |

Stage timings at 16 threads: k-mer filter + first index ~4.9 s (1.5–3x CPU, poorly
parallel), three EC rounds ~15 s (10–12x CPU), final overlap ~1.3 s, graph <1 s.

Facts that shape the plan:

- The final overlap pass (`ha_ec_ff` → `cal_ov_r` → `worker_hap_dc_ec_gen_new_idx`) does
  no DP: it chains and then verifies with `memcmp` (`exact_ec_check`). All DP is in EC.
- `ksw2` is not compiled. `POA.cpp` is dead on the default HiFi path (live only for
  `--ul` and `--sec-in`). `number_of_pround` is hard-coded to 0, so `worker_hap_dc_ec`,
  `worker_hap_dc_ec0` and `update_scb0` never run.
- The position index is rebuilt 4 times per run (3 EC rounds + final) and every build
  sketches every read twice (count pass, then fill pass).

## 2. Targets, ranked

### T1. First-pass window edit distance (do first)

`align_hc_ed_post_extz` (`Correct.cpp:12951-13010`) loops overlap → window and calls
`ed_band_cal_semi_64_w_absent_diag(t, t_l, q, q_l, thre, aux_beg, exz)` once per window.

Problem shape with HiFi defaults (`WINDOW_HC`=775, `max_ov_diff_ec`=0.04,
`THRESHOLD_MAX_SIZE`=31): query window ≤775 bp, `thre` ≤31, band 2·31+1 = 63 bits —
exactly one 64-bit word — and the target slice is `q_l + 2·thre` ≤ 837 bp. Per read per
round: ~43 candidate overlaps × ~11 windows ≈ 490 independent DP jobs. ONT:
`WINDOW_OHC`=375, e=0.07, thre=26.

Why it ports cleanly: the kernel (`Levenshtein_distance.h`, `HA_ED_INIT(64)` family) is
register-only Myers/Hyyrö bit-vector edit distance with no allocation, no SIMD, and one
global (`seq_nt4_table` → `__constant__`). One thread per window. Per job the host only
needs `(x_id, y_id, q_s, t_s, strand, thre, aux_beg)` ≈ 24 B; both sequences come from
GPU-resident 2-bit packed reads, which also removes the 115M `recover_UC_Read_sub_region`
decodes (2.7%). Output per job: `(err, ps, pe, ts, te)`.

Control-flow change needed: today each window's result is consumed immediately
(`is_align` → `push_hc_wlst_exz` gap-fill + traceback). Split into: (a) collect all
windows for a batch of reads, (b) GPU batch distance, (c) CPU gap-fill/traceback only for
windows that passed. The `_w_trace` variants `kv_resize` a 5-word-per-column path
(~33 KB per HiFi window) so traceback stays on the CPU in v1.

Steps:
1. Standalone `.cu` port of `ed_band_cal_semi_64_w_absent_diag` + record/replay harness
   (`gpu/`). Record real jobs from a CPU run (`HA_ED_DUMP=file`), replay on GPU, diff
   `err/pe`, measure throughput. **Done.** Result on 300k recorded chr11 jobs
   (77.5% aligned, max pn=837 / tn=775 as predicted): 0 mismatches on both the host
   build of the port and the GPU. CPU original 3.57 µs/job (1 thread); RTX 500 kernel
   0.185 µs/job = 5.4 M jobs/s = 3.9 G columns/s (19x one core, ~parity with 16
   cores), 0.45 µs/job including transfer of the raw sequences. The transfer cost
   disappears once reads are GPU-resident (step 3). Kernel is memory-access bound
   (one uncoalesced byte load per column per thread); next kernel-level win is packed
   2-bit loads (16 bases per 32-bit word) and/or staging windows in shared memory.
2. Batch collection in `gen_hc_r_alin`/`align_hc_ed_post_extz`: emit jobs into a
   per-thread queue, flush to GPU per `REFRESH_N`-sized read batch, resume CPU
   post-processing from results.
3. GPU-resident 2-bit reads (`R_INF.read_sperate` concatenated + offsets) so jobs carry
   coordinates, not bases.

### T2. Chain-gap alignment (`hc_aln_exz_adv_hc`)

151M calls from `hc_ovlp_base_direct` (`Correct.cpp:17473-17502`), one per
anchor-to-anchor segment. 94M are a `memcmp` (`cal_exact_exz`); 14M are real DP with
traceback and data-dependent retry thresholds (×1, ×e_rate, ×2, ×0.51, `maxe`), lengths
up to `MAX_SIN_L`=10000 and up to 64 words (`_infi_`). ~8% of time. Divergent: use
warp-per-segment, compute the final threshold in one pass instead of retrying.

### T3. Minimizer sketching and index lookup (~18%)

`mz1_ha_sketch` is minimap2-style with HPC and a per-k-mer filter lookup. Per-read
sketching is embarrassingly parallel and has known GPU implementations. The 4096-shard
khashl position index (`htab.cpp:303-314`) is already CSR-like (prefix-sum offsets into
contiguous `ha_idxpos_t` arrays), so `ha_pt_get` is a hash probe + contiguous gather.
CPU-only quick win first: cache each read's sketch between the count and fill passes of
`ha_pt_gen`.

### T4. Anchor sort and chaining (~13%)

Per-read radix sort of ~10^4 `anchor1_t` → CUB segmented sort. Chaining is
minimap2-style O(n·h) with `max_iter`=5000, `max_skip`=25; branchy fast path
(`quick_ck_lchain`) and multi-copy backtracking. Do after T1–T3.

### Not worth porting

`rphase_hc` (phasing), `wcns_gen`/`cns_gen_full0` (consensus) and the whole graph stage
(`build_string_graph_without_clean` → `clean_graph`) are serial CIGAR-walking and
pointer-chasing graph code. The only data-parallel pieces there are
`hap_alignment_advance_worker` in purge_dups (O(N²) from a per-unitig `memset`,
`Purge_Dups.cpp:3008`) and Hi-C read mapping (`worker_for_alignment`).

## 3. Fork-specific constraints

- `mark_hc_ovlp_dirty` (`ecovlp.cpp:2838-2853`) does a non-atomic read-modify-write of
  `dirty_reads[other]` from multiple threads. Needs `__sync_fetch_and_or` on CPU,
  `atomicOr` on GPU, or a deferred mark list.
- Likely bug: `ecovlp.cpp:4044` tests `!R_INF.dirty_reads[i]` without the `&0x3F` mask
  used at `ecovlp.cpp:3999`; bits 6–7 hold the last-corrected round, so a clean read
  with those bits set skips `reverse_non_dirty_ovlps`.
- In `-j` / `--keep-alive` mode each pass only visits new + dirty reads, so GPU batches
  must span many reads to fill the device. `kt_for_ec` already has the new-reads-then-
  dirty-reads ordering that a batched design must preserve.
- Pass ordering to keep: hap_ec → sl_ec → update_dc_ec → post_rev. `sl_ec` and
  `post_rev` mutate read storage, so GPU-resident reads must be re-uploaded (or updated
  in place) after each.

## 4. Hardware and expectations

Dev machine: RTX 500 Ada laptop GPU, 4 GB, sm_89, CUDA 12.6 at `/usr/local/cuda` (not on
`PATH`). A human-scale 2-bit read set (~25 GB) is not resident there; develop on the
chr11 subset and stream target windows. Production needs a 40–80 GB card.

Amdahl: with 16 CPU threads the DP is already parallel, so T1 alone should cut
EC-round time by roughly 35–45%; T1+T2+T3 together is where a 2–3x end-to-end EC gain
lives. Index building (T3) is the stage with the worst CPU parallelism today.
