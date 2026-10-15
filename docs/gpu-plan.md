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
needs `(q_s, t_s, strand, thre, aux_beg)` plus buffer offsets; both sequences come from
2-bit packed reads (originally planned device-resident, now streamed per batch, see
step 2+3), which also removes the 115M `recover_UC_Read_sub_region` decodes (2.7%).
Output per job: `(err, pe)`.

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
2. + 3. **Done: `--gpu` (`make GPU=1`).** Implementation in `gpu/ed_batch.{h,cu}`,
   hooked into `Correct.cpp` (`enum_hc_ed_wins_gpu`, `gen_hc_r_alin`,
   `gen_hc_r_alin_nec`, `align_hc_ed_post_extz`), `Assembly.cpp` (`ha_gpu_init` /
   `ha_gpu_new_round` per EC round / `ha_gpu_destroy`) and `CommandLines.cpp`.
   - Sequences are streamed with the jobs. A context carries the 2-bit words of its
     query read (once) and, per overlap, the forward-strand word range of the target
     that its windows read, copied straight from `read_sperate` (2 KB-ish per overlap,
     ~185 B per window measured), plus the N positions in that range from `N_site`. The
     device accessor reproduces `recover_UC_Read_sub_region` including reverse strand
     and N restoration; a job is 60 bytes of coordinates and buffer offsets. Nothing
     stays on the device between batches, so the read set can be any size.
   - Per read, `gen_hc_r_alin` enumerates every first-pass window of every overlap
     (same loop as `align_hc_ed_post_extz`), submits them as one context, and the
     unchanged CPU code then consumes `(err, pe)` from the result array instead of
     calling the kernel. Gap-fill, traceback, phasing and consensus are untouched.
   - Worker threads never call CUDA. A single service thread merges every context
     waiting in the queue into one launch (mean 1.3–10k jobs) and scatters results
     back; workers block on a condition variable. Two pinned/device staging slots
     (153 MB each) alternate, so the host-side gather and transfers of one chunk
     overlap the kernel of the previous one. This matters: per-read batches from
     16 streams do **not** overlap on the device (`gpu/ed_batchbench.cu`: 0.84 →
     1.06 M jobs/s going from 1 to 16 threads at batch 500, vs 3.9 M jobs/s at batch
     8000). `kthread` creates fresh pthreads per pass, so contexts come from a pool,
     not `thread_local`.
   - `HA_GPU_VERIFY=1` runs the CPU kernel as well for every window and aborts on the
     first `err`/`pe` mismatch with full coordinates.
   - Validation (chr11-2M, 11.1M windows per run): verify mode passes on HiFi, on reads
     with injected N bases, and in `--ont` mode; `*.p_ctg.gfa`, `ovlp.source.bin` and
     `ovlp.reverse.bin` are byte-identical to the CPU build (`ec.bin` differs by ~120
     bytes between any two runs, CPU or GPU — pre-existing nondeterminism).
   - Timing on the RTX 500 (16 SMs, WSL2), interleaved with CPU runs: steady-state EC
     rounds are at parity with 16 cores (round 2 +0.2 s, round 3 −0.4 s); the only net
     loss is the one-off 2.75 s CUDA context creation. Per-round GPU stats
     (`[M::ha_gpu_new_round] ... kernel X s, service Y s, worker wait Z thread-s`)
     show the device saturated: ~0.34 µs/window in situ, ~0.25 ms WSL2 overhead per
     launch, and every worker waiting ~0.5 s per round for results. A V100/A100 has
     ~6–10x the throughput and ~10x lower launch latency, which is where the predicted
     EC-round gain (~25–35%) should appear. `scripts/gadi/run_e1_sr_gpu.pbs` runs the
     E1 job with `--gpu` on `dgxa100`.
   - Next kernel-level wins if the Gadi run is GPU-bound: cheaper device accessor
     (hoist the word reload out of the per-base path), warp-level work for the
     `aux_beg`/tail loops, and overlapping worker wait with other reads
     (oversubscribing `-t` to ~2x cores is a zero-code approximation; `-t32` on 22
     cores was ~5% faster here).

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
  `post_rev` mutate read storage; streaming the sequences with each batch (rather than
  keeping them device-resident) means the GPU always sees the current `read_sperate`.

## 4. Hardware and expectations

Dev machine: RTX 500 Ada laptop GPU, 4 GB, sm_89, CUDA 12.6 at `/usr/local/cuda` (not on
`PATH`). Develop on the chr11 subset.

Device memory was the limit of the first (device-resident) design: the whole 2-bit read
set was uploaded each EC round. The first Gadi run (E1 incremental, `--ont -j`, gpuvolta
V100-32GB, 2026-10-14) died with `cudaMalloc` out of memory at the first upload, after
an hour of k-mer counting: the 3.36 G minimizers at w=51 imply ~115–125 Gbp of reads,
i.e. 29–31 GB of 2-bit sequence against 31.7 GB on the card. The streamed design
replaced it on 2026-10-15: device memory is now a fixed ~310 MB of staging whatever the
input, at the cost of ~185 B of sequence per window over PCIe (0.9 GB per round on
chr11; ~0.5 TB per round projected for 40x human, overlapped with the kernel). On the
RTX 500 the per-round service time rose ~25% (host gather) with kernel time unchanged;
the resident version is in git history if a measurement on an A100 ever shows the
transfer mattering. The per-round stats line reports the staged volume
(`X GB of sequence staged`) and `N contexts (M free)`, which should stay at the thread
count (a climbing number means a caller is not releasing its context).

Amdahl: with 16 CPU threads the DP is already parallel, so T1 alone should cut
EC-round time by roughly 35–45%; T1+T2+T3 together is where a 2–3x end-to-end EC gain
lives. Index building (T3) is the stage with the worst CPU parallelism today.
