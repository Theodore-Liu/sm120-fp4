# Backlog

Ranked work for the stages in `PLAN.md`. The top item is always in progress; when an item closes, the next one starts
and the list is refilled from the stage gates, so the queue is never empty. Each item names the measurement that
closes it. Dates are when an item was added, not estimates.

## In progress

3. **Stage 3 survey: the three FP4 sites DeepGEMM routes to `tcgen05`** (2026-10-02). Read the current DeepGEMM
   sources for the FP8xFP4 GEMM, the FP4 attention path and the FP4 einsum: their interfaces, tile shapes, the
   `tcgen05`/TMEM features they depend on, and what SM120 lacks (no TMEM, 99 KB shared memory, `mma.sync` only).
   Closes with `docs/stage3-survey.md` naming, per site, the SM120 design and the first kernel to write.
   State 2026-10-02: the survey is written against DeepGEMM `057ca5964aae` (the einsum site is FP8, not FP4;
   tiles, stages, recipe defaults confirmed), the first kernel and its interface are named, and the UE8M0 reference
   (`scripts/ue8m0_reference.py`) matches DeepGEMM's helpers bit for bit. The first kernel exists at its minimum
   (`scripts/fp8_fp4_gemm_sm120.py`: one warp per 16 x 8 tile, M <= 16, gran_k 128) and is correct against the
   reference to bf16 output rounding on five shapes; the MMA's fragment layout and e2m1 container convention were
   measured (`scripts/probe_f8f6f4*.py`: the code sits in bits 5:2 of its byte, not the low nibble). The tiled v1
   (32 x 128 tile, 4-stage cp.async, 8 warps) is correct and bit-identical to v0, and runs at 108 to 360 GB/s on the decode shapes,
   5.0 to 16.5 times the byte floor (`reports/fp8-fp4-gemm-v1-rtx5090-20261002.json`): the grid is too small
   (N/128 blocks). Split-K v2 (`reports/fp8-fp4-gemm-v2-rtx5090-20261002.json`) is 1.7 to 3.5 times v1 at 338 to
   633 GB/s, 2.8 to 5.3 times the floor, bit-identical to v1 except one element one ulp off; the grid is
   no longer the bound. The swizzled A rows add 10 to 17 percent on the K = 7168 shapes (K permutation the same, not additive;
   one barrier per stage within noise), best 743 GB/s, 2.4 to 4.9 times the floor
   (`reports/fp8-fp4-gemm-v3-rtx5090-20261002.json`). Two stages instead of four (four blocks per SM) add 30 to 34 percent on
   the N = 7168 shapes, pairs of blocks 21 to 24 and not on top; best 852 GB/s, 2.1 times the floor on M16
   N7168 (`reports/fp8-fp4-gemm-v4-rtx5090-20261002.json`). The N = 2048 shapes are bound by two launches and the 22-way
   reduce at 17 to 19 us: the reduce alone is 41 to 47 percent of the whole, half the splits gain 9 to 12.5 percent there, and
   a last-block fused reduce is 29 to 42 percent slower because it is latency-bound on N/BN blocks
   (`reports/fp8-fp4-gemm-v5-rtx5090-20261003.json`). The fused reduce with float4 loads is still 19 to 34 percent slower, so
   the pattern is closed for decode shapes; the planner at one block per SM gains 11 to 13 percent on the 2048-wide shapes and
   loses on none (`reports/fp8-fp4-gemm-v6-rtx5090-20261003.json`); v7 makes it the two-stage default (+2.2 to +13.3 percent
   on four shapes, -1.6 on one; 454 to 897 GB/s; `reports/fp8-fp4-gemm-v7-rtx5090-20261003.json`). The MQA-logits kernel's
   v0 (`scripts/fp8_fp4_mqa_logits_sm120.py`) is correct against DeepGEMM's test reference on six shapes (relative error at
   most 1.5e-7) and slow by design (one warp per query row: 0.4 to 8 TFLOP/s; `reports/fp8-fp4-mqa-logits-v0-rtx5090-20261003.json`);
   v1 stages a 256-row kv segment in shared memory for sixteen query rows and is bit-identical to v0; its first report's timings
   included two host syncs and are superseded (re-timed: 25 to 32 us, 2.9 to 175 times v0). v2 adds a tile rule (64-row segments,
   8 or 16 rows) and a double-buffer arm: 7.6 to 29.3 us on the four shapes, 1.1 to 3.3 times v1, 18 to 147 TFLOP/s, bit-identical
   (`reports/fp8-fp4-mqa-logits-v2-rtx5090-20261003.json`). v3 is the paged form (block tables, context lengths; one page per
   warp per step), bit-identical to v0 on the flat layout through random page permutations, 13 to 32 us on three decode shapes
   at 0.7 to 1.1 TB/s of kv rows (`reports/fp8-fp4-paged-mqa-logits-v3-rtx5090-20261003.json`). Both indexer forms now exist
   for SM120. The einsum site is read (survey 2.3): DeepGEMM's `fp8_einsum` covers four expressions, FP8 on both operands, no FP4;
   `bhr,hdr->bhd` is a per-head GEMM with per-token and per-block scales, so its SM120 v0 is the GEMM's block with an e4m3 B
   operand; v0 is done (`scripts/fp8_einsum_sm120.py`: six shapes within two bf16 half-ulps of torch.einsum on the
   dequantised operands; 48 to 256 us, 709 GB/s at B 8 falling to 156 at B 128 because y is re-read per 16-row tile;
   `reports/fp8-einsum-v0-rtx5090-20261003.json`). Every tcgen05 kernel family now has a correct SM120 form here. The einsum's tiled v1 (y and x staged in shared memory per
   128-R block, 8 warps x 128 columns x 128 b rows) is bit-identical to v0 and 1.33 times faster at B 128, where v0 re-read y eight
   times, but 7 to 9 percent slower at B 8 and 32 because its grid is 64 blocks (`reports/fp8-einsum-v1-rtx5090-20261003.json`);
   the next step there is a 64-column tile or a split over R. Adoption (items 4 to 6) now comes before further kernel tuning.
   Done 2026-10-06: v2 is v1 with a 64-column d tile and a 64-row b chunk (8 warps x 8 columns, 36.9 KB of shared memory, so two blocks
   per SM and twice v1's grid along d), each element's accumulation order unchanged: bit-identical to v0 on all seven selftest shapes;
   over three runs 31.5 to 33.4 / 42.7 to 43.6 / 86.8 to 86.9 us at B 8 / 32 / 128 (H 8, D 1024, R 4096) against v1's 52.0 to 52.1 /
   76.5 to 76.7 / 193.2 to 193.3 (1.56 to 1.79 times, and 2.23 times at B 128, where v2 reads y once per 64-row chunk but keeps
   twice the blocks resident) and v0's 47.9 to 49.8 / 66.6 to 68.4 / 258.9 to 261.2 (`reports/fp8-einsum-v2-rtx5090-20261006-run1..3.json`);
   the form to use at every measured batch; y, the weights, is shared across b rows by the operation itself, so its L2 reuse is real here.
   The other benchmarks were checked for the shared-data problem the paged ones had: the flat indexer and the GEMMs read data every row
   shares by definition, so only the paged benchmarks needed a cache per row.
   The einsum had no pytest test, so the nightly CI never ran it: tests/test_einsum.py (2026-10-06) checks v0, v1 and v2 on five shapes
   (B 1 to 200, including B not a multiple of 16 or 64) against torch.einsum on the dequantised operands and v1 and v2 bit for bit
   against v0; 15 pass. Nsight Compute on v2 at B 128 (reports/ncu-einsum-v2-b128-rtx5090-20261006.txt): 256 blocks on 170 SMs at two
   per SM leave about half the SMs with one block (theoretical occupancy 33 percent, achieved 27), DRAM throughput 25 percent, L2 hit
   82 percent, long scoreboard 4.19 cycles per issue from the per-kb x scale read inside the compute loop; next, measured one at a time:
   a 32-row b chunk at large B (twice the blocks, y read four times from L2) and the x scales read ahead of the k32 steps.
   The 32-row chunk first (v3, 2026-10-06: v2's template at TB 32, 27.6 KB; bit-identical, 20 tests pass): over three runs 31.5 to 32.4 /
   41.7 to 42.7 / 93.2 to 94.0 us at B 8 / 32 / 128 against v2's 33.6 to 34.0 / 41.7 to 42.7 / 87.5 to 87.6 in the same runs
   (reports/fp8-einsum-v3-rtx5090-20261006-run1..3.json): 1.04 to 1.08 times at B 8, level at B 32, 1.07 times slower at B 128, so
   balancing the grid at B 128 costs more in re-read y than it gains; v2 stays the default, v3 kept for the measurement; next the
   x scales read ahead of the k32 steps, on v2.
   v4 (2026-10-06: v2 with every m tile's x scales for the k block loaded at the top of the k-block iteration; bit-identical, 25 tests
   pass): over three runs 35.6 to 37.0 / 43.3 to 43.8 / 86.8 to 87.0 us against v2's 35.5 to 35.6 / 41.7 / 88.7 to 88.8 in the same
   runs (reports/fp8-einsum-v4-rtx5090-20261006-run1..3.json): level, 1.04 to 1.05 times slower, 1.02 times faster; the scale
   reads the profiler showed as long-scoreboard stalls are not what bounds the kernel, a recorded negative; v2 stays the default.
   v2's own times drift between sessions (B 8: 31.5 to 34.0 earlier today, 35.5 to 35.6 here), so variants are compared within
   one run only.

## Ready, in order

5. **Adoption 2: one stage-3 kernel inside an engine on a model that runs on an SM120 card** (second). The kernels are correct
   and measured but nothing calls them; pick the path a RTX PRO 6000 can hold (an FP4 checkpoint whose engine needs the FP8 x
   FP4 GEMM or the MQA-logits indexer), wire the kernel in behind a flag, and measure end to end against the engine's own path,
   as stage 2 was. Until this, stage 3 is a reference implementation, not a product.
   State 2026-10-03: in progress, the survey first (`docs/stage3-engine-wiring.md`). vLLM 0.28 calls the kernels at five sites:
   the sparse-attention indexer (`sparse_attn_indexer.py:500` prefill, `:603` decode; our MQA-logits v2 and v3), the FP4 MoE
   experts (`deep_gemm_moe.py:596, 616`, grouped; our GEMM needs a grouped driver), MegaMoE (SM100 only by construction) and the
   V4 `o_proj` FP8 einsum. No shipped checkpoint smaller than DeepSeek-V4-Flash (284B, NVFP4 about 165 GB) exercises the
   indexer, so the end-to-end host is two RTX PRO 6000 at TP=2, the configuration the public record runs. Order: (1) unit wiring
   of the indexer into vLLM's own code path on the 5090 behind `SM120FP4_INDEXER=1`, (2) the FP8-q format first (MXFP4 q needs
   an FP4 x FP4 instruction form, measured 2026-10-04: `scripts/probe_f4f4.py`, both `kind::f8f6f4` e2m1 x e2m1 and the packed
   `kind::mxf4.block_scale` with its scale-register map, wiring doc step 2; the MXFP4-q kernel v6 on it is correct on five shapes
   and 1.29 to 1.65 times v4 (6.9 / 17.2 / 19.2 us), its paged form bit-identical to it and 1.18 to 1.23 times the paged v4; an 80-byte stride on the flat v6 removes its bank conflicts
   and leaves its time unchanged (closed); Nsight Compute puts the flat v6's remaining time in the per-tile epilogue (shuffle reduction and
   predicated writes; no single stall reason above 2.4 of 13.1 cycles per issue); the packed epilogue v6e (3 shuffles per tile instead of 6) is
   bit-identical and 1.11 to 1.14 times v6 on the two larger shapes; in the paged v6 the same epilogue is bit-identical and 1.05 times on
   the two larger shapes; the paged kernel's time is in waiting for its own page (long scoreboard 5.92 of 11.6 cycles per issue), a warp that walks
   several pages double-buffered (v6f) is bit-identical but slower or level at every group size: measured, the prefetch cuts the page wait from 5.91 to
   1.69 cycles per issue but halves occupancy (16.7 against 33.3 percent); half-page double buffers at v6e's occupancy (v6g) are bit-identical
   but slower on the two larger shapes (45.8 / 43.8 against 43.8 / 40.0 us): measured, the reloads of q, weights and scales per half add 19
   percent instructions and 48 percent global load requests, outweighing the shorter wait; closed; v6h loads them once per head block and keeps
   the overlap: bit-identical, 1.04 times v6e on S 64 N 32768 H 8 in three of three runs, level on the other two shapes; Nsight Compute puts
   its remaining limit at occupancy (5.4 KB of shared memory per warp, 33 percent theoretical); quarter-page double buffers (v6i, 2.7 KB per
   warp) are bit-identical and 1.19 to 1.24 times v6e over three runs (8.9 / 35.6 / 33.5 us), the paged form to use; Nsight Compute on S 64 N 32768 H 8 (2026-10-05, reports/ncu-paged-v6i-rtx5090-20261005.txt): achieved occupancy 41.6 percent against 66.7 theoretical (v6e 20.7 against 33.3), registers and shared memory each limit it to four blocks, so cutting shared memory alone buys nothing; 4080 blocks are exactly six waves of 680, so the achieved gap is not a wave tail; 18.8 million instructions against v6e's 14.0 (34 percent more), long scoreboard 4.39 against 5.65 and short scoreboard 2.07 against 1.07 cycles per issue, SM throughput 41 and DRAM 8 percent; per-opcode counts (reports/ncu-paged-v6i-opcodes-rtx5090-20261006.txt) put all 4.8 million extra instructions in integer and control work, none in the MMA: IMAD +0.91, BRA +0.51, ISETP +0.41, BSSY/BSYNC +0.46, S2R +0.23 million, from masks, bounds and a rolled tile loop recomputed every quarter; v6j (2026-10-06) runs a full quarter's two tiles unrolled and unmasked, the tail quarter alone masked, staging unrolled: bit-identical on the four selftest shapes and at H 24 and 32, 14.2 million instructions (v6i 18.8, v6e 14.0), 62 registers and four blocks per SM as v6i (reports/ncu-paged-v6j-opcodes-rtx5090-20261006.txt); over three runs (reports/bench-paged-v6j-rtx5090-20261006-run1..3.json) 8.9 / 33.8 to 35.2 / 31.5 us against v6i's 8.9 / 35.5 to 35.6 / 33.5 (level, 1.01 to 1.05, 1.06 times), the paged form to use: a quarter fewer instructions buys at most 6 percent because the kernel waits on memory; Nsight Compute (reports/ncu-paged-v6j-rtx5090-20261006.txt) shows long scoreboard 7.49 cycles per issue against v6i's 4.39, cycles per issued instruction 17.7 against 13.1, achieved occupancy unchanged at 41.7 percent of 66.7; measured on the CPU from the benchmark's context lengths (scripts/paged_grid_occupancy.py, reports/paged-grid-occupancy-20261006.json; max_pages checked against the profiled grid, 510): 3.0 / 8.1 / 4.6 percent of blocks have no live warp, and the others average 4.85 / 4.74 / 4.15 live warps of 8, so a block slot runs at about 0.61 / 0.59 / 0.52 of its warps; on S 64 N 32768 that estimate (0.59) is close to the measured achieved over theoretical occupancy (41.7 of 66.7, 0.625), so the gap is the grid, not the scheduler; v6k (2026-10-06) compacts the (row, page) list: warp w takes the w-th live pair by binary search over the inclusive prefix sum of the rows' page counts, the grid sized by S x max_pages with no host sync; bit-identical on the four selftest shapes and at H 24 and 32; first timed with the prefix sum inside the call it was slower than v6j on every shape (19.2 / 39.7 / 37.6 against 8.9 / 35.4 / 31.5 us, reports/bench-paged-v6k-inline-meta-*), and Nsight Compute (reports/ncu-paged-v6k-rtx5090-20261006.txt) put the cause outside the kernel: the kernel alone 31.7 us against v6j's 35.8 with achieved occupancy 56.3 against 41.5 percent, the rest five small launches (clamp, divide, add, scan init, scan); with the list moved to a metadata call computed once per batch (fp4_fp4_paged_mqa_logits_sm120_v6k_meta), three runs give 8.9 / 30.8 to 31.3 / 27.4 to 27.5 us against v6j's 8.9 / 34.9 to 35.5 / 31.5 (level, 1.12 to 1.13, 1.15 times; reports/bench-paged-v6k-rtx5090-20261006-run1..3.json) and the metadata call 11.0 to 11.5 us; the metadata is now one single-block scan kernel (k_v6k_meta: warp-shuffle scans in chunks of 1024 rows with a carry), equal element for element to torch.cumsum of the clamped page counts at S 1, 33, 1024, 1025, 3000 (clamped to 4 pages) and 5000; over three runs it costs 2.8 us per batch (3.0 to 4.3 on a first call) against the five launches' 11.0 to 11.5 (reports/bench-paged-v6k-meta-rtx5090-20261006-run1..3.json), and the v6k kernel in the same runs 8.9 to 9.0 / 29.5 to 31.5 / 27.7 to 29.2 us against v6j's 8.9 / 35.5 to 35.6 / 31.4 to 31.5, so a single call with its own list is about level on the smallest shape (8.9 plus 2.8) and still ahead on the two larger ones; the engine wiring computes the list once per step and shares it across layers; Nsight Compute on v6k (reports/ncu-paged-v6k-stalls-rtx5090-20261006.txt): achieved occupancy 55.1 percent of 66.7, the largest stall 7.8 cycles per issue on L1TEX scoreboard (global loads), memory pipeline busy 75 percent, L2 hit 95.7 percent, 28.6 active threads per warp; one reading was the dependent prologue each one-page warp runs (a binary search over the prefix sum, then the context length and the block table) before its first copy, so v6l writes the list out as packed (row << 16 | page) entries after the scan (one block per row fills them) and a warp reads its pair and the live total with two independent loads: bit-identical on seven shapes, over three runs 8.9 / 29.4 / 27.4 us against v6k's 8.9 / 30.4 to 31.5 / 27.4 to 28.9 (reports/bench-paged-v6l-rtx5090-20261006-run1..3.json), steadier and up to 1.07 times, with the list at 5.0 to 6.8 us per batch against 2.8; the prologue explains a little of the wait, not most of it; v6l is the form when one list serves every layer, v6k otherwise; v6m (2026-10-06) issues each head block's first quarter copy before the block's q fragments, weights and scales are read: bit-identical on seven shapes and level with v6l in all three runs (8.9 / 29.4 / 27.4 to 27.5 us, reports/bench-paged-v6m-rtx5090-20261006-run1..3.json), because the compiler had already scheduled it so: both kernels compile to 432 SASS instructions with the first LDGSTS at the same place, differing only in register numbers; a recorded negative, kept for the comparison; PC sampling on v6l (reports/ncu-paged-v6l-pcsamp-sass-rtx5090-20261006.csv, per SASS instruction, 1568 not-issued samples) puts 23 percent of the stalled samples on the prologue chain (the packed entry, then the context length and the block table: 8.9, 8.4 and 5.7 percent on the three instructions that consume them), 13 percent at the copy wait and about 15 percent on the epilogue's shuffle additions; v6n removes the chain (one 16-byte entry per work item holding row, page index, physical page and valid rows, written by the metadata call, which now reads the block table): bit-identical on seven shapes and level with v6l in all three runs (8.9 / 29.4 to 29.5 / 27.4 us, reports/bench-paged-v6n-rtx5090-20261006-run1..3.json), its list 4.9 to 7.0 us; so the stalled samples on the prologue are hidden by the other resident warps and are not on the kernel's critical path; a recorded negative; it is (reports/ncu-paged-v6l-memory-rtx5090-20261006.txt, S 64 N 32768 H 8): L2 throughput 78 percent of peak, L2 hit 96 percent, 115 MB read through L2 against the 77 MB the 17,775 live pages hold (1.49 times), an effective 2.53 TB/s, above the card's DRAM bandwidth; the reason is the benchmark: every row reads the same flat k through its own page permutation, so after the first reader a page is in L2, and the paged timings above measure an L2-resident read, not an engine's batch where each row has its own cache; the latency work on this benchmark is done; measured 2026-10-06 (`--bench-paged-distinct`: every row's pages copied to physical pages of their own, values unchanged, outputs bit-identical to the shared-cache run for every kernel; reports/bench-paged-distinct-rtx5090-20261006-run1..3.json): with no page shared, every variant is level, 15.2 to 17.2 / 78.6 to 80.6 / 72.0 to 72.4 us on the three shapes, v6e no slower than v6j, v6k or v6l (78.6 against 79.9 to 80.6 on S 64 N 32768), at about 950 GB/s of the rows' pages, close to half the card's DRAM bandwidth; so the v6i to v6l gains are an L2-resident effect and do not carry to an engine's batch, and the paged kernel's real limit is the bytes it keeps in flight per SM; measured first rather than assumed (reports/ncu-paged-distinct-v6l-*-rtx5090-20261006.txt, S 64 N 32768 H 8 on the distinct cache): DRAM throughput 81 percent of peak, L2 hit 36 percent, 97.2 MB moved through DRAM in 68.7 us under the profiler (about 1.41 TB/s, 79 percent of the card's 1.79 TB/s), of which about 8.4 MB is the logits written and about 89 MB read against the 77.4 MB the live pages hold (1.15 times; the q fragments, weights and scales each one-page warp reads are about 13 percent at H 8); so in an engine's batch the paged kernel is DRAM-bound near the practical ceiling, the 950 GB/s above counted only the rows' pages, and more bytes in flight would gain little; the remaining lever is the 15 percent of extra bytes (a warp covering several pages of one row would read q and the weights once, at an occupancy cost v6f measured), worth at most about a tenth; measured 2026-10-07 by adding v6f (two or four pages of one row per warp, q, weights and scales read once per warp) to `--bench-paged-distinct` (outputs bit-identical to the shared-cache run; reports/bench-paged-distinct-v6f-rtx5090-20261007-run1..3.json): G=4 17.2 / 80.6 to 80.9 / 73.7 to 74.5 us and G=2 19.2 / 80.6 to 82.5 / 74.5 to 75.7 against v6e's 17.1 to 17.2 / 78.4 to 78.7 / 72.4 in the same runs, level at best and up to 1.12 times slower, so the saved bytes do not pay for the lost occupancy; a recorded negative, and the paged fp4 kernel is done until an engine's batch shape says otherwise; the fp8 path's distinct-cache rerun (fp8_mqa_logits_v5_sm120.py --bench-paged-distinct, every output bit-identical to the shared-cache run; reports/fp8-paged-distinct-rtx5090-20261006-run1..3.json): v5r and v5s tie at 58.1 / 203.5 to 204.4 / 203.5 to 206.6 us, v5h 64.3 / 205.7 to 207.6 / 205.6 to 207.6, the full-page v5 64.2 to 64.3 / 209.7 to 212.2 / 213.0 to 214.8, at up to 1.36 TB/s of cache rows, near the DRAM ceiling; v5s's loss to v5r on the shared cache (70.5 against 62.4 us on S 128 N 16384) was an L2-resident effect and is gone; v5r stays the default (the simpler of the two tied forms); not wired while vLLM refuses the MXFP4 indexer cache on SM120), (3) the two-GPU end-to-end run, (4) the grouped GEMM after step 3 has a number.
   Second reading 2026-10-03 (wiring doc 3b): the engine's k operand is MXFP4 with a UE8M0 scale per 32 along the head, or e4m3 with an
   fp32 row scale; ours is e2m1 with one UE8M0 scale per 128-wide row. The adapter therefore waits on a kernel variant (step 0: the
   per-32 scale fold, q's scale read from `weights`), which is the next piece of work on this item.
   Step 0 done 2026-10-03: `scripts/fp8_fp4_mqa_logits_v4_sm120.py` (k scale per 32 columns; at most 1.7e-7 against the reference; bit-identical
   to v2 where the block scales coincide; level with or ahead of v2 on three shapes, `reports/fp8-fp4-mqa-logits-v4-rtx5090-20261003.json`).
   The paged form landed the same day (`fp8_fp4_paged_mqa_logits_sm120_v4`: bit-identical to the flat v4 through random page
   permutations, 23 to 75 us on three decode shapes, `reports/fp8-fp4-paged-mqa-logits-v4-rtx5090-20261003.json`). Step 0 is closed.
   Next: step 1, the adapter `sm120fp4/indexer.py` with vLLM's calling convention (the `(data, scale)` q pair with the scale in
   `weights`, the int8/int32 zero-copy views of k and its scales, 2-D `context_lens`, an ignored `schedule_metadata`, `clean_logits`,
   `indices`) bound into `vllm.utils.deep_gemm` behind `SM120FP4_INDEXER=1`, and the unit test through the engine's own wrapper.
   Correction 2026-10-04 (wiring doc 3c): vLLM refuses the MXFP4 indexer cache on SM120 (`dsa_indexer_uses_fp4`), so the path the
   engine runs here is FP8 k with an fp32 row scale; `scripts/fp8_mqa_logits_v5_sm120.py` is that form (flat; within 4.5e-7 of the
   reference, 10.5 / 33.5 / 37.5 us on the three shapes, `reports/fp8-mqa-logits-v5-rtx5090-20261004.json`). The paged v5 in the
   132-byte entry layout landed 2026-10-04 (bit-identical to the flat v5 through page permutations; 62 to 241 us on the three decode
   shapes, three times the paged v4 because the 132-byte rows are staged with 4-byte loads; `reports/fp8-paged-mqa-logits-v5-rtx5090-20261004.json`).
   Step 1 landed 2026-10-04: `sm120fp4/indexer.py` serves vLLM's three indexer entry points with v5 (flat logits expanded to the
   engine's [M, N]; decode batch flattened to one row per query; empty schedule metadata), bound by the plugin's `register()` under
   `SM120FP4_INDEXER=1`; the engine's fp32 `weights` cost 1.2e-3 to 2.8e-3 relative when read as bf16, so v5 gained an fp32-weights
   variant (within 3.0e-7) that the adapter uses by default (wiring doc 3d; `tests/test_indexer_adapter.py`, 9 tests, through
   `vllm.utils.deep_gemm`'s own wrappers). The three MQA-logits kernel modules and `ue8m0_reference.py` moved into
   `sm120fp4/kernels/` with shims in `scripts/`. Step 1 closed the same day at the kernel level: the engine's own top-k kernels
   over our logits select the same index sets as over the reference on every row (`tests/test_indexer_topk.py`, 6 tests; the
   indexer function itself reads the engine's forward context and is first called by step 3's run). The paged staging moved to
   16-byte lane-strided loads the same day: 33.6 / 115.5 / 123.7 us against 62.4 / 234.2 / 240.7, now 1.44 to 1.66 times the paged
   v4 (wiring doc 3e; a first 16-byte form with a lone 33rd load per group on lane 0 was slower and is recorded there). The pinned
   engine declares block 64 for the V3.2 indexer backend and 256 for V4's, the same 132-byte rows; the adapter now accepts any
   multiple of 64 and maps a block to consecutive 64-row pages (tested bit for bit with 256-row blocks against the flat call;
   wiring doc 3e). v5f (fp32 weights) costs nothing measurable against v5 (9.9 / 33.5 / 37.6 us against 11.0 / 33.5 / 37.6).
   The engine's operands were read against the adapter's assumptions (wiring doc 3f): q one e4m3 group per (token, head), weights
   fp32 carrying q's scale and both softmax factors, k e4m3 with an fp32 row scale, V4's decode lengths compressed; nothing
   contradicts the adapter. Next: step 3's two-GPU run (DeepSeek-V4-Flash NVFP4 at TP=2 on two RTX PRO 6000; the plan is
   `docs/stage3-step3-plan.md`), then the grouped GEMM.
   Paged v5 profile 2026-10-04 (wiring doc 3g): L2-resident, occupancy capped at one block per SM by the 67.6 KB staging; the
   lever is shared memory per warp. The half-page form (`fp8_paged_mqa_logits_sm120_v5h`: two 32-row halves per page, 33.8 KB per
   block) is in and bit-identical to the flat v5 on five shapes including a 70-row context; timed 2026-10-04 at 33.5 / 103.4 / 113.4 us
   against the full page's 33.6 / 115.5 / 124.9 (wiring doc 3h): ten percent on the two large shapes; the adapter's paged call now
   uses it. The register-prefetch arm (v5d) lost, 39.8 / 136.0 / 146.2, because 200 registers per thread put the occupancy back to one
   block per SM (wiring doc 3i); it stays as a recorded negative. The row-width levers are written up in `docs/stage3-paged-v5-row-width.md`
   (the two-heads-per-tile lever was withdrawn on rereading: the A fragment already holds sixteen heads; the row as two 64-byte
   halves on the v4 path remains; its first form, raw-row staging v5r, closed the gap: 21.2 / 62.4 / 72.5 us against v5h's 33.2 / 103.4 / 113.4, ahead of the paged v4; the adapter uses it; Nsight Compute: 2.2 times fewer instructions, 2.5 times fewer shared-load bank conflicts, same occupancy; the flat v5 has no scatter to remove). The fp4 path's quarter-page double buffers ported to the fp8 rows (v5s, 2026-10-06: raw rows in quarters of 16 with cp.async double buffering in v5r's 4.2 KB per warp, q loaded once per head block) are bit-identical to the flat v5 on seven shapes but lose on S 128 N 16384 H 8 (70.4 to 70.6 against 62.4 to 63.1 us over three alternating runs), tie on S 64 N 8192 and win only on S 32 N 65536 H 16 (67.9 to 68.0 against 72.5), the one shape with two head blocks; Nsight Compute on the losing shape (reports/ncu-fp8-paged-v5-{raw,quarter}-s128-rtx5090-20261006.txt) shows the same instruction count (33.8 against 33.3 million), the same occupancy limit (two blocks per SM by shared memory) and the same shared-load bank conflicts, and the difference in the memory wait (long scoreboard 4.41 against 2.87 cycles per issue); our reading, not separately tested, is that at a fixed 4.2 KB per warp the double buffer keeps half as many bytes in flight per warp as v5r's one 4.2 KB batch, and the overlap does not pay for it unless q is reused across head blocks; v5r stays the default. The adapter's
   paged default, v5r since 2026-10-04 (sm120fp4/indexer.py calls it with raw=True; this sentence read v5h until 2026-10-07), is covered by the 64-row and 256-row block tests already (18 passing).
6. **Adoption 3: minimal CI** (third). The selftests (`fp8_fp4_gemm_sm120.py`, `fp8_fp4_mqa_logits_sm120.py`,
   `fp8_einsum_sm120.py`, `tests/test_vllm_backend.py`) run on every push on a self-hosted SM120 runner (this machine's WSL, as a
   scheduled task that polls), with the result badge in the README. Nobody depends on a kernel library whose tests only its
   author runs.
   Plan 2026-10-04 (`docs/ci-plan.md`): start as a nightly scheduled task on this machine that runs the suite in a dedicated
   checkout, writes `reports/ci/<date>.json` and pushes it (no new credential, no runner process); a self-hosted Actions
   runner when the repository has a second contributor. Landed 2026-10-04: `scripts/ci_nightly.py` (the dedicated checkout reset to
   origin/master, the GPU-idle check, pytest with junit, `reports/ci/<date>.json`, the README's CI status line, `--push` from the
   working copy), scheduled task `Sm120Ci` at 05:30 daily; the first hand run: 67 passed, 0 failed, 15 skipped (the checkpoint-bound
   tests without the shard) in 11 s on f033af8. The 15 skips are `tests/test_mm_fp4.py` on FlashInfer's `mm_fp4` backends (cuDNN, TensorRT-LLM, CuTe DSL), which the
   pinned FlashInfer 0.6.16.post3 declines on this device; that is stage 1's finding, not a CI defect, and the report now
   groups the skip reasons so a reader sees it. A self-hosted runner remains the step after.
   Gap closed 2026-10-06: the item names the stage-3 selftests, but the nightly runs `pytest tests/` only, so until today none of
   them ran in CI. tests/test_einsum.py (the einsum, v0 to v4, 25 cases) and tests/test_kernel_selftests.py (each module's own
   `main(["--selftest"])`: the UE8M0 reference, the FP8xFP4 GEMM, the indexer v0 to v3, v4, the fp4 v6 family with its paged forms,
   and the fp8 v5 family; 6 cases, 129 checks printed ok in one run, 4 s with the extensions cached) now carry them into the
   suite the nightly runs.
   Fixed 2026-10-07: the nightly's commit had failed since 2026-10-06 (WSL's git has no author identity, `empty ident name`; it has no push credential either), so the 10-06 and 10-07 reports and the README line stayed staged while the task reported CI_RC=1; ci_nightly.py --stage-only now stages them and writes the message, and run-ci-nightly.cmd commits and pushes with the Windows git (absolute path). The 10-07 run itself passed: 98 passed (the einsum's 25 and the six selftest modules included), 15 skipped.
7. **Stage 2, 8 to 16 tokens against FlashInfer's `compute_120f` grouped path by a stated margin** (2026-10-02). The
   gate clause not met: at 16 tokens the layer is level with Marlin and FlashInfer W4A16. The remaining gap is FC2
   against its own byte floor (`docs/stage2-design.md`); the candidate is a tensor-core FC2 that keeps two blocks per
   SM with the one-group tile. Closes with the 16-token rows ahead of both by the margin the report states.
   State 2026-10-07, from the measurements in docs/stage2-design.md: at 16 random tokens the prefetch + split FC2 takes 86.8 us against a
   50.2 us read of its codes (1.7x; v1 was 105.1, 2.1x), and without the activation loads it is still 1.27x to 1.36x its read floor from 8
   tokens up, so the remaining gap is the per-pair arithmetic and the per-pair warp reductions, not the weight stream; the layer at 16 tokens
   is level with Marlin (164.2 against 167.7 us on the checkpoint's layer 0, reports/real-ckpt-layer0-prefill-rtx5090-20261007-run1.json).
   Corrected 2026-10-08 after reading fc2_mma_pf.py: the kernel is already an m16n8k16 MMA with the weights' 16 rows as A and the token pairs as
   the 8 columns of B (NT tiles of 8 pairs), so the tokens are on the MMA's column side today. What it re-reads is the activation fragment:
   every warp, for every expert it walks and every 128-wide chunk, loads its pairs' activation columns from global (L2-resident, the 16 x I
   bf16 block) while the weights stream once. The next kernel keeps the MMA shape and stops that re-read: the activations staged once per
   block in shared memory (16 tokens x 768 bf16 is 24 KB, under the 32-column FC2's budget with two blocks per SM) and read by every warp and
   expert from there, or a block covering more output columns so one fragment serves more weight rows. It is written only when the GPU is
   free to time it against the prefetch + split kernel on the same activations, and judged by the 16-token layer row.
   Corrected again 2026-10-08, by arithmetic before any kernel was written: the activation block FC2 reads is per (token, expert) pair,
   not per token (FC1 writes one SwiGLU row per pair; `act` has M x top_k rows), so at 16 tokens and top_k 8 it is 128 x 768 bf16 =
   192 KB (256 KB at I = 1024), twice SM120's 99 KB of shared memory; staging it once per block is not possible, and the 24 KB above
   was the per-token figure. Two bounds on what any staging can return: the activation traffic cost 27 us of v1's 105.2 at 16 tokens
   (2026-09-29, docs/stage2-design.md's switch that keeps every multiply-add and drops only the loads), and the 32-column block, which
   halves the re-read, gained 16.4 us alone at 16 random tokens (88.8 to 72.4) and 6.0 in the layer, losing on concentrated routing
   (reports/fc2-cols32-rtx5090-2026-09-30.json, real-ckpt-layer0-cols32-rtx5090-2026-09-30.json), so the re-read is worth about that much. The
   form that fits is per expert: a block's eight warps take the same expert and eight different 16-column slabs (128 output columns per
   block, grid H / 128 x G over the experts), the expert's at most 16 pair rows (24 KB at I = 768) staged once in shared memory and read
   by all eight warps, so the re-read falls eight times at the cost of a 16-block grid per G (G = 8 gives 128 blocks). Written only
   when the GPU is free to time it against the prefetch + split kernel on the same activations, judged by the 16-token layer row.
   Measured 2026-10-08 (`sm120fp4/kernels/fc2_xs.py`, G = 4, 8, 16; reports/fc2-xs-rtx5090-20261008.json; same session as the prefetch kernel and the
   read floor, graph replay, L2 flushed): correct and deterministic on every row (error equal to v1's, bit-identical over 50 replays), and level
   with the prefetch kernel where it counts, 90.8 us at 16 random tokens against 90.9 (prefetch) and 88.8 (prefetch, two groups) with the
   codes' read at 47.9; 52.0 against 52.0 at 8 tokens, 35.6 against 35.5 at 4; 2 us ahead at one token (13.1 to 13.3 against 15.1) and behind on
   concentrated routing (fixed8 at 16 tokens 27.3 against 21.1; G = 4 is slower everywhere, too few blocks). So cutting the activation re-read eight
   times buys nothing at 16 random tokens: the re-read is not the remaining cost, and the 27 us the no-load switch saved in v1 was the loads'
   latency and instructions, not their bytes (the switch removed both; the staging keeps the instructions, from shared memory). The 32-column
   kernel's 16.4 us must then come from its other change, half as many column tiles walking the experts' routing (offsets, pairs, weights,
   alpha per expert per warp), which is the per-expert overhead the chain variant also targets. A recorded negative; the kernel stays as a
   measured form, not wired. The remaining lever for the 16-token row is that per-expert, per-warp routing work.
   Measured 2026-10-08 (`scripts/fc2_cols32.py` gains `build(chain=True)`: the prefetch kernel's routing prefetch, Meta and load_meta verbatim, on the
   32-column kernel; reports/fc2-cols32-chain-rtx5090-20261008.json, same session as the plain 32-column and the 16-column prefetch kernels, graph
   replay, L2 flushed): bit-identical to the plain 32-column kernel at every G and shape (it changes only when loads are issued), and within the
   timer's noise, 16 random tokens 87.8 against 93.4 (G 1), 72.4 against 75.5 (G 2), 74.5 against 72.4 (G 4) us; 8 tokens 45.8 against 47.9 (G 2),
   47.9 against 45.9 (G 4); concentrated routing at 16 tokens 19.2 to 21.2 against 21.2 to 24.1. So loading the routing one expert ahead is worth at
   most about 3 us where it helps and costs about 2 where it does not, on a kernel whose best row is 72.4 us against a 49.9 us read of its codes. The
   per-expert routing loads are not the remaining 22 us either; with the activation re-read (fc2_xs) and the routing chain both measured and level,
   what is left between the 32-column kernel and its read floor is the arithmetic and the warp reductions per pair, the instruction-bound part the
   fc2_chain breakdown named (math-only 47.9 at 16 random tokens with every load removed, 35.6 with the activations made rather than read). The
   option stays in the bench, off; the layer's choice is unchanged.
   Measured 2026-10-08 with Nsight Compute on the 32-column kernel at 16 random tokens, four groups, in the layer's cache state
   (reports/ncu-fc2-cols32-opcodes-16-random-rtx5090-20261008.txt; 19.33 million warp instructions over 147 thousand cycles): the decode of the
   FP4 codes is where the instructions go, not the tensor cores. By opcode: F2FP 4.23 million (21.9 percent), HADD2 4.23 (21.9), FMUL 3.99 (20.7),
   LOP3 1.71 (8.8), SHF 1.17 (6.0), HMMA 0.995 (5.1), LDG 0.55 (2.8), PRMT and CS2R 0.50 each, IMAD 0.31; everything else under 1 percent. The five
   decode opcodes (the e2m1 conversion, the half-to-float widening, the scale multiply, the byte extraction and the bf16 pack) are 79 percent of
   the instructions and the MMAs 5 percent, and the thread-level counts say the same (129.6 million fp32 against 109.4 million integer thread
   instructions). So the per-pair cost the fc2_chain breakdown left unbroken is `decode_pairs`: for each 16-element code group it converts two
   codes to f16x2, widens to two floats, multiplies each by the scale and packs the pair back to bf16x2, about four instructions per element
   beside one MMA per 16 x 8 x 16 tile. The next kernel keeps the weights' scale in half2 and multiplies the converted f16x2 pair in one HMUL2,
   or decodes through a 256-entry shared-memory table of byte to bf16x2 with the scale applied once per 16 (one LDS and one HMUL2 per pair),
   either of which removes the widening and the two FMULs; the MMA then takes bf16, so the f16-to-bf16 pack stays unless the MMA is switched to
   f16 inputs with f32 accumulation, which removes the pack as well and is the form to try first (the codes are exact in f16, the scales e4m3 are
   exact in f16 above 2^-14 and clamp below). Judged, as before, by the 16-token layer row against Marlin.
8. **Prefill: a path for more than 16 tokens** (2026-10-02). Today the FC2 kernel takes at most 16 tokens, so the
   layer is a decode layer. Either a second FC2 kernel for 17 to 256 tokens or a documented hand-off to the engine's
   path, measured at 32, 64 and 128 tokens against FlashInfer and Marlin.
   State 2026-10-07: the slicing fallback measured (`scripts/real_ckpt_layer.py --prefill`, the engine's own
   `vllm_backend.Weights.forward`); over three runs (reports/real-ckpt-layer0-prefill-rtx5090-20261007-run1..3.json) the backend's sliced path costs 298.8 to 299.7 / 599.8 to 601.3 / 1154.8 to 1156.6 us at 32 / 64 / 128 randomly routed tokens against Marlin's 217.9 to 218.9 / 242.4 to 242.7 / 254.9 to 255.7 on the whole batch, 1.37 / 2.47 / 4.53 times, bit-identical over 50 calls and at the decode rows' error (2.0e-3 against Marlin's 3.7e-3); each slice re-reads every routed expert's weights, so the cost grows with the slice count while Marlin's barely moves; so slicing is no prefill path, and the hand-off is the
   direction: Marlin needs a repacked second copy of the codes that does not fit beside the model
   (`docs/engine-integration-notes.md`), FlashInfer's CUTLASS W4A4 needs only the 128x4 scales; next, that path
   timed at 32, 64 and 128 tokens on the same weights.
   Measured 2026-10-07: on the baseline bench's weights (scripts/bench_moe_baseline.py --tokens 32,64,128, CUDA graph, L2 flushed; reports/moe-baseline-prefill-rtx5090-20261007-run1..3.json), FlashInfer's CUTLASS W4A4 takes 256.6 to 257.6 / 284.4 to 286.5 / 292.6 to 294.7 us at 32 / 64 / 128 tokens and b12x W4A4 237.2 to 237.3 / 278.3 to 280.3 / 302.8 to 304.9, against Marlin's 220.9 / 249.6 to 250.5 / 254.9 to 255.2 in the same runs (1.14 to 1.16 and 1.07 to 1.19 times Marlin); Marlin here matches Marlin on the checkpoint's weights (217.9 to 218.9 / 242.4 to 242.7 / 254.9 to 255.7), so the ratios carry across the two benches; the W4A4 paths quantize the activations (normwise error against unquantized activations 0.157 to 0.158, W4A16's 0.0045 to 0.0046, on these weights); so a CUTLASS W4A4 hand-off would cost about 1.15 times Marlin at prefill against
   the slicing fallback's 1.37 to 4.53; next, the hand-off wired in the backend (the 128x4 scale copy at load, the
   CUTLASS call above 16 tokens) and checked on the 300 retrieval items.
   State 2026-10-07: `Weights.enable_cutlass_prefill(a1_scale, a2_scale)` in `sm120fp4/vllm_backend.py` does it (the 128x4
   scale copy, the input scales folded into the alphas; forward() hands any batch above 16 tokens to
   `cutlass_fused_moe` once it is enabled); on the checkpoint's layer 0 with its own activation input scales (scripts/real_ckpt_layer.py --prefill, three runs, reports/real-ckpt-layer0-prefill-cutlass-rtx5090-20261007-run1..3.json), the hand-off takes 253.7 to 254.7 / 277.7 to 282.4 / 289.5 to 291.6 us at 32 / 64 / 128 tokens against the slices' 298.8 to 299.8 / 599.8 to 601.9 / 1154.8 to 1156.8 and Marlin's 218.0 to 219.8 / 242.3 to 243.5 / 256.8 to 257.8 (1.12 to 1.16 times Marlin), bit-identical over 20 calls, normwise error 0.110 to 0.117 against the fp32 W4A16 reference (the activations are quantized; the W4A16 slices are at 2.0e-3). Not yet enabled by the vLLM
   method, which still drops the input scales at load: next, keep them, enable the hand-off, and run the 300
   retrieval items through the engine with a prefill above 16 tokens.
   Done 2026-10-07 behind `SM120FP4_PREFILL=cutlass` (the method keeps the checkpoint's input scales and enables the
   hand-off): inside vLLM 0.28 (`SM120FP4_MOE=1 SM120FP4_PREFILL=cutlass`, `scripts/vllm_model_compare.py`, the engine log shows the hand-off enabled on all 48 routed-experts layers and called) the full model answers all 300 retrieval items, on the same items as stock vLLM, two runs identical on all 350 sequences, and generates the 350 prompts in 13.6 and 13.6 s against 20.5 s on the slices the same day (reports/vllm-compare-sm120-prefill-cutlass-20261007*.json, vllm-compare-sm120-slices-20261007.json). Made the default the same day (SM120FP4_PREFILL=slices keeps the slices; a checkpoint
   without input scales falls back to them): the engine run with the defaults matches the explicit run on all 350
   sequences (reports/vllm-compare-sm120-default-20261007.json). Prefill runs W4A4 as stock vLLM does, decode W4A16.
   Item 8 closed by the hand-off; a prefill kernel of our own stays open only if a measurement asks for it.
   Measured in the engine 2026-10-07: scripts/vllm_prefill_latency.py (a batch of N prompts of about L tokens, max_tokens=1, median of 3, one engine per mode, reports/vllm-prefill-{stock,sm120-cutlass,sm120-slices}-20261007.json): at N x L of 1x512, 1x1024, 4x512, 4x1024, 16x512 tokens the wall time of the call is stock 35 / 36 / 79 / 119 / 225 ms, the hand-off 30 / 33 / 69 / 98 / 156 ms and the slices 101 / 184 / 386 / 705 / 1458 ms, so the hand-off runs at 0.69 to 0.91 of stock's time and the slices at 2.9 to 6.5 times it (3.4 to 9.3 times the hand-off); the wall time holds one prefill plus one decode step and the call's overhead, and the decode step is faster on this backend (the README's decode table), so part of the gap to stock at the small shapes is the decode step, not prefill.
9. **Report the b12x W4A4 nondeterminism upstream** (2026-10-01). `reports/b12x-nondeterminism-rtx5090-2026-09-29.json`
   and `docs/stage2-baselines.md`: 20 identical calls, 20 outputs, 1.8 to 3.8% apart; the atomic scatter the source
   describes plus a lost or duplicated 8-column group in 2 of 100 calls. The issue text is drafted for the maintainer
   to review before anything is posted.
10. **Stage 2 on a second checkpoint family** (2026-10-02). A Gemma-class or Mixtral-class NVFP4 MoE checkpoint through
   the same table, to show the layer is not tuned to one expert shape.

## Closed

- **Adoption 1, the stage-2 backend as an installable vLLM plugin** (closed 2026-10-03): the kernel sources moved into
  `sm120fp4/kernels/` with shims left in `scripts/`; `scripts/plugin_install_test.py` installs the repository into a fresh venv with
  stock `vllm==0.28.0`, editable and as a wheel, and probes it (entry point listed; switch off leaves vLLM's config; switch on installs
  `SM120Fp4Config` and compiles the five kernels from site-packages): `reports/plugin-install-test-20261003.json` and
  `reports/plugin-install-test-noneditable-20261003.json`, both pass. README has the Install section. **The upstream text (a vLLM
  issue or PR) waits until the project is essentially complete (author, 2026-10-03): nothing is drafted or posted before then.**
- Engine integration, the layer as an optional MoE backend in vLLM (closed 2026-10-02): `sm120fp4/vllm_backend.py`,
  `sm120fp4/vllm_classes.py`, `scripts/vllm_model_compare.py`, `scripts/vllm_decode_throughput.py`,
  `reports/vllm-compare-20261002.json`, `reports/vllm-decode-compare-20261002.json`, README section "Inside vLLM".
- Stage 2 gate, PRO 6000 reproduction of the real-weights table (closed 2026-10-02): `scripts/pod_real_ckpt.sh`,
  `reports/rtxpro6000-realckpt-2026-10-02/`, README section "The same table on an RTX PRO 6000".
- Stage 1 (layouts, conformance suite, RTX 5090 and RTX PRO 6000) - `docs/conformance-report-*.md`.
- Stage 2 single-kernel benches reproduced on an RTX PRO 6000 - `reports/rtxpro6000-stage2-2026-10-01/`.
