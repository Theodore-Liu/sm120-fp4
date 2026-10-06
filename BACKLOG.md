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
   warp) are bit-identical and 1.19 to 1.24 times v6e over three runs (8.9 / 35.6 / 33.5 us), the paged form to use; Nsight Compute on S 64 N 32768 H 8 (2026-10-05, reports/ncu-paged-v6i-rtx5090-20261005.txt): achieved occupancy 41.6 percent against 66.7 theoretical (v6e 20.7 against 33.3), registers and shared memory each limit it to four blocks, so cutting shared memory alone buys nothing; 4080 blocks are exactly six waves of 680, so the achieved gap is not a wave tail; 18.8 million instructions against v6e's 14.0 (34 percent more), long scoreboard 4.39 against 5.65 and short scoreboard 2.07 against 1.07 cycles per issue, SM throughput 41 and DRAM 8 percent; per-opcode counts (reports/ncu-paged-v6i-opcodes-rtx5090-20261006.txt) put all 4.8 million extra instructions in integer and control work, none in the MMA: IMAD +0.91, BRA +0.51, ISETP +0.41, BSSY/BSYNC +0.46, S2R +0.23 million, from masks, bounds and a rolled tile loop recomputed every quarter; v6j (2026-10-06) runs a full quarter's two tiles unrolled and unmasked, the tail quarter alone masked, staging unrolled: bit-identical on the four selftest shapes and at H 24 and 32, 14.2 million instructions (v6i 18.8, v6e 14.0), 62 registers and four blocks per SM as v6i (reports/ncu-paged-v6j-opcodes-rtx5090-20261006.txt); over three runs (reports/bench-paged-v6j-rtx5090-20261006-run1..3.json) 8.9 / 33.8 to 35.2 / 31.5 us against v6i's 8.9 / 35.5 to 35.6 / 33.5 (level, 1.01 to 1.05, 1.06 times), the paged form to use: a quarter fewer instructions buys at most 6 percent because the kernel waits on memory; Nsight Compute (reports/ncu-paged-v6j-rtx5090-20261006.txt) shows long scoreboard 7.49 cycles per issue against v6i's 4.39, cycles per issued instruction 17.7 against 13.1, achieved occupancy unchanged at 41.7 percent of 66.7; measured on the CPU from the benchmark's context lengths (scripts/paged_grid_occupancy.py, reports/paged-grid-occupancy-20261006.json; max_pages checked against the profiled grid, 510): 3.0 / 8.1 / 4.6 percent of blocks have no live warp, and the others average 4.85 / 4.74 / 4.15 live warps of 8, so a block slot runs at about 0.61 / 0.59 / 0.52 of its warps; on S 64 N 32768 that estimate (0.59) is close to the measured achieved over theoretical occupancy (41.7 of 66.7, 0.625), so the gap is the grid, not the scheduler; v6k (2026-10-06) compacts the (row, page) list: warp w takes the w-th live pair by binary search over the inclusive prefix sum of the rows' page counts, the grid sized by S x max_pages with no host sync; bit-identical on the four selftest shapes and at H 24 and 32; first timed with the prefix sum inside the call it was slower than v6j on every shape (19.2 / 39.7 / 37.6 against 8.9 / 35.4 / 31.5 us, reports/bench-paged-v6k-inline-meta-*), and Nsight Compute (reports/ncu-paged-v6k-rtx5090-20261006.txt) put the cause outside the kernel: the kernel alone 31.7 us against v6j's 35.8 with achieved occupancy 56.3 against 41.5 percent, the rest five small launches (clamp, divide, add, scan init, scan); with the list moved to a metadata call computed once per batch (fp4_fp4_paged_mqa_logits_sm120_v6k_meta), three runs give 8.9 / 30.8 to 31.3 / 27.4 to 27.5 us against v6j's 8.9 / 34.9 to 35.5 / 31.5 (level, 1.12 to 1.13, 1.15 times; reports/bench-paged-v6k-rtx5090-20261006-run1..3.json) and the metadata call 11.0 to 11.5 us; next: one fused metadata kernel (a single-block scan over S) to cut the 11 us, and the engine wiring computes the list once per step; not wired while vLLM refuses the MXFP4 indexer cache on SM120), (3) the two-GPU end-to-end run, (4) the grouped GEMM after step 3 has a number.
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
   halves on the v4 path remains; its first form, raw-row staging v5r, closed the gap: 21.2 / 62.4 / 72.5 us against v5h's 33.2 / 103.4 / 113.4, ahead of the paged v4; the adapter uses it; Nsight Compute: 2.2 times fewer instructions, 2.5 times fewer shared-load bank conflicts, same occupancy; the flat v5 has no scatter to remove). The adapter's
   paged default (v5h) is covered by the 64-row and 256-row block tests already (18 passing).
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
7. **Stage 2, 8 to 16 tokens against FlashInfer's `compute_120f` grouped path by a stated margin** (2026-10-02). The
   gate clause not met: at 16 tokens the layer is level with Marlin and FlashInfer W4A16. The remaining gap is FC2
   against its own byte floor (`docs/stage2-design.md`); the candidate is a tensor-core FC2 that keeps two blocks per
   SM with the one-group tile. Closes with the 16-token rows ahead of both by the margin the report states.
8. **Prefill: a path for more than 16 tokens** (2026-10-02). Today the FC2 kernel takes at most 16 tokens, so the
   layer is a decode layer. Either a second FC2 kernel for 17 to 256 tokens or a documented hand-off to the engine's
   path, measured at 32, 64 and 128 tokens against FlashInfer and Marlin.
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
