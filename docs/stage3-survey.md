# Stage 3 survey: the three FP4 sites DeepGEMM routes to `tcgen05`, and what SM120 would need

Backlog item 3, opened 2026-10-02. A reading of the upstream sources, written before any stage-3 kernel exists here. The
first draft read rendered pages; the second pass (the same day) checked every claim against a checkout of
`deepseek-ai/DeepGEMM` at commit `057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7` ("Public release 26/09/30"), cloned to
`~/oss/DeepGEMM` in WSL; file and line references below are to that commit.

## 1. The three sites, as the vLLM tracking issue names them

vLLM's tracking issue for DeepSeek-V4-Flash on consumer Blackwell ([vllm-project/vllm#41063]) lists three DeepGEMM
dispatch sites that have an SM100 kernel and no SM120 one:

| site | dispatch | SM100 kernel today | SM120 today |
|---|---|---|---|
| FP8xFP4 GEMM | `fp8_fp4_gemm_nt` in `csrc/apis/gemm.hpp` | `sm100_fp8_fp4_gemm_1d1d` | none; `DG_HOST_UNREACHABLE` |
| FP4 attention (MQA logits, the indexer) | three sites in `csrc/apis/attention.hpp` (around lines 67, 177, 367) | `sm100_mqa_logits`, `sm100_paged_mqa_logits`, `launch_sm100_sparse_mqa_logits` | none for FP4; SM90 has FP8-only `sm90_fp8_mqa_logits` / `sm90_fp8_paged_mqa_logits` |
| einsum | `csrc/apis/einsum.hpp` | `sm100_bmn_bnk_mn_gemm` (BF16, line 55-57) and `sm100_fp8_bmm` (FP8) | none upstream; the issue's `sm120_fp8_einsum` is in the reporter's fork. **There is no FP4 einsum upstream at this commit**: `fp8_bmm` asserts both operands are `torch::kFloat8_e4m3fn` (lines 216-217 of the file), so the third site is an FP8 kernel, not FP4 |

`csrc/jit_kernels/impls/` at this commit holds `sm90_*` and `sm100_*` kernels only, 19 files, no `sm120_*`; FP4 appears in
`sm100_fp8_fp4_gemm_1d1d.hpp`, `sm100_fp8_fp4_mega_moe.hpp` and the MQA logits kernels. The ISA wall the issue quotes is exact: `ptxas: Instruction 'tcgen05.fence' not supported on .target 'sm_120f'` and
`Feature '.block32' not supported`. Everything an SM100 kernel does through `tcgen05` (MMA into tensor memory, TMEM
allocation, the `cta_group` pair, UTCCP scale-factor copies) has no SM120 counterpart; SM120 has `mma.sync` (and the
`mma.sync` FP4 shapes, which this repository's stage-2 kernels already use), TMA, mbarriers, 99 KB of shared memory per
block and 128 KB per SM, no TMEM, no 2-CTA MMA.

## 2. What each SM100 kernel does, and what has to change

### 2.1 `sm100_fp8_fp4_gemm_1d1d`

Read in the checkout: `csrc/apis/gemm.hpp`, `csrc/apis/layout.hpp`, `csrc/utils/layout.hpp`,
`csrc/jit_kernels/heuristics/sm100.hpp`, `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh` (579 lines) and
`deep_gemm/utils/math.py`:

- Operands: A and B each an (FP8 or packed FP4) tensor with a scale-factor tensor; "an FP4xFP4 pair uses the MXF4 MMA;
  other combinations use the MXF8F6F4 MMA" (`kind::mxf8f6f4`, `is_mxf4_mma()`). FP4 is two codes per byte.
- Scales: packed UE8M0, four exponent bytes per `int32` (`pack_ue8m0_to_int` in `deep_gemm/utils/math.py`: the fp32
  power-of-two scale's exponent byte, element 0 in the low byte). The recipe is `(gran_mn, gran_n?, gran_k)`; the default on
  SM100 for int scales is `(1, 1, 128)` (`csrc/utils/layout.hpp:86`), so one scale per row per 128 K elements; `gran_k`
  32 is also accepted (`csrc/apis/layout.hpp:46,54`). "1D1D" is per-row, per-K-block scaling on both operands. A scale
  is `ceil_to_ue8m0(amax.clamp_min(1e-4) / fmax)` with fmax 448 (e4m3) or 6 (e2m1): the smallest power of two that
  brings the block's largest magnitude inside the format (`math.py:13` and the two `per_token_cast_to_*`). The kernel's
  MMA applies the scales in hardware: `cute::UMMA::make_instr_desc_block_scaled<..., cutlass::float_ue8m0_t>`
  (`.cuh:305-307`), with the SF tiles delivered by TMA and UTCCP into TMEM (`get_sf_uttcp_aligned_block_sizes`,
  `tmem_sf_cols` in `sm100.hpp:127-129`).
- Tiles (`sm100.hpp`): `block_k = 128 * 8 / element_bits` (128 for an FP8 operand, 256 when both are FP4 and the MXF4
  MMA is used); `block_m` 32, 64 or 128 by M for a dense GEMM (`m <= 32`, `<= 64`, else), `block_n` 128 or multiples of
  32 from 16 up; FP4 operands need a 128-byte swizzle alignment (`swizzle_a_requirement = kPackedFP4 ? 128 : 64`).
  `UMMA_K` is 64 for MXF4, 32 otherwise (`.cuh:72`); MXF4 is K-major only (`.cuh:88`). Shared-memory capacity is
  taken as 232448 bytes (`sm100.hpp:16`), stages up to 32 (`.cuh:311`), a TMEM allocator for one or two CTAs
  (`cute::TMEM::Allocator1Sm/2Sm`, `.cuh:49`), TMA multicast across the cluster, a persistent grid of one CTA per SM,
  and a pluggable `epilogue_class`. The accumulator lives in TMEM.
- Reference and tests: `tests/test_fp8_fp4.py` drives `deep_gemm.fp8_fp4_gemm_nt` over `tests/generators.py`'s
  `QuantConfig` (gran_k per operand, FP4 per operand), with inputs from `per_token_cast_to_fp4` / `per_token_cast_to_fp8`
  and a tolerance per configuration (`QuantConfig.max_diff`). That is the conformance target for an SM120 kernel.

**The SM120 design for this site.** The MMA is `mma.sync.aligned.m16n8k32.kind::f8f6f4` with e4m3 A fragments and e2m1
B fragments (the SM120 instruction takes mixed 8/6/4-bit operands unscaled; there is no `mxf8f6f4` block-scaled form
and no `m16n8k64` MXF4 form on SM120), fp32 accumulators in registers. UE8M0 is applied outside the MMA: with `gran_k`
128 and `k32` MMA steps, four MMA steps share one (row scale of A) x (row scale of B) pair, so the kernel accumulates
four steps into a per-block fp32 partial and folds it into the running accumulator with one `exp2`-free shift per
128-K block (a UE8M0 product is an exponent add, applied as a float multiply by the decoded power of two); at `gran_k`
32 the fold is every step. Scale bytes come from the packed int32 stream the API already produces, so no scale
re-layout is needed, only the MN-major TMA-aligned padding of `get_mn_major_tma_aligned_packed_ue8m0_tensor`, which
an SM120 kernel can read as a plain row-major [rows, K/gran_k] byte array (it is that, padded). Tile: `BLOCK_M` 16 or
32 for decode shapes (the stage-2 FC1/FC2 shapes), `BLOCK_N` 128, `BLOCK_K` 128, 4 to 6 `cp.async` stages of A and B
codes in shared memory (an FP8 A tile 32x128 is 4 KB, a packed FP4 B tile 128x128 is 8 KB, so 6 stages fit in 72 KB of
the 99 KB); 8 warps per block, each warp owning 16 N columns across the whole K loop; one block per 128-N stripe per
M tile, no persistent scheduler at first. This is the FC1 kernel of stage 2 with the activation side switched from BF16
to e4m3 fragments and the E4M3-per-16 block scale replaced by the UE8M0-per-128 fold; the byte floor is the same
formula with the activation bytes halved. Prefill shapes (M in the hundreds) take the same kernel with `BLOCK_M` 128
and the register budget re-split (two warps per 16-N strip); that is a second configuration, not a second kernel.

**Measured on the RTX 5090 (2026-10-02), before the tiled kernel.** `scripts/probe_f8f6f4.py` and
`scripts/probe_f8f6f4_onehot.py` run single `mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e2m1.f32`
instructions on fragments built from a hypothesis and compare with the exact product. Two facts came out, one of them
not what the first draft assumed. The fragment layout is the PTX ISA's for m16n8k32 with 8-bit containers: lane
`4g + t`; `a0` row `g` k `4t..4t+3`, `a1` row `g+8` same k, `a2` row `g` k `16+4t..`, `a3` row `g+8` k `16+4t..`;
`b0` col `g` k `4t..4t+3`, `b1` col `g` k `16+4t..`; `c0,c1` row `g` cols `2t, 2t+1`, `c2,c3` row `g+8`. The e2m1
container is **not** the low nibble: the hardware reads each 8-bit container as a 6-bit field in bits 5:0 (sign at
bit 5, two exponent bits, three mantissa bits), so an e2m1 code goes in bits 5:2 (`code << 2`) with bits 1:0 and 7:6
zero; `0x08` reads as 1.0, `0x3C` as -6.0, `0x02` as 0.25 (a mantissa bit of the wider field), and a code left in the
low nibble produces numbers a few times too small. With that convention the probe matches the exact product to 0.0
on random inputs, and the other seven hypotheses miss by 113 to 157.

`scripts/fp8_fp4_gemm_sm120.py` is the first kernel at its minimum: one warp per 16 x 8 tile over the whole K,
direct global loads, no shared memory, M <= 16, N a multiple of 8, K a multiple of 128, `gran_k` 128, packed UE8M0
scales in DeepGEMM's layout (padded to a multiple of four scales per row), the fold outside the MMA once per 128-K
block, bf16 output. Against `ue8m0_reference.mx_gemm_reference` on five synthetic instances (M 1 to 16, N 8 to 512, K
128 to 2048) the error is bf16 output rounding: maximum absolute error 38 to 1498 against a bf16 half-ulp at the
matrix maximum of 50 to 1685 (one case 1345 against 1214, fp32 summation order), relative Frobenius error 1.2e-3 to
2.0e-3. DeepGEMM's own tolerance for the mixed configuration is `max_diff` 0.01 against an fp32 reference of the
original values; this test is against the exact dequantized product, so the bar is tighter and different in kind.

**v1, the tiled kernel, measured (2026-10-02, RTX 5090, `reports/fp8-fp4-gemm-v1-rtx5090-20261002.json`).** Block of 8 warps on a 32 x 128 output tile, each warp 16 columns for both m16 tiles; A (32 x 128 bytes) and B (128 x 64 packed bytes) per 128-K block through a 4-stage `cp.async` pipeline (48 KB of shared memory); fragments from shared memory with the e2m1 containers formed on the way; the UE8M0 fold once per 128-K block as in v0. It is correct against the reference to bf16 output rounding on six shapes (M 7 to 32, N 128 to 2048, K 128 to 7168) and bit-identical to v0 on every shape v0 covers. Its speed is where a first tiled version lands, not where the kernel has to be: cold-L2 medians of 20 single launches,

| M | N | K | bytes moved | v1 median (us) | achieved GB/s | floor at 1792 GB/s (us) | v1 / floor |
|---|---|---|---|---|---|---|---|
| 16 | 2048 | 7168 | 7.64 MB | 70.4 | 108 | 4.3 | 16.5x |
| 16 | 7168 | 7168 | 26.44 MB | 73.3 | 360 | 14.8 | 5.0x |
| 32 | 2048 | 7168 | 7.82 MB | 70.4 | 111 | 4.4 | 16.1x |
| 32 | 7168 | 7168 | 26.78 MB | 74.4 | 360 | 14.9 | 5.0x |
| 16 | 4096 | 2048 | 4.42 MB | 23.3 | 190 | 2.5 | 9.4x |

Achieved bandwidth is 108 to 360 GB/s against the card's 1792, 5.0 to 16.5 times the byte floor. The cause is in the grid: one block per 128 columns means 16 blocks for N = 2048 and 56 for N = 7168 on a 170-SM card, so most SMs idle and the N = 2048 rows read at 108 to 111 GB/s; the K = 7168 rows also take 56 serial 128-K blocks per block, each with a `__syncthreads`-bounded fold. The next version splits K across blocks (each block a slice of the 128-K blocks, partials reduced in fp32 before the bf16 store, which keeps the fold's arithmetic and so the bit-for-bit agreement per slice) and shrinks `BLOCK_N` to 64 for small N, the two moves the stage-2 FC2 kernel made for the same reason; the shared-memory fragment loads also want an XOR swizzle (the A tile's 128-byte rows put lanes of equal `t` on one bank).

**v2, split-K, measured (2026-10-02, RTX 5090, `reports/fp8-fp4-gemm-v2-rtx5090-20261002.json`).** v1's block, templated on BN (128 or 64 columns), with the grid (N/BN, splits): each block folds its slice of the 128-K blocks into an fp32 partial in a workspace [splits, M, N], and a reduce kernel sums the slices in a fixed order and stores bf16. Inside a slice the fold is v1's, so v2 agrees with v1 bit for bit on six of seven self-test shapes and differs by one bf16 ulp on one element of the seventh (the slice order of the fp32 sum before the round); BN 64 with 11 splits and BN 128 with 7 splits agree with each other bit for bit. `splits` is the smallest count that gives at least two blocks per SM. Cold-L2 medians of 20 launches (v2 includes its reduce kernel):

| M | N | K | v1 (us) | v2 config | v2 (us) | v2 GB/s | floor (us) | v2 / floor | v2 / v1 |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 2048 | 7168 | 66.6 | BN 128, 22 splits, 352 blocks | 19.1 | 401 | 4.3 | 4.5x | 3.5x |
| 16 | 7168 | 7168 | 70.4 | BN 128, 7 splits, 392 blocks | 41.8 | 633 | 14.8 | 2.8x | 1.7x |
| 32 | 2048 | 7168 | 68.4 | BN 128, 22 splits, 352 blocks | 19.2 | 406 | 4.4 | 4.4x | 3.6x |
| 32 | 7168 | 7168 | 70.7 | BN 128, 7 splits, 392 blocks | 43.8 | 612 | 14.9 | 2.9x | 1.6x |
| 16 | 4096 | 2048 | 23.3 | BN 64, 6 splits, 384 blocks | 13.1 | 338 | 2.5 | 5.3x | 1.8x |

v2 is 1.7 to 3.5 times v1 and reads 338 to 633 GB/s, 2.8 to 5.3 times the byte floor. The registered expectation (900 GB/s at N 2048, 1200 at N 7168) was not met: the grid is no longer the bound, so what is left is inside the block. The two candidates, in order: the shared-memory fragment loads (the A tile's 128-byte rows put lanes of equal `t` on one bank; an XOR swizzle on the chunk index fixes it, and the B loads are 2-byte and should become 4-byte with the two n8 tiles' nibbles fetched together), and the per-128-K-block `__syncthreads` pair around the fold, which can move to one per stage. Each is a measured step, not a rewrite.

**v3, the three single-change arms, measured (2026-10-02, RTX 5090, `reports/fp8-fp4-gemm-v3-rtx5090-20261002.json`).** Each arm is a template parameter on v2's tile and is bit-identical to v2 on three shapes (the K permutation included: the MMA's k32 sum is order-independent at these magnitudes). Cold-L2 medians of 20 launches, each arm alone and all together:

| M | N | K | v2 (us) | A rows swizzled | K permutation | one barrier | all three | best GB/s | best / floor |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 2048 | 7168 | 19.2 | 17.2 (+12%) | 17.2 (+12%) | 18.4 (+4%) | 17.2 (+12%) | 445 | 4.0x |
| 16 | 7168 | 7168 | 41.7 | 35.6 (+17%) | 35.6 (+17%) | 41.7 (+0%) | 35.6 (+17%) | 743 | 2.4x |
| 32 | 2048 | 7168 | 19.2 | 19.2 (+0%) | 19.1 (+1%) | 19.2 (+0%) | 18.4 (+4%) | 424 | 4.2x |
| 32 | 7168 | 7168 | 43.8 | 39.7 (+10%) | 38.7 (+13%) | 43.8 (+0%) | 37.6 (+16%) | 712 | 2.5x |
| 16 | 4096 | 2048 | 13.1 | 12.0 (+9%) | 12.6 (+4%) | 13.1 (+0%) | 12.8 (+2%) | 368 | 4.9x |

Swizzling the A rows and permuting K inside each k32 step (one 8-byte A load and one 4-byte B load per lane) buy the same 10 to 17 percent on the K = 7168 shapes and do not add, so they remove the same bank conflict; the one-barrier arm is within noise (under 5 percent everywhere), as registered. The registered expectation for the swizzle (at least 25 percent at N 7168) was not met: the best configuration reads 743 GB/s, 2.4 to 4.9 times the byte floor, so the shared-memory access pattern was a sixth of the gap, not most of it. What is left is latency: 48 KB of shared memory per block allows two blocks per SM, sixteen warps, and each 128-K block is four dependent MMA steps followed by a fold and a barrier. The next arms are pipeline depth (2 stages, 24 KB, four blocks per SM) and two 128-K blocks in flight per block before the fold; the swizzle is kept, the permutation and the single barrier are not.

**v4, occupancy and pipeline depth, measured (2026-10-02, RTX 5090, `reports/fp8-fp4-gemm-v4-rtx5090-20261002.json`).** Two arms on v2 with the swizzle, both bit-identical to v2 on three shapes: two pipeline stages instead of four (24 KB of shared memory, four blocks per SM) and pairs of 128-K blocks computed per barrier pair through the four-stage ring (two independent MMA chains, one barrier per block). Cold-L2 medians of 20 launches:

| M | N | K | v2 (us) | + swizzle | + swizzle, 2 stages | + swizzle, pairs | best GB/s | best / floor |
|---|---|---|---|---|---|---|---|---|
| 16 | 2048 | 7168 | 17.2 | 17.0 (+1%) | 17.1 (+0%) | 17.1 (+0%) | 449 | 4.0x |
| 16 | 7168 | 7168 | 41.5 | 35.6 (+17%) | 31.0 (+34%) | 33.5 (+24%) | 852 | 2.1x |
| 32 | 2048 | 7168 | 19.1 | 17.8 (+7%) | 18.6 (+3%) | 19.2 (-0%) | 439 | 4.1x |
| 32 | 7168 | 7168 | 43.0 | 38.9 (+11%) | 33.0 (+30%) | 35.6 (+21%) | 813 | 2.2x |
| 16 | 4096 | 2048 | 13.1 | 11.7 (+12%) | 11.0 (+19%) | 13.1 (-0%) | 401 | 4.5x |

Two stages are the larger step on every shape with K = 7168 and N = 7168 (+30 to +34 percent over v2, +12 to +18 over the swizzle alone) and on the K = 2048 shape; the pairs give +21 to +24 percent on the same shapes and nothing elsewhere, so the bound was occupancy, not the per-block dependency chain, and the two arms are not kept together. On the N = 2048, K = 7168 shapes neither moves anything: those runs sit at 17 to 19 us whatever the kernel does, which is the cost of two launches and a reduce over 22 split partials, so their next step is a fused reduce or fewer splits with the freed occupancy, not more tile work. The best configuration (swizzle, two stages) reads 852 GB/s on M16 N7168 K7168, 2.1 times the byte floor, and 4.5 times on the overhead-bound N 2048 shapes. The registered expectations: (d) at least 20 percent at N 2048 missed (0 percent, for the reason above); (e) at least 15 percent at N 7168 met (21 to 24).

**v5, the split-K fixed cost, measured (2026-10-03, RTX 5090, `reports/fp8-fp4-gemm-v5-rtx5090-20261003.json`).** Three arms on the v4 best (swizzle, two stages): the tile kernel and the reduce kernel timed alone; half the splits; and a fused reduce, where each block publishes its fp32 partial, takes a ticket on its column tile, and the last block to arrive sums the partials in split order and stores bf16 (bit-identical to the separate reduce on all three selftest shapes, counters self-resetting). Cold-L2 medians of 20 launches:

| M | N | K | splits | v4 best (us) | tile alone | reduce alone | half splits | fused reduce | half + fused |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 2048 | 7168 | 22 -> 11 | 17.2 | 13.2 | 7.0 (41%) | 15.3 (+12%) | 24.3 (-29%) | 23.3 (-26%) |
| 16 | 7168 | 7168 | 7 -> 4 | 31.4 | 27.5 | 7.1 (23%) | 29.4 (+7%) | 49.1 (-36%) | 46.0 (-32%) |
| 32 | 2048 | 7168 | 22 -> 11 | 19.1 | 15.1 | 9.0 (47%) | 17.5 (+9%) | 32.7 (-42%) | 29.4 (-35%) |
| 32 | 7168 | 7168 | 7 -> 4 | 32.8 | 28.6 | 8.9 (27%) | 31.8 (+3%) | 54.9 (-40%) | 48.2 (-32%) |
| 16 | 4096 | 2048 | 6 -> 3 | 11.2 | 9.4 | 5.4 (48%) | 13.1 (-15%) | 17.2 (-35%) | 15.6 (-28%) |

The reduce kernel alone is 5 to 9 us on every shape, 41 to 47 percent of the whole on the 2048-wide shapes, and the two kernels timed apart sum to 3 to 5 us more than the pair launched together, which is the launch latency the stream hides. Halving the splits helps where the block count stays at or above one per SM (+12.5 and +9.2 percent on the 2048-wide shapes, +6.7 and +3.1 on the 7168-wide) and hurts where it falls below (-15 percent at K = 2048, 96 blocks on 170 SMs), so the planner's rule for two stages is the smallest split count that keeps one block per SM, not two. The fused reduce is 29 to 42 percent slower than the separate one on every shape: it leaves the whole reduce to N/BN blocks (32 on the 2048-wide shapes), each thread walking 22 partials for 16 outputs with scalar loads, so the sum is latency-bound on 4,096 threads where the separate kernel spreads it over every SM. The registered expectations: (f) reduce plus launch at least 40 percent of the whole, met (41 to 47 on the 2048-wide shapes); (g) half splits at least 15 percent, missed (12.5 at best); (h) fused reduce at least 25 percent, missed in the other direction. The fused arm stays in the file as an arm, not a default; vectorised loads and more outputs per thread in flight are its next test.

**v6, the fused reduce with vector loads, and the one-block planner (2026-10-03, RTX 5090, `reports/fp8-fp4-gemm-v6-rtx5090-20261003.json`).** Two arms on the v4 best: the fused last-block reduce rewritten with float4 loads along N and four independent accumulators per thread (bit-identical to the separate reduce on the three selftest shapes, the split order unchanged), and the planner asked for one block per SM instead of two for the two-stage kernel (it may then also pick BN 128 where BN 64 was needed to reach two). Cold-L2 medians of 20 launches:

| M | N | K | default plan (BN x splits) | v4 best (us) | half splits | fused, scalar | fused, float4 | one-block plan (BN x splits) | one-block plan + fused float4 |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 2048 | 7168 | 128 x 22 | 17.1 | 15.2 (+13%) | 23.9 (-28%) | 21.2 (-19%) | 15.1 (+13%) (128 x 11) | 19.2 (-11%) |
| 16 | 7168 | 7168 | 128 x 7 | 31.3 | 29.4 (+6%) | 49.2 (-36%) | 46.3 (-32%) | 29.3 (+7%) (128 x 4) | 44.9 (-30%) |
| 32 | 2048 | 7168 | 128 x 22 | 19.2 | 17.2 (+12%) | 33.2 (-42%) | 26.9 (-29%) | 17.2 (+11%) (128 x 11) | 22.1 (-13%) |
| 32 | 7168 | 7168 | 128 x 7 | 32.7 | 32.8 (-0%) | 54.0 (-39%) | 49.8 (-34%) | 31.9 (+3%) (128 x 4) | 46.8 (-30%) |
| 16 | 4096 | 2048 | 64 x 6 | 11.0 | 13.4 (-18%) | 17.2 (-36%) | 14.3 (-23%) | 11.0 (+0%) (128 x 6) | 15.1 (-27%) |

The vector form recovers a third to a half of the scalar fused reduce's loss and no more (-19 to -34 percent against -28 to -42), so the fused pattern does not pay at decode shapes on this part: N/BN blocks, 16 on the 2048-wide shapes, carry the whole sum while the separate kernel spreads the same bytes over every SM, and a 3 to 5 us launch saving cannot cover that. It stays in the file as an arm. The one-block planner is the result that carries: +13.2 and +11.2 percent on the 2048-wide shapes, +6.7 and +2.7 on the 7168-wide, and 0.0 at K = 2048 where it picks the same split count at a wider tile, so no shape is worse; it matches the half-splits arm where that helped and avoids its -17.6 percent where that fell below a block per SM. The registered expectations: (i) fused within 5 percent of the separate reduce, missed; (j) at least 9 percent on the 2048-wide shapes with no shape worse than -2 percent, met. The planner's default moves to one block per SM for the two-stage kernel in the next version; the MQA-logits kernel is next.

**Interface of the first kernel, `sm120_fp8_fp4_gemm_1d1d`.** The same arguments `fp8_fp4_gemm_nt` passes the SM100
launcher at `gemm.hpp:128`: `a` codes (e4m3, `[M, K]`), `sfa` (packed UE8M0 `[M, K/gran_k_a]` after the layout
transform), `b` codes (packed e2m1, `[N, K/2]`), `sfb`, optional `c`, output `d` (bf16 or fp32, N-major), `m, n, k`,
`gran_k_a`, `gran_k_b`, `major_a`, `major_b` (K-major only at first, as MXF4 is upstream), `compiled_dims`, the epilogue
class (identity first). Dispatch: an `arch_major == 12` branch beside the SM100 one, reached only when `sfa` is
`torch::kInt` (the UE8M0 path); everything else unchanged.

**Conformance plan.** (1) `scripts/ue8m0_reference.py` (in this repository, done 2026-10-02): the scale rule, the e2m1
rounding, the nibble packing and the UE8M0 packing mirrored from `deep_gemm/utils/math.py` and checked bit for bit
against it on six configurations (`gran_k` 32 and 128, with and without UE8M0, packed and fp32 scales); its
`mx_gemm_reference` is the fp32 answer the kernel must reproduce. (2) The kernel's own test runs `tests/test_fp8_fp4.py`'s
`generate_normal` cases through the new branch and holds `QuantConfig.max_diff`. (3) The stage-1 habit: the same
inputs through DeepGEMM's SM100 kernel on a rented B200 or GB10 once, so the SM120 result is compared with the upstream
kernel's output and not only with the fp32 reference.

### 2.2 `sm100_mqa_logits` (FP4 attention)

Read from `csrc/jit_kernels/impls/sm100_mqa_logits.hpp` and `csrc/apis/attention.hpp` (rendered):

- Inputs: `q` and `kv` each a (tensor, scale-factor) pair; the config checks `qk_dtype == kPackedFP4 or qk_dtype ==
  torch::kFloat8_e4m3fn`; for FP4 the head dimension is 64 or 128, for FP8 32, 64 or 128. The paged variant takes a fused
  KV cache whose row is `kv_head_dim + sf_bytes` with `sf_bytes` 4 (MX) or `sizeof(float)`. The output `weights` /
  `logits` is the score matrix the indexer ranks (this is DeepSeek's sparse-attention indexer, not the softmax attention
  itself).
- Tiling: a 128-row (token, head) tile with `block_q = 128 / num_heads`, 384 KV tokens per KV tile (`kMQALogitsSplitKV`);
  scale factors MX-style with `kNumUTCCPAlignedElems = 128`; TMA 2D/3D descriptors, mbarriers, `num_tmem_stages`.
- Variants: contiguous, paged (block table + context lengths), sparse (`sparse_block_kv` 8 or 16, up to 4096 blocks) and
  paged-sparse; the sparse pair is SM100-only even upstream.

For SM120 this is a batched small-K GEMM (head_dim 64/128) between an FP4 query tile and FP4 keys with a per-row score
output, which is close to stage 2's FC1 shape class (one activation tile against many weight rows, result per row), with
two differences: both operands are FP4 (so `kind::f8f6f4` on both fragments, no BF16 side), and the K dimension is tiny,
so the kernel is bound by key-stream bandwidth and by how many 128-row tiles a block keeps in flight, not by MMA. The
paged variant adds a gather through the block table, which the TMA 3D descriptor does on SM100 and a plain per-page
load does on SM120.

### 2.3 einsum (FP8, not FP4)

Confirmed in the checkout: `csrc/apis/einsum.hpp` has `einsum` (BF16; `"bmk,bnk->mn"`, `"bhr,hdr->bhd"`,
`"bhd,hdr->bhr"`, `"bhd,bhr->hdr"`) and `fp8_einsum` (the last three expressions, two of them SM100-only), which
permutes to `(batch, m, n, k)` and calls `fp8_bmm`; `fp8_bmm` asserts `a` and `b` are `torch::kFloat8_e4m3fn`. No FP4
einsum exists upstream at this commit; the tracking issue's third site is the FP8 batched GEMM, and the reporter's
fork names its SM120 version `sm120_fp8_einsum`. For this repository the site is an FP8xFP8 batched GEMM with UE8M0
scales on SM100 (fp32 on SM90), the same scale handling as 2.1 with both operands e4m3, and it ranks after the two FP4
sites.

## 3. What this repository already has for each site

- The `mma.sync` FP4 fragment path, the E4M3 block-scale application, the TMA-free streaming of packed codes, PDL between
  dependent kernels, and the conformance suite for 128x4 scale layouts (stage 1) and for FC1/FC2 shapes (stage 2).
- Not yet: UE8M0 (MX) scales anywhere; FP4 on both MMA operands (stage 2 is FP4 weights x BF16 activations); any paged
  gather; any persistent-grid scheduler.

## 4. The order of work, and the first kernel

1. Done 2026-10-02: DeepGEMM checked out at `057ca5964aae` and the three open points settled above (no FP4 einsum
   upstream; tile, stage and shared-memory figures from `sm100.hpp` and the `.cuh`; default recipe `(1, 1, 128)`).
2. Done 2026-10-02: `scripts/ue8m0_reference.py`, bit for bit with `deep_gemm/utils/math.py` on six configurations
   (`python scripts/ue8m0_reference.py --selftest` with the checkout at `~/oss/DeepGEMM` or `DEEPGEMM=`).
3. Started 2026-10-02 (the minimal correct version above; the tiled version is next). First kernel: **the FP8xFP4 dense GEMM
   (`sm120_fp8_fp4_gemm_1d1d`)**, NT layout, M <= 16 first (the decode shape this
   repository knows how to measure) and then the prefill shapes, because it is the site whose SM100 counterpart is one
   GEMM with a known reference, because the MoE path (`fp8_fp4_mega_moe`) is built from it, and because the attention and
   einsum sites reduce to batched forms of it. The attention indexer follows, the einsum last.
4. Each kernel closes with the conformance test, the single-kernel bench against the same byte floor the stage-2 design
   note uses, and a row in a report; the gate is DeepGEMM's own test passing through a `sm120_` dispatch branch in a
   fork, not a number here.

## Sources

- [vllm-project/vllm#41063](https://github.com/vllm-project/vllm/issues/41063), the tracking issue (dispatch sites, the
  ptxas errors, the linked PRs #41062, #41028, #40923).
- [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM): README (architectures, UE8M0 packing, API list);
  `csrc/apis/gemm.hpp`, `csrc/apis/attention.hpp`, `csrc/apis/einsum.hpp`; `csrc/jit_kernels/impls/` (the file list and
  the two SM100 kernels read above).
