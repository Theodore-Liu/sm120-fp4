# Reduction order of the SM120 FP8 GEMMs, read with a known-value probe (2026-10-16)

Measured on an RTX 5090 (compute capability 12.0), vLLM 0.28.0 `ops.cutlass_scaled_mm` (FP8 e4m3, per-tensor scales 1.0, bf16 output) and `torch._scaled_mm` 2.13.0+cu130 (cuBLASLt, fp32 output), WSL2. Script `scripts/fp8_reduction_order_probe.py`; report `reports/fp8-reduction-order-rtx5090/probe-k4096.json`. This is the smallest form of the known-value reconstruction of Yang, Riasanovsky, Deng and Sarkar (arXiv 2609.11356), applied to the two FP8 GEMM rows of the census; it reads the accumulation's rounding and its order across k-tiles, not the full descriptor.

## The pattern

One output element receives three non-zero products and 4,093 zeros: P0 = +2^16 at position p0, P1 = 3 x 2^-9 (0.75 ulp of 2^16 in fp32) at p1, P2 = -2^16 at p2. The exact sum is 3 x 2^-9. Three outcomes are distinguishable and all are exactly representable in bf16 and fp32:

| kernel returns | meaning |
|---|---|
| 3 x 2^-9 | P1 was added after P0 and P2 had cancelled, or the three were summed exactly |
| 2^-7 | P1 was added to a running sum holding +-2^16 under round-to-nearest (0.75 ulp rounds up to one ulp) |
| 0 | P1 was added to a running sum holding +-2^16 under truncation (align to the larger exponent, drop the low bits) |

K = 4096 in 32-wide tiles (the k-extent of `mma.sync` for fp8); p0 and p2 sit in tiles t0 < t2, p1 moves over tiles before t0, inside t0, between, inside t2, and after t2. M = N = 16; row 0 and column 0 carry the pattern. 50 (t0, t1, t2) cases per kernel, plus an all-zero control.

## Result

| p1's tile relative to the others | cutlass_scaled_mm (50 cases) | torch._scaled_mm (50 cases) |
|---|---|---|
| t1 < t2, including t1 < t0 and t1 == t0 | 0 on every case (truncation) | 0 on every case (truncation) |
| t1 == t2 (same tile as P2) | 3 x 2^-9 (exact) | 3 x 2^-9 (exact) |
| t1 > t2 | 3 x 2^-9 (exact) | 3 x 2^-9 (exact) |
| all-zero control | 0 | 0 |

No case returned 2^-7, so no cross-tile addition rounds to nearest. Reading the three rows together:

- **Within one 32-wide tile the sum is exact.** With P2 and P1 in the same tile and P0 already in the accumulator, the instruction returns the exact 3 x 2^-9; alignment-and-truncation inside the instruction would have returned 0.
- **Every addition of a tile's result into the fp32 accumulator truncates.** With P1 in an earlier tile than P2 (whether before, inside or after P0's tile), the running sum +-2^16 + 3 x 2^-9 loses the small term, and the later -2^16 leaves 0. This is the behaviour Fasi, Higham, Mikaitis and Pranesh describe for the tensor-core accumulation (alignment to the largest exponent, truncation), now read on SM120 for fp8 operands; the oracle's unit u = 2^-23 per addition (a full ulp, not the half ulp of round-to-nearest) is the right one.
- **Tiles are summed in index order, with no split-K at this shape.** The outcome depends only on whether t1 is before or at/after t2, at every anchor pair tried (tiles 0 and 127, 0 and 64, 32 and 96, 63 and 64, 126 and 127). A split-K reduction would have made the result depend on which partial the tiles fell into; a tree order would have broken the monotone pattern. Both kernels read the same.

## The registered shapes, and a descriptor fit (same day)

The probe was repeated at (M, N) = (1, 4096), (256, 4096), (2048, 4096), (2048, 14336) with K = 4096 and at (256, 8192) with K = 8192 (`reports/fp8-reduction-order-rtx5090/probe-k*-m*-n*.json`, 31 cases per kernel per shape). `scripts/fp8_reduction_order_fit.py` then fits a descriptor to each report: T tiles go to S partials (tile t to partial t mod S, or to contiguous blocks); inside a partial, tiles add in index order with the tensor-core semantics read above (exact within a tile, truncation into the accumulator); the partials combine sequentially or as a tree, with round-to-nearest or truncation. The fitter reports every descriptor that predicts all 31 readings.

| kernel | shape | descriptor that predicts all 31 readings |
|---|---|---|
| cutlass_scaled_mm | all six shapes | S = 1: one sequential pass over the tiles (no split) |
| torch._scaled_mm (cuBLASLt) | M = 16, 256, 2048 (five shapes) | S = 1: one sequential pass over the tiles |
| torch._scaled_mm (cuBLASLt) | M = 1, N = 4096, K = 4096 | **S = 2, interleaved**: even tiles to one partial, odd tiles to the other, each a truncating sequential pass; one final combine. The combine's rounding (nearest or truncation) and order are not identifiable from a two-partial, three-product pattern; all four variants predict the 31 readings. No other S or assignment does. |

So the first reading's "index order, no split" holds for CUTLASS at every shape tried and for cuBLASLt at M >= 16; at M = 1 cuBLASLt picks a split-K-by-parity schedule. The raw classification in the probe's output labelled eight of the M = 1 readings "round-to-nearest"; that label was a guess about one order, and the fit shows the readings are the truncating two-partial order instead (a value of 2^-7 also arises when -2^16 is truncated against a small positive term and +2^16 is added afterwards). The classifier's labels are therefore kept only as raw values; the descriptor is what the fitter says.

## The NVFP4 GEMMs (same day, `scripts/nvfp4_reduction_order_probe.py`)

The same three products built from e2m1 codes and E4M3 block scales (P0: codes 4.0 x 4.0 with scales 2^6 x 2^6; P1: codes 1.5 x 1.0 with scales 2^-6 x 2^-2; P2: 4.0 x -4.0 with 2^6 x 2^6; per-tensor scales 1.0), each in its own 16-block, K = 4096 in 64-wide tiles (the e2m1 `mma.sync` k-extent), M = 16, N = 32 (vLLM's kernel requires N divisible by 32), 31 cases per kernel. A single-P0 control reads 2^16 on every kernel, so the block scales are applied as intended. Reports `reports/nvfp4-reduction-order-rtx5090/*.json`; descriptors from the same fitter.

| kernel | descriptor that predicts all 31 readings |
|---|---|
| vLLM 0.28.0 `cutlass_scaled_fp4_mm` | S = 1: one sequential pass, exact within a 64-wide tile, truncation into the accumulator |
| FlashInfer 0.7.1 `mm_fp4` cutlass | S = 1, the same |
| FlashInfer 0.7.1 `mm_fp4` b12x | S = 1, the same |
| TensorRT-LLM 1.3.0rc9 `nvfp4_gemm_cutlass` | S = 1, the same |
| TensorRT-LLM 1.3.0rc9 `nvfp4_gemm_cublaslt` | **contiguous split-K**: the K range cut into at least two contiguous chunks, each a truncating pass, combined by truncating adds (2, 4, 8 and 16 chunks with a truncating tree all fit; with round-to-nearest only 2 chunks fit). The anchors sit at halves and quarters of K, so the chunk count is not resolved beyond "at least two". |

So at this shape every CUTLASS-derived NVFP4 kernel, including FlashInfer's b12x, is one truncating pass over the tiles, and the two libraries whose NVFP4 GEMMs the census found bit-identical (FlashInfer and TensorRT-LLM cutlass) also share the order, which is what bit identity on random operands requires. cuBLASLt's NVFP4 path splits K; its outputs can therefore differ from the CUTLASS kernels' in the last bit on operands where the order matters, which the census's random-operand identity rows did not contain (TensorRT-LLM's default list resolved to cutlass on those shapes).

## What this does and does not say

- The order is read at six shapes, one of which (cuBLASLt at M = 1) differs; other shapes, other K, and other output dtypes may pick other schedules. The rounding reading (truncation per accumulator add, exact within a tile) is a property of the instruction and held at every shape and schedule seen.
- It says nothing about the INT8 kernel (integer accumulation, exact). The NVFP4 kernels are read above at one shape; the cuBLASLt chunk count needs anchors at finer K fractions.
- For R1's bound: the measured rounding is what the bound assumes (gamma_K at u = 2^-23). The measured order is tighter than the bound's "any order" and is not used to narrow it; the bound stays order-free so that a kernel upgrade that changes the schedule does not invalidate the verdicts.
