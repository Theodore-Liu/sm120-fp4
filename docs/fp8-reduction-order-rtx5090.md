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

## What this does and does not say

- It reads the order at M = N = 16. CUTLASS and cuBLASLt choose tile configurations and split-K by problem shape; the census's larger shapes may use a different schedule, and the probe should be repeated at the registered shapes before the paper states the order for them. The rounding reading (truncation per accumulator add, exact within a tile) is a property of the instruction and does not depend on the schedule.
- It says nothing about the INT8 kernel (integer accumulation, exact) or about the NVFP4 kernels (`mma.sync` with 64-wide k-tiles for e2m1; the same probe with e2m1 operands and block scales is the next step).
- For R1's bound: the measured rounding is what the bound assumes (gamma_K at u = 2^-23). The measured order is tighter than the bound's "any order" and is not used to narrow it; the bound stays order-free so that a kernel upgrade that changes the schedule does not invalidate the verdicts.
