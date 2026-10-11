# FlashInfer's b12x fused MoE, W4A4 mode, against the W4A4 fused-MoE oracle (RTX 5090, 2026-10-11)

Script: `scripts/flashinfer_b12x_w4a4_moe_probe.py`. Oracle: `sm120fp4/moe_oracle_w4a4.py` (selftest in `tests/test_moe_oracle.py`).
Report: `reports/flashinfer-b12x-w4a4-moe-rtx5090-fi071-20261011.json`. Stack: FlashInfer 0.7.1, PyTorch 2.9.1+cu128, driver
610.47, WSL2; `b12x_fused_moe(..., quant_mode="nvfp4")`, weights quantised at global scale 1 (every alpha 1), 8 experts, hidden
1024, intermediate 512, top-2, random routing, 5 runs per token count.

## Results

| tokens | routed rows | backend (by the dispatcher's cutovers) | elements | outside the bound, each of 5 runs | max err / bound | distinct outputs |
|---|---|---|---|---|---|---|
| 1 | 2 | micro | 1,024 | 0 | 0.64 | 1 |
| 4 | 8 | micro | 4,096 | 0 | 0.79 | 1 |
| 16 | 32 | static | 16,384 | 0 | 0.82 | 1 |
| 64 | 128 | static | 65,536 | 0 | 0.87 | 1 |

Conformant on every element of every run under the model of each backend's own rounding points, and deterministic at every
token count on this shape (the non-determinism the stage-2 baselines found was at top-8 with 128 experts, where many routes
land on the same output row; at top-2 with 8 experts the bf16 atomic scatter of the micro backend adds one value to each
output and rounds once).

## How the model was found, and what it says about the kernel

The oracle follows the kernel's rounding points; each had to be read from the source and then confirmed against the kernel,
because three of them are not what the source alone suggests and one is a precision design worth knowing:

1. **The two block quantisers are the fast path and can be emulated bit for bit.** `rcp.approx.ftz.f32` on this device equals
   the fp32 round-to-nearest reciprocal for 6 and for every E4M3 mantissa, and scales exactly with powers of two
   (`scripts/measure_rcp_approx.py`, `reports/rcp-approx-rtx5090-20261011.json`), so the scale `gs * (max * rcp(6))`, the
   E4M3 `cvt.rn`, the codes `value * (rcp(scale) * gs)` and the E2M1 `cvt.rn.satfinite` are reproduced in fp32 arithmetic. This
   matters because bf16 inputs produce exact ties often (a bf16 block maximum over 6 lands on an E4M3 midpoint for about one
   block in nine on random data); an oracle that treated the last bit of the reciprocal as unknown had to call those blocks
   ambiguous and lost all its resolution.
2. **The activation is written as bf16 before the mid quantiser.** With identity FC1 weights and identity W2 (exactly
   representable at global scale 1 as 0.9375), the kernel's output is its quantised activation: the model "fp32 SiLU, bf16
   store, fast quantiser" matched on 8,192 of 8,192 elements; a model without the bf16 store differed on 198.
3. **The static backend computes FC2 in K-chunks of 256 and rounds each partial to bf16**, multiplies each weighted partial in
   bf16 (`mul.rn.bf16x2` with `bf16(topk_weight * alpha)`), and sums every weighted partial of every route in fp32 before one
   output rounding. Found by fitting: with identity FC1 the kernel's top-1, weight-1 output matched "bf16 partials per 256"
   on every element (0 differing of 16,384 at K = 512; 0 at K = 256 where that is one chunk; 0 at K = 1024 with four chunks),
   while an fp32 FC2 differed on 37 percent. The combine then matched bit for bit only with the weight applied per partial
   (0 differing at M = 16 and M = 64, top-2 random weights); applying it to the summed partials left 39 percent differing.
   **The precision cost of this design, measured on the same probe (K = 512, top-1, weight 1):** against an fp32-accumulated
   FC2, 36.7 percent of outputs differ by more than half a bf16 ulp and 11.6 percent by more than one; the relative error has
   median 0.21 percent and 95th percentile 1.1 percent. It is a design the oracle models, not a defect; a caller comparing
   this path with the W4A16 path (which keeps FC2 in fp32) will see it as a small systematic noise floor.
4. **The micro backend keeps FC2 in one fp32 accumulation**, multiplies the bf16-held route result by the fp32 weight, rounds
   to bf16 and adds with a bf16 atomic: top-1 and top-2 outputs matched this model on every element (0 differing).
5. **`w1_alpha` is not an FC1 output multiplier in this mode.** With w13 quantised at global scale 1.5 and `w1_alpha = 1/1.5`,
   the kernel's activation was that of an FC1 output 1.5 times the dequantised product (block scale 1.375 where the oracle
   with the alpha applied expected 0.625, and every code consistent with that); the dispatcher's docstring says
   `input_global_scale` "overrides w1_alpha as the FC1 input-quant scale". `w2_alpha` does multiply FC2 (the identity-W2
   output came back in the oracle's units). Callers with w13 global scales other than 1 must fold them into the weights or
   the scales themselves. Recorded as a contract to know, not a defect.

Two corrections to the oracle itself came out of the same fitting and apply to every oracle in this repository:

- **The accumulation unit is 2^-23, not 2^-24.** Tensor cores accumulate in fp32 by aligning to the largest exponent and
  truncating, so one addition can err by a full ulp of the running sum (Fasi, Higham, Mikaitis and Pranesh, "Numerical
  behavior of NVIDIA tensor cores", 2021). The dense census rows had sat at 0.95 of the half-ulp bound and one fused-MoE
  element crossed it by 4 percent; with the truncation unit every row and element is inside (`sm120fp4/oracle.py`).
- **The output rounding's half ulp is taken at the top of the interval**, since the true sum may sit in the binade above the
  interval's midpoint; the selftest's faithful emulation had landed 5 percent outside on one element for that reason alone.

## The stage-2 shape: 128 experts, top-8, hidden 2048, intermediate 768

`reports/flashinfer-b12x-w4a4-moe-top8-rtx5090-fi071-20261011.json`, 10 runs per token count. M = 1 and 4 go to the micro
backend: 10 distinct outputs in 10 runs at both, every run inside the micro model's bound (max 0.20 and 0.27 of it). M = 16
goes to the static backend: one output in 10 runs, inside the bound (max 0.27). So the stage-2 non-determinism is placed: it
is the micro backend's bf16 atomic scatter, which rounds every partial sum in whatever order the atomics land; with eight
routes per token the roundings differ run to run. The oracle's micro model allows k bf16 roundings and the kernel stays well
inside it, so under the API's contract (bf16 output, no determinism promised) this is a precision design, not a defect. A
caller that needs run-to-run identical outputs at decode batch sizes does not get them from this path.

## What this row does not say

Two shape families, random operands, global scales of 1. The dynamic backend (past the static cutover of 1,024 routed rows)
was not exercised.
