# vLLM 0.28.0's INT8 activation quantiser (`ops.scaled_int8_quant`) on the RTX 5090 at shapes, dtypes and rows its test does not enumerate (2026-10-15)

Read under the registration `mlsys-5090-research/docs/d1-vllm-scaled-int8-quant-registration-2026-10-15.md` with
`scripts/vllm_scaled_int8_quant_probe.py`; the static kernels' arithmetic localised with `scripts/vllm_scaled_int8_quant_diag.py`.
vLLM 0.28.0, torch 2.13.0+cu130, the vllm028 environment under WSL2. Reports: `reports/vllm-scaled-int8-quant-rtx5090/`.

## The sweep

1,056 cases, 7,156,784,616 codes, deterministic (two runs each), no refusal, no error. Dtypes bf16, fp16 (absent from the
test), fp32; tokens {1, 3, 1000, 4097}; hidden {1, 2, 8, 15, 17, 24, 40, 1000, 8200, 16384, 32768}; dynamic symmetric,
dynamic asymmetric, static symmetric (scales 0.1, 2.1, 1e-3), static asymmetric ((scale, azp) in {(0.1, -255), (2.1, 54),
(0.05, 0)}); special rows (all zero, one nonzero, all positive, all negative, maximum last) at the first, second and last
token. Instrument check: the test's own case (7 x 1025, bf16, dynamic symmetric) bit-exact against the step model, and the
model within one code of vLLM's own test reference.

| kernel | cases | codes off the step model of its source | what it computes |
|---|---|---|---|
| dynamic symmetric | 132 | **0** (scales 0 off) | `scale = absmax / 127`; codes `rne(fp32(x) * fp32(127 / absmax))`, saturated; `absmax = 0` gives scale 0 and codes 0 |
| dynamic asymmetric, rows with two distinct values | 118 of 132 | **0** (scales and zero points 0 off) | `s = (max - min) / 255`; `zp = rne(-128 - min / s)`; codes `sat8(rne32(fp32(x) * fp32(1 / s)) + zp)` |
| dynamic asymmetric, a constant **nonzero** row (only reachable at hidden 1 and 2 here) | 14 | every such row | `s = 0`, `1 / s = inf`, `zp = (int) inf`; the row's codes come out `-1` whatever its value, so the row dequantises to 0 and its value is lost. An all-zero row (the realistic constant row: padding) gives codes 0 and is harmless. No contract is stated for this input; recorded as a contract gap, not a defect class (no realistic hidden size produces a nonzero constant row) |
| static symmetric | 396 | **0** | `rne(fp32(x) / scale)`: a true division (the diag fits the division model with 0 off and the reciprocal model with 24,821 off on 2.4e9 codes) |
| static asymmetric | 396 | 63 cases, 602,325 of 1,102,297,121 codes, always one code | `sat8(rne32(fp32(x) * fp32(1 / scale)) + azp)`: the **reciprocal product**, which is what the asymmetric kernel's source writes (`inv_s = 1.0f / scale; v = src * inv_s`, `scaled_quant.cu:128-137` at v0.28.0, the same on main) while the symmetric kernel divides (`src / scale`, line 115). The registration had modelled both as the division because the first reading of the asymmetric kernel stopped at its last line; the diag fits the reciprocal model with 0 off and the division model with 24,821 off. The two differ only where `x / scale` sits within an fp32 rounding of a tie |

Beside the kernel's own arithmetic, the ideal quantiser `round(x / scale_returned)` in float64 differs from the kernel by
one code on 13,145,065 of 7.16e9 codes (0.18 percent; dynamic symmetric 164,318, static symmetric 4,759,550, static
asymmetric 7,773,462, dynamic asymmetric 447,735 excluding the constant rows), never by more: the fp32 product or quotient
lands on the other side of a tie. vLLM's own test allows one code and so sees none of this.

## Readings and what they are

- Every registered risk read clean: hidden 1 and 2 (one- and two-thread blocks under the 256-wide reduction), the odd
  hidden sizes 15 and 17 in bf16 and fp16 (misaligned row starts, the scalar prefix), hidden 32768 (128 elements per
  thread), fp16 inputs, the all-zero row, saturation under the 1e-3 static scale, and the asymmetric zero point for
  all-positive and all-negative rows.
- One precision design, fitted bit for bit: the static asymmetric kernel multiplies by the fp32 reciprocal of the scale
  (`inv_s = 1.0f / scale`, hoisted out of the element loop) while the static symmetric kernel divides per element; both
  are in the source at v0.28.0 and on main. The effect is one code at exact ties, inside the test's tolerance; the
  dynamic kernels use the reciprocal as well. Not a defect under the census rule; the inconsistency between the two static
  kernels is the thing a maintainer might want to know. (An earlier version of this note said the two kernels were
  written the same way and the compiler had chosen differently; that was a misreading of the asymmetric kernel, which
  had been read only at its last line.)
- One contract gap: a constant nonzero row in the dynamic asymmetric mode. Degenerate input (hidden size 1 or 2, or an
  artificial row); recorded, not counted.

No defect class. The census's first quantiser row; with the GEMM row and the end-to-end run of the same day, the whole
INT8 W8A8 forward on this part (quantiser, GEMM, selection) has been read.

## Instrument notes

- The step model is torch fp32 arithmetic one operation at a time (each a single correctly rounded IEEE operation on
  both CPU and GPU) with `torch.round` for `cvt.rni` and `nearbyintf` (both round half to even). It was built from the
  source once and fitted two kernels at once; the static asymmetric mismatch was found because the model followed one
  kernel's text for both. Lesson: **read every line of every kernel you model, in the source of the version you
  measure**; the asymmetric kernel's `v = src * inv_s` sits two lines above the line that was read.
- Hypothesis fitting on the 218 stored example elements before the whole-array diag: the reciprocal model explained
  218 of 218, every alternative (float64 division, round half away) 191 of 218.
