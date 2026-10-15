# vLLM 0.28.0's FP8 activation quantiser (`ops.scaled_fp8_quant`) on the RTX 5090 at shapes, dtypes, strides and rows its test does not enumerate (2026-10-15)

Read under the registration `mlsys-5090-research/docs/d1-vllm-scaled-fp8-quant-registration-2026-10-15.md` with
`scripts/vllm_scaled_fp8_quant_probe.py`; the per-token kernel's conversion localised with `scripts/vllm_scaled_fp8_quant_diag.py`.
vLLM 0.28.0, torch 2.13.0+cu130, the vllm028 environment under WSL2. Reports: `reports/vllm-scaled-fp8-quant-rtx5090/`
(`full-20261015.json` the settled run; `full-20261015-run1-per-token-modelled-as-reciprocal.json` the first, see the instrument notes).

## The sweep

756 cases, 3,970,782,942 codes, deterministic (two runs each), no refusal, no error. Dtypes bf16, fp16 (absent from the
test), fp32; tokens {1, 3, 1000, 4097}; hidden {1, 2, 8, 15, 17, 24, 40, 1000, 8200, 32768}; dynamic per-tensor, dynamic
per-token with and without `scale_ub = 0.5`, static per-tensor (scales 0.01, 1.0, 1e-5), static `(1, 128)` groups at hidden
{1024, 8192}; strided input (row stride hidden + 8) at hidden {1000, 8200}; `num_token_padding` 4100 at 4097 tokens; special
rows (all zero, one nonzero, a 600 beyond the format's 448) at the first, second and last token. Before any verdict, torch's
cast to `float8_e4m3fn` was checked against a hand-rolled round-to-nearest-even e4m3fn on 2^20 values in (-448, 448): 0
mismatches; and the test's own case (7 x 1025, bf16, per-token) read bit-exact.

| kernel | cases | codes and scales off the step model of its source | what it computes |
|---|---|---|---|
| dynamic per-tensor | 129 | **0** | `scale = absmax / 448` (an `atomicMax` over blocks); codes `cvt_rne_satfinite(clamp(fp32(x) * fp32(1 / scale), ±448))`. An **all-zero tensor** (30 cases, tokens = 1 with the zero row) gives `scale = 0`, `1 / scale = inf`, `0 * inf = NaN`, and `fminf(NaN, 448)` is 448: every code comes out `0x7E` = +448 with scale 0. It dequantises to 0 (448 x 0), so the value is kept; the codes are the format's largest finite number. Recorded as a contract gap, harmless to a consumer that applies the scale |
| dynamic per-token, with and without `scale_ub` | 249 | **0** | `s = max(min(absmax, scale_ub) / 448, 1 / (448 x 512))`; codes `cvt_rne_satfinite(clamp(fp32(x) / s, ±448))`: a **true division** (`scaled_fp8_conversion<false>`), unlike every other kernel in the file. 232 rows hit the floor scale (the all-zero rows), codes 0 |
| static per-tensor | 360 | **0** | `cvt_rne_satfinite(clamp(fp32(x) * fp32(1 / scale), ±448))`; at the 1e-5 scale most codes saturate at ±448, as the clamp says |
| static `(1, 128)` groups | 18 | **0** | the same reciprocal product, one cached reciprocal per group of 128 |

Every named risk read clean: hidden 1 and 2, the odd hidden sizes in bf16 and fp16 (misaligned rows, the scalar path), the
strided input through the per-row reduction kernel, `scale_ub` below a row's absmax (the clipped elements saturate at
±448), fp16 inputs, the 64-group row at hidden 8192, and the padded output (rows past the token count are not judged).

No defect class. With the INT8 quantiser row, both of vLLM's activation quantisers on this part are read; with the FP8
`cutlass_scaled_mm` edge-shape row (2026-10-11), the FP8 W8A8 forward is read end to end as the INT8 one was.

## Instrument notes

- **The first run modelled every kernel as the reciprocal product** because `common.cuh` writes the conversion as
  `is_scale_inverted ? val * scale : val / scale` and the per-tensor dynamic and static kernels instantiate `<true>`; the
  per-token kernel instantiates `<false>` with `token_scale` (`common.cu`, the line after its `__syncthreads`), which the
  first reading of the source missed. 62 per-token cases read one e4m3 step off; the diag fitted the division with 0 off
  on 2.4e8 elements and the reciprocal with 5,665 off (and the reverse for the per-tensor kernel). Same lesson as the INT8
  row, from the other side: **read each kernel's instantiation, not the shared template**; the model is per kernel.
- **The cast check caught two faults in the hand-rolled reference before it judged anything**: `log2` of an exact power of
  two in float64 lands a hair below the integer (`floor` then loses one exponent; 1,801 mismatches), fixed with `frexp`;
  and CUDA's double `pow(2, -4)` came out inexact (`q = 0.49999999999999994`; 71 mismatches), fixed with `ldexp`; then `-0.0`
  (the reference dropped the sign bit; 1 mismatch), fixed with `signbit`. Three attempts are kept as
  `run-20261015-castcheck-attempt{1,2,3}.log`. A reference that has not been checked against something is not a reference.
- The model's NaN path had to follow `fminf` / `fmaxf` semantics (a NaN operand is dropped), not IEEE comparison
  semantics (a NaN propagates): the first run predicted NaN codes for the all-zero tensor and the kernel returned 448.
