# vLLM 0.28.0's (1, group) FP8 activation quantiser (`per_token_group_quant_fp8`, fp32 and UE8M0 scales) on the RTX 5090 at shapes and group contents its tests do not enumerate (2026-10-15)

Read under the registration `mlsys-5090-research/docs/d1-vllm-group-quant-fp8-registration-2026-10-15.md` with
`scripts/vllm_group_quant_fp8_probe.py`; the conversion localised with `scripts/vllm_group_quant_fp8_diag.py`. vLLM 0.28.0,
torch 2.13.0+cu130, the vllm028 environment under WSL2. Reports: `reports/vllm-group-quant-fp8-rtx5090/` (`full-20261015.json`
settled; two earlier attempts kept, see the instrument notes).

## The sweep

482 cases, 3,230,380,032 codes and 37,847,016 scales, deterministic (two runs each), no refusal. Dtypes bf16, fp16, fp32;
tokens {1, 3, 1000, 4097}; hidden {128, 384, 1152, 8320, 16384}; group sizes 64 and 128 (the register kernel for 16-bit
group-128 inputs, the shared-memory kernel otherwise); fp32 and UE8M0 scales; row-major and column-major scale layouts;
`eps` 1e-10 and 1e-4. Every case carries an all-zero group, a single-element group, a group whose absmax is exactly 14.0
(`y_s = 2^-5`, an exact power of two, the UE8M0 ceil boundary) and one at 14.0625 (a hair above). Instrument check: a
test-enumerated configuration (2048 x 1024, group 128, bf16, fp32 scales) bit-exact.

| what | reading |
|---|---|
| codes off the step model | **0** of 3,230,380,032, in both arms, both layouts, both group sizes, all three dtypes |
| scales off the model | **0** of 37,847,016 |
| UE8M0 scales that are powers of two | all (fraction 1.0 in every UE8M0 case); the 14.0 group gives `2^-5`, the 14.0625 group `2^-4`, as `exp2f(ceilf(log2f(y_s)))` says |
| all-zero group | scale `eps / 448` (fp32 arm) or its power-of-two ceiling; codes 0 |
| determinism | one distinct output in two runs everywhere; no NaN codes |

What the kernel computes at v0.28.0 (`csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu`): `absmax =
max(eps, max |fp32(x)|)` over the group; `y_s = absmax / 448`; with UE8M0 `y_s = exp2f(ceilf(log2f(max(|y_s|, 1e-10))))`
(the power of two **at or above** `y_s`; the Python docstring says "round down"); codes `cvt_rne(clamp(fp32(x) / y_s,
±448))`, a division per element. Upstream main (PR #61343, "multiply by reciprocal instead of dividing per element") now
computes `fp32(x) * fp32(1 / y_s)`, one e4m3 step apart from 0.28.0 at ties; that is the next release's arithmetic, not
this one's. No defect class.

With today's INT8 and FP8 whole-row rows this closes vLLM's three activation quantisers on this part; the fused
RMSNorm + group quantiser (the kernel a block-scaled FP8 model actually runs per layer) is the next record.

## Instrument notes

- **Read the source at the tag you measure.** The registration was written from main, whose kernel multiplies by a
  reciprocal; the measured 0.28.0 kernel divides. The instrument check read 880 of 2^21 codes off with every scale matching;
  the diag fitted the division with 0 off on 4.2e7 elements across both kernel paths and all dtypes, and `git show
  v0.28.0:csrc/libtorch_stable/.../per_token_group_quant.cu` then showed `src / y_s` at line 85. The same mistake, main read
  for 0.28.0 plus one kernel read only at its last line, had produced the INT8 quantiser note's wrong explanation ("the
  compiler chose differently" between two kernels written alike); the INT8 note was corrected the same night: the
  asymmetric kernel's own source hoists `inv_s = 1.0f / scale`, at both versions. Attempt 1 kept as
  `run-20261015-attempt1-reciprocal-model.log`.
- The probe transposed the column-major scales when the scale tensor happened to be square (3 tokens, three groups), although
  the wrapper already returns a `[T, H/g]` view in both layouts; six cases read 30 scales misplaced, none wrong in value.
  Attempt 2 kept as `full-20261015-attempt2-square-colmajor-misread.json`.
