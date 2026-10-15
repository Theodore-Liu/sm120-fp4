# vLLM 0.28.0's fused RMSNorm + dynamic quantiser on the RTX 5090: bit-exact to its own source, and how far it sits from the unfused composition (2026-10-15)

Read under the registration `mlsys-5090-research/docs/d1-vllm-fused-rmsnorm-quant-registration-2026-10-15.md` with
`scripts/vllm_fused_rmsnorm_quant_probe.py`; localisations `scripts/vllm_fused_rmsnorm_quant_diag.py` (rms rounding) and
`scripts/vllm_fused_rmsnorm_quant_diag2.py` (the per-block conversion). vLLM 0.28.0 (source read at the tag), torch
2.13.0+cu130, the vllm028 environment under WSL2. Reports: `reports/vllm-fused-rmsnorm-quant-rtx5090/` (`full-20261015.json`
settled; `full-20261015-run1-block-modelled-as-reciprocal.json` the first run).

## The sweep

792 cases: 648 served and judged (3,042,676,374 codes, 7,526,700 scales), 144 refused by contract (`scale_ub` is accepted
only with an fp8 output: `fused_layernorm_dynamic_per_token_quant.cu:192`; the registration had paired it with int8).
Input dtypes bf16, fp16, fp32; tokens {1, 3, 1000, 4097}; per-token hidden {8, 40, 1000, 1023, 8200, 16384} x fp8 / int8 x
residual on/off x `scale_ub` none / 0.5; per-block fp8 at hidden {128, 8192, 16384} x groups {64, 128} x residual x transposed
scales. Every case carries an all-zero row, a 1e-4 row (epsilon binding) and a one-dominant-element row. Two oracles: the
step model of the fused source (rms taken as any fp32 within two ulps of the exact `rsqrt`, since cub's summation order is
not reproducible) and the composition of the unfused kernels (`rms_norm` / `fused_add_rms_norm`, then `scaled_fp8_quant`
per-token, `scaled_int8_quant` or `per_token_group_quant_fp8`) run on the same inputs.

| what | reading |
|---|---|
| codes off the step model | **0** of 3,042,676,374, every dtype, both paths, with and without residual, both scale layouts |
| scales off the step model | 77 of 7,526,700, all fp32-input cases (22 of them): the fp32 absmax moves with the last ulp of `rms`, and two ulps of candidates did not cover every row; the diag puts the per-thread partial-sum rms within 0.01 ulps of the exact one on average, so this is the candidate window, not the kernel |
| residual written back | 0 off (`scalar_t(x + residual)`) |
| determinism | one distinct output in two runs everywhere |
| **fused versus the unfused composition** | codes differ on **1.10 percent** overall: bf16 input 2.93 percent, fp16 0.375 percent, fp32 0.000 percent (337 codes); per-token 1.08, per-block 1.19 percent; never by more than one e4m3 step (32 at the top binade) or two int8 codes. Scales differ on 1,939,937 of 7,526,700 (26 percent of rows or groups), by one input-dtype ulp of the absmax |

## What the fused kernel computes, and why it is not the composition

Read at v0.28.0 (`layernorm_utils.cuh`): the normalised value is `y = fp32( scalar_t( fp32(x + res) * rms ) * weight )`: the
product with `rms` is **rounded to the input dtype before the weight multiply**, and the weight multiply is itself done in
the input dtype (c10 `bf16 * bf16` rounds again). The unfused `rms_norm` writes `scalar_t(x * rms * w)`: one rounding. So on
bf16 and fp16 inputs the fused kernel's `y` carries one extra rounding, which moves about 3 percent (bf16) or 0.4 percent
(fp16) of the quantised codes by one step and a quarter of the row or group scales by one input ulp; on fp32 input there is no
intermediate rounding and the two agree to the last rms tie. A precision design in the source, not a defect: the test's
tolerances (`atol 1` int8, `rtol 0.125` fp8, scales `rtol 1e-5`) sit above it. A serving stack that mixes the fused layer-norm
path with the unfused one (different layers, different engines, a reference run) will see this 1-step disagreement on a few
percent of activations; it is the fused kernel's definition, not noise.

Conversions, per path (all confirmed by fit): per-token fp8 **divides** by the token scale (`ScaledQuant<false>`, "exact
match with FBGemm"); per-token int8 multiplies by `1 / scale` after `scale = max(absmax / 127, FLT_EPSILON)`; per-block fp8
**divides** by the group scale as well, since the fp8 launch passes `is_scale_inverted = false` and the "invert the scale
here" branch runs only for int8 (the registration had read it as a reciprocal product; 72 cases off until diag2 fitted the
division with 2 to 12 codes off per 8e6 to 1.6e7, the rms ties).

Named risks all read clean: hidden 1023 (the non-vectorised per-token kernel), hidden 8, hidden 16384, fp16 input, the
epsilon-binding row (`rms` at `rsqrt(eps)` scale), the dominant-element row, transposed group scales, and the int8 `1 /
(absmax / 127)` against the unfused `127 / absmax` (two codes apart at most, in the composition column).

No defect class. This closes the per-layer activation path of a block-scaled FP8 model on this part: norm, quantiser, GEMM.

## Instrument notes

- **Source at the measured tag, every line of the path that runs**: the per-block fp8 conversion was read from the
  `is_scale_inverted ? 1.0f / scale : scale` line without following which instantiation fp8 launches; `git show v0.28.0:`
  on the host file settled it. Third instance today of the same lesson; the register of lessons has it once.
- The rms candidate window (two fp32 ulps) is the model's one approximation; for fp32 inputs the absmax is exact in the
  last ulp of `rms` and 77 scales fell outside it. A wider window or an emulation of cub's tree would close it; the codes did
  not need it.
- `scale_ub` with int8 output is a contract refusal, registered as cases by mistake; the 144 refusals are counted as such.
