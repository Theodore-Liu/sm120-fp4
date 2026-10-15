# vLLM 0.28.0's INT8 W8A8 path on the RTX 5090: the chooser selects a kernel that refuses, and the Triton kernel it never reaches reads conformant at 2,404 shapes (2026-10-15)

Read under the registration `mlsys-5090-research/docs/d1-vllm-int8-scaled-mm-registration-2026-10-15.md` with
`scripts/vllm_int8_scaled_mm_probe.py`. vLLM 0.28.0, torch 2.13.0+cu130, the vllm028 environment under WSL2. Reports:
`reports/vllm-int8-scaled-mm-rtx5090/` (`full-20261015.json` is the settled run; `full-20261015-run1-ulp-no-subnormal-floor.json`
the first run, see the instrument note).

## 1. What an INT8 checkpoint runs into on this part

- `ops.cutlass_scaled_mm` with int8 operands refuses: `scaled_mm_helper.hpp:34, Int8 not supported on SM120. Use FP8
  quantization instead, or run on older arch (SM < 100)`. The SM100 and SM120 dispatchers pass `nullptr` for the int8
  kernel (`scaled_mm_c3x_sm100.cu:19`, `scaled_mm_c3x_sm120.cu:19`, same on upstream `main`).
- `ops.cutlass_scaled_mm_azp` (the asymmetric form) with the same operands raises `cutlass_gemm_caller.cuh:62, Error
  Internal`: its dispatcher sends every device at or above SM90 to the SM90 kernel (`scaled_mm_entry.cu`,
  `if (version_num >= 90) cutlass_scaled_mm_azp_sm90(...)`), so on SM120 a Hopper kernel is launched and fails inside
  CUTLASS instead of being refused with a reason.
- vLLM's INT8 kernel chooser (`_POSSIBLE_INT8_KERNELS`, CUDA: CUTLASS, then Triton, then Humming) selects
  `CutlassInt8ScaledMMLinearKernel` at compute capability 120 for all four layer configurations (channelwise or
  per-tensor, static or dynamic input, symmetric or asymmetric), because its `is_supported` returns true for any CUDA
  device without looking at the capability. The layer therefore fails at its first forward with the message above; the
  Triton kernel next in the list, whose `is_supported` is true on any CUDA-alike device, is never reached. With
  `VLLM_DISABLED_KERNELS=CutlassInt8ScaledMMLinearKernel` the chooser returns `TritonInt8ScaledMMLinearKernel` (checked
  for the symmetric-dynamic and asymmetric-static configurations).
- Public record: vllm issue #21097 ("w8a8 quantization not supporting sm120", closed not planned 2025-12-10) and the
  vLLM forum thread 1981 (2025-11-19) both report the message and recommend FP8; neither mentions the Triton kernel in
  the list or the environment variable.

A dispatch defect, not a numeric one: the chooser's own ordering says the Triton kernel is the fallback "when available",
and the CUTLASS class claims availability it does not have on SM100 and SM120. Fixed locally with a regression test (vLLM
worktree `C:\Users\jingz\oss\vllm-int8-select`, branch `fix/int8-scaled-mm-selection-sm100`; the PR text awaits the
user's review in `mlsys-5090-research/docs/pr-drafts/`).

## 2. The Triton kernel the fallback would run: `triton_scaled_mm` at 2,404 shapes

| what | reading |
|---|---|
| cases | 2,404 (bf16 and fp16 outputs; per-tensor x per-tensor and per-token x per-channel scales; the weight as a contiguous `[K, N]` and as the transposed view of an `[N, K]` buffer, the form the layer passes; with and without bias; M in {1, 7, 33, 129, 1000}; N in {8, 24, 40, 1000, 4160, 8200}; K in {16, 40, 96, 1152, 4224}; plus the tensor-descriptor path forced off at four eligible shapes) |
| judged elements | 1,257,784,320 |
| outside the census bound (three fp32 roundings plus the output rounding, plus one output rounding for the bias added after the cast) | **0**; max 0.99999 of the bound |
| bit-exact against the epilogue model read from source (`out(out(fp32(fp32(fp32(acc) * sa) * sb)) + bias)`) | **1.0 on every case** |
| refused shapes | none (no alignment rule on M, N or K; K = 16 and 40 and N = 8 served) |
| determinism | two runs per case, one distinct output everywhere; no NaN |

The kernel does what its source says and nothing else: the integer accumulation is exact, the scales are applied in
fp32 in the order `scale_a` then `scale_b`, the result is cast once, and a bias is added in the output dtype after the
cast (one output rounding more than a fused epilogue would spend; a design, visible in the model fitting bit for bit, not a
defect). The named risks (one k-block at K = 16, the masked non-descriptor path at K = 40, the partial n-tile at N = 8
and 24 with per-channel scales, the `N >= 8192` tile branch, the transposed-view tile transpose, the bias path) all read
0 outside. One registered expectation was not exercised: |acc| never exceeded 2^24 (max 2,215,712 at K = 4224 with
full-range operands), so the predicted `fp32(acc)` rounding did not occur in this sweep.

## For a user of INT8 W8A8 checkpoints on this part

vLLM 0.28.0 fails at the first forward with "Int8 not supported on SM120"; the Triton kernel that vLLM already ships
serves the same layer bit-for-bit as its source describes, and is one environment variable away
(`VLLM_DISABLED_KERNELS=CutlassInt8ScaledMMLinearKernel`) until the chooser is fixed. Its speed is not read here.

## Instrument notes

- The first run read 152 fp16 cases "outside": every flagged element had |output| in (1e-5, 6e-5), the fp16 subnormal
  range, where the probe's output-ulp formula (`2^(floor(log2 |x|) - 10)`) gave half or less of the true spacing
  (`2^-24`). Each flagged element was bit-exact against the epilogue model. The ulp function now floors at the subnormal
  spacing (fp16 `2^-24`, bf16 `2^-133`); the rerun with that one change reads 0 outside on the same 2,404 cases. The first
  run's report is kept beside the settled one. Lesson filed: an output-ulp bound carries the hidden premise that the
  output is normal; small scales times small sums reach fp16 subnormals in an INT8 GEMM routinely.
- The exact oracle (float64 matmul of the int8 operands) was checked against an int64 CPU matmul at 32 x 4224 x 128
  before any verdict: identical.
