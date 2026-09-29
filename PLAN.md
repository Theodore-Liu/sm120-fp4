# Plan

Three stages, each with an acceptance gate. No dates: a stage is done when its gate is met on the hardware named,
and the next stage starts then.

## Stage 1: layouts and conformance

Deliverables

- `docs/scale-layouts.md`: every scale-factor layout in circulation for NVFP4 and MXFP4, each with the source it was
  taken from (file, function, commit or version) and the index formula quoted, plus a worked example small enough to
  check by hand. Layouts: 128x4 swizzled (CUTLASS `Sm1xxBlockScaledConfig`, cuDNN "128x4 tiled", TensorRT-LLM /
  FlashInfer `get_sf_out_offset_128x4`); row-major (checkpoints from ModelOpt / compressed-tensors; FlashInfer's
  `is_sf_swizzled_layout=False`); DeepGEMM's packed scale variants; Marlin's repacked weight and scale order. The
  document states which of these are the same layout under different names, because that is the finding that matters.
- `sm120fp4/layouts.py`: pure-PyTorch converters with padding and inverse, verified by round trip and against the
  library that owns each layout.
- `sm120fp4/reference.py`: NVFP4 quantize (E2M1 grid, E4M3 block scale of 16, fp32 per-tensor scale), dequantize,
  and a reference GEMM with fp32 accumulation; MXFP4 (E8M0 scale of 32) alongside.
- `tests/`: the conformance suite. Each test names the failure class it catches and the public report of that failure:
  - values: FlashInfer `fp4_quantize` against the reference quantizer, per element;
  - layout: FlashInfer's swizzled scale tensor against `to_128x4` of its linear one;
  - GEMM: `mm_fp4` on each available backend against the reference GEMM, including the all-zeros failure
    (flashinfer-ai/flashinfer#2577);
  - CUDA-graph replay: a captured GEMM replayed N times, every replay checksummed against eager
    (deepseek-ai/DeepGEMM#444; flashinfer-ai/flashinfer#4841);
  - tactics: which autotuner tactics initialise and run on this device, as a list the caller can pin
    (flashinfer-ai/flashinfer#4841: 14 of 60 could not run on SM120).
- A command that runs the suite and writes one JSON report with the device, driver, library versions and every result.

Gate: the suite runs green on an RTX 5090 with the installed FlashInfer and PyTorch, every layout claim in the
specification is backed by a passing test, and one external SM12x machine (RTX PRO 6000 or DGX Spark) has reproduced
the report.

Status (2026-09-29): met. The suite is green on the RTX 5090; the RTX PRO 6000 Blackwell Workstation Edition
reproduced every verdict (`docs/conformance-report-rtxpro6000.md`). A byte-level comparison of the two machines' GEMM
outputs is still to be run, now that operands are generated on the CPU.

## Stage 2: grouped block-scaled GEMM and fused MoE for SM120

Deliverables

- A grouped NVFP4 GEMM (expert-batched, variable M per group) built on the SM120 collective mainloops
  (`sm120_blockscaled_mma_array_tma`), with tile shapes chosen for the 99 KB budget and a fixed, tested tactic set.
- A fused MoE path (routing, grouped GEMM, activation, second grouped GEMM) for the checkpoints people run on these
  cards (Qwen3-30B-A3B class, Gemma 4 MoE class), with the small-M regime handled explicitly: a weight-only path for
  M below the instruction's row count and a rule for switching to native FP4.
- The stage-1 suite extended to grouped shapes and to the fused path.

Gate, measured on the RTX 5090 and reproduced on an RTX PRO 6000: correct output on the suite; no corruption over
CUDA-graph replays; at batch 1 not slower than the best weight-only (Marlin W4A16) path the engines ship; at batch 8
to 16 faster than the FlashInfer `compute_120f` grouped path by a margin the report states.

Status (2026-09-29): baselines measured before any kernel is written (`docs/stage2-baselines.md`). FlashInfer 0.6.16
already ships three SM120 MoE paths (b12x W4A4 and W4A16, CUTLASS W4A4) that produce correct output, so the target
narrows to what they leave open on the RTX 5090: decode at 1 to 4 tokens, where the best path reaches 31 to 55% of the
DRAM weight-read floor under CUDA graphs with a cold L2, and determinism, which the fastest W4A4 path does not have.
Marlin W4A16 is still to be added to the comparison set.

## Stage 3: the FP4 kernels DeepGEMM does not ship for SM120

Deliverables: SM120 implementations of the three FP4 sites DeepGEMM routes to `tcgen05` today (the FP8xFP4 GEMM, the FP4
attention path, the FP4 einsum), with the stage-1 suite extended to them.

Gate: a DeepSeek-V4-class checkpoint runs end to end on SM120 hardware through these kernels with output matching the
reference within the suite's tolerance.

## Working rules

- Every layout or performance claim in the documents is backed by a test or a measurement in this repository; where
  a number is quoted from someone else's report it is marked as theirs.
- Kernels are developed on an RTX 5090 (170 SMs) and validated on an RTX PRO 6000 (188 SMs) or DGX Spark before a
  gate is called met.
- Nothing in this repository refers to serving-engine internals beyond their public APIs.
