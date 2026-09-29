# Conformance report: RTX 5090 (SM120), 2026-09-28 and 2026-09-29

Machine: NVIDIA GeForce RTX 5090 (170 SMs, 32 GB), driver 610.47, WSL2 (Linux 6.18), Python 3.12, PyTorch 2.13.0+cu130,
FlashInfer 0.6.16.post3, CUDA 13.0 toolkit for FlashInfer's JIT. JSON: `reports/conformance-rtx5090-2026-09-29.json`
(regenerate with `python -m sm120fp4.cli conformance`) and `reports/tactics-rtx5090-2026-09-29.json` (regenerate with
`PYTHONPATH=. python scripts/probe_tactics.py`).

| check | result |
|---|---|
| 128x4 layout: `to_128x4` vs `flashinfer.nvfp4_block_scale_interleave`, 5 shapes incl. multi-tile and padded | identical bytes |
| `fp4_quantize` swizzled output vs `to_128x4` of its linear output, 3 shapes | identical |
| `fp4_quantize` block scales vs reference | identical, 3 shapes |
| `fp4_quantize` codes vs reference | identical except ulp-level midpoint ties: 225 of 262,144 on one tensor, 0 of 65,536 and 0 of 524,288 on the others (39 of 65,536 on the first in the earlier run, whose inputs came from the GPU generator) |
| `e2m1_and_ufp8sf_scale_to_float` vs reference dequantiser | identical when given the reciprocal per-tensor scale; off by `global_scale**2` when given the quantizer's |
| `mm_fp4` backend `cutlass`, shapes 128x256x512, 16x4096x4096, 1x1024x2048, 512x512x1024 | runs; max relative error 0.24% to 0.36% of the output's max against the fp32 reference GEMM; no all-zero output |
| `mm_fp4` `cutlass` under CUDA graph, 64x2048x2048, 200 replays | every replay equals eager |
| `mm_fp4` backend `b12x` (FlashInfer's SM12x-specific NVFP4 GEMM, CuTe-DSL based), same 4 shapes | runs; correct against the reference; 200 CUDA-graph replays equal eager |
| `mm_fp4` `cutlass` vs `b12x` outputs, 4 shapes | byte-identical on every shape (`out_sha256_16` in the JSON) |
| `mm_fp4` backend `auto` on this device | runs and is correct (the docstring says SM12x `auto` prefers `b12x`, then `cutlass`, then `cudnn`) |
| `mm_fp4` backend `cute-dsl` | refused up front: `does not support backend 'cute-dsl' with capability 120`; the refusal is correct, not conservative (see the tactics section: the kernels it would build do not compile for `sm_120a`) |
| `mm_fp4` backend `cudnn` | unavailable: cuDNN declines every engine (`FORT_NATIVE_9X engine is only supported since Hopper` for 900 <= arch < 1000; `a_scale_layout != "RowMajor" \|\| b_scale_layout != "ColumnMajor"`; `port_scale.tensor->getReordering() != CUDNN_TENSOR_REORDERING_NONE`) |
| `mm_fp4` backend `trtllm` | unavailable by design: `BackendSupportedError: mm_fp4 does not support backend 'trtllm' with capability 120` |

## Autotuner tactics, tactic by tactic

`scripts/probe_tactics.py` builds the same runners `mm_fp4` builds (`flashinfer.gemm.gemm_base`), asks each for its
tactic list on a shape, runs every tactic alone against the reference GEMM, and times it (median of 30 CUDA-event
timings after 5 warm-ups). The runner's own default choice (`tactic=-1` for CUTLASS, `None` for b12x) is timed the
same way. Five shapes: the four above plus 4096x4096x4096. Nothing is left to the autotuner.

| runner | tactics presented | ran | correct | note |
|---|---|---|---|---|
| CUTLASS (`fp4_gemm_tactic_num()`) | 32 on every shape | 32 | 32 | none of the flashinfer#4841 failure class (that report was the fused MoE; the dense GEMM's 32 all run here) |
| b12x (`Sm120B12xBlockScaledDenseGemmKernel`) | 8 on every shape: tiles (64,64), (64,128), (128,64), (128,128) x prefetch off/on, cluster (1,1) | 8 | 8 | the list is tile x prefetch; `can_implement` accepted every tile on every shape |
| cute-dsl (`_cute_dsl_gemm_fp4_runner`, built directly, bypassing `mm_fp4`'s gate) | 6 to 16 per shape, every one tagged `sm100` | 0 | 0 | each fails at compile: `expects arch to be one of [sm_100a, sm_100f, sm_103a, sm_110a], but got sm_120a` (error code `MmaMXF4NV`). So `mm_fp4`'s refusal of this backend on capability 120 is the kernel's real limit |

Timing, 4096x4096x4096 (the one shape here that is not launch-bound; the four small shapes finish in 8 to 48 us and
their tactic ranking changes between runs):

| runner | fastest tactic | fastest | slowest | default | default / fastest |
|---|---|---|---|---|---|
| CUTLASS | 16 | 143.3 us (959 TFLOP/s) | 294.2 us, tactic 2 (467 TFLOP/s) | 145.8 us | 1.02 |
| b12x | ((128,128), (1,1), swap_ab False, prefetch False) | 147.8 us (930 TFLOP/s) | 253.7 us, (64,64) with prefetch (542 TFLOP/s) | 154.2 us | 1.04 |

What the numbers say: on this device every tactic either runner presents runs and is correct, so an autotuner cannot pick
a broken one; what it can pick is a slow one, and the slowest is 2.05x (CUTLASS) and 1.72x (b12x) the fastest at
4096-cube. Both defaults are within 4% of the best presented tactic at that shape, so pinning buys little there; at
the small shapes the defaults trail the best tactic by 1.1x to 2.0x, but those are launch-bound microseconds. For b12x
the ranking at 4096-cube is by tile area: (128,128) then (128,64) and (64,128) then (64,64), with prefetch worth less
than 2% either way. The per-shape pinnable lists are in the JSON under `backends.<runner>.shapes.<shape>.pinnable`.

What this says about SM120 today, on this stack:

1. The dense NVFP4 GEMM path that works on a GeForce Blackwell card is the CUTLASS one. The all-zeros failure reported
   for the CUTLASS backend in flashinfer-ai/flashinfer#2577 (February 2026, RTX PRO 6000) does not reproduce on this
   RTX 5090 with FlashInfer 0.6.16.post3; whether the fix is in FlashInfer, in CUDA 13, or a difference between the two
   SM120 cards is not settled by one machine, which is why the gate for stage 1 asks for a second SM12x report.
2. cuDNN's block-scale matmul engines reject the 128x4 reordered scale layout on this architecture (the reasons name
   the layout and the reordering explicitly) and its FP8/FP4 "FORT" engines are gated to Hopper and Blackwell datacenter
   parts. A caller that lets FlashInfer choose the backend gets CUTLASS here; a caller that asks for cuDNN gets an
   exception, not wrong numbers.
3. The TensorRT-LLM generated kernels refuse capability 120 outright, so nothing routes to them silently. FlashInfer's
   own SM12x dense path, the `b12x` backend, runs and matches the CUTLASS path to the same relative error on every
   shape; `auto` on this device resolves to it.
4. The CuTe-DSL `cute-dsl` backend is refused for capability 120 by the wrapper, and probing past the wrapper shows why:
   its kernels are `sm100`-only at the MMA level. Nothing is lost by the refusal.
5. Two conventions in the same library point in opposite directions for the per-tensor scale; see
   `docs/scale-layouts.md`, section 1.

A second SM120 machine, an RTX PRO 6000 Blackwell Workstation Edition, reproduces every verdict here: see
`docs/conformance-report-rtxpro6000.md`. Operands are generated on the CPU since that run, because PyTorch's CUDA
generator gives different numbers on GPUs with different SM counts.

Not measured here: grouped (MoE) GEMM paths, which are stage 2.
