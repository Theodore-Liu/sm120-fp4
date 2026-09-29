# Conformance report: RTX 5090 (SM120), 2026-09-28

Machine: NVIDIA GeForce RTX 5090 (170 SMs, 32 GB), driver 610.47, WSL2 (Linux 6.18), Python 3.12, PyTorch 2.13.0+cu130,
FlashInfer 0.6.16.post3, CUDA 13.0 toolkit for FlashInfer's JIT. JSON: `reports/conformance-rtx5090-2026-09-28.json`
(regenerate with `python -m sm120fp4.cli conformance`).

| check | result |
|---|---|
| 128x4 layout: `to_128x4` vs `flashinfer.nvfp4_block_scale_interleave`, 5 shapes incl. multi-tile and padded | identical bytes |
| `fp4_quantize` swizzled output vs `to_128x4` of its linear output, 3 shapes | identical |
| `fp4_quantize` block scales vs reference | identical, 3 shapes |
| `fp4_quantize` codes vs reference | identical except ulp-level midpoint ties: 39 of 65,536 on one tensor, 0 of 524,288 and 0 of 262,144 on the others |
| `e2m1_and_ufp8sf_scale_to_float` vs reference dequantiser | identical when given the reciprocal per-tensor scale; off by `global_scale**2` when given the quantizer's |
| `mm_fp4` backend `cutlass`, shapes 128x256x512, 16x4096x4096, 1x1024x2048, 512x512x1024 | runs; max relative error 0.26% to 0.35% of the output's max against the fp32 reference GEMM; no all-zero output |
| `mm_fp4` `cutlass` under CUDA graph, 64x2048x2048, 200 replays | every replay equals eager |
| `mm_fp4` backend `cudnn` | unavailable: cuDNN declines every engine (`FORT_NATIVE_9X engine is only supported since Hopper` for 900 <= arch < 1000; `a_scale_layout != "RowMajor" \|\| b_scale_layout != "ColumnMajor"`; `port_scale.tensor->getReordering() != CUDNN_TENSOR_REORDERING_NONE`) |
| `mm_fp4` backend `trtllm` | unavailable by design: `BackendSupportedError: mm_fp4 does not support backend 'trtllm' with capability 120` |

What this says about SM120 today, on this stack:

1. The dense NVFP4 GEMM path that works on a GeForce Blackwell card is the CUTLASS one. The all-zeros failure reported
   for the CUTLASS backend in flashinfer-ai/flashinfer#2577 (February 2026, RTX PRO 6000) does not reproduce on this
   RTX 5090 with FlashInfer 0.6.16.post3; whether the fix is in FlashInfer, in CUDA 13, or a difference between the two
   SM120 cards is not settled by one machine, which is why the gate for stage 1 asks for a second SM12x report.
2. cuDNN's block-scale matmul engines reject the 128x4 reordered scale layout on this architecture (the reasons name
   the layout and the reordering explicitly) and its FP8/FP4 "FORT" engines are gated to Hopper and Blackwell datacenter
   parts. A caller that lets FlashInfer choose the backend gets CUTLASS here; a caller that asks for cuDNN gets an
   exception, not wrong numbers.
3. The TensorRT-LLM generated kernels refuse capability 120 outright, so nothing routes to them silently.
4. Two conventions in the same library point in opposite directions for the per-tensor scale; see
   `docs/scale-layouts.md`, section 1.

Not measured here: grouped (MoE) GEMM paths, which are stage 2; the 8x4 scale layout; MXFP4.
