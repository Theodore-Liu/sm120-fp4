# What TensorRT-LLM's NVFP4 GEMM costs on the RTX 5090: the TensorRT-LLM half of R1's cost column

Measured 2026-10-16 on NVIDIA GeForce RTX 5090 (compute capability 12.0), tensorrt_llm 1.3.0rc9, torch 2.9.1+cu128, WSL2, clocks not pinned; CUDA-event timings, 10 warm-ups then 50 iterations, median with inter-quartile range. Operands and backend table reused from the census probe `trtllm_nvfp4_gemm_probe.py` (128x4-swizzled E4M3 scales, alpha = 1 / (gs_a gs_b)). Script `scripts/trtllm_kernel_throughput.py`; data `reports/trtllm-kernel-throughput-rtx5090/full-20261016.json` (28 rows; the 7 refusals are all `cute_dsl_nvfp4_gemm_blackwell`, which states it requires SM100 or SM103). Same dense shapes as the vLLM, FlashInfer and DeepGEMM tables.

## Reading the numbers

- **TensorRT-LLM's NVFP4 GEMM is the fastest 4-bit path measured on this card**: 801 to 837 TFLOPS at M = 2048 (cutlass, cuBLASLt and the default list within 4 percent of each other), against FlashInfer's 588 to 728 for the same operands and shapes (14 to 36 percent behind) and DeepGEMM's mixed fp8 x fp4 at 372 to 567. At 256 x 8192 x 8192 the default list reaches 609 against FlashInfer's 442.
- At M = 256 the order is already set: default 199 TFLOPS, cuBLASLt 193, cutlass 153, against FlashInfer's 85 to 129 at the same shape.
- The default backend list ("cutlass,cublaslt,cuda_core") is never slower than its best member by more than the IQR, so TensorRT-LLM's dispatch does not carry the per-call host cost FlashInfer's `auto` does.
- The cutlass backend alone is the noisiest at the small shapes (105 us median with a 42 to 130 us IQR at 16 x 4096 x 4096), where cuBLASLt is steady at 45 us; below M = 256 all three are launch-bound.
- 837 TFLOPS is about half of the dense FP4 rate the card's specification advertises; no library here gets closer.

## TFLOPS and medians (median us, IQR)

| M x N x K | cutlass | cuBLASLt | default list |
|---|---|---|---|
| 1 x 4096 x 4096 | 42 us (38-48), 1 TFLOPS | 48 us (42-66), 1 TFLOPS | 41 us (39-50), 1 TFLOPS |
| 16 x 4096 x 4096 | 105 us (42-130), 5 TFLOPS | 45 us (42-52), 12 TFLOPS | 41 us (39-45), 13 TFLOPS |
| 256 x 4096 x 4096 | 56 us (46-131), 153 TFLOPS | 45 us (42-61), 193 TFLOPS | 43 us (40-63), 199 TFLOPS |
| 2048 x 4096 x 4096 | 86 us (65-135), 801 TFLOPS | 89 us (87-98), 773 TFLOPS | 88 us (85-126), 784 TFLOPS |
| 16 x 14336 x 4096 | 130 us (97-183), 14 TFLOPS | 63 us (57-100), 30 TFLOPS | 59 us (58-68), 32 TFLOPS |
| 2048 x 14336 x 4096 | 290 us (283-314), 830 TFLOPS | 289 us (285-306), 831 TFLOPS | 287 us (284-297), 837 TFLOPS |
| 256 x 8192 x 8192 | 93 us (55-126), 369 TFLOPS | 59 us (58-64), 582 TFLOPS | 56 us (54-74), 609 TFLOPS |

`cute_dsl_nvfp4_gemm_blackwell`: refused at every shape ("requires SM 100 (B200) or SM 103 (B300), but got SM 120").
