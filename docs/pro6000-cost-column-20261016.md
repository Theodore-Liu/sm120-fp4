# The same cost column on a second SM120 card: RTX PRO 6000 Blackwell Server Edition against the RTX 5090 (R1's second architecture)

Measured 2026-10-16 on a rented RTX PRO 6000 Blackwell Server Edition (96 GB, 188 SMs, driver 595.91.07, host CUDA 13.2, bare-metal Linux, RunPod US-PA-1, $2.49 per hour, about one hour including three installation attempts), with the four throughput scripts and the two cross-library scripts of the 5090 tables at the same library versions: vLLM 0.28.0 with torch 2.13.0+cu130; FlashInfer 0.7.1 with torch 2.14.1+cu130 and cuDNN 9.24 (pip resolved a newer torch than the 5090 environment's 2.9.1); the vllm-project/DeepGEMM dev fork at a7a9277 built with CUDA 13.2; TensorRT-LLM 1.3.0rc9 with torch 2.9.1+cu128. Runner: `scripts/pod_cost_column.sh` (its three fixes along the way are in its header); data under `reports/pro6000-cost-column-20261016/`. The 5090 numbers come from WSL2 with unpinned clocks, the PRO 6000 numbers from bare metal; both are medians of 50 CUDA-event timings after warm-up.

## Dense GEMMs at the shapes that are not launch-bound (TFLOPS; ratio = PRO 6000 / 5090)

| kernel | shape | PRO 6000 | RTX 5090 | ratio |
|---|---|---|---|---|
| vLLM 0.28.0: cutlass_scaled_mm fp8 | 256x4096x4096 | 169 | 228 | 0.74 |
| vLLM 0.28.0: triton_scaled_mm int8 (the SM120 fallbac | 256x4096x4096 | 115 | 149 | 0.77 |
| vLLM 0.28.0: marlin_gemm W4A16 gptq g128 | 256x4096x4096 | 139 | 86 | 1.62 |
| vLLM 0.28.0: cutlass_scaled_mm fp8 | 2048x4096x4096 | 544 | 278 | 1.96 |
| vLLM 0.28.0: triton_scaled_mm int8 (the SM120 fallbac | 2048x4096x4096 | 417 | 334 | 1.25 |
| vLLM 0.28.0: marlin_gemm W4A16 gptq g128 | 2048x4096x4096 | 286 | 192 | 1.49 |
| vLLM 0.28.0: cutlass_scaled_mm fp8 | 2048x14336x4096 | 676 | 402 | 1.68 |
| vLLM 0.28.0: triton_scaled_mm int8 (the SM120 fallbac | 2048x14336x4096 | 535 | 507 | 1.05 |
| vLLM 0.28.0: marlin_gemm W4A16 gptq g128 | 2048x14336x4096 | 317 | 213 | 1.48 |
| vLLM 0.28.0: cutlass_scaled_mm fp8 | 256x8192x8192 | 333 | 257 | 1.30 |
| vLLM 0.28.0: triton_scaled_mm int8 (the SM120 fallbac | 256x8192x8192 | 301 | 372 | 0.81 |
| vLLM 0.28.0: marlin_gemm W4A16 gptq g128 | 256x8192x8192 | 251 | 152 | 1.65 |
| FlashInfer 0.7.1: mm_fp4 cutlass | 256x4096x4096 | 83 | 129 | 0.64 |
| FlashInfer 0.7.1: mm_fp4 b12x | 256x4096x4096 | 87 | 126 | 0.69 |
| FlashInfer 0.7.1: mm_bf16_fp4 cudnn (W4A16) | 256x4096x4096 | 42 | 54 | 0.79 |
| FlashInfer 0.7.1: mm_bf16_fp4 cute-dsl (W4A16) | 256x4096x4096 | 70 | 60 | 1.16 |
| FlashInfer 0.7.1: mm_fp4 cutlass | 2048x4096x4096 | 471 | 588 | 0.80 |
| FlashInfer 0.7.1: mm_fp4 b12x | 2048x4096x4096 | 488 | 588 | 0.83 |
| FlashInfer 0.7.1: mm_bf16_fp4 cudnn (W4A16) | 2048x4096x4096 | 156 | 39 | 4.05 |
| FlashInfer 0.7.1: mm_bf16_fp4 cute-dsl (W4A16) | 2048x4096x4096 | 208 | 143 | 1.46 |
| FlashInfer 0.7.1: mm_fp4 cutlass | 2048x14336x4096 | 866 | 728 | 1.19 |
| FlashInfer 0.7.1: mm_fp4 b12x | 2048x14336x4096 | 912 | 723 | 1.26 |
| FlashInfer 0.7.1: mm_bf16_fp4 cudnn (W4A16) | 2048x14336x4096 | 211 | 44 | 4.75 |
| FlashInfer 0.7.1: mm_bf16_fp4 cute-dsl (W4A16) | 2048x14336x4096 | 253 | 184 | 1.37 |
| FlashInfer 0.7.1: mm_fp4 cutlass | 256x8192x8192 | 285 | 442 | 0.64 |
| FlashInfer 0.7.1: mm_fp4 b12x | 256x8192x8192 | 291 | 430 | 0.68 |
| FlashInfer 0.7.1: mm_bf16_fp4 cudnn (W4A16) | 256x8192x8192 | 19 | 27 | 0.68 |
| FlashInfer 0.7.1: mm_bf16_fp4 cute-dsl (W4A16) | 256x8192x8192 | 160 | 102 | 1.57 |
| FlashInfer 0.7.1: grouped_mm_fp4 cudnn | 8x128x4096x4096 | 293 | 404 | 0.72 |
| FlashInfer 0.7.1: grouped_mm_fp4 cudnn | 8x1024x4096x4096 | 933 | 707 | 1.32 |
| FlashInfer 0.7.1: grouped_mm_fp4 cudnn | 64x128x2048x7168 | 490 | 460 | 1.07 |
| FlashInfer 0.7.1: grouped_mm_fp4 cudnn | 128x64x2048x7168 | 278 | 285 | 0.98 |
| FlashInfer 0.7.1: grouped_mm_fp4 cudnn | 128x512x1536x4096 | 1062 | 961 | 1.10 |
| DeepGEMM fork: fp8_gemm (fp8 x fp8, (1,128) scales) | 256x4096x4096 | 89 | 103 | 0.86 |
| DeepGEMM fork: fp8_fp4_gemm (fp8 x fp4, (1,128) scales) | 256x4096x4096 | 96 | 153 | 0.63 |
| DeepGEMM fork: fp8_gemm (fp8 x fp8, (1,128) scales) | 2048x4096x4096 | 319 | 293 | 1.09 |
| DeepGEMM fork: fp8_fp4_gemm (fp8 x fp4, (1,128) scales) | 2048x4096x4096 | 358 | 372 | 0.96 |
| DeepGEMM fork: fp8_gemm (fp8 x fp8, (1,128) scales) | 2048x14336x4096 | 586 | 466 | 1.26 |
| DeepGEMM fork: fp8_fp4_gemm (fp8 x fp4, (1,128) scales) | 2048x14336x4096 | 646 | 567 | 1.14 |
| DeepGEMM fork: fp8_gemm (fp8 x fp8, (1,128) scales) | 256x8192x8192 | 210 | 311 | 0.68 |
| DeepGEMM fork: fp8_fp4_gemm (fp8 x fp4, (1,128) scales) | 256x8192x8192 | 252 | 387 | 0.65 |
| DeepGEMM fork: k_grouped_fp8_gemm_nt_contiguous | 8x128x4096x4096 | 144 | 136 | 1.06 |
| DeepGEMM fork: k_grouped_fp8_gemm_nt_contiguous | 8x1024x4096x4096 | 310 | 326 | 0.95 |
| DeepGEMM fork: k_grouped_fp8_gemm_nt_contiguous | 64x128x2048x7168 | 210 | 219 | 0.96 |
| DeepGEMM fork: k_grouped_fp8_gemm_nt_contiguous | 128x64x2048x7168 | 131 | 141 | 0.93 |
| DeepGEMM fork: k_grouped_fp8_gemm_nt_contiguous | 128x512x1536x4096 | 309 | 274 | 1.13 |
| TensorRT-LLM 1.3.0rc9: trtllm nvfp4_gemm cutlass | 256x4096x4096 | 141 | 153 | 0.92 |
| TensorRT-LLM 1.3.0rc9: trtllm nvfp4_gemm cublaslt | 256x4096x4096 | 165 | 193 | 0.85 |
| TensorRT-LLM 1.3.0rc9: trtllm nvfp4_gemm default | 256x4096x4096 | 145 | 199 | 0.73 |
| TensorRT-LLM 1.3.0rc9: trtllm nvfp4_gemm cutlass | 2048x4096x4096 | 707 | 801 | 0.88 |
| TensorRT-LLM 1.3.0rc9: trtllm nvfp4_gemm cublaslt | 2048x4096x4096 | 634 | 773 | 0.82 |
| TensorRT-LLM 1.3.0rc9: trtllm nvfp4_gemm default | 2048x4096x4096 | 635 | 784 | 0.81 |
| TensorRT-LLM 1.3.0rc9: trtllm nvfp4_gemm cutlass | 2048x14336x4096 | 1080 | 830 | 1.30 |
| TensorRT-LLM 1.3.0rc9: trtllm nvfp4_gemm cublaslt | 2048x14336x4096 | 1075 | 831 | 1.29 |
| TensorRT-LLM 1.3.0rc9: trtllm nvfp4_gemm default | 2048x14336x4096 | 1065 | 837 | 1.27 |
| TensorRT-LLM 1.3.0rc9: trtllm nvfp4_gemm cutlass | 256x8192x8192 | 468 | 369 | 1.27 |
| TensorRT-LLM 1.3.0rc9: trtllm nvfp4_gemm cublaslt | 256x8192x8192 | 484 | 582 | 0.83 |
| TensorRT-LLM 1.3.0rc9: trtllm nvfp4_gemm default | 256x8192x8192 | 451 | 609 | 0.74 |

## Reading

- **At M = 2048 the PRO 6000 is ahead on every GEMM** (ratios 0.80 to 4.75, median 1.26): 188 SMs against 170, and a 96 GB part's memory system. TensorRT-LLM's NVFP4 GEMM reaches 1,065 to 1,080 TFLOPS at 2048 x 14336 x 4096 (837 on the 5090); vLLM's CUTLASS FP8 reaches 676 (402).
- **At M = 256 the order reverses for most kernels** (ratios 0.63 to 1.65): with fewer tiles than SMs the lower-clocked Server Edition cannot fill its machine and the 5090's clock wins. The cost column is therefore not a per-card scalar; the crossover sits between M = 256 and M = 2048 for these shapes.
- **FlashInfer's cuDNN W4A16 backend is 4 to 4.8x faster on the PRO 6000 than on the 5090 at M = 2048** (156 and 211 TFLOPS against 39 and 44), while its cute-dsl backend moves 1.4x. cuDNN 9.24 on the pod against 9.27 on the 5090, and bare metal against WSL2, are both in play; the 5090's cuDNN W4A16 number is re-measured under a pinned cuDNN before it is written as the kernel's cost.
- The quantisers and the fused norm (not in the table above) are within about 20 percent either way at 2048 tokens and above, and launch-bound below.
- Refusals are the same on both cards: FlashInfer's cuDNN `mm_fp4` refuses every shape; TensorRT-LLM's cute_dsl backend refuses SM120; TensorRT-LLM's cutlass refuses N not divisible by 32.

## Cross-library rows on the second card

- GEMM outputs: 43 of 43 compared pairs bit-identical (7 skipped by the N % 32 refusal), the same as on the 5090: FlashInfer cutlass and b12x, TensorRT-LLM cutlass, cuBLASLt and default agree to the bit on this card too.
- Quantisers: the three GPU quantisers are byte-identical on every input (18 of 18 GPU-only pairs); 9 pairs differ, all involving the CPU reference, as on the 5090 (the exact-tie cases).

## Instrument notes (three launches to get a clean run)

1. One venv for vLLM 0.28.0 and FlashInfer 0.7.1: pip refused the resolution (vLLM pins FlashInfer 0.6.16.post3); split into venvs A and C.
2. The image's CUDA 12.8 toolkit: FlashInfer refused b12x (needs 12.9) and its JIT failed; the DeepGEMM build needs nvcc 13; TensorRT-LLM dlopens libcublasLt.so.13. The system CUDA 13.2 toolkit from NVIDIA's apt repository, as `pod_conformance.sh` already did, fixed all three; OpenMPI and libdw-dev were the two other missing system libraries.
3. pip's editable install of the DeepGEMM fork does not link the CUTLASS headers into `deep_gemm/include` (develop.sh does); without them every JIT compile failed with an opaque `command failed with exit code 1`. The 5090 checkout had the links from develop.sh, which is why the difference was invisible there.
4. The first version of the runner masked the DeepGEMM build's exit code behind `| tail -2` and reported RC=0 for a failed build.
