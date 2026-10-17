# What the vLLM 0.28.0 kernels the census read cost on the RTX 5090: the R1 throughput column

Measured 2026-10-16 on NVIDIA GeForce RTX 5090 (compute capability 12.0), vLLM 0.28.0, torch 2.13.0+cu130, WSL2, clocks not pinned; CUDA-event timings, 10 warm-ups then 50 iterations, median with the inter-quartile range and the minimum. Script `scripts/vllm_kernel_throughput.py`; data `reports/vllm-kernel-throughput-rtx5090/full-20261016.json` (57 rows, 0 errors). The semantics of every kernel here is in `docs/census-ledger.md`; this file is what they cost, so a reader of the R1 paper has both.

## Reading the numbers

- Below about 256 tokens every kernel is launch-bound on this host: medians of 10 to 35 us with inter-quartile ranges as wide as the median and minimums near 5 us. Those rows measure the WSL2 launch path, not the kernel; the GEMM TFLOPS and quantiser GB/s columns are meaningful from M = 256 upward and are to be read as such.
- The INT8 Triton fallback (the kernel vLLM reaches on SM120 because no CUTLASS INT8 kernel is built for it; ledger row of 2026-10-14) is slower than the CUTLASS FP8 kernel at small and medium M and faster at the largest shapes: 334 against 278 TFLOPS at 2048 x 4096 x 4096, 507 against 402 at 2048 x 14336 x 4096, 372 against 257 at 256 x 8192 x 8192. The FP8 CUTLASS kernel's best row is 402 TFLOPS, about 0.48 of the card's dense FP8 peak (838 TFLOPS); the epilogue applies per-token and per-channel scales.
- Marlin W4A16 (GPTQ uint4b8, group 128) is the slowest GEMM at every shape from M = 16 up (191 to 213 TFLOPS at M = 2048); its case is memory, not speed.
- The fused RMSNorm + per-token FP8 quantiser is slower than the unfused pair at 2048 tokens (33 against 24 us at hidden 4096; 47 against 39 at 8192) and faster at 16384 tokens (272 against 312 us) and at 1 to 256 tokens. The fusion's two intermediate roundings (ledger row of 2026-10-15) therefore buy nothing at the mid-size prefill shapes on this card.
- Per-tensor dynamic FP8 quantisation costs 1.5 to 2.4 times per-token at five of the six token counts (26 vs 11 us at 1 x 4096, 27 vs 18 at 2048 x 4096, 249 vs 153 at 16384 x 4096; the global absmax is a second pass) and is faster at 256 x 4096 (59 vs 65 us), where the per-token kernel's launch dominates. (Corrected 2026-10-17: the line had said "about twice at every shape", which its own table below contradicts.)

## GEMMs (median us; TFLOPS = 2MNK / median)

| M x N x K | cutlass_scaled_mm fp8 | triton_scaled_mm int8 | marlin_gemm W4A16 g128 |
|---|---|---|---|
| 1 x 4096 x 4096 | 18 us (16-32), 2 TFLOPS | 32 us (27-45), 1 TFLOPS | 22 us (17-37), 2 TFLOPS |
| 16 x 4096 x 4096 | 18 us (17-29), 30 TFLOPS | 34 us (31-43), 16 TFLOPS | 37 us (25-67), 14 TFLOPS |
| 256 x 4096 x 4096 | 38 us (34-48), 228 TFLOPS | 58 us (55-61), 149 TFLOPS | 100 us (94-105), 86 TFLOPS |
| 2048 x 4096 x 4096 | 247 us (243-255), 278 TFLOPS | 206 us (198-215), 334 TFLOPS | 358 us (355-372), 192 TFLOPS |
| 16 x 14336 x 4096 | 27 us (25-39), 70 TFLOPS | 44 us (39-104), 43 TFLOPS | 72 us (65-84), 26 TFLOPS |
| 2048 x 14336 x 4096 | 598 us (589-612), 402 TFLOPS | 474 us (467-494), 507 TFLOPS | 1128 us (1118-1142), 213 TFLOPS |
| 256 x 8192 x 8192 | 134 us (110-152), 257 TFLOPS | 92 us (88-116), 372 TFLOPS | 226 us (209-269), 152 TFLOPS |

## Quantisers and the fused norm (median us; GB/s = bf16 input bytes / median, input read only)

| tokens x hidden | scaled_fp8_quant per-token | scaled_fp8_quant per-tensor | scaled_int8_quant dynamic sym | per_token_group_quant_fp8 (1,128) | rms_norm_dynamic_per_token_quant fp8 (fused) | rms_norm + scaled_fp8_quant per-token (unfused) |
|---|---|---|---|---|---|---|
| 1 x 4096 | 11 us, 1 GB/s | 26 us, 0 GB/s | 11 us, 1 GB/s | 17 us, 0 GB/s | 15 us, 0 GB/s | 19 us, 0 GB/s |
| 16 x 4096 | 23 us, 6 GB/s | 44 us, 3 GB/s | 14 us, 10 GB/s | 16 us, 8 GB/s | 19 us, 7 GB/s | 26 us, 5 GB/s |
| 256 x 4096 | 65 us, 32 GB/s | 59 us, 36 GB/s | 14 us, 146 GB/s | 16 us, 131 GB/s | 15 us, 136 GB/s | 22 us, 97 GB/s |
| 2048 x 4096 | 18 us, 942 GB/s | 27 us, 612 GB/s | 14 us, 1198 GB/s | 19 us, 885 GB/s | 33 us, 503 GB/s | 24 us, 708 GB/s |
| 2048 x 8192 | 19 us, 1811 GB/s | 38 us, 895 GB/s | 17 us, 1949 GB/s | 26 us, 1286 GB/s | 47 us, 710 GB/s | 39 us, 855 GB/s |
| 16384 x 4096 | 153 us, 878 GB/s | 249 us, 540 GB/s | 154 us, 871 GB/s | 163 us, 823 GB/s | 272 us, 494 GB/s | 312 us, 430 GB/s |

## Not measured here

The NVFP4 GEMM and MoE paths (FlashInfer, DeepGEMM, CUTLASS examples, TensorRT-LLM) and the attention kernels; those are the next columns of the same table, one script per library. A second card (a rented PRO 6000 or 5080) runs the same script for the ISPASS paper's second architecture.
