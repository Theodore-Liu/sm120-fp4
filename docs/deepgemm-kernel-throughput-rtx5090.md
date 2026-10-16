# What the vllm-project/DeepGEMM SM120 kernels the census read cost on the RTX 5090: the DeepGEMM half of R1's cost column

Measured 2026-10-16 on NVIDIA GeForce RTX 5090 (compute capability 12.0), vllm-project/DeepGEMM `dev` checkout at `~/oss/DeepGEMM-vllm` (the fork that carries the SM120 kernels; CUDA 13 JIT), torch 2.13.0+cu130, WSL2, clocks not pinned; CUDA-event timings, 10 warm-ups then 50 iterations, median with inter-quartile range. Operands come from the fork's own test generators (`tests/generators.py`: `generate_normal`, `generate_k_grouped_contiguous`), K-major, UE8M0 scales as the SM120 kernel type selects. Script `scripts/deepgemm_kernel_throughput.py`; data `reports/deepgemm-kernel-throughput-rtx5090/full-20261016.json` (19 rows, 0 refusals). Dense shapes match the vLLM and FlashInfer tables; the k-grouped shapes are the same tuples as FlashInfer's m-grouped rows but mean something else (experts x m x n x K-per-expert, output accumulated over the group), so they are not comparable across the two tables.

## Reading the numbers

- Dense FP8 x FP8 with (1, 128) scales: 293 TFLOPS at 2048 x 4096 x 4096 and 466 at 2048 x 14336 x 4096, against 278 and 402 for vLLM's CUTLASS FP8 kernel at the same shapes (its scales are per-token x per-channel, a cheaper epilogue), 5 to 16 percent ahead. At 256 x 8192 x 8192 the two are 311 and 257.
- Dense FP8 x FP4 (the mixed form the SM120 port adds): 372 and 567 TFLOPS at M = 2048, between the FP8 x FP8 kernel and FlashInfer's NVFP4 x NVFP4 (588 and 728). The 4-bit operand halves one stream of bytes, not both.
- Below M = 256 both dense kernels are launch-bound here (34 to 114 us medians, wide IQR); the FP8 x FP4 kernel is consistently the faster of the two at those sizes as well (34 against 65 us at M = 1).
- The k-grouped FP8 kernel (an fp32 accumulator is required on SM120, so the output is accumulated over the group): 136 to 326 TFLOPS; the shape with 128 experts of 64 rows is the slowest (141 TFLOPS), the 8 x 1024 case the fastest.
- Not timed: `m_grouped_fp8_fp4_gemm_nt_contiguous` and the masked form (the census's grouped rows); the second card.

## Dense GEMMs (median us, IQR; TFLOPS = 2MNK / median)

| M x N x K | fp8 x fp8 (1,128) | fp8 x fp4 (1,128) |
|---|---|---|
| 1 x 4096 x 4096 | 65 us (56-113), 1 TFLOPS | 34 us (25-81), 1 TFLOPS |
| 16 x 4096 x 4096 | 114 us (88-186), 5 TFLOPS | 53 us (42-86), 10 TFLOPS |
| 256 x 4096 x 4096 | 83 us (72-120), 103 TFLOPS | 56 us (43-75), 153 TFLOPS |
| 2048 x 4096 x 4096 | 235 us (228-250), 293 TFLOPS | 185 us (176-191), 372 TFLOPS |
| 16 x 14336 x 4096 | 74 us (60-95), 26 TFLOPS | 52 us (50-87), 36 TFLOPS |
| 2048 x 14336 x 4096 | 517 us (508-548), 466 TFLOPS | 424 us (412-430), 567 TFLOPS |
| 256 x 8192 x 8192 | 111 us (90-140), 311 TFLOPS | 89 us (78-104), 387 TFLOPS |

## k-grouped FP8 GEMM (median us, IQR; TFLOPS over the whole grouped product)

| experts x m x n x K per expert | median | TFLOPS |
|---|---|---|
| 8 x 128 x 4096 x 4096 | 253 us (245-275) | 136 |
| 8 x 1024 x 4096 x 4096 | 843 us (792-864) | 326 |
| 64 x 128 x 2048 x 7168 | 1100 us (1062-1133) | 219 |
| 128 x 64 x 2048 x 7168 | 1704 us (1680-1965) | 141 |
| 128 x 512 x 1536 x 4096 | 3009 us (2537-3168) | 274 |

## Instrument notes

- The first version of this script built operands by hand with `per_token_cast_to_fp8/fp4` and got a cuBLASLt allocation failure and an unspecified launch failure; both were the instrument's operand layout, not the kernels (the fork's generators produce what the kernels take). Hand-built operands for a library with its own generators are a known way to blame the wrong side.
- The SM120 k-grouped entry refuses without an fp32 accumulator (`c.has_value()`), as the DeepGEMM PR draft notes; the timing passes `d` as `c`.
