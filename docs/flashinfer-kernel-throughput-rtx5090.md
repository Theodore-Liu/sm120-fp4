# What the FlashInfer NVFP4 kernels the census read cost on the RTX 5090: the FlashInfer half of R1's cost column

Measured 2026-10-16 on NVIDIA GeForce RTX 5090 (compute capability 12.0), FlashInfer 0.7.1, torch 2.9.1+cu128, cuDNN 92700, WSL2, clocks not pinned; CUDA-event timings, 10 warm-ups then 50 iterations, median with inter-quartile range. Script `scripts/flashinfer_kernel_throughput.py`; data `reports/flashinfer-kernel-throughput-rtx5090/full-20261016.json` (47 rows, 7 refusals, all of them the cuDNN `mm_fp4` backend). The same shapes as the vLLM table (`docs/vllm-kernel-throughput-rtx5090.md`), so the two tables read side by side; the environments differ (torch 2.9.1+cu128 here, 2.13.0+cu130 there), which is why only ratios within a table and the order of magnitude across tables are to be read.

## Reading the numbers

- Below about 256 rows every kernel is launch-bound on this host (medians 57 to 160 us with wide inter-quartile ranges); the TFLOPS column is meaningful from M = 256 upward.
- **The `auto` backend of `mm_fp4` is slower than either explicit backend at every shape** (20 to 45 percent in the table: 149.5 against 116.9 us at 2048 x 4096 x 4096, 126.6 against 77.7 us at 256 x 8192 x 8192). Resolved the same night (`scripts/flashinfer_auto_dispatch_cost.py`, logs `auto-dispatch-m16-20261016.log`, `auto-dispatch-m2048-20261016.log`): on SM120 `auto` resolves to `[b12x, cutlass, cudnn]` and dispatches b12x, so the kernel is the same one as the explicit b12x row; the gap is host work. FlashInfer v0.7.1's `backend_requirement` wrapper (`flashinfer/utils.py`, `suitable_auto_backends`) runs every backend's requirement checker and the heuristic on every call with no cache; measured wall time per call auto minus b12x is 39 us at 16 x 4096 x 4096 and 21 us at 2048 x 14336 x 4096 (cProfile puts about 20 to 30 us per call in `suitable_auto_backends` and the autotuner cache lookups). A fixed host cost of this size is 20 to 45 percent at decode-size shapes and about 5 percent at M = 2048. The CUDA-event rows in the table include it because the GPU idles while the host resolves. Caching the resolution per argument signature would remove it; that is a FlashInfer change, not a kernel one.
- `mm_fp4` cutlass and b12x are within 2 percent of each other at M = 2048 (588 and 728 TFLOPS at 4096 x 4096 and 14336 x 4096); b12x is ahead at M = 256 on the 4096 x 4096 shape (109 against 85 TFLOPS). Against vLLM's CUTLASS FP8 kernel at the same shapes (278 and 402 TFLOPS), the NVFP4 GEMM delivers about 2x, which is the arithmetic ratio of the formats.
- The cuDNN `mm_fp4` backend refuses every one of these shapes with a block-scale dequantize descriptor error for the 128x4 reordering (recorded verbatim in the data); it is not a usable path for these calls on this part.
- **W4A16 (`mm_bf16_fp4`)**: cute-dsl reaches 143 to 184 TFLOPS at M = 2048, the cuDNN backend 39 to 44 TFLOPS at the same shapes (1.8 and 5.4 ms against 0.48 and 1.30 ms), a 3.7 to 4.6x gap; at M = 1 and 16 the two are within launch noise. vLLM's Marlin W4A16 at 2048 x 14336 x 4096 measured 213 TFLOPS in its own environment, so Marlin and cute-dsl are the two W4A16 paths worth comparing further; cuDNN's is not.
- `grouped_mm_fp4` (cuDNN, 128-aligned segments): 405 to 961 TFLOPS over the five MoE shapes, the highest at 128 experts x 512 tokens x 1536 x 4096; the 128 x 64 case (short segments) is the slowest at 285 TFLOPS.

## Dense GEMMs (median us, IQR; TFLOPS = 2MNK / median)

| M x N x K | mm_fp4 cutlass | mm_fp4 b12x | mm_fp4 auto | mm_bf16_fp4 cudnn (W4A16) | mm_bf16_fp4 cute-dsl (W4A16) |
|---|---|---|---|---|---|
| 1 x 4096 x 4096 | 70 us (63-104), 0 TFLOPS | 64 us (59-76), 1 TFLOPS | 84 us (81-101), 0 TFLOPS | 141 us (102-198), 0 TFLOPS | 57 us (52-70), 1 TFLOPS |
| 16 x 4096 x 4096 | 69 us (62-112), 8 TFLOPS | 111 us (64-158), 5 TFLOPS | 89 us (82-116), 6 TFLOPS | 98 us (85-150), 5 TFLOPS | 67 us (62-89), 8 TFLOPS |
| 256 x 4096 x 4096 | 67 us (62-96), 129 TFLOPS | 68 us (63-118), 126 TFLOPS | 147 us (95-208), 59 TFLOPS | 160 us (149-215), 54 TFLOPS | 143 us (126-200), 60 TFLOPS |
| 2048 x 4096 x 4096 | 117 us (108-149), 588 TFLOPS | 117 us (111-168), 588 TFLOPS | 149 us (134-213), 460 TFLOPS | 1783 us (1699-1929), 39 TFLOPS | 482 us (476-504), 143 TFLOPS |
| 16 x 14336 x 4096 | 84 us (80-108), 22 TFLOPS | 72 us (69-96), 26 TFLOPS | 93 us (91-104), 20 TFLOPS | 109 us (104-161), 17 TFLOPS | 81 us (77-106), 23 TFLOPS |
| 2048 x 14336 x 4096 | 330 us (326-363), 728 TFLOPS | 333 us (324-373), 723 TFLOPS | 363 us (354-409), 663 TFLOPS | 5411 us (5137-5706), 44 TFLOPS | 1304 us (1237-1335), 184 TFLOPS |
| 256 x 8192 x 8192 | 78 us (75-88), 442 TFLOPS | 80 us (78-91), 430 TFLOPS | 127 us (105-186), 271 TFLOPS | 1257 us (1233-1310), 27 TFLOPS | 337 us (331-374), 102 TFLOPS |

`mm_fp4 cudnn`: refused at all seven shapes (block-scale dequantize descriptor, 128x4 reordering).

## Grouped MoE GEMM, cuDNN (median us, IQR; TFLOPS over the whole grouped product)

| experts x tokens/expert x N x K | median | TFLOPS |
|---|---|---|
| 8 x 128 x 4096 x 4096 | 85 us (79-122) | 404 |
| 8 x 1024 x 4096 x 4096 | 389 us (375-407) | 707 |
| 64 x 128 x 2048 x 7168 | 522 us (507-537) | 460 |
| 128 x 64 x 2048 x 7168 | 845 us (818-891) | 285 |
| 128 x 512 x 1536 x 4096 | 858 us (837-892) | 961 |

## Instrument notes

- First use of `mm_fp4 cutlass` JIT-compiles `fp4_gemm_cutlass_sm120`; FlashInfer's ninja runs with no `-j`, which on this 16-core host started enough nvcc processes to fill the WSL memory cap (40 GB) in two minutes and froze the VM. The run uses a PATH shim (`~/bin-ninja-j3/ninja`, `exec ninja -j3`); with it the build held at 6 GB. Any script that can trigger a FlashInfer JIT build needs the shim.
- Not measured: `b12x_fused_moe` (W4A4 MoE; the census found it non-deterministic at M = 1 and 4, so its timing needs a determinism arm first), TensorRT-LLM, DeepGEMM, CUTLASS examples; a second card.
