# Why the best NVFP4 GEMM on the RTX 5090 reads half the specified FP4 rate (2026-10-17)

Question (discovery queue `r1-why-half-the-rated-fp4-rate`): TensorRT-LLM's NVFP4 GEMM reaches 837 TFLOPS at 2048 x 14336 x 4096 on the RTX 5090, and NVIDIA's specification lists 1,676 TFLOPS dense FP4 (3,352 with sparsity). Is the gap the host's clock, warm-L2 operands in our timing loop, memory bandwidth, or the tensor pipe's issue rate? Scripts `scripts/nvfp4_gemm_rate_gap.py` (timing under three conditions with the SM clock sampled), `scripts/ncu_trtllm_nvfp4_driver.py` and `scripts/ncu_vllm_fp8_driver.py` (Nsight Compute drivers); reports in `reports/nvfp4-rate-gap-rtx5090/`. WSL2 host, Nsight Compute 2026.1.1 (counters are readable under WSL2 on this machine).

## What the specification says (RTX Blackwell GPU Architecture whitepaper v1.1, Appendix A, Table 3, RTX 5090 column; "peak rates are based on GPU Boost Clock", 2,407 MHz, 170 SMs)

| row | dense / sparse TFLOPS | FLOP per clock per SM at boost |
|---|---|---|
| FP4 Tensor with FP32 accumulate ("FP4 AI TOPS") | 1,676 / 3,352 | 4,096 |
| FP8 Tensor with FP16 accumulate | 838 / 1,676 | 2,048 |
| FP8 Tensor with FP32 accumulate | 419 / 838 | 1,024 |

So the sheet itself halves FP8 when the accumulator is fp32 (the GeForce segmentation that Ada also had), and lists FP4 with fp32 accumulate at four times the FP8-fp32 rate. The `mma.sync` e2m1 instruction the SM120 kernels use accumulates in fp32 only.

## Measured (`trtllm.json`; CUDA events, median of 50 after 10 warm-ups; SM clock sampled with nvidia-smi during the loop)

| backend | shape | warm TFLOPS | L2 flushed TFLOPS | clock MHz | FLOP/clock/SM (warm) | of the sheet's FP4 rate | of the FP8-fp16acc rate |
|---|---|---|---|---|---|---|---|
| cutlass | 2048 x 14336 x 4096 | 858 | 829 | 2,542 | 1,986 | 0.49 | 0.97 |
| cublaslt | 2048 x 14336 x 4096 | 843 | 818 | 2,557 | 1,940 | 0.47 | 0.95 |
| default | 2048 x 14336 x 4096 | 807 | 775 | 2,557 | 1,856 | 0.45 | 0.91 |
| cutlass | 2048 x 4096 x 4096 | 689 | 474 | 2,482 | 1,633 | 0.40 | 0.80 |
| cublaslt | 2048 x 4096 x 4096 | 748 | 445 | 2,482 | 1,772 | 0.43 | 0.87 |
| cutlass | 256 x 8192 x 8192 | 494 | 294 | 2,542 | 1,142 | 0.28 | 0.56 |
| default | 256 x 8192 x 8192 | 643 | 290 | 2,542 | 1,487 | 0.36 | 0.73 |

- **Not the clock.** The SM clock during the timed loops was 2,450 to 2,610 MHz, above the 2,407 MHz boost the sheet's figures assume. Per clock per SM the best kernel delivers 1,986 FLOP: 48 percent of the 4,096 the FP4 row implies, and 97 percent of the 2,048 of the FP8-with-fp16-accumulate row.
- **Warm L2 matters at the smaller shapes, not at the largest.** Writing 256 MB between iterations costs 3 to 4 percent at 2048 x 14336 x 4096 (operands 35 MB, re-read from DRAM once per call either way) but 31 to 40 percent at 2048 x 4096 x 4096 and 41 to 55 percent at 256 x 8192 x 8192. The cost column's numbers at those shapes are warm-L2 numbers; the flush also leaves the kernel cold (instruction cache, TLB), and the two effects are not separated here.

## Nsight Compute, one call each at 2048 x 14336 x 4096 (ncu locks the clock to 1.68 to 2.27 GHz while profiling, so read ratios, not rates)

| kernel | tensor pipe cycles active (% of peak sustained) | tensor instructions issued (% of peak sustained) | DRAM throughput (% of peak) | L2 sectors (% of peak) | warps active (% of peak) |
|---|---|---|---|---|---|
| TensorRT-LLM cutlass NVFP4 (`Sm12xOnly<GemmUniversal<... MainloopSm120TmaWarpSpecializedBlockScaled<4, 3 ...`) | 86.0 | 86.0 | 24.0 | 71.4 | 18.8 |
| TensorRT-LLM cuBLASLt NVFP4 (`cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1 ... 128x128x256`) | 81.2 | 81.2 | 17.4 | 70.8 | 20.9 |
| vLLM CUTLASS FP8 (`enable_sm120_family<GemmUniversal<... MainloopSm120TmaWarpSpecialized<2, 2 ...`), 403 TFLOPS warm, 401 flushed | 96.4 | 48.1 | 10.5 | 45.1 | 20.8 |

- The NVFP4 kernels keep the tensor pipe busy 81 to 86 percent of active cycles and issue tensor instructions at 81 to 86 percent of what Nsight counts as the sustained peak, with DRAM at a quarter of peak. **The kernels are not the gap**: a kernel at 86 percent issue cannot double its throughput. The remaining 14 percent is the ordinary prologue, epilogue and tail of a 112-CTA grid on 170 SMs.
- The FP8 kernel with fp32 accumulation issues tensor instructions at 48 percent of peak while its pipe is active 96 percent of cycles: the fp32-accumulate half-rate of the sheet's FP8 row, seen as an instruction issue rate. Its 403 TFLOPS is 96 percent of the sheet's 419 for FP8 with fp32 accumulate (at the measured 2.5 GHz, 92 percent).

## Reading

The best NVFP4 kernel on this part runs the `mma.sync` e2m1 path at 97 percent of 2,048 FLOP per clock per SM, which is the sheet's FP8-with-fp16-accumulate rate and twice its FP8-with-fp32-accumulate rate, with the pipe 86 percent busy. The sheet's FP4 figure, 4,096 per clock per SM, is twice what this instruction path delivers at near-full issue. Either the specified FP4 rate needs an instruction form these kernels do not use, or the figure does not apply to fp32 accumulation on this part the way the FP8 rows do. Which of the two is not decided here; what is decided is that the gap between 837 and 1,676 TFLOPS is not the WSL2 host's clock, not warm L2 at this shape, not memory bandwidth, and not an under-utilised kernel.

## What this does and does not say

- One shape profiled per kernel; the smaller shapes' gap has a second component (the L2 flush effect) that the profile does not separate from the pipe rate.
- Nsight's "percent of peak sustained" for the tensor pipe is relative to its own model of the pipe's peak; the FLOP-per-clock arithmetic above uses the sheet's figures and the measured clock, independently of that model, and the two agree on the conclusion.
- FlashInfer's and vLLM's NVFP4 kernels were not profiled; their throughput in the cost column (588 to 728 TFLOPS) sits below TensorRT-LLM's, so their gap has a kernel component as well.
- A microbenchmark of a bare `mma.sync.kind::f8f6f4` loop (no memory traffic) would settle the instruction's issue rate directly; it is the next step and is cheap.
