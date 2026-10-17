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

## The bare instruction (`sm120fp4/kernels/mma_issue_rate.cu`, same day)

A register-only microbenchmark issues each `mma.sync` form in eight independent accumulator chains per warp, 4,096 iterations, with no memory traffic, and reads `clock64()` around the loop. With one block of eight warps per SM the per-SM accounting is exact; the wall-clock TFLOPS are at whatever clock the part ran (about 2.8 GHz for these sub-millisecond kernels, judged from the FP8-fp16 row against its sheet rate; nvidia-smi's 100 ms samples read 2,535 MHz). Report `reports/mma-issue-rate-rtx5090/mma_issue_rate.jsonl`.

| instruction | clocks per instruction per SM | FLOP per clock per SM | sheet row at boost | bare wall TFLOPS (about 2.8 GHz) |
|---|---|---|---|---|
| `m16n8k32.kind::f8f6f4` e4m3 x e4m3, **f32** accumulate | 8.0 | 1,024 | 419 (FP8, fp32 accumulate) | 492 |
| `m16n8k32.kind::f8f6f4` e4m3 x e4m3, **f16** accumulate | 4.0 | 2,048 | 838 (FP8, fp16 accumulate) | 924 |
| `m16n8k32.kind::f8f6f4` e2m1 x e2m1 in 8-bit containers, f32 accumulate | 8.0 | 1,024 | (no row; equals FP8 fp32) | 498 |
| `m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::4X` e2m1 x e2m1, ue4m3 scales, f32 accumulate | 4.0 (first run) / 2x the FP8-fp16 rate by wall clock (second run) | 4,096 | 1,676 (FP4 with fp32 accumulate) | 1,943 to 1,986 |

The sheet's four rows are the instruction set's four rates, exactly: fp32 accumulation halves FP8; e2m1 in containers buys nothing over FP8; the block-scaled k64 instruction, the one the SM120 CUTLASS kernels use (`s16864` in their names), issues 4,096 FLOP per clock per SM, the sheet's FP4 figure. The second run's one-block row read 6.2 clocks (2,635 FLOP/clock) while its wall TFLOPS stayed at 1,960; with 170 blocks on 170 SMs the block placement is not guaranteed one per SM, so the two-block rows and the wall clock are the reading there.

## Reading

The instruction reaches the sheet's FP4 rate; the kernels reach 48 percent of it. The best NVFP4 GEMM issues the block-scaled `mma.sync` at half the rate the bare instruction sustains, while Nsight Compute reports its tensor pipe 86 percent active and 86 percent of "peak sustained" instruction issue: Nsight's peak model for the tensor pipe does not match the bare k64 instruction's rate (it is consistent with the k32 rate), so its percentages are not a throughput reading for this instruction. The gap between 837 and 1,676 TFLOPS is therefore inside the kernels: the operand delivery to the tensor cores (shared-memory and register-file bandwidth for A and B fragments and their block scales), the tile schedule, and the epilogue, not the host's clock, not warm L2 at this shape, not DRAM bandwidth, and not the instruction. The FP8 side reads the opposite way: vLLM's FP8 kernel with fp32 accumulation runs at 96 percent of its instruction's rate (403 of 419 TFLOPS at boost), so FP8 on this part is at its ceiling and the ceiling is the fp32-accumulate half rate.

## What this does and does not say

- One shape profiled per kernel; the smaller shapes' gap has a second component (the L2 flush effect) that the profile does not separate from the pipe rate.
- Nsight's "percent of peak sustained" for the tensor pipe is relative to its own model of the pipe's peak; the FLOP-per-clock arithmetic above uses the sheet's figures and the measured clock, independently of that model, and the two agree on the conclusion.
- FlashInfer's and vLLM's NVFP4 kernels were not profiled; their throughput in the cost column (588 to 728 TFLOPS) sits below TensorRT-LLM's, so their gap has a kernel component as well.
- The microbenchmark's one-block-per-SM accounting assumes one block per SM; with exactly 170 blocks the scheduler does not guarantee it (the second run's 6.2-clock row). The wall-clock TFLOPS and the two-block rows do not depend on that assumption.
- Which part of the kernels' operand delivery costs the other half (shared-memory bandwidth for the fragments, the scale loads, the warp-specialised schedule) is the next question; a kernel-side ncu read of `smsp__inst_executed_pipe_lsu`, shared-memory bandwidth and warp stall reasons would separate them.
