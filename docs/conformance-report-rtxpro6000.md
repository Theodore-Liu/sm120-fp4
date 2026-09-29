# Conformance report: RTX PRO 6000 Blackwell Workstation Edition (SM120), 2026-09-29

The second SM12x machine for the stage 1 gate. Machine: NVIDIA RTX PRO 6000 Blackwell Workstation Edition (188 SMs,
96 GB), driver 610.57.04, a rented cloud instance (Ubuntu 24.04, Python 3.12), PyTorch 2.13.0+cu130, FlashInfer
0.6.16.post3, CUDA 13.2 toolkit for FlashInfer's JIT. The software is pinned to the versions of the RTX 5090 report, so
the two reports differ only in the machine. Run with `scripts/pod_conformance.sh`; everything it wrote is in
`reports/rtxpro6000-2026-09-29/`.

## Result: every verdict of the RTX 5090 report reproduces

| check | RTX 5090 (170 SMs) | RTX PRO 6000 (188 SMs) |
|---|---|---|
| 128x4 layout vs `nvfp4_block_scale_interleave`, 5 shapes | identical bytes | identical bytes |
| `fp4_quantize` scales vs reference | identical | identical |
| `fp4_quantize` codes vs reference | ulp-level midpoint ties only | ulp-level midpoint ties only (39, 0, 0 of 65,536, 524,288, 262,144) |
| `mm_fp4` `cutlass`, `b12x`, `auto`, 4 shapes | correct, max relative error 0.24% to 0.36% | correct, 0.26% to 0.34% |
| CUDA-graph replay, 200 replays, the same three backends | every replay equals eager | every replay equals eager |
| `mm_fp4` `cute-dsl` / `cudnn` / `trtllm` | refused | refused, with the same messages |
| tactics: CUTLASS / b12x presented, ran, correct | 32/32/32 and 8/8/8 on 5 shapes | 32/32/32 and 8/8/8 on 5 shapes |
| cute-dsl tactics built past the wrapper | 0 compile (`expects arch ... sm_100a ... got sm_120a`) | 0 compile, same error |
| pytest | 36 passed, 15 skipped | 36 passed, 15 skipped |

Timing at 4096x4096x4096 (median of 30, CUDA events; the one launch-bound-free shape):

| runner | fastest | slowest / fastest | default / fastest |
|---|---|---|---|
| CUTLASS | tactic 24, 111.8 us (1229 TFLOP/s) | 1.93 | 1.10 |
| b12x | (128,128), prefetch on, 113.9 us (1206 TFLOP/s) | 1.58 | 1.02 |

On the RTX 5090 the fastest CUTLASS tactic was 16 and the default was within 2% of it; here the fastest is 24 and the
default is 10% slower. The best tactic is device-specific even between two SM120 parts, which is the case for pinning a
tactic per device and shape rather than per architecture.

## What was not like-for-like, and the fix

The absolute errors in this run are not comparable with the 5090's for three of the four GEMM shapes. The reference
output's maximum differs between the machines (280 against 309 on the 16x4096x4096 shape), a gap far beyond fp32
summation order, so the two machines multiplied different matrices. The operands came from `torch.randn` on the GPU
with a fixed seed, and PyTorch's CUDA generator does not produce the same numbers on GPUs with different SM counts; only
the smallest shape's operands happened to coincide (its errors match to the last bit). The verdicts above do not depend
on this, since each machine checks its own kernels against its own reference.

The suite now generates operands on the CPU and copies them to the device, and the report records a checksum of every
GEMM output (`out_sha256_16`). With that change, on the RTX 5090, the CUTLASS and b12x backends return byte-identical
outputs on all four shapes. Whether the RTX PRO 6000 returns the same bytes is the next run.

## Running this on a cloud image: three traps

Each one stopped a run here before the fourth succeeded; `scripts/pod_conformance.sh` now handles all three, and the
failed runs' logs are kept as `reports/rtxpro6000-2026-09-29/env-attempt*-run.log`.

1. The image's `/usr/local/cuda` was CUDA 12.8. FlashInfer's JIT takes its toolkit from `CUDA_HOME`, else `nvcc` on
   `PATH`, else `/usr/local/cuda`; with 12.8 it logs `SM 12.x requires CUDA >= 12.9` and then fails with
   `FlashInfer requires GPUs with sm75 or higher`, which names the wrong cause.
2. Pointing `CUDA_HOME` at the pip CUDA packages instead fails differently: the pip `nvcc` (13.3) and the runtime
   headers that PyTorch's wheel pins (13.0) are different releases, and CCCL stops with `CUDA compiler and CUDA toolkit
   headers are incompatible`. A system toolkit from one release (13.2 from NVIDIA's apt repository) works.
3. FlashInfer runs `ninja` by name; a virtualenv that is not activated leaves it off `PATH`.
