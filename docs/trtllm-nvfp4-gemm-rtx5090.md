# TensorRT-LLM `nvfp4_gemm` on the RTX 5090, judged by the exact oracle (2026-10-10)

TensorRT-LLM issue 14154 reports `torch.ops.trtllm.nvfp4_gemm` numerically wrong on consumer Blackwell: correlation 0.113
against a bf16 reference on an RTX 5060 Ti (sm_120, TRT-LLM main of 2026-05-14, CUDA 13.2, driver 595) and deterministic garbage
on a GB10 (sm_121), the same on the CUTLASS, cuBLASLt and CuTe-DSL backends; a later reproduction on an RTX PRO 6000 (sm_120,
tensorrt_llm 1.2.1, torch 2.9.1) found the op correct (lossless operands bit-exact, correlation 0.9915 on random bf16). This
note adds the RTX 5090 (sm_120) with the oracle of `sm120fp4/oracle.py`, which needs no tolerance: lossless operands have an
exact integer answer, and random operands have an exact value with a per-element bound an fp32-accumulating kernel must respect.

## Setup

| item | value |
|---|---|
| device | NVIDIA GeForce RTX 5090 (sm_120, 170 SMs), driver 610.47 |
| tensorrt_llm | 1.3.0rc9 (NVIDIA pip index), torch 2.9.1+cu128, Python 3.12, WSL2 Ubuntu |
| operands | `sm120fp4.reference.quantize_nvfp4` (E2M1 codes, E4M3 block scales of 16, fp32 per-tensor scale), scales swizzled 128x4 with `sm120fp4.layouts.to_128x4`, alpha = 1 / (gs_a * gs_b) |
| shapes (M x N x K) | 16x4096x4096, 128x4096x4096, 1x4096x4096, 64x512x1024, 3x256x2048 |
| backends | `nvfp4_gemm_cutlass`, `nvfp4_gemm_cublaslt`, `nvfp4_gemm` (allowed "cutlass,cublaslt,cuda_core"), `cute_dsl_nvfp4_gemm_blackwell` |
| script, report | `scripts/trtllm_nvfp4_gemm_probe.py`, `reports/trtllm-nvfp4-gemm-rtx5090-20261010.json` |

## Result: correct on the RTX 5090, on every backend that runs; CuTe-DSL refuses SM120 loudly

| operands | backends | elements outside the oracle's bound | max error / bound | correlation with the exact value |
|---|---|---|---|---|
| lossless (codes on the E2M1 grid, all scales 1.0), 5 shapes | cutlass, cublaslt, default | 0 of 626,176 | 0.45 to 0.88 | 0.9999986 |
| random bf16 quantised, 5 shapes | cutlass, cublaslt, default | 0 of 626,176 | 0.44 to 0.87 | 0.9999986 |
| either | cute_dsl | refused: "CuteDSL NVFP4 backend requires SM 100 (B200) or SM 103 (B300), but got SM 120" | | |

On the lossless operands the largest absolute error is exactly half a bf16 ulp of the output magnitude (8.0 at outputs near
2900, 4.0 near 2100, 3.25 to 3.75 near 1200 to 1400), which is the output rounding and nothing else: the kernels return the
exact integer sum rounded once to bf16. The three runnable backends return identical numbers on every shape (identical error
statistics to the last digit), so on this device they share the result, as the issue's three backends shared their failure.

The issue's own reading, correlation against the bf16 product of the unquantised operands, is dominated by the quantisation
error and not by the kernel: the exact NVFP4 product correlates with that reference at the level the RTX PRO 6000 report
measured (about 0.99), and the kernel correlates with the exact product at 0.999999.

## What this says about issue 14154

The op is correct on a second sm_120 part with the current release candidate, under a tolerance-free check. Together with
the RTX PRO 6000 result this leaves the reported failure to the reporting environments (an RTX 5060 Ti under CUDA 13.2 /
driver 595 with a main-branch build, and a GB10 at sm_121), not to the SM120 source path; the thread's own toolchain
hypothesis (an E2M1 conversion that compiles everywhere and is wrong where the instruction is absent) was checked by one
participant on the PTX side and did not hold for sm_121a. What would settle it is the same probe on the failing machines:
the script is self-contained and needs only tensorrt_llm and this repository on the path.

Reproduce:

```
PYTHONPATH=/path/to/sm120-fp4 python scripts/trtllm_nvfp4_gemm_probe.py --out reports/trtllm-nvfp4-gemm-<device>-<date>.json
```
