# SM120 kernel census: the ledger

One row per shipped kernel path read against the exact oracle (`sm120fp4/oracle.py`) on the RTX 5090 (sm_120, driver 610.47). The four-week
test of the direction (opened 2026-10-10): fewer than three distinct confirmed defects by 2026-11-07 downgrades it. "Conformant" means every
output element inside the oracle's bound (exact product plus fp32-accumulation slack plus output rounding), on the shapes and operands the
report names, and is read with the detectability floor the oracle module documents (a one-code fault at K = 4096 is below fp32 noise).

| date | library, version | path | shapes, operands | verdict | report | note |
|---|---|---|---|---|---|---|
| 2026-10-10 | TensorRT-LLM 1.3.0rc9, torch 2.9.1+cu128 | `nvfp4_gemm` cutlass / cublaslt / default | 5 shapes x (lossless, random bf16) | conformant, 0 of 626,176 outside | `reports/trtllm-nvfp4-gemm-rtx5090-20261010.json` | issue 14154 does not reproduce here; CuTe-DSL refuses SM120 with a message |
| 2026-10-10 | vLLM 0.28.0, torch 2.13+cu130 | `cutlass_scaled_fp4_mm` | 5 shapes x (vLLM-quantised, lossless) | conformant, 0 of 626,176 outside | `reports/vllm-nvfp4-rtx5090-20261010.json` | alpha must be a CUDA tensor (interface, not numerics) |
| 2026-10-10 | vLLM 0.28.0 | `scaled_fp4_quant` | 5 operand pairs | conformant once ties are classified | same, plus `reports/vllm-fp4-quant-*-20261010.json` | convention: GPU quantisers round E4M3 scale ties and E2M1 code ties half away from zero, torch's cast to even; vLLM and FlashInfer agree with each other (`docs/vllm-nvfp4-rtx5090.md`) |
| 2026-09-29, 2026-10-10 | FlashInfer 0.6.16.post3 | `mm_fp4` cutlass / b12x / auto | 4 shapes | conformant (stage 1 relative check, now the oracle form in `tests/test_mm_fp4.py`, 12 pass) | `docs/conformance-report-rtx5090.md` | cute-dsl / cudnn / trtllm refuse SM120 |
| 2026-09-29, 2026-10-10 | FlashInfer 0.6.16.post3 and 0.7.1 | `b12x_fused_moe` W4A4 | top-8, intermediate 768, M = 1 / 4 / 16 | **non-deterministic** at M = 1 and 4 (20 of 20 distinct, up to 1.84 percent), bit-stable at M = 16 | `reports/b12x-nondeterminism-rtx5090-*.json` | bf16 atomic scatter order; the lost 8-column group of 0.6.16 did not recur on 0.7.1 in 100 runs (`docs/stage2-baselines.md`) |
| 2026-10-10 | FlashInfer 0.7.1, cuDNN 9.27 | `grouped_mm_fp4` cudnn | 9 segment patterns, n 256, k 512 | **defect**: every segment starting off a 128-row boundary wrong across the segment, deterministic, no error | `reports/flashinfer-grouped-fp4-alignment-rtx5090-fi071-20261010.json` | aligned segments conformant; `grouped_mm_fp8` same segments conformant (`docs/flashinfer-grouped-fp4-rtx5090.md`) |
| 2026-10-10 | FlashInfer 0.7.1, cuDNN 9.27 | `grouped_mm_mxfp8` cudnn | 5 segment patterns | **defect**, the same rule; NaNs on the 16/48/64 case | `reports/flashinfer-grouped-mxfp8-alignment-rtx5090-fi071-20261010.json` | the block-scale handling for both block-scaled formats |
| 2026-10-10 | FlashInfer 0.7.1, cuDNN 9.27 | `grouped_mm_fp8` cudnn | 5 segment patterns | conformant | `reports/flashinfer-grouped-fp8-alignment-rtx5090-fi071-20261010.json` | the control for the two rows above |
| 2026-10-10 | vLLM 0.28.0 | `cutlass_fp4_moe_mm` (grouped NVFP4, vLLM's own data preparation) | 4 routings, segments of arbitrary length | conformant, 0 of 610,304 outside | `reports/vllm-fp4-moe-mm-rtx5090-20261010.json` | the control for the cuDNN finding: per-expert scale regions padded to 128 rows (`blockscale_offsets`) |
| 2026-10-10 | FlashInfer 0.7.1, cuDNN 9.27 | `mm_bf16_fp4` (W4A16) cudnn / cute-dsl / cute-dsl-native | 5 shapes, bf16 activations x reference-quantised NVFP4 weights | conformant, 0 of 365,312 outside on each backend | `reports/flashinfer-mm-bf16-fp4-rtx5090-fi071-20261010.json` | blackwell-native / blackwell-tiled refuse SM120 with a message; the W4A16 oracle is `exact_gemm_bf16_nvfp4` |
| 2026-10-10 | FlashInfer 0.6.16.post3 | `mm_bf16_fp4` cute-dsl | the same 5 shapes | conformant, identical numbers to 0.7.1 | `reports/flashinfer-mm-bf16-fp4-rtx5090-fi0616-20261010.json` | cudnn refused (needs 9.23.1); the other backend names do not exist in 0.6.16 |
| 2026-10-10 | FlashInfer 0.6.16.post3 | `grouped_mm_fp4` cudnn | any | refused: cuDNN 9.20 below the required 9.21 | `reports/flashinfer-grouped-fp4-rtx5090-20261010.json` | a clear message, not a defect |
| 2026-10-11 | torch 2.13.0+cu130 and 2.9.1+cu128 (cuBLASLt) | `torch._scaled_mm` FP8 e4m3, per-tensor and rowwise scales | 5 shapes, 1,085,440 elements per path | conformant, 0 outside on both torch versions, max 0.961 of bound | `reports/fp8-gemm-rtx5090-{vllm028,fi071}-20261011.json` | `docs/fp8-gemm-rtx5090.md` |
| 2026-10-11 | vLLM 0.28.0 | `cutlass_scaled_mm` FP8: per-tensor, per-token x per-channel, 1x128 x 128x128 blockwise | the same 5 shapes | conformant, 0 outside on all three scale kinds | `reports/fp8-gemm-rtx5090-vllm028-20261011.json` | the blockwise path needs the activation scale column-major, as vLLM's own caller writes it; row-major reads wrong for every m > 1 (an operand contract that lives in the caller, not the docstring) |
| 2026-10-11 | FlashInfer 0.6.16.post3 | `mm_fp8` trtllm_low_latency | 5 shapes | **no capability guard**: SIGSEGV at m = 1, 128, 256 (sm100f cubins on sm_120), RuntimeError at m = 16, 512 | same | **fixed upstream**: 0.7.1 refuses SM120 with a message (`BackendSupportedError`); logged, not reported |

## Tally toward the four-week test

Distinct defect classes found and reproduced: **2** (the b12x W4A4 scatter non-determinism; the cuDNN grouped block-scaled GEMM's mid-tile
segments). Confirmed by maintainers: 0 (both issue drafts await the author's review before posting). Conventions documented, not defects: 1
(E4M3 / E2M1 tie rounding, GPU quantisers against torch). Reports that did not reproduce: 1 (TensorRT-LLM issue 14154 on this part). Crashes on an unsupported part already guarded in the next release: 1 (FlashInfer 0.6.16 `mm_fp8`).

## Not yet read

FlashInfer b12x W4A16 fused MoE (the dense W4A16 GEMM above is conformant); CUTLASS's
SM120 block-scaled examples; FlashInfer's cuDNN / CUTLASS dense FP8 and MXFP8 GEMMs (`mm_fp8` has only the TRT-LLM backend; the cuDNN and SM120 CUTLASS FP8 runners sit behind `bmm_fp8`); DeepGEMM; llama.cpp q4_0 / q8_0 GEMV; attention kernels (an exact
oracle for attention is the hard instrument and is not written).

## An instrument note

The first W4A16 run read every backend non-conformant with identical statistics on both backends. Two independent backends agreeing with each
other against the oracle is the signature of a wrong operand on the caller's side, not of two kernels wrong the same way; the cause was the
scale layout (`prepare_bf16_fp4_weights` takes the 128x4-swizzled scales, as its docstring says; the probe had passed them row-major). Fixed, every
backend is conformant. The rule for the census: a verdict of non-conformant is written down only after the operand contract has been read
from the library's own documentation and a second backend or library disagrees with the first.

A second note (2026-10-11, `docs/fp8-gemm-rtx5090.md`): a fault that vanishes exactly at m = 1 is a layout fault in a per-row operand,
not a kernel fault, since at one row every layout of an (m, g) tensor is the same bytes. vLLM's blockwise FP8 path takes its
activation scale column-major, which its docstring does not say and its own caller does; read the caller before the verdict.
