# Dense FP8 GEMM paths on the RTX 5090 against the exact oracle (2026-10-11)

Script: `scripts/fp8_gemm_census.py`. Reports: `reports/fp8-gemm-rtx5090-vllm028-20261011.json` (torch 2.13.0+cu130, vLLM 0.28.0,
FlashInfer 0.6.16.post3) and `reports/fp8-gemm-rtx5090-fi071-20261011.json` (torch 2.9.1+cu128, FlashInfer 0.7.1; vLLM not
installed there). Device: NVIDIA GeForce RTX 5090 (sm_120), driver 610.47, WSL2.

## The oracle

An e4m3 value is exact in fp64, and so is an fp32 scale, so the exact value of a scaled FP8 GEMM element is the fp64 sum of the
quantised products times the scales. The bound a correct fp32-accumulating kernel must respect is the census's usual
`K * 2^-24 * sum |a_i b_i|` (scaled), plus two fp32 roundings for the epilogue's scale multiplies (`2 * 2^-24 * |exact|`), plus
half an ulp of the bf16 output, plus the fp64 arithmetic's own slack. Every element is judged; no tolerance.

Five shapes (m, n, k): (1, 4096, 4096), (16, 2048, 7168), (128, 4096, 1024), (256, 1024, 2048), (512, 512, 512); 1,085,440
output elements per path. Operands: a (m, k) and w (n, k) drawn on the CPU with a per-shape seed and cast to e4m3; scales
positive fp32 in [0.005, 0.025).

## Results

| path | scales | torch 2.13 / vLLM 0.28 env | torch 2.9.1 / FlashInfer 0.7.1 env |
|---|---|---|---|
| `torch._scaled_mm` (cuBLASLt) | per-tensor | conformant, 0 of 1,085,440 outside, max 0.954 of bound | conformant, same numbers |
| `torch._scaled_mm` | rowwise (m,1) x (1,n) | conformant, 0 outside, max 0.961 | conformant, same numbers |
| vLLM `cutlass_scaled_mm` | per-tensor | conformant, 0 outside, max 0.954 | - |
| vLLM `cutlass_scaled_mm` | per-token x per-channel | conformant, 0 outside, max 0.961 | - |
| vLLM `cutlass_scaled_mm` | 1x128 activations x 128x128 weights (DeepSeek-V3 style) | conformant, 0 outside, max 0.966 | - |
| FlashInfer `mm_fp8` `trtllm_low_latency` | per-tensor alpha | **no capability guard**: SIGSEGV in `cuFuncSetAttribute` at m = 1, 128, 256 after downloading `sm100f` cubins; `RuntimeError: Error occurred when running low latency TRTLLM-GEN GEMM` at m = 16, 512 | refused with a message: `BackendSupportedError: mm_fp8 does not support backend 'trtllm_low_latency' with capability 120` |

The torch and vLLM per-tensor and rowwise rows return bit-identical maxima on every shape; the two libraries' SM120 FP8 paths
agree with each other and with the oracle.

The FlashInfer 0.6.16 row is a crash on an unsupported part rather than wrong output, and 0.7.1 already carries the guard
(`@supported_compute_capability` on `mm_fp8`), so it is logged as fixed upstream, not reported.

## An instrument note: the column-major activation scale

The first blockwise run read vLLM's 128-block path non-conformant on every shape but (1, 4096, 4096), with 97 to 99.6 percent
of elements outside the bound. A fault that disappears exactly at m = 1 is a layout fault in a per-row operand, not a kernel
fault: at one row every layout of an (m, k/128) tensor is the same bytes. vLLM's own caller (`CutlassFp8BlockScaledMMKernel`)
quantises activations with `QuantFP8(..., column_major_scales=True)`, so `scale_a` is the (m, k/128) tensor in column-major
order, and passes `scale_b = Bs.T`, the transposed view of the (n/128, k/128) weight scales. With the activation scale laid out
column-major the path is conformant on all five shapes. The `cutlass_scaled_mm` docstring describes the broadcasting shapes
but not this layout; the contract lives in the caller.

A second caller-side fault found on the way: the oracle's first version inferred the scale kind from the tensor shapes, and on
the (1, 4096, 4096) shape a (1, n) column scale has the same last dimension as k, so it was read as a per-element scale. The
oracle now takes the scale kind explicitly.

## Addendum (2026-10-11, later): `bmm_fp8` on its four backends

`scripts/fp8_gemm_census.py --bmm-backends ...` reads FlashInfer 0.7.1's `bmm_fp8` (A (b, m, k) row-major, B (b, k, n) column-major, per-tensor
scales) on the same five shapes, `reports/fp8-gemm-bmm-rtx5090-fi071-20261011.json`:

| backend | result |
|---|---|
| cublas | conformant, 0 of 1,085,440 outside; bit-identical to the `torch._scaled_mm` rows |
| cudnn | conformant, 0 outside; bit-identical to cublas |
| cute-dsl | refuses SM120 with `BackendSupportedError` |
| **cutlass** | **takes the WSL2 virtual machine down.** `scripts/bmm_fp8_isolate.py` writes a marker to the Windows side before and after one call; for cutlass the "calling" marker is the last line twice over (`reports/bmm-fp8-isolate-rtx5090-fi071-20261011.log`), Windows then reports `Wsl/Service/E_UNEXPECTED` for every WSL command until the distribution is terminated (`wsl -t Ubuntu`) and restarted. The first attempt to read all four backends in one process left no report and no error, which is how the crash was noticed. |

The crash is a defect by any contract (a GEMM call must not end the process, let alone the machine it runs in), but it was observed
under WSL2's GPU paravirtualisation only; the census owes a bare-metal Linux reading, where the same fault would surface as an Xid in
the kernel log or an exception, before anything is reported upstream. The census script leaves cutlass out of its default backend
list because no try/except can catch this.
