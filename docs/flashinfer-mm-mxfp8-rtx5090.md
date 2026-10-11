# FlashInfer's dense MXFP8 GEMM (`mm_mxfp8`) on its backends (RTX 5090, 2026-10-11)

Script: `scripts/mm_mxfp8_isolate.py`, one backend per process with a marker file on the Windows side before every call (the lesson
of `bmm_fp8`'s cutlass backend, which took the WSL2 virtual machine down and left nothing else behind). Stack: FlashInfer 0.7.1,
PyTorch 2.9.1+cu128, driver 610.47, WSL2. Operands from `mxfp8_quantize(..., True, 32)` as the docstring asks (A row-major with its
1D swizzled 128x4 scales, B the transposed view of a row-major (n, k) with its 1D swizzled scales), the oracle the exact float64
product of `mxfp8_dequantize_host`'s values with the census's accumulation bound. Five shapes, 1,085,440 elements per backend.

| backend | result |
|---|---|
| cudnn | conformant, 0 outside, max 0.927 of bound |
| cutlass | conformant, 0 outside, max 0.927 |
| b12x | conformant, 0 outside, max 0.927 |
| auto | conformant, 0 outside, max 0.927 (the first call took 56 s: autotuning) |
| cute-dsl, cutedsl_low_latency, trtllm | refuse SM120 with `BackendSupportedError` |

No backend crashed. The CUTLASS MXFP8 path therefore runs on this device where the CUTLASS `bmm_fp8` path does not survive a
single call; the `bmm_fp8` crash is that path's own, not a property of CUTLASS on SM120 under WSL2. Marker log:
`reports/mm-mxfp8-isolate-rtx5090-fi071-20261011.log`.
