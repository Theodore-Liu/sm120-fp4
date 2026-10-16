# The NVFP4 x NVFP4 GEMMs of FlashInfer and TensorRT-LLM return the same bits on the RTX 5090 (cross-library row of R1's semantics table)

Measured 2026-10-16. Operands: one fixed seed, `sm120fp4.reference.quantize_nvfp4` on the CPU (packed e2m1 codes, E4M3 block scales, the global scales), saved once and fed to every backend unchanged (`reports/nvfp4-cross-library-rtx5090/operands.pt`, not committed; the script regenerates it from the seed). Backends: FlashInfer 0.7.1 `mm_fp4` cutlass and b12x (torch 2.9.1+cu128); TensorRT-LLM 1.3.0rc9 `nvfp4_gemm_cutlass`, `nvfp4_gemm_cublaslt` and `nvfp4_gemm` with its default list "cutlass,cublaslt,cuda_core" (torch 2.9.1+cu128); bf16 output. Shapes: 16 / 256 / 2048 x 4096 x 4096, 256 x 8192 x 8192, and the off-grid 1000 x 4104 x 4128. Scripts `scripts/nvfp4_gemm_cross_library.py` (dump per library, then compare) and `scripts/nvfp4_cross_library_sanity.py`; data `cross-library-20261016.json`, `sanity-20261016.json`, the dump logs.

## Result

- **Every pair of backends that both accept a shape returns bit-identical bf16 output**: 43 of 43 compared pairs identical, 0 elements differing anywhere, over the five shapes (7 pairs skipped because TensorRT-LLM's cutlass backend, and therefore its default list, refuses N = 4104: "Expected n to be divisible by 32"; cuBLASLt and both FlashInfer backends take it).
- Every backend is deterministic across two calls at every shape.
- The outputs are not trivial: full density (non-zero fraction 1.0), 2,484 to 4,078 distinct bf16 values per output, absmax 576 to 948; at 256 x 4096 x 4096 every backend is 0 of 1,048,576 outside the exact oracle's fp32-accumulation bound (`sanity-20261016.json`).

## Reading

- The census judged each kernel against the exact oracle *within a bound*, which leaves room for different summation orders between libraries. This measurement closes that room for these two libraries on this part: FlashInfer's CUTLASS-based SM120 kernel, its b12x kernel, TensorRT-LLM's CUTLASS kernel and cuBLASLt's NVFP4 GEMM produce the same result to the bit, on-grid and off-grid, at every shape they all accept. A model served through any of them sees the same numbers from this layer.
- What this does and does not say: it is an observation on these operands and shapes (random bf16 inputs through one quantiser), not a proof of identical accumulation order; a difference would have been a finding, its absence is a row. DeepGEMM's mixed fp8 x fp4 and vLLM's own NVFP4 path are not in this comparison (different operand formats).
- Instrument note: TensorRT-LLM's cutlass and default ops take the swizzled scales as uint8 and refuse a float8_e4m3fn view ("mat1Scale dtype"); cuBLASLt takes either. The first dump passed fp8 views to all three and recorded refusals that were the instrument's, not the library's; corrected before the comparison.

## Pairs (identical at every accepted shape)

| pair | 16x4096x4096 | 256x4096x4096 | 2048x4096x4096 | 256x8192x8192 | 1000x4104x4128 |
|---|---|---|---|---|---|
| flashinfer b12x vs flashinfer cutlass | identical | identical | identical | identical | identical |
| flashinfer b12x vs trtllm cublaslt | identical | identical | identical | identical | identical |
| flashinfer b12x vs trtllm cutlass | identical | identical | identical | identical | skipped |
| flashinfer b12x vs trtllm default | identical | identical | identical | identical | skipped |
| flashinfer cutlass vs trtllm cublaslt | identical | identical | identical | identical | identical |
| flashinfer cutlass vs trtllm cutlass | identical | identical | identical | identical | skipped |
| flashinfer cutlass vs trtllm default | identical | identical | identical | identical | skipped |
| trtllm cublaslt vs trtllm cutlass | identical | identical | identical | identical | skipped |
| trtllm cublaslt vs trtllm default | identical | identical | identical | identical | skipped |
| trtllm cutlass vs trtllm default | identical | identical | identical | identical | skipped |
