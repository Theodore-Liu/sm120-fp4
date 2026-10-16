# The NVFP4 activation quantisers of vLLM, FlashInfer and TensorRT-LLM emit the same bytes on the RTX 5090; they differ from the CPU reference only at exact ties (cross-library quantiser row of R1's semantics table)

Measured 2026-10-16. Inputs: bf16 matrices from one seed (16 / 256 / 2048 x 4096, 1000 x 4128, 4097 x 8192) plus one 1 x 128 block built on e2m1 midpoints, each with its per-tensor global scale, saved once and fed to every quantiser (`inputs.pt`, not committed; regenerated from the seed). Quantisers: vLLM 0.28.0 `scaled_fp4_quant` (torch 2.13.0+cu130), FlashInfer 0.7.1 `nvfp4_quantize` with the linear scale layout (torch 2.9.1+cu128), TensorRT-LLM 1.3.0rc9 `fp4_quantize` with `isSfSwizzledLayout=False` (torch 2.9.1+cu128), and the CPU reference `sm120fp4.reference.quantize_nvfp4`. Packed e2m1 codes and E4M3 block scales compared byte for byte. Scripts `scripts/nvfp4_quantize_cross_library.py` (dump per environment, then compare) and `scripts/nvfp4_quantize_cross_library_diag.py`; data `quant-cross-library-20261016.json`, `diag-20261016.json`, the dump logs.

## Result

- **The three GPU quantisers are byte-identical to each other on every input**: codes and scales, all six shapes, 0 bytes differing in any pair (vLLM vs FlashInfer, vLLM vs TensorRT-LLM, FlashInfer vs TensorRT-LLM).
- **Block scales are identical across all four quantisers, the CPU reference included**, on every input.
- Codes differ between the GPU quantisers and the CPU reference on 12,666 of 47,193,472 elements (0.0268 percent), on three of the six inputs (0 on 256 x 4096, on 1000 x 4128 and on the constructed tie block). Every differing element is a tie: the scaled value sits exactly at the midpoint between two adjacent e2m1 values (distance from the midpoint at most 2.4e-7 of the step) and the two codes are one step apart. The GPU convention at these ties is the one the census documented for vLLM's quantiser (`docs/census-ledger.md`, row of 2026-10-10: ties classified before the conformance verdict); this measurement shows FlashInfer and TensorRT-LLM share it to the byte.

## Reading

- Together with the GEMM row (`nvfp4-gemm-cross-library-rtx5090.md`): on this part, a bf16 activation quantised and multiplied through vLLM, FlashInfer or TensorRT-LLM returns the same bits end to end. The libraries are interchangeable at this layer, and a numerical difference between two of these stacks must come from elsewhere (attention, norms, the KV cache, sampling).
- The CPU reference is the odd one out, and only at exact midpoints, which random bf16 data reaches at a rate of about one in ten thousand elements here. A reference written in exact arithmetic has to choose a tie rule; the hardware's `cvt` instruction chose it already, and three libraries inherit it.
- Scope: per-tensor global scale with 16-wide E4M3 block scales (the NVFP4 recipe); the linear scale layout (the swizzled layouts are a permutation of the same bytes, covered by the GEMM row).

## Pairs

| pair | 16x4096 | 256x4096 | 2048x4096 | 1000x4128 | 4097x8192 | 1x128 |
|---|---|---|---|---|---|---|
| flashinfer vs reference | 23 code bytes, 0 scales | identical | 9609 code bytes, 0 scales | identical | 3012 code bytes, 0 scales | identical |
| flashinfer vs trtllm | identical | identical | identical | identical | identical | identical |
| flashinfer vs vllm | identical | identical | identical | identical | identical | identical |
| reference vs trtllm | 23 code bytes, 0 scales | identical | 9609 code bytes, 0 scales | identical | 3012 code bytes, 0 scales | identical |
| reference vs vllm | 23 code bytes, 0 scales | identical | 9609 code bytes, 0 scales | identical | 3012 code bytes, 0 scales | identical |
| trtllm vs vllm | identical | identical | identical | identical | identical | identical |
