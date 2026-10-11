# vLLM's NVFP4 quantiser and GEMM on the RTX 5090, judged by the exact oracle (2026-10-10)

vLLM 0.28.0 (torch 2.13.0+cu130, driver 610.47, sm_120): `vllm._custom_ops.scaled_fp4_quant` and `cutlass_scaled_fp4_mm`, checked with
`sm120fp4/oracle.py`. Scripts `scripts/vllm_nvfp4_probe.py`, `scripts/diag_vllm_fp4_quant_scales.py`, `scripts/diag_vllm_fp4_quant_mismatch.py`;
reports `reports/vllm-nvfp4-rtx5090-20261010.json`, `reports/vllm-fp4-quant-scales-rtx5090-20261010.json`, `reports/vllm-fp4-quant-mismatch-rtx5090-20261010.json`.

## GEMM: conformant on every shape

| operands | shapes (M x N x K) | elements outside the bound | max error / bound |
|---|---|---|---|
| quantised by vLLM's own `scaled_fp4_quant` | 16/128/1 x 4096 x 4096, 64x512x1024, 3x256x2048 | 0 of 626,176 | 0.44 to 0.87 |
| lossless (codes on the E2M1 grid, scales 1.0) | the same five | 0 of 626,176 | 0.45 to 0.88; largest error exactly half a bf16 ulp |

`cutlass_scaled_fp4_mm` on SM120 returns the exact product rounded once to bf16 within fp32-accumulation slack, as TensorRT-LLM's
`nvfp4_gemm` does (`docs/trtllm-nvfp4-gemm-rtx5090.md`). The op requires `alpha` on the device (a CPU tensor raises
"alpha: must be a CUDA tensor"); that is an interface rule, not a numerical one.

## Quantiser: conformant, with one convention the oracle made visible

On four of five operand pairs vLLM's codes and E4M3 block scales equal the reference quantiser's byte for byte (the codes that
differ sit on E2M1 midpoints, where two roundings are valid). On one operand (64 x 1024, `global_scale` 307.2) **41 of 4,096 block
scales differ by one E4M3 step, every one of them at an exact E4M3 midpoint**: block amax 2.265625 gives a scale of exactly 116.0,
halfway between 112 and 120; amax 4.53125 gives 232.0, halfway between 224 and 240. The reference (torch's `float8_e4m3fn` cast)
rounds these ties to even (112, 224); vLLM rounds them up (120, 240), and so does FlashInfer 0.6.16's `fp4_quantize` on the same
blocks (41 of 41 agree with vLLM, 0 with the reference). The 117 code differences on that operand are the consequence of the
41 scale choices, not independent code errors: both libraries then encode the block consistently with the scale they chose.

The codes show the same two conventions one level down. On the 512 x 1024 operand, 444 codes differ with the scales identical,
every one at an exact E2M1 midpoint (|x| x gs / sf = 2.5, 1.25, 0.25): the reference rounds to the even code, vLLM and FlashInfer
to the odd one, and the two libraries agree with each other at all 444. On the 64 x 1024 operand a further 33 codes differ
within one fp32 ulp of a midpoint (0.2500000298 against 0.25): the reference multiplies by the rounded quotient gs / sf, the
kernels evaluate x x gs / sf in another order, and the two land on different sides of the midpoint; again the libraries agree
with each other.

So the two CUDA quantisers agree with each other, round E4M3 scale ties and E2M1 code ties half away from zero, and evaluate the
scaled value in an order that differs from the torch reference by one ulp at midpoints; a torch-based quantiser rounds ties to even. Both are valid roundings of the same number and neither is a defect; it is an interoperability fact: the same bf16 tensor
quantised on the GPU and on the host can differ in a block's scale and codes wherever the scale lands on a midpoint, which a
"nice" global scale (448 x 6 / amax with a bf16 amax) makes common. `oracle.quantizer_conformance` now reports scale ties apart
(`scales_differ_at_tie`, both values neighbours of the midpoint), counts the codes of such blocks with the tie, and treats a code within
two fp32 ulps of an E2M1 midpoint as a tie; the verdict for vLLM's quantiser is conformant on all five pairs, and a scale or code that
differs off a midpoint still fails it.

Why the stage 1 suite called the scales "identical": its operands were plain `randn` in fp32 with a `randn`-derived global scale,
under which exact midpoints are rare; the probe's bf16 operands with a bf16 amax make the global scale and the block scales land
on short binary fractions, where ties are common. A census that wants to see conventions has to feed the values that reach them.
