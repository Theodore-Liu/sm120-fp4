# FlashInfer `grouped_mm_fp4` on the RTX 5090: wrong output for expert segments that start mid-tile (2026-10-10)

FlashInfer 0.7.1, cuDNN 9.27.0 (the op requires cuDNN 9.21 or later; the pinned 0.6.16 environment's 9.20 is refused with a clear
message), PyTorch 2.9.1+cu128, RTX 5090 (sm_120), driver 610.47. Scripts `scripts/flashinfer_grouped_fp4_probe.py`,
`scripts/flashinfer_grouped_fp4_alignment.py`, `scripts/flashinfer_grouped_fp4_hypothesis.py`, `scripts/flashinfer_grouped_fp4_solve_scales.py`,
`scripts/flashinfer_grouped_fp4_search_addressing.py`; reports `reports/flashinfer-grouped-fp4-*-rtx5090-fi071-20261010.json`. The oracle is
`sm120fp4.oracle.exact_grouped_gemm_nvfp4` (the exact product expert by expert, with the fp32-accumulation bound).

## What was measured

Tokens quantised by the reference quantiser under one global scale, expert weights each quantised under one shared global scale (the op
takes a single alpha), scales swizzled 128x4 over all tokens at once (the layout FlashInfer's own `tests/grouped_mm/test_grouped_mm_fp4.py`
builds with `nvfp4_quantize(..., sfLayout=layout_128x4)` on the whole `(cum_m, k)` tensor), routing as `m_indptr`, n = 256, k = 512.

| segments (tokens per expert) | elements outside the bound, per segment | verdict |
|---|---|---|
| 128, 128, 128, 128 | 0, 0, 0, 0 | conformant |
| 256, 128, 384, 256 (FlashInfer's own test case) | 0, 0, 0, 0 | conformant |
| 0, 128, 0, 128 | 0, 0, 0, 0 | conformant |
| 64, 64 | 0, 16,297 of 16,384 | the second segment wrong |
| 120, 136 | 0, 34,591 of 34,816 | the second segment wrong |
| 16, 48, 64 | 0, 12,232 of 12,288, 16,291 of 16,384 | the second and third wrong |
| 32, 96 | 0, 24,393 of 24,576 | the second wrong |
| 128, 64, 64 | 0, 0, 16,286 of 16,384 | the third wrong (starts at row 192) |
| 192, 64 | 0, 16,258 of 16,384 | the second wrong (starts at row 192) |

The rule the table shows: **a segment is correct when its first row is a multiple of 128 and wrong otherwise**, independent of its length
(64, 96, 136 rows all wrong when mid-tile; 64 rows correct when aligned, as the 0, 128, 0, 128 case's second block of 128 and the first
probe's 2-expert case with segments 0 and 32 show). The first probe (`flashinfer_grouped_fp4_probe.py`) saw the same with 8 experts of
random lengths, where 24,576 of 131,072 outputs were NaN. The op raises nothing; its documentation states no alignment requirement on
`m_indptr`; its test suite uses segment lengths that are multiples of 128 only and judges with a cosine similarity of 0.9.

## What the wrong values are, and are not

- Deterministic: two identical calls give identical wrong outputs.
- Not a row permutation: each wrong output row is nearest to its own token's exact product (max error about 20 on outputs of
  magnitude about 30), and matches no token's exact product within the bound.
- Not a re-addressing of the scale buffer that these tests can name: the scales the kernel applied were solved per row by least squares
  (the inner products with the scale factored out are known exactly; residual 0.2) and snap to E4M3 values, but no rule of the form
  "the 128x4 or 8x4 or row-major buffer read at a constant element offset, with global or segment-local rows" reproduces more than 12
  percent of them against a chance level of 10 percent. Nor do scales or codes taken from the segment's tile base. The error is in the
  kernel's handling of a segment that does not start on a 128-row scale tile, and its exact form is not recovered from the outside.

## Why this matters

MoE routing produces segment lengths that are arbitrary; a serving path that calls this op with real routing gets, for every expert whose
first token is not a multiple of 128 rows into the sorted token list, an output that is wrong by a factor-of-one-scale magnitude for the
whole segment and raises nothing. The only callers that would never see it are those that pad every expert to 128 tokens.

## Where the cause sits: the FP4 block scales, not the segment handling

`grouped_mm_fp8` (the same cuDNN MoE grouped GEMM with e4m3 operands and one alpha, no block scales), run through the same segment sweep
(`scripts/flashinfer_grouped_fp8_alignment.py`, `reports/flashinfer-grouped-fp8-alignment-rtx5090-fi071-20261010.json`), is conformant on every
case, 64/64, 16/48/64, 192/64 and 120/136 included (0 elements outside the bound, max error under 0.97 of it). So the grouped kernel handles
arbitrary `m_indptr` correctly when there are no block scales, and the FP4 path's defect is in how the per-16-element scales of a segment that
does not start on a 128-row scale tile are addressed.

## The MXFP8 path shares it

`grouped_mm_mxfp8` (e4m3 elements with UE8M0 scales per 32 elements, the same cuDNN grouped GEMM with a different block scale), run through
the same sweep (`scripts/flashinfer_grouped_mxfp8_alignment.py`, `reports/flashinfer-grouped-mxfp8-alignment-rtx5090-fi071-20261010.json`,
operands and reference from FlashInfer's own `mxfp8_quantize` / `mxfp8_dequantize_host`): 4 x 128 conformant; 64/64, 192/64 and 120/136 wrong
across the mid-tile segment (16,359 of 16,384; 16,354 of 16,384; 32,233 of 34,816 outside the bound, errors about 100 on outputs of about 500);
16/48/64 wrong on both later segments with 2,512 NaNs. So the defect is the block-scale handling of a segment that starts inside a 128-row scale
tile, for both block-scaled formats the path serves; the unscaled FP8 path is unaffected.

## The control: vLLM's own grouped NVFP4 GEMM takes arbitrary segments

vLLM 0.28.0's `cutlass_fp4_moe_mm` (its CUTLASS grouped block-scaled GEMM, the MoE path it ships for SM120), fed through vLLM's own data
preparation (`get_cutlass_moe_mm_data`, `shuffle_rows`, `scaled_fp4_experts_quant`) with random top-k routing, so that the expert segments
have the lengths routing produces (13/12/21/18; 29/31/23/32/34/37/39/31; 12/20; 83/67/76/73/74/77/80/70 tokens) and all but the first start
off a 128-row boundary, is conformant on every element (0 of 610,304 outside the bound; `scripts/vllm_fp4_moe_mm_probe.py`,
`reports/vllm-fp4-moe-mm-rtx5090-20261010.json`). The difference in the interface is the point: vLLM gives each expert its own
block-scale region padded to 128 rows (`blockscale_offsets`, separate from the row offsets), so a segment's scales always begin on a tile
boundary; FlashInfer's cuDNN path takes one scale tensor swizzled over all tokens and has nothing to say where a segment's tile begins.

## A second device, bare metal: RTX PRO 6000 Blackwell Server Edition (2026-10-11)

`reports/flashinfer-grouped-fp4-alignment-pro6000-fi071-20261011.json`, run on a rented RTX PRO 6000 (compute capability 12.0, driver
580.173.02, Linux, no virtualisation) with FlashInfer 0.7.1, PyTorch 2.14.1+cu130 and cuDNN 9.24: the same nine segment patterns give the
same rule, element for element in kind. Every segment that starts at a multiple of 128 is correct (4x128; 256/128/384/256; 0/128/0/128;
the first segment of every other case) and every segment that starts elsewhere is wrong across the segment: 16,281 of 16,384 for the second
half-tile of 64/64; 34,541 of 34,816 for 120/136; 12,221 and 16,270 for 16/48/64; 24,363 for 32/96; 16,265 for the third segment of
128/64/64; 16,241 for 192/64 (`reports/flashinfer-grouped-fp8-alignment-pro6000-fi071-20261011.json`: `grouped_mm_fp8` on the same
segments is correct everywhere, as on the 5090). The MXFP8 sweep on the same pod (`reports/flashinfer-grouped-mxfp8-alignment-pro6000-fi071-20261011.json`) gives the same rule with the same
2,512 NaNs on 16/48/64 as the 5090. So the defect is neither the 5090's, nor the driver's, nor WSL2's, nor cuDNN 9.27's: two
devices, two drivers, two cuDNN builds and two PyTorch builds agree.

## Status

A draft issue for the author's review is in the paper repository (`docs/pr-drafts/flashinfer-grouped-fp4-alignment-issue.md`). Not yet asked: whether the block-scaled grouped GEMM in FlashInfer's unified MoE API routes
real workloads through this op on SM120.
