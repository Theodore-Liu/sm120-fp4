# vLLM 0.28.0's routed Marlin W4A16 GEMM (`moe_wna16_marlin_gemm`) on the RTX 5090 under routing its test does not enumerate (2026-10-15)

Read under the registration `mlsys-5090-research/docs/d1-vllm-moe-marlin-registration-2026-10-15.md` with
`scripts/vllm_moe_marlin_w4a16_probe.py`; the weighting epilogue fitted with `scripts/vllm_moe_marlin_w4a16_diag.py`.
vLLM 0.28.0, torch 2.13.0+cu130, the vllm028 environment under WSL2. Reports: `reports/vllm-moe-marlin-w4a16-rtx5090/`.

## The sweep

882 calls, all served, two runs each, 475,938,688 judged elements, deterministic, no NaN. The op called directly in both
forms `fused_marlin_moe` uses: the gate-up call (input read by token, `top_k` rows per token, no weighting) and the down
call (input read by row, `top_k = 1`, `size_m = M * top_k`, output times the top-k weight). GPTQ `uint4b8` with and
without act_order, AWQ `uint4` with zero points; group sizes -1, 32, 128; m in {1, 7, 129, 300}; (n, k) in {(64, 128),
(320, 1152), (4160, 4224)}; routings: the test's top-k of a softmax (8 experts, top-k 1, 2; 16 experts, top-k 8), 64
experts with 56 unused and the rest holding 1 to 40 tokens, and one expert holding every token; `moe_block_size` by
vLLM's occupancy rule and, at one shape, each of 8, 16, 32, 64. Instrument check: the test's own routing (133 tokens,
5 experts, top-k 2, group 128) read 0 outside first.

## Readings and what they are

| set | cases | outside the one-rounding oracle | what the kernel does |
|---|---|---|---|
| gate-up, grouped (32, 128 with k > group) | 289 | **0** of 177,733,056 elements (max 0.99 of bound) | the routed GEMM itself: one output rounding after fp32 accumulation; conformant under every routing, m, n, k, variant and block size tried |
| down, grouped | 289 | 194 cases, 800,754 elements, 1 to 2 output ulps (max 4.5 x bound) | **a precision design of the weighting epilogue, fitted bit for bit** (`diag`, 0.9998 to 0.9999 identical): the GEMM result is rounded to the output dtype, the fp32 top-k weight is rounded to the output dtype, the product is rounded again. The one-rounding model fits 62 to 65 percent of elements, the two two-rounding models 70 to 76, the weight-on-activation models 45 to 52 |
| channelwise (group -1) and single-group (k == group), both forms | 304 | 259 cases, up to 56 x the bound at k = 128 | the dense Marlin row's design: the channel scale is applied after the accumulator is rounded to the output dtype (`docs/vllm-marlin-w4a16-rtx5090.md`); in the down form it compounds with the weighting design above |

No reading the kernel's fitted numerics do not account for; under the census rule, none is a defect. The routing never
produced a wrong row: empty experts, an expert holding one token at block size 64, every token in one expert, top-k 8
at m = 1, and every block size give the same conformant gate-up readings.

## For a user of 4-bit MoE checkpoints on this part

A GPTQ or AWQ MoE layer with group scales computes its expert GEMMs to one output rounding; the top-k weighting on the
down projection costs up to two more output ulps (three roundings in total: the GEMM output, the weight, the product).
vLLM's own test judges the fused path at an absolute 5e-2, which does not see either design; this note is the record.

## Instrument notes

The first smoke called the down form with `top_k` and `size_m = M` and was refused (`a.size(0) = M * top_k, size_m =
M`): the kernel's contract is `fused_marlin_moe`'s call shape (`top_k = 1`, `size_m = M * top_k`), not a reading. The
per-row float64 oracle in Python is the sweep's cost (about 75 minutes for 882 cases, the 4160 x 4224 experts' quantisation
and the row loop); a batched oracle would cut it to minutes if the sweep is repeated.
