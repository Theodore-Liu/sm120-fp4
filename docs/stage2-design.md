# Stage 2 design draft: a deterministic fused MoE for SM120 decode

Draft, 2026-09-29. It follows from `docs/stage2-baselines.md`: on the RTX 5090, for one Qwen3-30B-A3B-shaped layer,
the best existing NVFP4 MoE path reaches 31% of the DRAM weight-read floor at 1 token and 56% at 4 (CUDA graph, cold
L2), and the W4A4 path with atomic scatter is non-deterministic and occasionally drops or doubles a contribution.

## Prior art this builds on or must beat

| work | what it does | on SM120? | combine |
|---|---|---|---|
| FlashInfer `b12x_fused_moe` (b12x project) | CuTe-DSL fused MoE, micro/static/dynamic backends by row count; W4A4 and W4A16 | yes | W4A4: bf16x2 atomic add into the output (measured non-deterministic) |
| FlashInfer `cutlass_fused_moe` | CUTLASS grouped GEMMs + finalize | yes | deterministic in measurement |
| vLLM `fused_marlin_moe` | W4A16 grouped Marlin GEMMs from NVFP4 checkpoints | yes | deterministic in measurement |
| SGLang PR #36787 (open, September 2026) | one cooperative kernel for 1 to 16 routed tokens: W4A4 quantize, FC1, SiLU-mul with requantization, grouped FC2, routing-weight finalize, `mma.sync.kind::mxf4nvf4`; falls back to FlashInfer CUTLASS above 16 tokens or when cooperative launch is not provable. Reports 45.1 us vs 72.1 us (CUTLASS) at 1 token, 84.5 vs 111.6 at 4, 195.1 vs 216.1 at 16, on its own model and card | yes | not stated |
| MonoMoE (arXiv 2609.04244; FlashInfer `mono_moe`) | weight-major persistent megakernel: the whole decode token tile sits on the MMA's N dimension and CTAs partition expert-weight tiles; routing, top-k, quantization, both projections, activation and reduction in one launch, auxiliary work overlapped with the weight stream. H200: 1.54x over vLLM's Triton grouped GEMM | no: FlashInfer's build is SM90a only, block-FP8, one fixed shape | not stated |
| incoai/splash #178, omlx #2238 | decode MoE GEMMs sustaining 17 to 34% of DRAM bandwidth on other hardware; split-K and weight-major regrouping named as the levers | n/a | n/a |

The idea to carry over is MonoMoE's: at decode batch sizes the weights are the large operand and the tokens are few, so
the kernel should be organised around streaming each touched expert's weights exactly once at full bandwidth, with the
tokens riding along, not around token tiles.

## What SM120 allows

- The block-scaled MMA is `mma.sync.aligned.m16n8k64 ... kind::mxf4nvf4.block_scale.scale_vec::4X`. With weights as the
  16-row side and tokens as the 8-column side, one instruction serves up to 8 tokens; 16 tokens take two.
- About 99 KB of shared memory per SM and no TMA multicast or tensor memory: a weight-streaming pipeline is cp.async or
  TMA loads into a few shared-memory stages per CTA, several CTAs per SM to hide latency (flashinfer#4990 measured the
  CUTLASS grouped path at one CTA per SM and 88 to 90% of cycles with no eligible warp).
- W4A16 (dequantize FP4 weights to bf16 in registers, bf16 MMA) and W4A4 (quantize activations to NVFP4, FP4 MMA) read
  the same weight bytes. At 1 to 4 tokens the arithmetic is negligible either way; W4A16 avoids the activation
  quantization error (15 to 17% normwise against unquantized activations on the synthetic layer, 0.5% for W4A16) and
  was the faster existing path at every batch size measured. The first kernel is therefore W4A16 on NVFP4 checkpoints;
  W4A4 is a variant, not the default.

## Proposed structure

1. FC1 phase. CTAs partition (touched expert, intermediate tile). Each streams its slice of that expert's gate and up
   weights once, multiplies by the (at most 16) tokens routed to the expert, applies SiLU(gate) * up in fp32, and writes
   the intermediate for those tokens to a small global buffer (tokens x top-k x intermediate in bf16: 196 KB at 16
   tokens, top-8, 768; it stays in L2).
2. One grid-wide dependency (a cooperative-launch barrier, or two kernels chained with programmatic dependent launch).
3. FC2 phase. CTAs partition the output (hidden) columns. Each CTA streams, for its column tile, the FC2 weights of every
   touched expert in a fixed expert order, multiplies by that expert's intermediate rows, scales by the routing weight
   and accumulates in fp32 registers; then writes its tile once. Every output element has exactly one writer and one
   summation order, so the result is deterministic by construction, with no atomics.

Bytes read are the touched experts' weights and scales once plus the small intermediate; the fixed-order FC2 reduction
costs nothing extra in bandwidth. The risk is FC2 load balance at 1 token (8 experts x a column tile per CTA) and the
barrier's cost at small sizes; both are measurable before the full kernel exists.

## Before writing the kernel

1. Measure the achievable floor, not the datasheet one: a pure weight-streaming kernel that reads exactly the touched
   experts' FP4 bytes (8 to 81 chunks of 2.65 MB, cold L2) and does nothing else. The gap between that and the existing
   paths is the real room.
2. Measure the barrier: an empty two-phase kernel with the chosen grid-wide dependency.
3. Only then the kernel, validated against the reference in this repository with bitwise determinism across 1000 calls
   as a test, and the stage 1 CUDA-graph replay check.

## Open questions

- Whether SGLang #36787's single cooperative kernel is deterministic, and its numbers on this layer and card; if it
  lands, it is the W4A4 baseline to beat.
- Real checkpoints: per-expert global scales and non-unit alphas (FlashInfer 0.6.16 couples the b12x input scale to the
  first alpha); the design takes per-expert alphas from the start.
