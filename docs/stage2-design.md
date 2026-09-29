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

## Before writing the kernel: the measured floor and the price of two phases

`scripts/micro_floor.py`, JSON `reports/micro-floor-rtx5090-2026-09-29.json`. Every timing is one CUDA-graph replay
with L2 flushed first; the flush also keeps the GPU busy while the host enqueues the graph, so host submission latency
does not land between the timing events (timed on an idle GPU an empty kernel reads 7.2 us, which is that latency).

Fixed costs, graph replay with events around it:

| what | 1 block/SM | 2 | 4 | 8 |
|---|---|---|---|---|
| empty kernel | 2.82 us | 2.82 | 2.82 | 2.82 |
| empty kernel with a cooperative grid barrier | 4.19 | 4.61 | 4.86 | (exceeds co-residency) |
| empty pair chained with PDL | 2.82 | 2.85 | 4.86 | 4.86 |

The 2.82 us of an empty kernel is inside every number in this document and in `stage2-baselines.md`, so comparisons
between them are like for like. A grid barrier costs 1.4 to 2 us; a PDL pair costs nothing over one kernel at up to 2
blocks per SM and 2 us at 4.

Weight streaming: a kernel that only reads the touched experts' FP4 weights and scales (16-byte streaming loads, grid-
stride over the chunks, 256 threads per block), best over 1 to 16 blocks per SM, same routing as the baselines:

| tokens | experts | MiB read | datasheet floor | read once | GB/s | read in two phases, grid barrier | two phases, PDL | best existing path | existing / two-phase PDL |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 8 | 20.2 | 11.8 us | 18.2 us | 1168 | 21.2 us | 20.2 us | 37.6 us | 1.86 |
| 2 | 15 | 38.0 | 22.2 | 29.4 | 1352 | 31.5 | 31.5 | 49.9 | 1.58 |
| 4 | 30 | 75.9 | 44.4 | 55.3 | 1441 | 58.0 | 57.1 | 78.8 | 1.38 |
| 8 | 48 | 121.5 | 71.1 | 84.7 | 1504 | 88.8 | 87.8 | 113.9 | 1.30 |
| 16 | 81 | 205.0 | 120.0 | 140.0 | 1535 | 143.6 | 142.1 | 166.5 | 1.17 |

What this says:

- The reachable floor is well above the datasheet one at small sizes: 20 MiB cannot be read at 1792 GB/s once ramp-up,
  the tail and the 2.8 us replay overhead are counted. This simple streamer reaches 1168 GB/s at 1 token and 1535 at
  16; a better one may do somewhat better, so these are estimates of the floor, not bounds on it.
- One block per SM streams 1.2 to 1.6 times slower than two or more (29.4 against 18.2 us at 1 token), which matches
  flashinfer#4990's diagnosis of the CUTLASS grouped path (one CTA per SM, latency-bound). The kernel needs at least two
  resident blocks per SM, which with about 99 KB of shared memory bounds each block's staging at about 48 KB.
- The two-phase structure costs 2 to 4 us over a single read; PDL is the cheaper barrier here and does not need
  cooperative co-residency.
- The room is largest where decode lives: a kernel that ran at the two-phase floor would be 1.86x the best existing path
  at 1 token, 1.38x at 4 and 1.17x at 16. Past 16 tokens the existing paths are already close to the floor and there is
  little to win, which is where SGLang #36787 also hands over to CUTLASS.

## FC1 phase, first kernel (`scripts/fc1_w4a16.py`)

W4A16 on CUDA cores, not tensor cores: at 1 to 16 tokens the FC1 arithmetic is about 0.4 G multiply-adds at most, small
next to the weight reads. Each warp owns one or two intermediate columns of one touched expert, issues all its weight
loads (16-byte streaming loads of the up and gate rows) before decoding any, decodes FP4 with SM120's hardware
`cvt.rn.f16x2.e2m1x2` and the E4M3 block scales in registers, dots them with the expert's tokens, reduces across the
warp with a fixed butterfly, and writes SiLU(gate) * up for each (token, expert) pair. The batch size is a template
parameter bounding tokens per expert, so accumulators stay in registers (no spills in any instantiation, `-Xptxas -v`).

Correctness against the fp32 reference on dequantized weights with bf16 activations: 0.16% to 0.17% normwise at every
batch size (bf16 output rounding), and bit-identical over 50 calls at each.

| tokens | FC1 kernel | read-only time for the same FP4 codes | ratio |
|---|---|---|---|
| 1 | 15.1 us | 13.1 us | 1.15 |
| 2 | 23.3 | 19.2 | 1.21 |
| 4 | 43.5 | 34.6 | 1.26 |
| 8 | 62.2 | 52.0 | 1.20 |
| 16 | 158.7 | 84.7 | 1.87 |

The read-only column counts the codes only; the kernel also reads the block scales, one byte per 16 values, 12.5% more.
At 1 token the kernel is therefore within about 2.5% of the time to read what it must (13.1 x 1.125 = 14.7 us).

What got it there, each step measured:

1. Decoding with a `__constant__` lookup table: 58.1 us at 1 token. Lanes index it with different codes, and constant
   memory serialises divergent indices. The hardware conversion instead: 21.5 us.
2. Batch size as a template parameter, all of a warp's weight loads issued before decoding, 1 to 4 columns per warp:
   17.2 us at 1 token, but 160.5 at 16 against 138.0 before.
3. Moving the decode out of the per-token loop, on the guess that repeated decoding cost the 16-token case, made every
   size slower (66.3 us at 4 tokens against 47.9): the reloads of the token vector it forced cost more than the decodes
   it saved. Reverted.
4. Sweeping columns per warp (1, 2, 4) at each batch size: 2 at 1 token, 1 elsewhere; 4 is worst everywhere.

Open: 16 tokens, at 1.87x its read floor and slower than the first version (138.0 us). Not yet explained; the next
step there is a profile rather than another guess. Next overall: the FC2 phase (one writer per output column tile,
experts in a fixed order, fp32 accumulation), then both phases together against the existing paths.

## Open questions

- Whether SGLang #36787's single cooperative kernel is deterministic, and its numbers on this layer and card; if it
  lands, it is the W4A4 baseline to beat.
- Real checkpoints: per-expert global scales and non-unit alphas (FlashInfer 0.6.16 couples the b12x input scale to the
  first alpha); the design takes per-expert alphas from the start.
