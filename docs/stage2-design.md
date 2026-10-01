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
step there is a profile rather than another guess.

## FC2 phase (`scripts/fc2_w4a16.py`)

Same decode and loads as FC1. The output side is where determinism is decided: each output (hidden) column belongs to
one block, and nothing is added atomically.

- v0: one warp per output column walks every touched expert in ascending id order, reads that expert's row for its
  column, dots it with each routed pair's activation, and adds routing weight x alpha x the value into the token's
  accumulator in shared memory. 20.2 us at 1 token for weights that take about 6 us to read: each warp has only one
  row of 384 bytes per expert in flight (24 of 32 lanes busy at intermediate 768), and there are only 2048 warps.
- v1: one block per 1 or 2 output columns; its 8 warps split the touched experts round-robin with two experts' rows in
  flight each, keep their own partial sums in shared memory, and the block adds the 8 partials in warp order at the end.
  Every sum still has one order fixed by the routing. FC2 alone: 15.1 us at 1 token (from 20.2), 21.8 at 2 (33.5), 36.6
  at 4 (60.1), 60.2 at 8 (97.0), 101.1 at 16 (174.8). Two columns per block win at 1 token, one elsewhere.

## The whole layer on the GPU, like for like (`scripts/moe_w4a16.py`)

The first FC1 and FC2 timings left routing out: the pairs were grouped by expert with torch's `argsort` and
`unique_consecutive`, which synchronises with the host to size its output, outside the timed graph. The baselines'
calls include their own routing. So routing is now a kernel too: one block counts pairs per expert in shared memory,
scans the counts, and places each pair at its expert's base plus its rank among the earlier pairs of that expert.
Experts come out ascending and pairs in index order, identical to the torch router at every batch size tested (checked
element by element), with no atomics deciding a position. Buffers have a fixed size of min(experts, pairs) slots; unused
slots carry expert -1, which FC1 and FC2 skip. Nothing returns to the host, and routing, FC1 and FC2 are timed as one
CUDA graph with L2 flushed first, the method of `stage2-baselines.md`.

Qwen3-30B-A3B-shaped layer, RTX 5090, JSON `reports/moe-w4a16-rtx5090-2026-09-29.json`:

| tokens | normwise error vs fp32 | bit-identical over 50 calls | route + FC1 + FC2 | best existing path (same method) | speedup |
|---|---|---|---|---|---|
| 1 | 0.25% | yes | 29.3 us | 37.6 us, b12x W4A16 | 1.28x |
| 2 | 0.23% | yes | 41.7 | 49.9, Marlin W4A16 | 1.20x |
| 4 | 0.23% | yes | 74.5 | 78.8, Marlin W4A16 | 1.06x |
| 8 | 0.24% | yes | 113.2 | 113.9, Marlin W4A16 | 1.01x |
| 16 | 0.24% | yes | 244.5 | 166.5, Marlin W4A16 | 0.68x |

For scale: the existing W4A16 paths are 0.44% to 0.48% from the same reference, and b12x W4A4 is not bit-stable.

What this does and does not show:

- At 1 and 2 tokens the layer is 1.28x and 1.20x faster than any existing SM120 path measured, deterministic, and
  about twice as close to the fp32 result. Routing costs 2 to 4 us of that (the same kernels with host-side routing
  outside the graph measured 25.3 us at 1 token).
- At 4 and 8 tokens it is level with Marlin; at 16 it is slower, because FC1 at 16 tokens runs at 1.87x its read floor
  (open, above).
- The two-phase floor measured earlier is 20.2 us at 1 token, so there are still about 9 us to find at 1 token: routing
  (2 to 4 us), three kernel boundaries with no PDL yet, and FC2 (15.1 us alone at 1 token for about 7 MB of weights
  and scales) not yet timed against a read of exactly its own bytes.
- One layer shape, one card, synthetic weights with unit global scales, and the baselines measured earlier the same day
  by the same method (their run-to-run spread was up to 2 us).

## Open questions

- Whether SGLang #36787's single cooperative kernel is deterministic, and its numbers on this layer and card; if it
  lands, it is the W4A4 baseline to beat.
- Real checkpoints: per-expert global scales and non-unit alphas (FlashInfer 0.6.16 couples the b12x input scale to the
  first alpha); the design takes per-expert alphas from the start.

## Where the layer's time goes, and why 16 tokens is slow (`scripts/moe_breakdown.py`)

JSON `reports/moe-breakdown-rtx5090-2026-09-29.json`. Every number is one CUDA-graph replay with L2 flushed first, so
each carries the same 2.8 us replay overhead; the "read" columns are streaming reads of exactly the FP4 codes and
scales that kernel consumes for the touched experts, as two kernels (codes, scales), so they carry it twice.

| tokens | experts | router | FC1 | read of FC1's bytes | FC2 | read of FC2's bytes |
|---|---|---|---|---|---|---|
| 1 | 8 | 4.9 us | 15.1 | 17.2 | 15.1 | 13.1 |
| 2 | 15 | 4.9 | 23.8 | 25.3 | 21.2 | 17.2 |
| 4 | 30 | 4.9 | 43.8 | 42.8 | 35.6 | 25.9 |
| 8 | 48 | 4.9 | 62.2 | 62.2 | 60.2 | 36.6 |
| 16 | 81 | 6.2 | 160.5 | 99.1 | 99.1 | 58.1 |

- FC1 runs at the speed of reading its own bytes from 1 to 8 tokens (the two-kernel read is no faster), and 1.6x slower
  at 16.
- FC2 is 1.15x to 1.7x slower than reading its bytes at every batch size: that is the larger remaining gap from 2 to 16
  tokens.
- The router costs 4.9 us as a graph replay on its own, of which 2.8 us is the replay overhead every kernel pays.

Why FC1 is slow at 16 tokens. With the expert set fixed at 8 and every token routed to all 8, the bytes FC1 reads do not
change with the batch and only the arithmetic does:

| tokens per expert | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| FC1 | 15.1 us | 17.2 | 28.4 | 41.7 | 105.2 |

So FC1 becomes bound by instructions, not memory, as tokens per expert grow. At 16 tokens with 8 experts that is 128
(token, expert) pairs, the same arithmetic as the random 16-token batch (also 128 pairs), which takes 160.5 us while
reading ten times more experts: roughly 100 us of it is CUDA-core arithmetic (per pair, per weight: a bf16-to-fp32
conversion of the activation and two fused multiply-adds, repeated for every intermediate column). With random routing
at 1 to 8 tokens most experts receive one or two tokens, which is why FC1 is memory-bound there. The fix for 16 tokens
is to move FC1's (and FC2's) arithmetic onto tensor cores (`mma.m16n8k16` in bf16 after the in-register FP4 decode,
tokens as the 8-column side), not to tune the CUDA-core loop further.

## Chaining the kernels with programmatic dependent launch

FC1 waits for the router's output and immediately allows the next kernel to launch; FC2 issues its first round of
weight loads and only then waits for FC1 to complete before it reads the activations. The calls are no-ops without the
launch attribute, so the same kernels run either way. JSON `reports/moe-w4a16-rtx5090-2026-09-29.json` (same run as the
rows below; the plain column moved by up to 6 us from the earlier table at 16 tokens):

| tokens | route + FC1 + FC2 | with PDL | best existing path | speedup with PDL | output identical to plain launch |
|---|---|---|---|---|---|
| 1 | 29.2 us | 27.4 us | 37.6 us | 1.37x | yes |
| 2 | 41.7 | 41.4 | 49.9 | 1.20x | yes |
| 4 | 74.5 | 73.8 | 78.8 | 1.07x | yes |
| 8 | 111.4 | 111.4 | 113.9 | 1.02x | yes |
| 16 | 250.4 | 248.6 | 166.5 | 0.67x | yes |

PDL pays where the kernels are short (1.8 us at 1 token) and is noise elsewhere. Next, in order of what the numbers say
is left: FC2 at the speed of its bytes (2 to 16 tokens), tensor-core arithmetic for FC1 and FC2 (16 tokens), and a
cheaper router (1 token).

## FC2: what the activation re-reads cost

*Correction (2026-10-01): timed with L2 flushed, so the activations came from DRAM; in the layer FC1 has just written them and they are in L2. See "The activations are warm in the layer" at the end.*

FC2 v1 reads, for every output column, the FC1 activation row of every (token, expert) pair the column's experts
receive: at 16 tokens that is 2048 columns x 128 pairs x 1.5 KB of activation traffic against 58 MB of weights. To
measure its share (JSON `reports/moe-breakdown-fc2-activation-rtx5090-2026-09-29.json`), FC2 has a diagnostic switch (`fc2_set_skip_act`, never set in normal use) that replaces each
activation load with the constant 1.0 and keeps every multiply-add, so only the traffic disappears. Same method as the
breakdown above (graph replay, L2 flushed), one run:

| tokens | FC2 | FC2 without activation loads | read of FC2's bytes |
|---|---|---|---|
| 1 | 15.1 us | 13.1 | 13.0 |
| 2 | 22.0 | 19.2 | 17.2 |
| 4 | 36.9 | 29.4 | 27.3 |
| 8 | 63.2 | 47.8 | 37.6 |
| 16 | 105.2 | 78.5 | 57.9 |

The activation traffic costs 2 us at 1 token and 27 us at 16; without it FC2 is at its read floor at 1 token and 1.08x
at 4. From 8 tokens up, FC2 without activation loads is still 1.27x to 1.36x its read floor: the rest is the per-pair
arithmetic and the per-pair warp reductions, the same instruction-bound pattern FC1 shows. Both halves point at the
same change for 4 to 16 tokens: tensor-core arithmetic with the tokens as the MMA's 8-column side, where each activation
tile is loaded once per block and reused across the weight rows the block streams, rather than re-read per column.

## FC1 on tensor cores (`scripts/fc1_mma.py`)

The CUDA-core FC1 is at its read floor while experts receive one or two tokens and instruction-bound beyond (fixed 8
experts: 15.1 us at 1 token per expert, 102.4 at 16). The tensor-core version keeps the in-register FP4 decode and puts
the arithmetic on `mma.sync.m16n8k16` (bf16 in, fp32 accumulate), weights as the 16-row side and tokens as the 8-column
side (a second tile for 9 to 16). Three details make it work:

- The decoded weight is exact in bf16: an E2M1 value times an E4M3 scale has at most 6 significant bits.
- k is permuted identically for weights and activations (a dot product does not depend on the order of k) so that each
  lane reads 16-byte vectors, and the four lanes of an MMA quad read 64 contiguous bytes of a row per load. With the
  first permutation (a lane's 128 k contiguous) the quad's lanes read 16 bytes at a 64-byte stride and every 32-byte
  sector was fetched by two instructions: 19.2 us at 1 token instead of 15.1.
- Every weight load of a lane is issued before any is decoded; loading and computing one chunk at a time gave 25.3 us at
  1 token.

Each block owns 16 channels of one expert; its four warps split the hidden dimension into quarters and their partial
tiles are added in shared memory in warp order, so the result stays deterministic. One run, graph replay, L2 flushed
(`reports/fc1-mma-rtx5090-2026-09-29.json`; the 19.2 and 25.3 us of the two earlier versions come from runs whose logs
were not kept):

| routing | tokens | tensor-core FC1 | CUDA-core FC1 | read of FC1 codes |
|---|---|---|---|---|
| random | 1 | 15.1 us | 15.1 | 13.1 |
| random | 2 | 25.3 | 23.3 | 19.2 |
| random | 4 | 43.8 | 43.6 | 34.4 |
| random | 8 | 66.3 | 62.2 | 52.0 |
| random | 16 | 101.1 | 162.6 | 84.7 |
| 8 experts, every token to all 8 | 1 | 15.1 | 15.1 | 13.1 |
| same | 4 | 15.1 | 28.4 | 13.1 |
| same | 8 | 16.1 | 43.8 | 13.1 |
| same | 16 | 25.3 | 102.4 | 13.1 |

Normwise error against the fp32 reference 0.16% to 0.17% in every row, as for the CUDA-core version, and bit-identical
over 50 calls in every row. The tensor-core FC1 matches the CUDA-core one at 1 token, is up to 9% slower at 2 to 8
randomly routed tokens (25.3 against 23.3 us at 2), and is 1.6x faster at 16 random tokens and up to 4.0x faster when
tokens concentrate on few experts. Next: FC2 on tensor cores (the same shape with the intermediate as k), then the
layer's FC1 and FC2 chosen per batch size, timed against the existing paths as before.

## FC2 on tensor cores (`scripts/fc2_mma.py`)

The same machinery as FC1 on tensor cores, arranged for FC2: a block owns 16 output columns (the MMA's M), the N side is
one expert's (token, expert) pairs, and k is the intermediate dimension. The block's eight warps take the touched experts
round-robin; after each expert a warp adds weight x alpha x its result into per-token accumulators in shared memory and
synchronises before the next expert (a token can sit in a different lane for the next expert, and the synchronisation
fixes the order of the two adds); the warps' partials are then added in warp order and each output is written once, so
the result is deterministic without atomics. Each activation row is now read once per 16 output columns rather than
once per column.

One run, graph replay, L2 flushed, activations from the CUDA-core FC1 (`reports/fc2-mma-rtx5090-2026-09-29.json`):

| routing | tokens | tensor-core FC2 | CUDA-core FC2 | read of FC2 codes |
|---|---|---|---|---|
| random | 1 | 14.1 us | 15.1 | 9.0 |
| random | 2 | 22.3 | 22.6 | 13.1 |
| random | 4 | 37.6 | 37.6 | 21.2 |
| random | 8 | 52.2 | 62.2 | 31.4 |
| random | 16 | 105.2 | 103.2 | 49.9 |
| 8 experts, every token to all 8 | 1 | 15.1 | 15.1 | 9.0 |
| same | 4 | 15.1 | 23.2 | 9.0 |
| same | 8 | 17.2 | 39.7 | 9.0 |
| same | 16 | 21.2 | 71.8 | 9.0 |

Normwise error of the layer against the fp32 MoE reference 0.23% to 0.25% in every row,
the same as with the CUDA-core FC2, and bit-identical over 50 calls in every row. The tensor-core FC2 is faster than the
CUDA-core one or within 2% of it at every size, up to 3.4x when tokens concentrate on few experts (21.2 against 71.8 us at
16 tokens per expert). At 16 random tokens it is level (105.2 against 103.2 us) and still 2.1x a read of its bytes: each
warp walks about ten experts one after another, and the next expert's weights are loaded only after the current
expert's arithmetic. Prefetching the next expert's weights while the current one computes is the next change.

## FC2: prefetching the next expert, and two blocks per column tile (`scripts/fc2_mma_pf.py`)

The tensor-core FC2 loaded an expert's weights only after the previous expert's arithmetic, and at 16 random tokens
each warp walks about ten experts. Two changes, each measured against it:

- **prefetch**: while a warp computes one expert, the loads for its next expert are already issued (a register double
  buffer; the k loop is specialised on I / 128 so both buffers fit, 163 to 238 registers, no spills).
- **prefetch + split**: the grid becomes (H / 16) x 2. With H = 2048 the single-group grid has 128 blocks for the
  RTX 5090's 170 SMs; two groups give 256. Each group reduces its warps in warp order and writes an fp32 partial; the
  last group to finish a tile (an atomic counter) adds the two partials in group order and writes the output, so the
  summation order is fixed.

One run, graph replay, L2 flushed (`reports/fc2-mma-pf-rtx5090-2026-09-29.json`):

| routing | tokens | v1 | prefetch | prefetch + split | read of FC2 codes |
|---|---|---|---|---|---|
| random | 1 | 14.6 us | 15.1 | 15.1 | 9.0 |
| random | 2 | 22.8 | 21.2 | 23.3 | 13.3 |
| random | 4 | 37.6 | 35.6 | 35.6 | 21.3 |
| random | 8 | 53.6 | 52.0 | 49.9 | 31.5 |
| random | 16 | 105.1 | 89.3 | 86.8 | 50.2 |
| 8 experts, every token to all 8 | 1 | 13.1 | 15.1 | 15.1 | 9.0 |
| same | 4 | 15.1 | 17.2 | 17.2 | 9.0 |
| same | 8 | 17.2 | 17.2 | 18.5 | 8.9 |
| same | 16 | 21.2 | 21.2 | 21.2 | 9.0 |

Normwise error against the fp32 MoE reference 0.23% to 0.25% for all three, bit-identical
over 50 calls in every row. Prefetch alone performs v1's additions in v1's order and equals v1 bit for bit in every
row; the split adds the same terms in another fixed order and is not bit-identical to v1 at random 8 and random 16 (the same error
against the fp32 reference), and is stable across calls.

At 16 random tokens the two changes together take FC2 from 105.1 to 86.8 us
(17% less), from 2.1x to 1.7x a read of its bytes; at 8 random tokens from
53.6 to 49.9 us. At one token and when tokens concentrate on few experts they
do not help: 14.6 against 15.1 us at one random token, 15.1 against
17.2 us at 4 tokens on 8 experts. The layer therefore picks its FC2 by batch shape: v1 at one token,
prefetch + split once the batch spreads over many experts.

## The layer with each GEMM chosen by batch size (`scripts/moe_layer.py`)

GPU router, FC1 and FC2 in one graph, every launch after the router with programmatic dependent launch (the
tensor-core kernels gained the same launch switch as the CUDA-core ones; off, it is the plain launch). The kernel for
each GEMM is fixed by the batch size from the separate kernel measurements above, not picked per row from this run:
FC1 on CUDA cores up to 8 tokens and on tensor cores at 9 to 16; FC2 on CUDA cores at 1 token, the prefetch kernel at
2 to 4, prefetch + split at 5 to 16. The all-CUDA-core layer is timed in the same session beside it.

One run, graph replay, L2 flushed (`reports/moe-layer-rtx5090-2026-09-30.json`). The existing-path column is the best
of b12x, cutlass and Marlin from `reports/moe-baseline-rtx5090-2026-09-29.json` (the same timing method, an earlier session; random routing only):

| routing | tokens | FC1 | FC2 | layer | all-CUDA-core layer | best existing | ratio |
|---|---|---|---|---|---|---|---|
| random | 1 | CUDA core | CUDA core | 27.4 us | 27.4 | b12x-w4a16 37.6 | 1.37x |
| random | 2 | CUDA core | prefetch | 39.7 | 41.7 | marlin-w4a16 49.9 | 1.26x |
| random | 4 | CUDA core | prefetch | 70.4 | 74.5 | marlin-w4a16 78.8 | 1.12x |
| random | 8 | CUDA core | prefetch + split | 109.0 | 115.5 | marlin-w4a16 113.9 | 1.04x |
| random | 16 | tensor core | prefetch + split | 170.8 | 254.7 | marlin-w4a16 166.5 | 0.98x |
| 8 experts, every token to all 8 | 1 | CUDA core | CUDA core | 27.4 | 27.4 | - | - |
| same | 4 | CUDA core | prefetch | 39.7 | 47.9 | - | - |
| same | 8 | CUDA core | prefetch + split | 60.2 | 80.6 | - | - |
| same | 16 | tensor core | prefetch + split | 41.8 | 178.6 | - | - |

Normwise error against the fp32 MoE reference 0.23% to 0.25% and bit-identical over 50
calls in every row. With randomly routed tokens the layer is faster than the best existing path from 1 to 8 tokens
(1.37x at 1, 1.04x at 8) and
2.6% slower at 16 (170.8 against
marlin-w4a16 166.5 us), where the tensor-core kernels took the layer from
254.7 us. When tokens concentrate on few experts the tensor-core kernels matter most:
41.8 against 178.6 us at 16 tokens on 8 experts.

## The chosen layer against Marlin, piece by piece in one session (`scripts/moe_layer_breakdown.py`)

The layer section above compared against Marlin from an earlier session's report, timed there by another helper. Here
every piece and vLLM's Marlin W4A16 MoE are timed by one method (graph replay, L2 flushed) in one session, on the same
weights and routing (`reports/moe-layer-breakdown-rtx5090-2026-09-30.json`, microseconds; in brackets, a stream read of
the codes that GEMM touches):

| routing | tokens | router | FC1 (read) | FC2 (read) | layer | Marlin |
|---|---|---|---|---|---|---|
| random | 8 | 4.9 | 62.7 (52.0) | 50.3 (31.5) | 107.2 | 114.1 |
| random | 16 | 5.0 | 102.1 (84.7) | 87.9 (49.9) | 171.2 | 168.3 |
| 8 experts, every token to all 8 | 16 | 5.4 | 25.3 (13.0) | 25.3 (9.0) | 43.7 | 42.0 |

Normwise error against the fp32 MoE reference at most 0.46% for both. Measured this way the layer is
faster than Marlin at 8 random tokens (107.2 against 114.1 us) and
1.8% slower at 16 (171.2 against 168.3).
When the 16 tokens concentrate on 8 experts, Marlin gains as much as the tensor-core kernels do:
43.7 against 42.0 us, so the concentrated case in the layer section's table is a gain over
the all-CUDA-core layer, not over the existing path. At 16 random tokens FC2 is the furthest from its read
(87.9 against 49.9 us, 1.76x) and FC1
closer (102.1 against 84.7, 1.21x); timed as
separate graphs the parts sum to 195.1 us and the layer in one graph takes
171.2. FC2 at 16 tokens is the next target.

## FC2 at 16 tokens: groups per tile, and skipping empty pair tiles (`scripts/fc2_groups.py`)

Two explanations for FC2's distance from its read at 16 random tokens, each measured in one session against the
plain kernel (`reports/fc2-groups-rtx5090-2026-09-30.json`, microseconds):

- **Too little parallelism over experts.** The prefetch kernel takes G = 1 to 4 groups per column tile (128 to 512
  blocks); each group's warps walk 1/G of the touched experts.
- **Wasted tensor-core work.** At 9 to 16 tokens the kernel runs two 8-pair tiles per expert, and 128 pairs over 81
  touched experts leave the second tile nearly always empty. Built with `PF_SKIP_EMPTY`, a warp skips the MMAs of a tile
  its expert does not fill (the activation loads of empty slots were already skipped per lane).

| routing | tokens | v1 | G1 | G2 | G3 | G4 | G1 skip | G2 skip | read of FC2 codes |
|---|---|---|---|---|---|---|---|---|---|
| random | 8 | 54.0 | 52.0 | 49.9 | 53.8 | 54.0 | 52.5 | 50.0 | 31.5 |
| random | 16 | 105.2 | 88.8 | 87.6 | 86.8 | 86.8 | 88.8 | 87.7 | 49.9 |
| 8 experts, every token to all 8 | 16 | 21.2 | 20.5 | 21.2 | 21.2 | 21.2 | 21.0 | 21.2 | 9.0 |

Normwise error against the fp32 MoE reference 0.23% to 0.24% for every variant,
bit-identical over 50 calls, and the skip variants equal the plain kernel bit for bit. Neither explanation holds: at
16 random tokens the four group counts span 86.8 to 88.8 us and skipping the empty tile changes nothing
(87.7 against 87.6 us with two groups). What remains is occupancy: the 16-token
kernel holds 169 registers per thread (cuobjdump of the built extension, `reports/fc2-occupancy-rtx5090-2026-09-30.json`), so a block of 256 threads needs 43,264 of an SM's
65,536 registers and one block, eight warps, fits per SM; a warp's prefetch of its next expert is all the latency hiding
there is. Smaller blocks with more groups per tile test that next.
`PF_SKIP_EMPTY` stays a build option, off by default, since it neither helps nor changes a bit.

## FC2 at 16 tokens: occupancy (`scripts/fc2_occupancy.py`)

The registers of the 16-token instantiation, read from each built extension (`cuobjdump --dump-resource-usage`):
169 per thread with eight warps per block, 168 with four, and 128 with
`__launch_bounds__(256, 2)`, which fits two blocks per SM and spills 136 bytes of stack to do it (the other
builds spill none). Same session, microseconds (`reports/fc2-occupancy-rtx5090-2026-09-30.json`):

| routing | tokens | 8 warps, G2 | 4 warps, G2 | 4 warps, G4 | 8 warps, 2 blocks/SM, G2 | read of FC2 codes |
|---|---|---|---|---|---|---|
| random | 8 | 49.9 | 52.0 | 49.9 | 56.1 | 31.5 |
| random | 16 | 87.9 | 88.8 | 86.8 | 84.7 | 49.9 |
| 8 experts, every token to all 8 | 16 | 21.2 | 21.2 | 23.2 | 27.4 | 8.9 |

Normwise error against the fp32 MoE reference 0.23% to 0.24% for every build,
bit-identical over 50 calls. Twice the resident warps take 16 random tokens from 87.9 to
84.7 us and cost elsewhere (49.9 to 56.1 at 8 tokens,
21.2 to 27.4 at 16 tokens on 8 experts); smaller blocks change little. With
parallelism over experts, empty pair tiles and occupancy each moving FC2 at 16 random tokens by a few percent at most,
it stays at 1.76x a read of its codes. Performance counters are not
available on this machine, so the next step separates the two sides by construction: the kernel with its arithmetic
removed (loads only) and with its loads replaced by registers (arithmetic only), timed beside it.

## FC2 at 16 tokens: loads against arithmetic (`scripts/fc2_split.py`)

This machine has no performance counters, so the two sides are separated by construction: `fc2_mma_pf.build(mode=...)`
compiles the same kernel (eight warps, two groups) as **loads** (every weight, scale and activation load kept, the
decode and MMAs replaced by one fold of the loaded values), as **loads, contiguous** (the same bytes, the tile's 16
rows read at lane stride so each warp instruction covers 512 contiguous bytes of codes), and as **math** (decode and
MMAs kept, the codes made from the row index, no weight read). The variants give wrong answers by design and are timing
instruments only; the full kernel's error against the fp32 reference is 0.23% to 0.24%
in the same run. Same session, microseconds (`reports/fc2-split-rtx5090-2026-09-30.json`):

| routing | tokens | full | loads only | loads only, contiguous | math only | stream read of the codes |
|---|---|---|---|---|---|---|
| random | 8 | 49.9 | 41.7 | 41.7 | 19.1 | 31.5 |
| random | 16 | 87.8 | 69.6 | 70.4 | 47.9 | 51.9 |
| 8 experts, every token to all 8 | 16 | 21.2 | 19.1 | 19.1 | 13.1 | 9.0 |

Both sides cost. At 16 random tokens the kernel's loads take 69.6 us; the stream read reads only the codes, and
the kernel also reads one scale byte per eight code bytes, so the comparable read is about 58.4 us
(51.9 x 9/8) and the loads sit at about 1.19x of it. Reading the same bytes contiguously
changes nothing (70.4 us), so the access pattern within a tile is not the cost, and moving the tile through
shared memory for coalescing would not pay. The decode and MMAs alone take 47.9 us, and the full kernel's
87.8 us lies between the two sides overlapped (69.6) and serialised (117.5). With
tokens on 8 experts the loads are further from a read (19.1 against 9.0 us
for the codes alone), where each warp has one expert and the loads cannot be hidden behind another. The larger lever at
16 random tokens is the arithmetic: decode and MMAs that together cost about as much as reading the weights.

## The decode is cheap (`scripts/decode_bench.py`)

Two FP4 x E4M3 to bf16 decodes: **A**, the kernels' own (`cvt.rn.f16x2.e2m1x2`, to floats, times the scale, packed to
bf16x2), and **C**, a lookup (the bf16 bits of the eight E2M1 magnitudes in two byte tables selected with `prmt` by
each nibble, the sign by bit operations, one bf16x2 multiply by the scale). The product is exact in bf16, and C equals
A on every input: 65,024 code-byte and scale pairs (the E4M3 NaN scales excluded), 0
different. Timed on 64 MiB of codes, each value decoded 8 times under
different scales (1.07 billion decodes; `reports/decode-bench-rtx5090-2026-09-30.json`): A
59.2 us, C 56.1 us, a plain read of the codes 47.8 us. Eight
decodes per value barely lengthen a read, so decoding is not what FC2's arithmetic side spends: at 16 random tokens
FC2 decodes each of about 81 experts' 1.57 million values once. The lookup is not adopted (a few percent on a
benchmark the read nearly bounds). What the math-only variant keeps besides decode and MMAs is the chain of dependent
loads per expert (its id, its offsets, its pair list, then its activation rows), which the prefetch does not cover;
prefetching that chain with the weights is the next change.

## FC2 at 16 tokens: what the arithmetic side spends (`scripts/fc2_chain.py`)

Three more build options of `scripts/fc2_mma_pf.py`, each tested in one session (eight warps, two groups;
`reports/fc2-chain-rtx5090-2026-09-30.json`, microseconds):

- **chain**: the next expert's routing (offsets, the pair indices of the lane's tile rows, and for each accumulator slot
  its token and routing weight x alpha) loaded with its weights, since each expert's arithmetic waited on that chain.
  It changes only when loads are issued, and the output equals the plain kernel's bit for bit in every row.
- **math, skip**: math-only with the MMAs of unfilled pair tiles skipped, to see whether padded tensor-core work (16
  columns computed for about 1.6 pairs per expert at 16 random tokens) is what math-only spends.
- **math, no activations**: math-only with the activations made from the pair index instead of read.

| routing | tokens | full | full, chain | math | math, chain | math, skip | math, no activations | read of FC2 codes |
|---|---|---|---|---|---|---|---|---|
| random | 8 | 49.9 | 52.0 | 19.0 | 25.2 | 28.3 | 17.2 | 31.5 |
| random | 16 | 87.8 | 84.2 | 47.9 | 56.0 | 47.9 | 35.6 | 49.9 |
| 8 experts, every token to all 8 | 16 | 21.2 | 19.2 | 13.1 | 15.1 | 13.1 | 10.0 | 9.0 |

Full kernels: normwise error against the fp32 reference 0.23% to 0.24%, every variant
bit-identical over 50 calls. The chain is small and mixed (87.8 to 84.2 us at 16
random tokens, 21.2 to 19.2 on 8 experts, 49.9 to
52.0 at 8 random) and slows math-only, so it stays an option, off. Skipping padded tiles leaves
math-only where it was (47.9 against 47.9): padded MMAs are not its cost. Making
the activations instead of reading them takes math-only from 47.9 to 35.6 us:
reading activations is about 12 us of it. Each block covers 16 output
columns, so the 128 column tiles each read every pair's activation rows; covering 32 columns per block halves those
reads and is the next change. The 35.6 us left with neither weights nor activations read (decode,
MMAs, the shared-memory accumulation) is not yet broken down.

## Real weights: layer 0 of Qwen3-30B-A3B in NVFP4 (`scripts/real_ckpt_layer.py`)

The 128 experts of layer 0 from `nvidia/Qwen3-30B-A3B-NVFP4` (ModelOpt NVFP4: E2M1 codes, E4M3 scales per 16 values,
a per-tensor `weight_scale_2`), the same shape as every measurement above (hidden 2048, expert 768, top 8). Two checks
before any timing (`reports/real-ckpt-layer0-rtx5090-2026-09-30.json`):

- **Layout, against an independent source.** Each expert's gate, up and down projection dequantized with this
  library's convention (even element in the low nibble, row-major scales, value = code x scale x `weight_scale_2`)
  against the same weights of the bf16 checkpoint `Qwen/Qwen3-30B-A3B`: relative error 0.094 to 0.095 for every
  projection, the size of FP4 rounding. With the nibbles swapped it is 1.414, about the square root of two that two
  unrelated matrices of equal norm give, so the check can fail and did not.
- **Global scales.** FC1 applies one `weight_scale_2` to both halves; gate and up carry the same value in
  128 of 128 experts, so that is exact for this checkpoint.

The layer as chosen by batch size, and vLLM's Marlin W4A16 MoE on the same codes, scales and global scales, both timed
in one session by graph replay with L2 flushed; error against an fp32 MoE on the dequantized weights; the layer is
bit-identical over 50 calls in every row:

| tokens | FC1 / FC2 | layer error | Marlin error | layer (us) | Marlin (us) | Marlin / layer |
|---|---|---|---|---|---|---|
| 1 | cuda core / cuda core | 0.17% | 0.30% | 27.4 | 37.5 | 1.37x |
| 2 | cuda core / prefetch | 0.21% | 0.38% | 39.7 | 49.9 | 1.26x |
| 4 | cuda core / prefetch | 0.22% | 0.37% | 70.4 | 78.8 | 1.12x |
| 8 | cuda core / prefetch split2 | 0.21% | 0.37% | 107.3 | 113.4 | 1.06x |
| 16 | tensor core / prefetch split2 | 0.20% | 0.36% | 170.8 | 166.7 | 0.98x |

On real weights the layer is faster than Marlin from 1 to 8 randomly routed tokens, closer to the fp32 reference at
every size, and 2.4% slower at 16 tokens, as with the synthetic weights
above.

## FC2: 32 columns per block (`scripts/fc2_cols32.py`)

Every column tile of FC2 reads each of its experts' activations, and the activation reads were the one cost left
standing at 16 tokens (above). Here each warp computes two 16-row tiles of the same expert from one read of the
activations, so the grid has half the column tiles. Two tiles of weights are as many loads in flight per warp as the
prefetch kernel's current and next expert, so this kernel does not also prefetch; expert assignment, warp order and
group order are the prefetch kernel's, and at the same group count its output is bit-identical to it in every case
below, and bit-identical over 50 calls. Registers (cuobjdump): 87 to 127 against
163 to 239 for the prefetch kernel, no local memory in either.

One session, graph replay, L2 flushed (`reports/fc2-cols32-rtx5090-2026-09-30.json`, microseconds; G is the number of
expert groups per column tile; the last column is the best 32-column time over the best 16-column time):

| routing | tokens | 16 col, G=1 | 16 col, G=2 | 32 col, G=1 | 32 col, G=2 | 32 col, G=4 | read floor | ratio |
|---|---|---|---|---|---|---|---|---|
| random | 1 | 15.1 | 15.1 | 17.2 | 19.2 | 18.2 | 9.2 | 1.14 |
| random | 2 | 21.2 | 23.3 | 27.4 | 21.2 | 21.2 | 13.1 | 1.00 |
| random | 4 | 35.6 | 35.6 | 43.8 | 33.5 | 31.5 | 23.2 | 0.88 |
| random | 8 | 52.0 | 50.9 | 60.1 | 45.8 | 45.8 | 31.5 | 0.90 |
| random | 16 | 95.0 | 88.8 | 92.9 | 75.5 | 72.4 | 49.9 | 0.82 |
| fixed8 | 1 | 15.1 | 15.1 | 17.2 | 19.0 | 17.2 | 8.9 | 1.14 |
| fixed8 | 16 | 21.2 | 21.2 | 23.3 | 25.3 | 25.3 | 9.0 | 1.10 |

At 4 to 16 randomly routed tokens the 32-column kernel with 2 or 4 groups is faster, by
16.4 us at 16 tokens (88.8 to
72.4, against a 49.9 us read of the codes). At one token, and when 16
tokens concentrate on 8 experts, it is slower; which of its two changes (no prefetch, half the tiles) costs those cases is
not measured here. So it joins the layer's choice by batch size for the random-routing sizes it wins, measured on the whole layer
next.

## The 32-column FC2 in the layer, on real weights (`scripts/real_ckpt_layer.py --with-cols32`)

The layer of the real-weights section (Qwen3-30B-A3B NVFP4, layer 0) timed three ways in one session: as
chosen by batch size, with FC2 replaced from 4 tokens up by the 32-column kernel at 4 groups, and vLLM's Marlin; on
random routing and with every token sent to the same 8 experts. Both versions of the layer are bit-identical over 50
calls (`reports/real-ckpt-layer0-cols32-rtx5090-2026-09-30.json`, microseconds):

| routing | tokens | layer as chosen | with 32-column FC2 | difference | Marlin |
|---|---|---|---|---|---|
| random | 4 | 68.9 | 74.3 | +5.4 | 79.3 |
| random | 8 | 107.3 | 107.3 | +0.0 | 113.4 |
| random | 16 | 172.6 | 166.7 | -6.0 | 167.8 |
| fixed8 | 4 | 37.6 | 43.7 | +6.1 | 37.6 |
| fixed8 | 16 | 41.7 | 45.8 | +4.1 | 41.7 |

In the layer the kernel gains 6.0 us at 16 random tokens, against
16.4 us when FC2 is timed alone; that brings the layer to 166.7 us against Marlin's
167.8. It loses at 4 tokens and when tokens concentrate on 8 experts
(+4.1 us at 16). Where the rest of the isolated gain goes inside the layer is not
measured here. The layer chooses by batch size alone, and at 16 tokens the 32-column kernel helps random routing and hurts
concentrated routing, so the choice is left unchanged until the layer can tell the two apart (the number of experts a
batch touches is known only on the GPU, after routing).

## Where the 32-column kernel's gain goes in the layer

*Correction (2026-10-01): timed with L2 flushed, so the activations came from DRAM; in the layer FC1 has just written them and they are in L2. See "The activations are warm in the layer" at the end.*

The same run timed, in one session, the layer up to FC1 (routing and FC1, the same launches), and each FC2 alone on the
activations that layer had just produced. FC2's share of the layer is the layer minus that prefix; where it is smaller
than FC2 alone, the difference is FC2 time hidden behind the kernels before it
(`reports/real-ckpt-layer0-cols32-split-rtx5090-2026-10-01.json`, microseconds):

| routing | tokens | routing + FC1 | 16-col FC2 alone | its share of the layer | hidden | 32-col FC2 alone | its share of the layer | hidden |
|---|---|---|---|---|---|---|---|---|
| random | 4 | 45.8 | 37.6 | 24.6 | 13.0 | 31.5 | 28.6 | 2.9 |
| random | 8 | 66.3 | 51.7 | 41.0 | 10.8 | 46.0 | 41.0 | 5.1 |
| random | 16 | 105.2 | 90.7 | 67.6 | 23.1 | 72.4 | 61.4 | 11.0 |
| fixed8 | 4 | 29.4 | 17.0 | 8.2 | 8.8 | 19.2 | 14.3 | 4.9 |
| fixed8 | 16 | 27.4 | 25.3 | 14.3 | 10.9 | 25.3 | 18.4 | 6.9 |

At 16 random tokens the 32-column kernel is 18.2 us faster alone and 6.1 us faster in the
layer, because the kernel it replaces hides 23.1 us of its time behind routing and FC1 and the 32-column kernel
hides 11.0. Both launch with programmatic dependent launch; which part of either kernel runs under its predecessor,
and why the two hide different amounts, is not measured here.

## The activations are warm in the layer, and the timing alone flushed them

Two runs test where the hidden time above comes from. First, the same breakdown with every kernel launched without
programmatic dependent launch (`reports/real-ckpt-layer0-cols32-split-nopdl-rtx5090-2026-10-01.json`): the time each
FC2 hides is about the same as with it, so dependent launch is not the cause (microseconds hidden, FC2 alone minus its
share of the layer):

| routing | tokens | 16-col FC2 hidden, no PDL | 32-col FC2 hidden, no PDL |
|---|---|---|---|
| random | 4 | 11.6 | 2.8 |
| random | 8 | 5.9 | 4.9 |
| random | 16 | 23.1 | 6.9 |
| fixed8 | 4 | 7.0 | 4.9 |
| fixed8 | 16 | 9.0 | 4.9 |

Second, each FC2 timed alone with the activations FC1 wrote, and the routing buffers, read back into L2 after the
flush and before the timed replay, the weights left cold (`micro_floor.graph_time(after_flush=...)`;
`reports/real-ckpt-layer0-cols32-actwarm-rtx5090-2026-10-01.json`, with dependent launch as in the layer):

| routing | tokens | 16-col alone, flushed | 16-col alone, activations warm | 16-col share of the layer | 32-col alone, flushed | 32-col alone, activations warm | 32-col share of the layer |
|---|---|---|---|---|---|---|---|
| random | 4 | 35.8 | 28.3 | 22.6 | 31.5 | 30.4 | 28.7 |
| random | 8 | 49.9 | 43.7 | 43.0 | 45.8 | 45.8 | 42.0 |
| random | 16 | 90.9 | 71.4 | 69.4 | 72.4 | 71.7 | 63.5 |
| fixed8 | 4 | 17.2 | 13.3 | 8.4 | 19.2 | 19.2 | 14.3 |
| fixed8 | 16 | 25.3 | 21.1 | 16.2 | 25.3 | 25.1 | 20.2 |

With its activations in L2, as they are when it runs right after FC1, the 16-column FC2 at 16 random tokens takes
71.4 us instead of 90.9, close to its share of the layer
(69.4); the 32-column FC2, which reads each activation once, changes little
(72.4 to 71.7). The hidden time is the activations being in L2,
not overlap. This corrects the cost given to activation re-reads in the sections above, which timed FC2 with L2 flushed:
in the layer those re-reads are served from L2, and most of the 32-column kernel's gain alone was the flushed
activations. In the layer it is still 5.9 us faster at 16 random tokens (shares
69.4 and 63.5), by a margin these runs
do not attribute. Timing a kernel of the layer alone needs its inputs in the state its predecessor leaves them.

## Occupancy does not explain the 32-column kernel's place in the layer

The 32-column FC2 fits two blocks per SM (121 registers and 17,412 bytes of shared memory at 16 tokens, cuobjdump); the
prefetch FC2 it replaces, at 168 registers, fits one. A build of the 32-column kernel with 40 KB of shared memory no
thread reads (58,372 bytes, so one block per SM; `fc2_cols32.build(one_block=True)`) gives the same output and, in the
layer, the same time at 16 random tokens (`reports/real-ckpt-layer0-cols32-occupancy-rtx5090-2026-10-01.json`,
microseconds, one session):

| routing | tokens | layer as chosen | with 32-column FC2 | with 32-column FC2, one block per SM |
|---|---|---|---|---|
| random | 4 | 68.6 | 74.5 | 70.4 |
| random | 8 | 107.3 | 107.3 | 105.0 |
| random | 16 | 171.0 | 166.7 | 166.7 |
| fixed8 | 4 | 37.9 | 43.8 | 41.7 |
| fixed8 | 16 | 41.7 | 45.8 | 45.8 |

At 16 random tokens the layer takes 166.7 us with either build of the 32-column kernel, against
171.0 as chosen, so the second resident block is not what it gains there. At 4 random tokens the
one-block build is faster than the two-block one (70.4 against 74.5),
which these runs do not explain either.

## Both FC2 kernels read the same DRAM bytes in the layer (Nsight Compute)

`scripts/ncu_fc2_in_layer.py` runs routing and FC1, then the prefetch FC2; then routing and FC1 again, then the
32-column FC2 (synthetic weights of the same shape, L2 evicted before each sequence), under
`ncu --replay-mode application --cache-control none`, so each FC2 is measured with the activations FC1 just wrote still
in L2 and every metric pass starts from the same state. Megabytes per kernel, L2 hit rate in percent
(`reports/ncu-fc2-in-layer-*-rtx5090-2026-10-01.csv`):

| case | DRAM read, 16-col | DRAM read, 32-col | L2 traffic, 16-col | L2 traffic, 32-col | L2 hit %, 16-col | L2 hit %, 32-col |
|---|---|---|---|---|---|---|
| 16 random | 77.3 | 73.1 | 106.1 | 111.1 | 28.2 | 34.0 |
| 16 random, repeat | 77.2 | 75.8 | 105.8 | 113.8 | 27.1 | 33.9 |
| 4 random | 27.4 | 27.7 | 34.6 | 39.0 | 20.1 | 28.5 |
| 16 concentrated (8 experts) | 8.2 | 7.4 | 37.1 | 33.1 | 75.8 | 77.7 |

The two kernels read about the same bytes from DRAM in every case; between the two runs of the 16-token case a kernel's
DRAM bytes moved by up to 2.7 MB, as much as the two kernels differ there. The 32-column kernel moves slightly
more through L2 under random routing and slightly less when the tokens concentrate on 8 experts. So it does not gain by
reading fewer DRAM bytes: the prefetch kernel's repeated activation reads are
served from L2, as the warm timings above found, and the 32-column kernel's place in the layer comes from something these
counters do not attribute (how the loads are issued and overlapped, for example). Kernel durations under the profiler
are not used: it serializes the kernels and drops programmatic dependent launch, and its times do not match the
graph-replay timings above.

## What the 32-column kernel does differently, by stall reason (Nsight Compute)

The same driver, now also running the one-block-per-SM build, with the scheduler's issue rate and the stall reasons per
issued instruction (`reports/ncu-fc2-stall-reasons-*-rtx5090-2026-10-01.csv`; application replay, activations in L2):

| 16 random tokens | 16-col prefetch | 32-col | 32-col, one block per SM |
|---|---|---|---|
| warps active, % of peak | 16.65 | 25.84 | 16.65 |
| instructions executed (millions) | 22.66 | 19.33 | 19.33 |
| instructions issued per scheduler cycle | 0.17 | 0.23 | 0.22 |
| stalled on a memory result, per issue | 7.07 | 5.19 | 4.83 |
| stalled on the load queue, per issue | 0.26 | 3.12 | 0.01 |
| stalled on the math pipe, per issue | 0.77 | 1.46 | 0.90 |

| 4 random tokens | 16-col prefetch | 32-col | 32-col, one block per SM |
|---|---|---|---|
| warps active, % of peak | 16.61 | 25.48 | 16.60 |
| instructions executed (millions) | 7.46 | 6.89 | 6.90 |
| instructions issued per scheduler cycle | 0.17 | 0.20 | 0.19 |
| stalled on a memory result, per issue | 7.61 | 7.20 | 6.19 |
| stalled on the load queue, per issue | 0.53 | 3.35 | 0.01 |
| stalled on the math pipe, per issue | 0.19 | 0.30 | 0.17 |

At 16 random tokens the one-block build runs at the prefetch kernel's occupancy and still issues more per cycle: it
executes 15% fewer instructions (one activation load and the token bookkeeping serve two tiles) and its
warps wait less on memory results per instruction issued (two tiles of weight loads in flight per warp). The two-block
build's extra warps add no issue rate and meet the load queue instead (the load-queue stall is about 3 per issue, against 0.01 for the one-block build and 0.26 for the prefetch kernel), which is consistent with the layer time being the same with one block or two (above). At 4 tokens the
instruction saving is smaller and the two-block build meets the same load-queue stall (3.35, against 0.01 and
0.53), consistent with the one-block build
being the faster of the two in the layer there; that link is not tested here.

## The single-kernel FC2 results again, with the activations warm (`--act-warm`)

Every FC2 bench above timed the kernel alone behind an L2 flush, so its activations came from DRAM; in the layer FC1 has
just written them and they are in L2 (above). The four benches now take `--act-warm`: after each flush the activations
and the routing buffers are read back into L2, untimed, and the weights stay cold. Each bench ran plain and warm back to
back in one session (`reports/fc2-{groups,occupancy,split,chain}-rtx5090-2026-10-01.json` and the `-actwarm-` files
beside them). Two checks on the instrument: today's cold runs sit within about 2 us of the 2026-09-30 reports on every
row, so differences smaller than that are not read; and the math-only build that makes its activations instead of
reading them times the same cold and warm (35.6 and 35.9 us at 16 random tokens), as does the stream read of the codes
(49.6 and 49.9), so the warming itself costs the timed kernel nothing.

At 16 random tokens, microseconds, cold then warm:

| bench | variant | cold | warm |
|---|---|---:|---:|
| groups | v1 (no prefetch) | 106.2 | 71.7 |
| groups | prefetch, 1 group | 89.7 | 62.1 |
| groups | prefetch, 2 groups | 87.0 | 70.2 |
| groups | prefetch, 3 groups | 87.3 | 73.4 |
| groups | prefetch, 4 groups | 88.8 | 77.5 |
| occupancy | 8 warps, 2 groups | 87.0 | 69.4 |
| occupancy | 4 warps, 2 groups | 87.8 | 63.1 |
| occupancy | 4 warps, 4 groups | 85.8 | 71.2 |
| occupancy | 8 warps, two blocks per SM | 82.7 | 78.8 |
| split | full | 86.8 | 70.2 |
| split | loads only | 68.4 | 59.1 |
| split | loads only, contiguous | 70.3 | 58.0 |
| split | math only | 47.9 | 41.5 |
| chain | full | 86.8 | 69.7 |
| chain | full, chain prefetch | 82.7 | 67.6 |
| chain | math only | 47.9 | 42.0 |
| chain | math only, activations made | 35.6 | 35.9 |

What changes:

- **Groups per tile.** Cold, one to four groups span 87.0 to 89.7 us and the conclusion was that parallelism over experts
  does not matter. Warm, they order: one group 62.1, two 70.2, three 73.4, four 77.5. One group is also the fastest warm
  at 8 random tokens (36.9 against 42.7 for two) and on 8 experts (15.6 against 17.4). With the activations in L2, each
  added group costs more than the latency it hides.
- **Two blocks per SM.** Cold it was the fastest build at 16 random tokens (82.7); warm it is the slowest (78.8 against
  63.1 to 71.2 for the one-block builds), and it is the slowest warm at 8 random tokens (51.8) and on 8 experts (25.1)
  as well. Its 136 bytes of spill buy occupancy that only helped while the activations came from DRAM.
- **Loads against a read.** Warm, the loads-only build takes 59.1 us against a comparable read of 56.1 (49.9 x 9/8 for
  the scale bytes), about 1.05x where it was 1.23x cold; the loads are close to the read once the activations are in L2.
- **What the activations cost the arithmetic side.** Cold, reading activations was about 12 us of math-only (47.9 against
  35.6); warm it is about 6 (42.0 against 35.9).

What does not change: skipping the MMAs of empty pair tiles (within 1 us cold and warm at every routing), reading the
codes contiguously (58.0 against 59.1 warm), and the chain prefetch, whose 2.1 us warm gain at 16 random tokens is
within the cross-session spread and which is no faster at 8 random tokens (43.6 against 43.1).

The layer picks one group up to 4 tokens and two above (`moe_layer.choice`), so at 16 tokens it runs the 8-warp,
2-group prefetch build (69.4 to 70.2 warm here, and 69.4 as its share of the layer, above); the benches behind that rule
were timed cold. The one-group build and the 4-warp, 2-group build are 7 to 8 us faster alone with warm activations; whether
that holds inside the layer is the next measurement, with `scripts/real_ckpt_layer.py`.

## One group per tile in the layer (`scripts/real_ckpt_layer.py --fc2-groups-sweep`)

The layer on real weights (layer 0 of Qwen3-30B-A3B in NVFP4) with its FC2 run three ways in one session: 8 warps and
one group, 8 warps and two groups (the choice above 4 tokens until now), and 4 warps and two groups. Run twice
(`reports/real-ckpt-layer0-fc2groups-rtx5090-2026-10-01.json`, and with the 32-column kernels beside them in
`reports/real-ckpt-layer0-fc2groups-cols32-rtx5090-2026-10-01.json`); microseconds, second run, the first within 0.5 us
of it on every row:

| routing | tokens | 1 group | 2 groups | 4 warps, 2 groups | 32-col | 32-col, one block per SM | Marlin |
|---|---|---:|---:|---:|---:|---:|---:|
| random | 2 | 39.7 | 42.2 | 41.8 | n/a | n/a | 50.0 |
| random | 4 | 69.6 | 74.4 | 72.5 | 76.6 | 72.4 | 78.6 |
| random | 8 | 101.6 | 107.2 | 103.2 | 111.3 | 104.2 | 113.0 |
| random | 16 | 164.0 | 170.2 | 175.8 | 164.9 | 164.6 | 165.2 |
| fixed8 | 4 | 39.7 | 41.6 | 41.5 | 43.8 | 42.4 | 37.7 |
| fixed8 | 16 | 43.8 | 43.8 | 46.1 | 47.2 | 45.9 | 41.6 |

Within a session, the same build timed twice agrees to about 1 us (the layer as chosen against the same build in the
sweep: 107.3 and 107.2 at 8 random tokens, 170.3 and 170.2 at 16). One group is 5.6 and 6.2 us faster than two at 8 and
16 random tokens, equal on 8 experts, and at least as fast as every other FC2 at every row; the 32-column kernel's
advantage in the layer was over the two-group build only. Every variant's error against the fp32 reference is 0.20% to
0.22% and every one is bit-identical over 50 calls.

`moe_layer.choice` now runs one group from 2 to 16 tokens, which is every batch the layer takes: the FC2 kernels are
built for at most 16 tokens (`MAXM`). (Corrected 2026-10-01: this sentence first said two groups stayed in use above 16
tokens; no batch above 16 reaches FC2.) The layer
under the new rule (`reports/real-ckpt-layer0-rule-g1-rtx5090-2026-10-01.json`): 101.2 us at 8 random tokens and 163.1
at 16, against Marlin's 113.4 and 164.6 in the same run; at 16 random tokens the two are within the session's spread.
Marlin stays ahead on 8 concentrated experts (37.7 against 39.7 at 4 tokens, 41.6 against 43.8 at 16).

## Where Marlin's lead on concentrated routing comes from (`scripts/ncu_layer_vs_marlin.py`)

The layer and vLLM's Marlin MoE, each run three times after an L2 eviction under Nsight Compute (`--cache-control none
--clock-control none`, so each kernel is timed in the cache state its predecessor leaves; synthetic weights of the layer's
shape, `reports/ncu-layer-vs-marlin-{4-fixed8,16-fixed8,16-random}-rtx5090-2026-10-01.csv`). Kernel time only: no graph,
no dependent launch, no gaps between kernels, so the sums are not the graph-timed layer times above; and with the clocks
uncontrolled the repeats differ, by up to 10 percent on the stable rows and far more on one (below). Microseconds, three
repeats each:

| 4 tokens on 8 experts | ours | Marlin |
|---|---|---|
| routing | `k_route` 3.6 to 3.7 | align 3.1, count-and-sort 1.4 to 1.9 |
| FC1 | `k_fc1` (CUDA cores) 23.1 to 23.4 | Marlin 15.5 to 15.7, act-and-mul 1.6 to 1.8 |
| FC2 | `k_fc2_pf` 10.0 to 12.3 | Marlin 10.5 to 10.7, reduce 4.4 to 4.8 |
| sum | 36.7 to 39.1 | 36.8 to 37.9 |

The 2 us by which Marlin leads the layer on 8 experts at 4 tokens (37.7 against 39.7, above) is FC1's: on routing that
puts all 32 pairs on 8 experts, the CUDA-core FC1 the layer picks up to 8 tokens takes 7.6 us longer than Marlin's FC1,
while the layer's routing and FC2 together are about 5 us shorter than Marlin's five other kernels. At 16 tokens on 8
experts the tensor-core FC1 is level with Marlin's (15.9 to 17.2 against 16.8 to 19.8) and the kernel sums are level
within the spread (34.9 to 39.7 against 38.9 to 41.6), so the 2 us the graph-timed layer gives Marlin there (41.6
against 43.8) is not in any kernel's time; it is in the launch and dependency gaps, which this run does not see.

At 16 random tokens the layer's `k_fc2_pf` took 56.9, 70.8 and 81.7 us on the three repeats where Marlin's FC2 took 49.7 to
50.3 and the layer's FC1 91.6 to 95.6: the FC2 kernel's time moves by 25 us between identical runs under the profiler, the
others' by 2 to 4. The graph-timed layer does not show that spread (163.1 against 163.6 and 164.0 across sessions), so it
is a property of the profiled, unoverlapped run, and open: the kernel's cross-group counters and its expert walk are the
places to look.

Next: the tensor-core FC1 at 4 tokens on concentrated routing, where the CUDA-core one loses to Marlin.

## The tensor-core FC1 from 2 tokens up (`scripts/real_ckpt_layer.py --fc1-sweep`)

The ncu run above put the 2 us on 8 experts at 4 tokens in the CUDA-core FC1, which the layer picked up to 8 tokens on
random-routing timings. The layer on real weights with FC1 forced each way, in one session, then the same sweep again with
the new rule in place (`reports/real-ckpt-layer0-fc1sweep-rtx5090-2026-10-01.json`, `...-fc1sweep-rule-...json`);
microseconds, second run, the first within 0.5 us of it on every row but one (random, 1 token: 29.2 and 27.5 for the
tensor-core FC1, where Marlin's own row moved 0.2):

| routing | tokens | CUDA-core FC1 | tensor-core FC1 | Marlin |
|---|---|---:|---:|---:|
| random | 1 | 29.2 | 27.5 | 35.6 |
| random | 2 | 39.7 | 39.7 | 50.0 |
| random | 4 | 70.2 | 68.6 | 78.6 |
| random | 8 | 101.1 | 101.2 | 112.4 |
| random | 16 | 228.1 | 163.7 | 165.6 |
| fixed8 | 4 | 39.7 | 27.4 | 37.7 |
| fixed8 | 8 | 59.1 | 29.5 | 37.7 |
| fixed8 | 16 | 122.7 | 43.8 | 41.8 |

On random routing the two FC1 kernels time the same in the layer from 1 to 8 tokens (within 1.6 us), so the CUDA-core
choice bought nothing there; on routing that puts every token on the same 8 experts it cost 12.3 us at 4 tokens and 29.6
at 8 (a row the layer had not been timed on before), where the CUDA-core kernel's work per expert grows with the tokens
on it and only 8 blocks' worth of experts are live. Every variant's error against the fp32 reference is the same to four
digits and bit-identical over 50 calls. `moe_layer.choice` now runs the tensor-core FC1 from 2 tokens up; at 1 token
the CUDA-core pair stays. Under the new rule the layer is ahead of Marlin on every row but one: 27.4 against 37.7 and
29.5 against 37.7 on 8 experts at 4 and 8 tokens, 163.0 against 165.6 at 16 random tokens; on 8 experts at 16 tokens
Marlin keeps 2 us (41.8 against 43.8), which the ncu run above places in launch gaps rather than in any kernel.

## Dependent launch costs the layer 2 us at 16 tokens (`scripts/real_ckpt_layer.py --no-pdl`)

The layer on real weights with programmatic dependent launch on and off, each twice in separate sessions
(`reports/real-ckpt-layer0-fc1sweep-{rule,pdl-rep2}-...json` on; `...-fc1sweep-{nopdl,nopdl-rep2}-...json` off); the
FC1 and FC2 choices are the current rule's. Microseconds, both runs:

| routing | tokens | PDL on | PDL off | Marlin (same runs) |
|---|---|---:|---:|---:|
| random | 1 | 28.7, 28.8 | 30.1, 29.6 | 35.6 to 36.6 |
| random | 2 | 39.7, 39.9 | 41.8, 41.8 | 49.9 to 50.0 |
| random | 4 | 69.4, 70.2 | 70.4, 70.4 | 78.6 to 79.1 |
| random | 8 | 101.2, 101.3 | 101.9, 102.0 | 112.4 to 113.2 |
| random | 16 | 163.0, 163.6 | 161.2, 161.7 | 164.0 to 165.6 |
| fixed8 | 4 | 27.4, 27.4 | 27.4, 27.4 | 37.7 |
| fixed8 | 8 | 29.5, 29.5 | 29.6, 29.5 | 37.7 to 37.9 |
| fixed8 | 16 | 43.8, 43.7 | 42.0, 41.8 | 41.6 to 41.8 |

Dependent launch is worth 0.9 to 2.1 us at 1 and 2 tokens, nothing at 4 and 8, and costs 1.8 to 2.0 us at 16 tokens on
both routings, in both runs. The 2 us Marlin kept on 8 experts at 16 tokens is this: with dependent launch off the layer
takes 41.8 to 42.0 there against Marlin's 41.6 to 41.8, level within the spread. Why the early trigger helps short batches
and costs the long ones is not measured here; the FC2 kernel at 16 tokens runs longest, and a successor started early
holds its blocks while it waits. `moe_layer.use_pdl` now switches dependent launch off at 16 tokens and keeps it on below, and
`real_ckpt_layer.py` sets it per batch; under that rule (`reports/real-ckpt-layer0-fc1sweep-pdlrule-rtx5090-2026-10-01.json`)
the layer takes 161.5 us at 16 random tokens against Marlin's 163.5 in the same run, and 41.8 against 41.5 on 8 experts at
16 tokens; the other rows are unchanged (28.8, 39.9, 68.6, 101.4, 27.4, 29.5 against Marlin's 35.8, 49.9, 78.8, 112.7,
37.7, 37.7). Every row is now level with or ahead of Marlin; on 8 experts at 16 tokens the two are within 0.3 us, inside
the session's spread.

## The same benches on an RTX PRO 6000 Blackwell Workstation Edition (`scripts/pod_stage2.sh`)

A cloud RTX PRO 6000 Blackwell Workstation Edition (96 GB, 188 SMs, driver 580.178.04, CUDA 13.0 host, the same
torch 2.13.0+cu130 and CUDA 13.2 toolkit as the conformance run) ran the synthetic stage-2 benches with
`scripts/pod_stage2.sh`: the floor, the layer, the four FC2 benches cold and with `--act-warm`, and the breakdown
(`reports/rtxpro6000-stage2-2026-10-01/`). vLLM was not installed there, so there is no Marlin column; the layer
report's `best_existing` field is read from the RTX 5090 baseline file and is not a PRO 6000 number. Two cautions before
reading: identical configurations timed twice on the PRO 6000 differ by up to 4.4 us (the 1-token layer, 33.0 and 28.6
us for the same two kernels, where the RTX 5090 gave 27.4 twice), so PRO 6000 differences under about 5 us are not
read here; and `moe_layer.py` launched every kernel with dependent launch on in that run (it now follows
`use_pdl`), so its 16-token rows carry the cost the RTX 5090 run measured at 1.8 to 2.0 us.

Microseconds, 16 random tokens, FC2 alone, cold then with the activations warm; the RTX 5090 columns are this
document's 2026-10-01 runs:

| variant | PRO 6000 cold | PRO 6000 warm | RTX 5090 cold | RTX 5090 warm |
|---|---:|---:|---:|---:|
| stream read of the codes | 55.3 | - | 49.3 | - |
| prefetch, 1 group | 94.2 | 59.4 | 89.7 | 62.1 |
| prefetch, 2 groups | 90.8 | 69.6 | 87.0 | 70.2 |
| prefetch, 4 groups | 90.1 | 69.6 | 88.8 | 77.5 |
| 4 warps, 2 groups | 94.2 | 61.4 | 87.8 | 63.1 |
| 4 warps, 4 groups | 79.3 | 59.4 | 85.8 | 71.2 |
| 8 warps, two blocks per SM | 88.1 | 81.9 | 82.7 | 78.8 |
| loads only | 73.7 | 61.4 | 68.4 | 59.1 |
| math only | 45.1 | 34.8 | 47.9 | 41.5 |
| math only, activations made | 30.7 | 28.7 | 35.6 | 35.9 |

What carries over: with the activations warm, one group is 10 us faster than two (59.4 against 69.6; 62.1 against 70.2
on the 5090) and also fastest at 8 random tokens (36.9 against 43.0) and on 8 experts (14.3 against 16.4); the two-blocks
build is slowest warm (81.9); the loads-only build sits within 1.1x of the stream read warm (61.4 against 55.3) where it
was 1.33x cold; reading activations costs the arithmetic side about 6 us warm (34.8 against 28.7) and 14 cold; skipping
empty pair tiles and the contiguous read change nothing. What differs: the PRO 6000 reads the codes 12 percent slower
(55.3 against 49.3 us for the same bytes) and its cold kernels are further from its read than the 5090's (the 2-group
prefetch at 8 random tokens 69.6 cold against 43.0 warm, where the 5090 gave 50.0 and 43.1), so the warm numbers, not the
cold ones, are the ones that agree across the two parts; and the 4-warp, 4-group build is level with one group warm on
the PRO 6000 (59.4) where it trailed on the 5090 (71.2), a difference this run does not explain.

The layer (synthetic weights, the rule's kernels, dependent launch on): 165.9 us at 16 random tokens, 108.5 at 8, 73.7
at 4, 43.0 at 2 and 43.0 on 8 experts at 16 - within 5 us of the RTX 5090's 2026-09-30 synthetic run on every row from 2 tokens up except
16 random tokens, where it is 5 us faster (170.8); at 1 token it is 5.6 us slower (33.0 against 27.4), inside the PRO
6000's own repeat spread above. The breakdown's cold FC1 and FC2 are 6 to 9 us longer than the 5090's
(FC1 71.6 against 62.7 at 8 random tokens; FC2 94.2 against 87.9 at 16) while the layers are level, which is the warm
activations again: the layer never pays the cold price its pieces show.

## The FC2 kernel's outliers (`scripts/fc2_replay_dist.py`)

The 25 us spread of `k_fc2_pf` across three profiled repeats (above) is sporadic outliers, not a wide distribution, and
it is not the clocks. Under Nsight Compute with the clocks uncontrolled, ten repeats of the layer at 16 random tokens gave
the FC2 kernel 55.2 to 59.0 us on eight and 79.7 and 1659.7 on two, while FC1 spanned 89.7 to 93.7 and Marlin's two
kernels 93.6 to 95.6 and 49.1 to 50.7; with the clocks locked to base, all ten FC2 repeats fell in 73.0 to 76.2 (FC1 112.9
to 118.7), every kernel about 25 percent slower and none out of line
(`reports/ncu-jitter-16-random-clock-{none,base}-rtx5090-2026-10-01.csv`). The kernel has no cross-block wait to stall
on: with one group per tile its blocks are independent, and the counter path (the last group adds the partials) is a single
atomic per block, never a spin.

Outside the profiler, 400 graph replays each behind an L2 flush, the FC2 kernel with the activations warm, and FC1 and the
stream read of the codes beside it as controls (`reports/fc2-replay-dist-16-random-rtx5090-2026-10-01.json`):

| kernel | median | p90 | p99 | max | replays above 1.5x the median | above 2x |
|---|---:|---:|---:|---:|---:|---:|
| FC2 prefetch, 1 group, activations warm | 63.0 | 65.5 | 226.3 | 549.9 | 10 of 400 | 5 |
| FC1 tensor core, cold | 102.5 | 104.2 | 106.2 | 911.7 | 2 of 400 | 2 |
| stream read of the codes, cold | 49.8 | 50.9 | 56.1 | 206.8 | 2 of 400 | 2 |

Every kernel shows rare outliers on this machine, a desktop GPU that also drives the display (the baseline utilisation
of 24 to 46 percent noted elsewhere in this repository): about one replay in 200 for FC1 and for the read, with maxima of
4 to 9 times the median. The FC2 kernel's rate is five times that, one replay in 40 above 1.5x the median and one in 80
above 2x, with maxima of 3 to 9 times. The medians agree with the benches above (63.0 against 59.4 to 62.1 warm), so the
rule's choice stands; what is open is why this kernel is preempted, or stalls, more often than the others - its blocks run
longest (about 60 us each at one block per SM) and a kernel whose blocks are all resident for its whole duration has the
most to lose from a display-driven preemption, but that is a candidate, not a measurement. A run on a GPU with no display
attached would separate the two.
