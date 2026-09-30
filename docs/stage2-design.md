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
