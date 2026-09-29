# Stage 2 baselines: NVFP4 MoE on SM120 today (RTX 5090, 2026-09-29)

Before writing a grouped GEMM or a fused MoE for SM120, this measures what already exists. Scripts:
`scripts/bench_moe_baseline.py` (latency and error of each path), `scripts/diag_b12x_actquant.py`,
`scripts/diag_b12x_nondeterminism.py` and `scripts/diag_b12x_lost_update.py` (the findings below). vLLM 0.28's Marlin
W4A16 MoE is measured from the same FP4 bytes, prepared with vLLM's own NVFP4-to-Marlin conversion. JSON: `reports/moe-baseline-rtx5090-2026-09-29.json`,
`reports/b12x-nondeterminism-rtx5090-2026-09-29.json`. Stack as in the stage 1 reports: FlashInfer 0.6.16.post3,
PyTorch 2.13.0+cu130, driver 610.47.

## What already exists (survey, September 2026)

- FlashInfer 0.6.16 ships three SM120 MoE paths: `b12x_fused_moe` (CuTe-DSL kernels from the b12x project, W4A4 and
  W4A16, with micro, static and dynamic backends chosen by routed row count), `cutlass_fused_moe` with an SM120 module,
  and `group_gemm_nvfp4_nt_groupwise` (grouped GEMM only, tile_m fixed at 128).
- b12x itself (github.com/lukealonso/b12x, Apache-2.0) is a CuTe-DSL/Triton library for SM120/SM121 covering GEMM,
  attention and fused MoE; its dense GEMM is FlashInfer's `b12x` backend (stage 1 measured it).
- sm120_nvfp4_ops (github.com/xinyang-zhou/sm120_nvfp4_ops, MIT, research preview) has an NVFP4 GEMM, a grouped GEMM and
  a fused MoE for the RTX 5090; it states that CUDA-graph and race validation are still pending.
- Open problem reported upstream: flashinfer-ai/flashinfer#4990 (September 2026, no reply yet) measures the SM120/SM121
  CUTLASS grouped MoE at one CTA per SM and 88 to 90% "no eligible warp" cycles on a 512-expert model: latency-bound,
  with every one of its 32 to 63 tactics within 2%.
- Earlier reports that native NVFP4 MoE on SM120 produced garbage (flashinfer#2723, NVIDIA/cutlass#3096) or ran slower
  than Marlin W4A16 predate FlashInfer 0.6.16; on this stack both W4A4 paths below produce correct output.

So a new grouped GEMM or fused MoE is not the missing piece by itself. What the numbers below show is missing is decode
speed at small batch, and determinism.

## Setup

One MoE layer of Qwen3-30B-A3B's shape: 128 experts, top-8, hidden 2048, expert intermediate 768, SwiGLU. Weights are
quantized once with this repository's reference quantizer (global scale 1, the convention of FlashInfer's own MoE tests,
because FlashInfer 0.6.16's b12x path uses its first alpha both as weight scale and as input quantization scale) and the
same bytes are given to every path in the layout it expects. Tokens M = 1, 2, 4, 8, 16, random routing.

The DRAM floor is the time to read the touched experts' FP4 weights and scales once at 1792 GB/s:
2,654,208 bytes per expert.

## Latency: CUDA-graph replay, L2 flushed before each replay (microseconds)

| tokens | experts read | DRAM floor | b12x W4A4 | b12x W4A16 | CUTLASS W4A4 | Marlin W4A16 (vLLM) | best as a fraction of the floor |
|---|---|---|---|---|---|---|---|
| 1 | 8 | 11.8 | 43.8 | 37.6 | 45.8 | 37.6 | 31% |
| 2 | 15 | 22.2 | 64.3 | 50.1 | 66.3 | 49.9 | 45% |
| 4 | 30 | 44.4 | 90.9 | 79.6 | 95.2 | 78.8 | 56% |
| 8 | 48 | 71.1 | 123.9 | 115.7 | 132.0 | 113.9 | 62% |
| 16 | 81 | 120.0 | 174.8 | 168.7 | 196.4 | 166.5 | 72% |

(One run; the b12x and CUTLASS columns moved by up to 2 us against the first run of the same script.)

Why cold: inside a forward pass the other layers' weights evict this layer's from L2. The RTX 5090's 96 MB L2 holds
all the experts a batch of 1 to 4 tokens touches (21 to 80 MB), so timings with a warm L2 measure L2 rather than DRAM;
with a warm L2, b12x W4A16 and Marlin at 4 tokens come in under the DRAM floor (44.2 and 37.5 us against 44.4), which
is the tell.

At one token the best path runs at 31% of the rate at which the card can stream the weights it needs; the gap closes as
the batch grows. The two W4A16 paths, b12x and vLLM's Marlin, are the fastest at every batch size and within 2% of each
other, so on this layer native FP4 activations do not pay for themselves at decode batch sizes.

Without a CUDA graph the calls cost far more: 106 to 261 us (b12x W4A4), 248 to 419 us (b12x W4A16), 114 to 310 us
(CUTLASS) and 97 to 226 us (Marlin) at 1 to 16 tokens, so host-side work dominates at small batch for any caller that
does not capture graphs.

## Accuracy

Normwise relative error ||out - ref|| / ||ref|| against a fp32 reference on the dequantized weights:

| path | against its own arithmetic | against unquantized activations |
|---|---|---|
| CUTLASS W4A4 | 0.23% to 0.24% (reference with bf16 intermediates) | 15.0% to 16.4% |
| b12x W4A16 | 0.44% to 0.48% | 0.44% to 0.48% |
| Marlin W4A16 | 0.44% to 0.48% | 0.44% to 0.48% |
| b12x W4A4 | 0.8% to 1.7% per run (reference with the SwiGLU output rounded to bf16 before quantization) | 15.0% to 16.7% |

The 15 to 17% of both W4A4 paths is what NVFP4 activations cost on this synthetic layer; it is the same for both, so
neither W4A4 path is less accurate than the other against the model's intent.

## Finding 1: b12x W4A4 is not deterministic, and occasionally drops or doubles a contribution

Twenty identical calls of `b12x_fused_moe(quant_mode="nvfp4")` return twenty different outputs at 1, 4 and 16 tokens,
and every token's output moves. b12x W4A16, CUTLASS W4A4 and Marlin W4A16 return one output across the same calls. A
CUDA graph of the W4A4 call therefore also does not replay to the eager result; that is this non-determinism, not replay
corruption.

Where it comes from. The W4A4 micro and static kernels document their last step as "Scatter: bf16x2 atomic add
(directly into token-major output)", and FC2 runs per slice of the intermediate dimension. So an output element receives
one atomic bf16 add per (expert, slice) pair that reaches it. Varying that count isolates the cause
(`diag_b12x_nondeterminism.py`, 20 calls each):

| routing | intermediate size | adds per output element | distinct outputs in 20 calls |
|---|---|---|---|
| top-1 | 128 (one slice) | 1 | 1 |
| top-1 | 768 (several slices) | several | 20 |
| top-8 | 128 (one slice) | 8 | 20 |

With one add per element the output is bit-stable; with more than one it is not. Addition in bf16 is not associative,
and the order of atomics varies between calls.

Reordering does not explain all of it. Reordering moves an element by about one bf16 ulp of the largest partial sum it
passes through. In 100 identical top-1 calls (`diag_b12x_lost_update.py`, 4 tokens, intermediate 768), all but a few
elements stay within 8 such ulps of the median run, but 2 of the 100 calls each contain one aligned group of 8 adjacent
output columns in which 7 of the 8 elements are off by tens to hundreds of ulps, the size of a whole slice partial (the
largest: +54.0 at an element whose 128-wide FC2 partials include -52.0). The same pattern (one aligned 8-column group
per affected call) appeared in an earlier sample of 100, and in both samples the first call of the process was one of
the affected ones. A group of 8 bf16 values is the width of one `scatter_add_v4_bf16x2` store, so the pattern fits a
vector atomic update that is lost or applied twice; this has not been traced to a line of the kernel source.

What a caller sees: W4A4 outputs that differ between identical calls by 1 to 4% normwise, and, in about 2% of calls in the configuration tested (top-1, 4 tokens), an
8-column group of one token's output that is wrong by a whole partial contribution.

## Finding 2: where b12x W4A4 rounds

Against the reference that quantizes both GEMM inputs from fp32, b12x W4A4 is 2 to 4% off. Rounding the SwiGLU output
to bf16 before quantizing it for FC2 brings a single run to 0.93%, and quantizing only the FC1 input or only the FC2
input makes it worse (8.1%, 17.2%), so b12x quantizes both inputs and holds the SwiGLU output in bf16 first. The
remaining 0.8 to 1.7% is the run-to-run spread of finding 1. CUTLASS keeps the same quantity in a form that the bf16
reference matches to 0.24%.

## What stage 2 should build, given this

1. Decode at 1 to 4 tokens is the gap: 31 to 55% of the DRAM floor for the best existing path. A kernel that streams the
   touched experts' weights once, at full bandwidth, with both GEMMs and the SwiGLU fused, is where there is room.
2. Determinism is a requirement, not an extra: the W4A4 path with the lowest host overhead gives a different answer on
   every call and, occasionally, a wrong one. A deterministic combine (each output element written by one owner, or a
   fixed-order reduction) is part of the design, not an option.
3. The comparison set for the stage 2 gate is Marlin W4A16 and b12x W4A16 (fastest here, within 2% of each other),
   b12x W4A4 and CUTLASS W4A4, all as CUDA-graph replays with a cold L2.
