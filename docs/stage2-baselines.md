# Stage 2 baselines: NVFP4 MoE on SM120 today (RTX 5090, 2026-09-29)

Before writing a grouped GEMM or a fused MoE for SM120, this measures what already exists. Scripts:
`scripts/bench_moe_baseline.py` (latency and error of each path), `scripts/diag_b12x_actquant.py` and
`scripts/diag_b12x_nondeterminism.py` (the two findings below). JSON: `reports/moe-baseline-rtx5090-2026-09-29.json`,
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

| tokens | experts read | DRAM floor | b12x W4A4 | b12x W4A16 | CUTLASS W4A4 | best as a fraction of the floor |
|---|---|---|---|---|---|---|
| 1 | 8 | 11.8 | 41.8 | 37.6 | 46.8 | 31% |
| 2 | 15 | 22.2 | 64.1 | 49.9 | 66.3 | 45% |
| 4 | 30 | 44.4 | 89.0 | 80.6 | 95.1 | 55% |
| 8 | 48 | 71.1 | 123.7 | 115.5 | 131.8 | 62% |
| 16 | 81 | 120.0 | 173.8 | 168.7 | 195.3 | 71% |

Why cold: inside a forward pass the other layers' weights evict this layer's from L2. The RTX 5090's 96 MB L2 holds
all the experts a batch of 1 to 4 tokens touches (21 to 80 MB), so timings with a warm L2 measure L2 rather than DRAM;
with a warm L2, b12x W4A16 at 4 tokens even comes in under the DRAM floor (43.7 us against 44.4), which is the tell.

At one token the best path runs at 31% of the rate at which the card can stream the weights it needs; the gap closes as
the batch grows. W4A16 is the fastest path at every batch size here, so on this layer native FP4 activations do not pay
for themselves at decode batch sizes.

Without a CUDA graph the calls cost far more: 100 to 265 us (b12x W4A4), 256 to 479 us (b12x W4A16) and 107 to 304 us
(CUTLASS) at 1 to 16 tokens, so host-side work dominates at small batch for any caller that does not capture graphs.

## Accuracy

Normwise relative error ||out - ref|| / ||ref|| against a fp32 reference on the dequantized weights:

| path | against its own arithmetic | against unquantized activations |
|---|---|---|
| CUTLASS W4A4 | 0.23% to 0.24% (reference with bf16 intermediates) | 15.0% to 16.4% |
| b12x W4A16 | 0.44% to 0.48% | 0.44% to 0.48% |
| b12x W4A4 | 0.8% to 1.7% per run (reference with the SwiGLU output rounded to bf16 before quantization) | 15.0% to 16.7% |

The 15 to 17% of both W4A4 paths is what NVFP4 activations cost on this synthetic layer; it is the same for both, so
neither W4A4 path is less accurate than the other against the model's intent.

## Finding 1: b12x W4A4 is not deterministic

Twenty identical calls of `b12x_fused_moe(quant_mode="nvfp4")` return twenty different outputs at 1, 4 and 16 tokens.
Runs differ from each other by up to 1.8% normwise (up to 3.8% in another sample of 20), and every token's output moves,
not a few rows. b12x W4A16 and CUTLASS W4A4 return one output across the same twenty calls. A CUDA graph of the W4A4
call therefore also does not replay to the eager result; that is this non-determinism, not replay corruption.

Not yet established: the mechanism. The pattern fits run-to-run differences in accumulation order before a
quantization step, where a last-bit change moves a value across an FP4 rounding boundary and is amplified to a whole
FP4 step; this has not been checked against the kernel source.

## Finding 2: where b12x W4A4 rounds

Against the reference that quantizes both GEMM inputs from fp32, b12x W4A4 is 2 to 4% off. Rounding the SwiGLU output
to bf16 before quantizing it for FC2 brings a single run to 0.93%, and quantizing only the FC1 input or only the FC2
input makes it worse (8.1%, 17.2%), so b12x quantizes both inputs and holds the SwiGLU output in bf16 first. The
remaining 0.8 to 1.7% is the run-to-run spread of finding 1. CUTLASS keeps the same quantity in a form that the bf16
reference matches to 0.24%.

## What stage 2 should build, given this

1. Decode at 1 to 4 tokens is the gap: 31 to 55% of the DRAM floor for the best existing path. A kernel that streams the
   touched experts' weights once, at full bandwidth, with both GEMMs and the SwiGLU fused, is where there is room.
2. Determinism is a requirement, not an extra: the fastest W4A4 path today gives a different answer each call.
3. The comparison set for the stage 2 gate is b12x W4A16 (fastest here), b12x W4A4 and CUTLASS W4A4 as measured
   above, and Marlin W4A16 (not yet measured here), all as CUDA-graph replays with a cold L2.
