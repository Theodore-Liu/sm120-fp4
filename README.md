# sm120-fp4

FP4 (NVFP4 / MXFP4) building blocks for **consumer and workstation Blackwell** (SM120 / SM121: GeForce RTX 50 series,
RTX PRO 6000 Blackwell, DGX Spark), where the datacenter kernel stacks do not apply.

SM120 is not a subset of SM100. It has no `tcgen05.mma`, no Tensor Memory, no cluster multicast and about 99 KB of
shared memory per SM; its only block-scaled MMA is `mma.sync ... kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64`.
Kernels written for B200 either fail to compile for it or run only after patches, and the most common failure mode is
not a crash but silently wrong output: a scale-factor tensor in the wrong layout, a tactic that cannot run being
selected, a CUDA-graph replay that returns poisoned state.

This project is built in three stages, each usable on its own:

1. **Layouts and conformance** (`sm120fp4.layouts`, `sm120fp4.reference`, `tests/`): one written specification of
   every NVFP4/MXFP4 scale-factor layout in circulation (the 128x4 swizzled layout shared by CUTLASS, cuDNN and
   TensorRT-LLM/FlashInfer; the row-major checkpoint layout; DeepGEMM's packed variants; Marlin's repack), pure-PyTorch
   reference converters between them, a reference quantizer and reference GEMM, and a conformance suite that runs the
   installed kernel libraries against the reference on the GPU in front of it: values, layouts, CUDA-graph replay
   stability, and which autotuner tactics can actually run on this architecture.
2. **Grouped (MoE) block-scaled GEMM and fused MoE for SM120**, correct without an autotuner lottery and tuned for the
   99 KB shared-memory budget, including the small-batch decode regime (M of 1 to 16) where the m16n8k64 instruction
   wastes rows and the choice between weight-only (W4A16) and native FP4 (W4A4) paths has to be made per call.
3. **The FP4 kernels DeepGEMM does not ship for SM120**: the FP8xFP4 GEMM, FP4 attention and FP4 einsum paths that
   DeepSeek-V4-class models need.

Stage 1 is complete; stage 2 has a working W4A16 MoE layer, validated on a real NVFP4 checkpoint (below). Nothing
here depends on any particular serving engine; the conformance suite is meant to be run by engine developers against
their own builds.

## Requirements

- An SM120 / SM121 GPU with a driver for CUDA 13.0 or newer (`compute_120f` needs CUDA 13.0).
- PyTorch with CUDA 13 wheels.
- Optional, for the conformance suite: FlashInfer (any version that exposes `fp4_quantize` and `mm_fp4`), CUTLASS
  (C++ headers for the SM120 examples, or the CuTe DSL wheels).

## Layout of the repository

- `docs/scale-layouts.md`: the specification, with every formula quoted from its source and a worked example.
- `sm120fp4/layouts.py`: converters (`to_128x4`, `from_128x4`, padding helpers) and layout descriptors.
- `sm120fp4/reference.py`: the NVFP4 reference quantizer, dequantizer and reference GEMM (fp32 accumulate).
- `tests/`: conformance tests; each test states which failure class it exists to catch and cites the public report.

## What works on SM120 today (RTX 5090 and RTX PRO 6000, FlashInfer 0.6.16.post3, PyTorch 2.13 cu130, driver 610)

Every row was measured on an RTX 5090 and reproduced on an RTX PRO 6000 Blackwell Workstation Edition
(`docs/conformance-report-rtxpro6000.md`).

| item | result | where |
|---|---|---|
| 128x4 scale layout | one layout under three names (CUTLASS `Sm1xxBlockScaledConfig`, cuDNN "128x4 tiled", TensorRT-LLM/FlashInfer `get_sf_out_offset_128x4`); `to_128x4` matches FlashInfer byte for byte | `docs/scale-layouts.md` s.2, `tests/test_layouts.py` |
| 8x4 scale layout | tile 8 rows x 4 columns, `(row % 8) * 4 + col % 4`, rows padded to 8; derived by one-hot probing | `docs/scale-layouts.md` s.3, `tests/test_layouts_8x4.py` |
| NVFP4 quantizer | reference agrees with `fp4_quantize` on every block scale and every code except ulp-level midpoint ties; the sign bit is kept on zero | `tests/test_quantize.py` |
| per-tensor scale direction | FlashInfer's quantizer and dequantizer take it in opposite directions; the wrong one scales outputs by `global_scale**2` silently | `docs/scale-layouts.md` s.1 |
| `mm_fp4`, CUTLASS backend | correct on 4 shapes (max relative error 0.35%), 200 CUDA-graph replays identical to eager | `tests/test_mm_fp4.py` |
| `mm_fp4`, `b12x` backend (FlashInfer's SM12x-specific path) and `auto` | correct on 4 shapes; 200 CUDA-graph replays identical to eager; `auto` on SM12x resolves to this path first | `tests/test_mm_fp4.py` |
| `mm_fp4`, `cute-dsl` backend | refused for capability 120, and rightly: built directly, its kernels fail to compile for `sm_120a` (`sm100`-only MMA) | `docs/conformance-report-rtx5090.md` |
| autotuner tactics | CUTLASS presents 32, b12x 8, on every shape; all run and are correct, so the tuner can pick a slow one (2.05x at 4096-cube) but not a broken one; per-shape pinnable lists in `reports/tactics-*.json` | `scripts/probe_tactics.py`, `docs/conformance-report-rtx5090.md` |
| `mm_fp4`, cuDNN backend | unavailable: cuDNN declines every engine on this architecture (reasons name the scale layout and Hopper-only engines) | `docs/conformance-report-rtx5090.md` |
| `mm_fp4`, TensorRT-LLM backend | unavailable by design: `does not support backend 'trtllm' with capability 120` | same |

To run the whole set on a fresh Linux box or cloud instance with an SM12x GPU, `scripts/pod_conformance.sh` installs
the pinned stack and a matching CUDA toolkit and writes every report; its comments list the three environment traps it
avoids.

Regenerate the table's evidence with `python -m sm120fp4.cli conformance` and `PYTHONPATH=. python scripts/probe_tactics.py`;
the JSON lands in `reports/`.

## Stage 2: the MoE layer on real weights (RTX 5090)

A W4A16 MoE layer for SM120 (router, FC1 with the SwiGLU, FC2 with the weighted sum over experts; FP4 codes decoded with
SM120's `cvt.rn.f16x2.e2m1x2`), each GEMM chosen by batch size (`scripts/moe_layer.py`): FC1 on CUDA cores at one
token and on tensor cores from two tokens up, FC2 on CUDA cores at one token and otherwise a tensor-core kernel that
prefetches its weights and takes one expert group per tile, kernels chained with programmatic dependent launch below 16
tokens and launched plainly at 16 (each rule was measured on both routing shapes below; `docs/stage2-design.md`
records every experiment and the hypotheses ruled out). Validated on the 128 experts of layer 0 of
`nvidia/Qwen3-30B-A3B-NVFP4` (ModelOpt NVFP4), against vLLM's Marlin W4A16 MoE on the same codes, scales and global
scales, timed in one session by CUDA-graph replay with L2 flushed before each replay
(`scripts/real_ckpt_layer.py`, `reports/real-ckpt-layer0-fc1sweep-pdlrule-rtx5090-2026-10-01.json`):

| tokens | routing | FC1 | FC2 | this layer (us) | Marlin (us) | Marlin / this | error | Marlin error |
|---|---|---|---|---|---|---|---|---|
| 1 | random (top-8 of 128) | CUDA cores | CUDA cores | 28.8 | 35.8 | 1.24x | 0.17% | 0.30% |
| 2 | random (top-8 of 128) | tensor cores | tensor cores, prefetch, one group per tile | 39.9 | 49.9 | 1.25x | 0.21% | 0.38% |
| 4 | random (top-8 of 128) | tensor cores | tensor cores, prefetch, one group per tile | 68.6 | 78.8 | 1.15x | 0.22% | 0.37% |
| 8 | random (top-8 of 128) | tensor cores | tensor cores, prefetch, one group per tile | 101.4 | 112.7 | 1.11x | 0.21% | 0.37% |
| 16 | random (top-8 of 128) | tensor cores | tensor cores, prefetch, one group per tile | 161.5 | 163.5 | 1.01x | 0.20% | 0.36% |
| 4 | all tokens on the same 8 experts | tensor cores | tensor cores, prefetch, one group per tile | 27.4 | 37.7 | 1.38x | 0.21% | 0.36% |
| 8 | all tokens on the same 8 experts | tensor cores | tensor cores, prefetch, one group per tile | 29.5 | 37.7 | 1.28x | 0.21% | 0.36% |
| 16 | all tokens on the same 8 experts | tensor cores | tensor cores, prefetch, one group per tile | 41.8 | 41.5 | 0.99x | 0.20% | 0.36% |

- Error is normwise against an fp32 MoE on the dequantized weights; the layer's output is bit-identical over 50 calls
  at every row.
- The checkpoint layout is checked against an independent source first: every projection dequantized with this
  library's convention matches the bf16 original (`Qwen/Qwen3-30B-A3B`) to 0.094 to 0.095 relative error, the size
  of FP4 rounding; with the nibbles swapped it is 1.414.
- The two routing shapes are the two ends of a decode batch: tokens spread over the experts (each token its own
  top-8 of 128) and tokens that all land on the same 8 experts.

### The same table on an RTX PRO 6000

The same script, checkpoint shards and Marlin build on a cloud RTX PRO 6000 Blackwell Workstation Edition (188 SMs;
`NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 610.57.04, 97887 MiB, 12.0, 3090 MHz, 14001 MHz`), set up by `scripts/pod_real_ckpt.sh` and timed the same way (`reports/rtxpro6000-realckpt-2026-10-02/real-ckpt-layer0-fc1sweep-nvidia-rtx-pro-6000-blackwell-workstation-edition-2026-10-02.json`). The layout check gives the same
0.094 to 0.095 and 1.414 as on the RTX 5090, and every row is bit-identical over 50 calls:

| tokens | routing | this layer (us) | Marlin (us) | Marlin / this | error | Marlin error | RTX 5090: Marlin / this |
|---|---|---|---|---|---|---|---|
| 1 | random (top-8 of 128) | 28.7 | 38.9 | 1.36x | 0.17% | 0.30% | 1.24x |
| 2 | random (top-8 of 128) | 43.0 | 53.2 | 1.24x | 0.21% | 0.38% | 1.25x |
| 4 | random (top-8 of 128) | 75.8 | 88.1 | 1.16x | 0.22% | 0.37% | 1.15x |
| 8 | random (top-8 of 128) | 108.5 | 122.9 | 1.13x | 0.21% | 0.37% | 1.11x |
| 16 | random (top-8 of 128) | 165.9 | 174.1 | 1.05x | 0.20% | 0.36% | 1.01x |
| 4 | all tokens on the same 8 experts | 30.7 | 43.0 | 1.40x | 0.21% | 0.36% | 1.38x |
| 8 | all tokens on the same 8 experts | 30.7 | 43.0 | 1.40x | 0.21% | 0.36% | 1.28x |
| 16 | all tokens on the same 8 experts | 45.1 | 43.0 | 0.96x | 0.20% | 0.36% | 0.99x |

The layer is ahead of Marlin on 7 of the eight rows on both GPUs; the row it loses is the same on both (16 tokens on the same 8 experts, 45.1 against 43.0 us
here), and the ratios move the same way with batch size and routing.

### Inside vLLM: decode throughput of the whole engine

`scripts/vllm_decode_throughput.py`, `nvidia/Qwen3-30B-A3B-NVFP4` on the RTX 5090 under vLLM 0.28.0, one engine per run, the Triton attention backend in both, prefix caching off. For each concurrency N the engine decodes N sequences as one batch; decode throughput is the difference between a 128-token and a 32-token budget on the same prompts (so prefill and fixed overhead cancel), each the median of 3 runs after a warm-up. Stock vLLM runs this checkpoint on its `VLLM_CUTLASS` W4A4 path; under `SM120FP4_MOE=1` all 48 routed-experts layers run on this repository's W4A16 kernels, everything else unchanged (`reports/vllm-decode-stock-20261002.json`, `reports/vllm-decode-sm120-20261002.json`, `reports/vllm-decode-compare-20261002.json`):

| concurrent sequences | stock vLLM (W4A4 CUTLASS), decode tok/s | this backend (W4A16), decode tok/s | this / stock | ms per decode step, stock | ms per decode step, this |
|---|---|---|---|---|---|
| 1 | 150 | 206 | 1.37x | 6.67 | 4.85 |
| 2 | 264 | 355 | 1.35x | 7.58 | 5.63 |
| 4 | 522 | 671 | 1.28x | 7.66 | 5.96 |
| 8 | 1097 | 1435 | 1.31x | 7.29 | 5.58 |
| 16 | 2098 | 2534 | 1.21x | 7.63 | 6.31 |

The engine decodes 1.21 to 1.37 times faster with the layer in place, most at 1 sequence and least at 16, where the layer's own table above is level with Marlin. The two paths are different numerics (W4A4 against W4A16; the greedy comparison in `docs/engine-integration-notes.md` has both at 300 of 300 retrieval items), so this is the engine-level remainder of the per-layer gain, not a like-for-like kernel race; the like-for-like one is the layer table. Prefill is differenced out: the backend runs it through 16-token slices by design, and the 32-token budget's wall time at 16 sequences (0.51 s against stock's 0.32 s) shows that cost, which backlog item 5 addresses.

Limits, measured: the FC2 kernel takes at most 16 tokens, so the layer is a decode layer and does not cover prefill.
At 16 spread tokens the layer and Marlin are within 2 us (RTX 5090) and 9 us (RTX PRO 6000) of each other, and the
remaining gap to the layer's own weight-read floor is in FC2 (`docs/stage2-design.md`). Only the first MoE layer of one
checkpoint has been run on the layer bench. Inside vLLM 0.28 (`sm120fp4/vllm_backend.py`, opt-in `SM120FP4_MOE=1`) the full model answers all 300 retrieval items with every routed-experts layer on these kernels, as stock vLLM does on its W4A4 path; the first generated token agrees with stock on 293 of 300 and whole 64-token greedy sequences on 3 of 350, the two paths being different numerics (`docs/engine-integration-notes.md`). The decode-throughput table inside the engine is above.

## Status

Stage 1 complete. Stage 2: the W4A16 decode layer above is ahead of Marlin on seven of the eight measured rows of a real NVFP4 checkpoint on both the RTX 5090 and the RTX PRO 6000, behind on the same eighth row on both (16 tokens on the same 8 experts), and is deterministic; of the stage's gate, the RTX PRO 6000 reproduction is now met, the 8-to-16-token margin over FlashInfer's path is not (the layer is level there), and the layer runs inside vLLM 0.28 end to end as an opt-in backend (`SM120FP4_MOE=1`): the full model answers 300 of 300 retrieval items on it, the same as stock, and the engine decodes 1.21 to 1.37 times faster at 1 to 16 concurrent sequences than on vLLM's own W4A4 path. The stage's remaining clause is the 8-to-16-token margin over FlashInfer's path at the layer level (`BACKLOG.md`). Stage 3 has started: `docs/stage3-survey.md` reads DeepGEMM at a pinned commit and fixes the first kernel (`sm120_fp8_fp4_gemm_1d1d`) and its conformance plan; `scripts/ue8m0_reference.py` mirrors DeepGEMM's UE8M0 scale path bit for bit, and `scripts/fp8_fp4_gemm_sm120.py` holds an SM120 FP8xFP4 GEMM with UE8M0 scales in two versions, both correct against that reference to bf16 output rounding; the split-K one reaches 633 GB/s on a 7168-wide decode shape, 2.8 times its byte floor, with the shared-memory fragment loads the next thing to fix (the `mma.sync` `kind::f8f6f4` e2m1 container convention was measured on the card: the code sits in bits 5:2 of its byte). Stage 3 (the FP4 kernels DeepGEMM does not ship for SM120) has not started. See `PLAN.md` for the stage gates.

## License

Apache-2.0.
