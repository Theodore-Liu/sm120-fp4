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
SM120's `cvt.rn.f16x2.e2m1x2`, kernels chained with programmatic dependent launch), each GEMM chosen by batch size
(`scripts/moe_layer.py`). Validated on the 128 experts of layer 0 of `nvidia/Qwen3-30B-A3B-NVFP4` (ModelOpt NVFP4),
against vLLM's Marlin W4A16 MoE on the same codes, scales and global scales, timed in one session by CUDA-graph replay
with L2 flushed (`scripts/real_ckpt_layer.py`, `reports/real-ckpt-layer0-rtx5090-2026-09-30.json`):

| tokens | FC1 | FC2 | this layer (us) | Marlin (us) | Marlin / this | error | Marlin error |
|---|---|---|---|---|---|---|---|
| 1 | CUDA cores | CUDA cores | 27.4 | 37.5 | 1.37x | 0.17% | 0.30% |
| 2 | CUDA cores | tensor cores, prefetch | 39.7 | 49.9 | 1.26x | 0.21% | 0.38% |
| 4 | CUDA cores | tensor cores, prefetch | 70.4 | 78.8 | 1.12x | 0.22% | 0.37% |
| 8 | CUDA cores | tensor cores, prefetch, 2 blocks per tile | 107.3 | 113.4 | 1.06x | 0.21% | 0.37% |
| 16 | tensor cores | tensor cores, prefetch, 2 blocks per tile | 170.8 | 166.7 | 0.98x | 0.20% | 0.36% |

- Error is normwise against an fp32 MoE on the dequantized weights; the layer's output is bit-identical over 50 calls
  at every size.
- The checkpoint layout is checked against an independent source first: every projection dequantized with this
  library's convention matches the bf16 original (`Qwen/Qwen3-30B-A3B`) to 0.094 to 0.095 relative error, the size
  of FP4 rounding; with the nibbles swapped it is 1.414.

Limits, measured: at 16 randomly routed tokens the layer is 2.4%
slower than Marlin, and when 16 tokens concentrate on 8 experts it is slower too (43.7 against 42.0 us, measured on
synthetic weights of the same shape, `docs/stage2-design.md`). Measured on one RTX 5090 only; not yet integrated into a serving engine. The design, every
experiment and the hypotheses ruled out are in `docs/stage2-design.md`.

## Status

Stage 1 complete; stage 2 in progress. See `PLAN.md` for the stage gates.

## License

Apache-2.0.
