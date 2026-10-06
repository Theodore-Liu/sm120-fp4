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

## Install

The stage-2 MoE layer is an opt-in vLLM 0.28 plugin. Into a Python 3.12 environment that already has vLLM 0.28:

```
pip install git+https://github.com/Theodore-Liu/sm120-fp4
```

(or `pip install -e .` from a clone, for development).

The install registers the `vllm.general_plugins` entry point `sm120fp4_moe`. vLLM loads it at engine start in every process, and it
is a no-op unless the switch is set:

```
SM120FP4_MOE=1 vllm serve nvidia/Qwen3-30B-A3B-NVFP4 ...
```

With the switch set, vLLM's `modelopt_fp4` quantization config resolves to `sm120fp4.vllm_classes.SM120Fp4Config`, whose MoE
method runs this repository's W4A16 decode layer (`sm120fp4/vllm_backend.py`); without it, vLLM's own class stays in place. To
check an install without serving a model:

```
python -c "import importlib.metadata as m; print([e.name for e in m.entry_points(group='vllm.general_plugins')])"
# ... 'sm120fp4_moe' ...
SM120FP4_MOE=1 python -c "from vllm.plugins import load_general_plugins as l; l(); from vllm.model_executor.layers.quantization import get_quantization_config as g; print(g('modelopt_fp4').__module__)"
# sm120fp4.vllm_classes
```

`scripts/plugin_install_test.py` runs exactly these checks in a fresh venv with stock `vllm==0.28.0` from PyPI, once with an editable
install of the checkout and once (`--non-editable`) with a wheel build of it, the probe run from outside the checkout and compiling the
kernels from site-packages: `reports/plugin-install-test-20261003.json` and `reports/plugin-install-test-noneditable-20261003.json`
both record a pass on 2026-10-03 (entry point listed; the switch off leaves `vllm.model_executor.layers.quantization.modelopt.ModelOptNvFp4Config`;
the switch on installs `sm120fp4.vllm_classes.SM120Fp4Config`; the five kernel modules compile from `site-packages/sm120fp4/kernels/`).

Limits. The kernels compile at first use (`torch.utils.cpp_extension.load_inline`), so a CUDA toolkit with `nvcc` is needed on the serving
host and the first engine start takes a few minutes longer. The backend checks the shapes it was measured on (`sm120fp4/vllm_backend.py`:
at most 16 tokens per call, hidden size 2048, intermediate size 768 or 1024, the Qwen3-30B-A3B-NVFP4 layer) and was measured on the RTX 5090
and the RTX PRO 6000 only (the tables below).

## Layout of the repository

- `docs/scale-layouts.md`: the specification, with every formula quoted from its source and a worked example.
- `sm120fp4/layouts.py`: converters (`to_128x4`, `from_128x4`, padding helpers) and layout descriptors.
- `sm120fp4/reference.py`: the NVFP4 reference quantizer, dequantizer and reference GEMM (fp32 accumulate).
- `tests/`: conformance tests; each test states which failure class it exists to catch and cites the public report.
- `sm120fp4/kernels/`: the stage-2 W4A16 decode MoE layer (`moe_layer.py`) and the `fc1_*.py` / `fc2_*.py` / `moe_w4a16.py` kernels it
  loads, packaged since 2026-10-03 so a pip install carries them; `scripts/` keeps shims under the old names for the bench and diagnostic
  scripts, and `scripts/real_ckpt_layer.py` runs the layer on a real checkpoint beside Marlin.
- `sm120fp4/vllm_backend.py`, `sm120fp4/vllm_classes.py`: the layer as an opt-in vLLM 0.28 MoE backend (`SM120FP4_MOE=1`);
  `scripts/vllm_model_compare.py` and `scripts/vllm_decode_throughput.py` measure it inside the engine.
- `docs/stage3-survey.md`, `scripts/fp8_fp4_gemm_sm120.py`, `scripts/fp8_einsum_sm120.py`, `scripts/probe_f8f6f4*.py`: stage 3,
  the FP4 kernels DeepGEMM routes to `tcgen05` and SM120 lacks; the MQA-logits kernels (`fp8_fp4_mqa_logits_sm120.py` v0 to v3,
  `fp8_fp4_mqa_logits_v4_sm120.py`, `fp8_mqa_logits_v5_sm120.py`) and `ue8m0_reference.py` live in `sm120fp4/kernels/` since
  2026-10-04, with shims under the old `scripts/` names.
- `sm120fp4/indexer.py`: vLLM's sparse-attention indexer entry points (`fp8_fp4_mqa_logits`, `fp8_fp4_paged_mqa_logits`,
  `get_paged_mqa_logits_metadata`) served by the v5 kernel behind `SM120FP4_INDEXER=1` (adoption item 2, step 1; `docs/stage3-engine-wiring.md` 3d).
  `tests/test_indexer_adapter.py` and `tests/test_indexer_topk.py` are its tests (against the kernel's reference, through vLLM's wrappers, and under the engine's top-k kernels).
- `docs/stage3-paged-v5-row-width.md`: what the paged v5's remaining gap to the paged v4 is made of (the e4m3 row's bytes, 132 against 68) and the lever that could close part of it (the row as two 64-byte halves on the v4 path; a second lever, two heads per tile, turned out to be in place already); a design, no kernel.
- `docs/ci-plan.md`: the plan for a minimal CI on an SM120 runner (adoption item 3); a plan, not a run.
- `docs/stage3-step3-plan.md`: the plan for the two-GPU end-to-end run (host, software, the two measurements, the order on the pod); a plan, not a run.
- `docs/stage3-engine-wiring.md`: where vLLM 0.28 calls the stage-3 kernels (file and line, inputs, our kernel, the gap), what runs
  DeepSeek-V4-Flash on consumer Blackwell today, and the order of work for adoption item 2.
- `scripts/plugin_install_test.py` (`run-plugin-install-test.cmd`, `run-plugin-noneditable-test.cmd`): the clean-venv check that a pip
  install of this repository, editable or as a wheel, registers the vLLM plugin on a stock vLLM 0.28 and compiles its kernels from
  site-packages (adoption item 1); reports `reports/plugin-install-test-20261003.json` and `plugin-install-test-noneditable-20261003.json`.
- `BACKLOG.md`: the ranked work queue; `PLAN.md`: the stage gates; `reports/`: every measurement the tables above cite.

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

## Stage 3: the FP4 kernels DeepGEMM does not ship for SM120 (in progress)

DeepSeek-V4-Flash on consumer Blackwell is blocked by three DeepGEMM dispatch sites that have an SM100 kernel built on
`tcgen05` and no SM120 one (vLLM issue #41063). `docs/stage3-survey.md` reads DeepGEMM at commit `057ca5964aae` and settles
what each site needs: the FP8xFP4 GEMM with UE8M0 block scales (`sm100_fp8_fp4_gemm_1d1d`), the FP4 MQA-logits indexer
kernels, and an einsum that turns out to be FP8, not FP4, upstream. The first kernel is the GEMM, and three steps of it are
done on the RTX 5090:

- `scripts/ue8m0_reference.py`: the UE8M0 (MX) scale path - scale rule, e2m1 grid and rounding, nibble and scale packing -
  mirrored from `deep_gemm/utils/math.py` and checked bit for bit against it on six configurations.
- The `mma.sync kind::f8f6f4` operand convention, measured with one-hot probes (`scripts/probe_f8f6f4_onehot.py`): the card
  reads an e2m1 container as a six-bit field, so the code goes in bits 5:2 of its byte; the fragment layout is the PTX ISA's
  8-bit m16n8k32 one. A code left in the low nibble gives wrong, smaller numbers.
- The FP4 x FP4 forms for an MXFP4 q (`scripts/probe_f4f4.py`, `reports/probe-f4f4-rtx5090-20261004.json`): `kind::f8f6f4` with e2m1
  on both operands (bits 5:2 on both), and the packed `kind::mxf4.block_scale.scale_vec::2X` at k = 64 with the lane and byte each
  UE8M0 scale is read from; both exact on the RTX 5090. A probe that only uses unit scales cannot see a common permutation of k,
  so the scale arm is what pins the block layout.
- `sm120fp4/kernels/fp4_fp4_mqa_logits_v6_sm120.py` — the indexer kernel with MXFP4 on both sides (q [S, H, 64] packed e2m1 with UE8M0
  [S, H, 4], k as v4 takes it), two `kind::mxf4.block_scale` k64 MMAs per tile with the scales applied by the instruction; correct
  against the dequantised reference on five shapes (at most 7.9e-8), with a fault arm the selftest must catch; 6.9 / 17.2 / 19.2 us on three
  indexer shapes, 1.29 to 1.65 times v4 on the same k codes (`reports/fp4-fp4-mqa-logits-v6-rtx5090-20261005.json`); its paged form
  (`sf_cache` [pages, 64, 4], rows staged at an 80-byte stride) is bit-identical to it through random page permutations and 1.18 to 1.23
  times the paged v4 (`reports/fp4-fp4-paged-mqa-logits-v6-rtx5090-20261005.json`); the same stride on the flat kernel (v6s) removes its
  bank conflicts (1.57 million to 481) without changing its time, so the flat kernel is not shared-load bound; its stall counters spread
  over shuffles, fixed-latency waits and the shared-memory queue, which points at the per-tile epilogue; the packed epilogue (v6e, one butterfly
  for both column values) is bit-identical and runs 6.9 / 15.1 / 17.2 us, 1.11 to 1.14 times v6 on the two larger shapes, and 1.05 times in the
  paged form (11.0 / 43.6 / 41.6 us), where half of each issue cycle is spent waiting for the warp's own page; walking several pages per
  warp with a prefetch (v6f) is bit-identical but slower or level at every group size: the prefetch works (page wait 5.91 to 1.69 cycles per
  issue) but the second buffer halves occupancy (16.7 against 33.3 percent); half-page double buffers at v6e's occupancy (v6g) are
  bit-identical and slower on the two larger shapes (the per-half reloads add 19 percent instructions and 48 percent global load requests),
  and v6h, loading them once per head block while keeping the overlap, is bit-identical and 1.04 times v6e on one of three shapes in three runs, level
  on the others, and its remaining limit is occupancy set by 5.4 KB of shared memory per warp; quarter-page double buffers (v6i, 2.7 KB per warp,
  four blocks per SM) are bit-identical and 1.19 to 1.24 times v6e over three runs, 8.9 / 35.6 / 33.5 us against the paged v4's 13.2 / 54.0 / 54.0,
  the paged form to use; `--selftest`, `--bench`,
  `--bench-paged`, `--ncu-shape`.
- `scripts/fp8_fp4_gemm_sm120.py`: the GEMM in three versions, every one correct against the reference to bf16 output
- `scripts/fp8_fp4_mqa_logits_sm120.py` — the FP8 x FP4 MQA-logits (indexer) kernel for SM120: v0 (one warp per query row), v1 (sixteen query rows x a 256-row kv segment per block, kv staged in shared memory) v2 (a tile rule, segment groups, a double-buffer arm) and v3 (the paged form: block tables and context lengths), all bit-identical and correct against DeepGEMM's test reference; `--selftest`, `--bench`.
- `scripts/fp8_fp4_mqa_logits_v4_sm120.py` — v4 of the indexer kernel in the engine's k format (one UE8M0 scale per 32 columns, the MXFP4 layout vLLM's indexer passes), compiled with v0 to v3 so it is checked bit for bit against v2 where the scales coincide, and its paged form (`sf_cache` [pages, 64, 4]) checked bit for bit against it through random page permutations; `--selftest`, `--bench`, `--bench-paged`.
- `scripts/fp8_mqa_logits_v5_sm120.py` — v5, the FP8-k form of the indexer kernel (e4m3 k rows with an fp32 scale per row, q's scale in `weights`), the path vLLM runs on SM120 since it refuses the MXFP4 indexer cache there, flat and paged (the paged form reads vLLM's own 132-byte-entry cache layout and is bit-identical to the flat form through page permutations); correct against the dequantised reference; `--selftest`, `--bench`, `--bench-paged`.
- The paged v5 (vLLM's 132-byte-row fp8 indexer cache) runs 33.6 / 115.5 / 123.7 us on the three decode shapes after its staging moved
  to lane-strided 16-byte loads, against the paged v4's 23.3 / 72.4 / 74.5 (`reports/fp8-paged-mqa-logits-v5-stage16-rtx5090-20261004.json`,
  `reports/fp8-fp4-paged-mqa-logits-v4-rtx5090-20261003.json`); Nsight Compute puts its occupancy at one block per SM, capped by the
  67.6 KB of shared memory the eight staged pages take (`docs/stage3-engine-wiring.md` 3g).
- The half-page staging (`fp8_paged_mqa_logits_sm120_v5h`, two blocks per SM) runs 33.5 / 103.4 / 113.4 us against the full page's
  33.6 / 115.5 / 124.9 on the same shapes back to back (`reports/fp8-paged-mqa-logits-v5h-rtx5090-20261004.json`; wiring doc 3h).
- The register-prefetch arm of the half-page kernel (v5d) is slower, 39.8 / 136.0 / 146.2 us: 200 registers per thread put the occupancy
  back to one block per SM (`reports/ncu-paged-v5d-v5h-rtx5090-20261004.txt`; wiring doc 3i).
- Staging the half-pages as raw 132-byte rows (v5r: straight 16-byte copies, a 33-word read stride, no scatter) runs 21.2 / 62.4 / 72.5 us against
  the half-page kernel's 33.2 / 103.4 / 113.4 back to back, ahead of the paged MXFP4 kernel's 23.3 / 72.4 / 74.5; the adapter's paged call uses it
  (`reports/fp8-paged-mqa-logits-v5r-rtx5090-20261004.json`; wiring doc 3j).
- `scripts/fp8_einsum_sm120.py` — the FP8 einsum `bhr,hdr->bhd` for SM120 (DeepGEMM's `fp8_einsum` recipe: per-token x, per-block y): v0 (one warp per 16 x 8 tile) and v1 (y and x tiles staged in shared memory, 8 warps per 128 x 128 tile), both correct against torch.einsum on the dequantised operands and bit-identical to each other; `--selftest`, `--bench`.
  rounding - v0 (one warp per 16 x 8 tile, the convention check), v1 (32 x 128 tiles, 4-stage `cp.async`, bit-identical to
  v0) and v2 (split-K with a fixed-order reduce, bit-identical to v1 except one element one ulp off). Cold-L2 medians of 20
  launches on DeepSeek-V4-style decode shapes (`reports/fp8-fp4-gemm-v2-rtx5090-20261002.json`):

| M | N | K | v1 (one block per 128 columns), us | v2 (split-K), us | v2 GB/s | byte floor at 1792 GB/s, us | v2 / floor |
|---|---|---|---|---|---|---|---|
| 16 | 2048 | 7168 | 66.6 | 19.1 (128-wide tiles, 22 splits) | 401 | 4.3 | 4.5x |
| 16 | 7168 | 7168 | 70.4 | 41.8 (128-wide tiles, 7 splits) | 633 | 14.8 | 2.8x |
| 32 | 2048 | 7168 | 68.4 | 19.2 (128-wide tiles, 22 splits) | 406 | 4.4 | 4.4x |
| 32 | 7168 | 7168 | 70.7 | 43.8 (128-wide tiles, 7 splits) | 612 | 14.9 | 2.9x |
| 16 | 4096 | 2048 | 23.3 | 13.1 (64-wide tiles, 6 splits) | 338 | 2.5 | 5.3x |

v2 is 1.7 to 3.5 times v1 and 2.8 to 5.3 times the byte floor. Three single-change arms on top of it, each bit-identical to v2
(`reports/fp8-fp4-gemm-v3-rtx5090-20261002.json`): swizzling the A rows in shared memory adds 10 to 17 percent on the K = 7168
shapes, a K permutation that widens the fragment loads adds the same and not more (the same bank conflict), one barrier per
stage adds nothing. Two further arms (`reports/fp8-fp4-gemm-v4-rtx5090-20261002.json`): two pipeline stages instead of four,
which lets four blocks share an SM, add 30 to 34 percent on the 7168-wide shapes; pairs of 128-K blocks per barrier add 21 to
24 and not on top of that. The best configuration reads 852 GB/s on M16 N7168 K7168, 2.1 times the byte floor;
the 2048-wide shapes sit at the cost of two launches and a 22-way reduce. Decomposed (`reports/fp8-fp4-gemm-v5-rtx5090-20261003.json`),
the reduce kernel alone is 41 to 47 percent of those shapes' time; halving the splits recovers 9 to 12.5 percent where a block per SM
remains, and a fused last-block reduce, bit-identical to the separate one, is 29 to 42 percent slower because it is latency-bound on
a few blocks, so it stays an arm rather than the default; with float4 loads it is still 19 to 34 percent slower
(`reports/fp8-fp4-gemm-v6-rtx5090-20261003.json`), which closes that pattern for decode shapes. The planner asked for one block per
SM instead of two is the two-stage default since v7 (`reports/fp8-fp4-gemm-v7-rtx5090-20261003.json`): +2.2 to +13.3 percent on
four of the five decode shapes and -1.6 on one, 454 to 504 GB/s on the 2048-wide shapes and 805 to 897 on the 7168-wide. The
MQA-logits kernel's v0 (`reports/fp8-fp4-mqa-logits-v0-rtx5090-20261003.json`) computes DeepGEMM's indexer logits (ReLU of the
per-head FP8 x FP4 scores, weighted and summed over heads, per-row kv spans) to a relative error of 1.5e-7 against the test
reference on six shapes; v1 stages a 256-row kv segment in shared memory for sixteen query rows at a time (its first report's
timings included two host syncs and are superseded by the v2 report, which re-times it at 25 to 32 us); v2
(`reports/fp8-fp4-mqa-logits-v2-rtx5090-20261003.json`) picks 64-row segments and 8 or 16 rows per block and runs 7.6 to 29.3 us
on the four shapes, 18 to 147 TFLOP/s, bit-identical to v0; v3 (`reports/fp8-fp4-paged-mqa-logits-v3-rtx5090-20261003.json`) is the
paged form the decode path needs, bit-identical to v0 through random page permutations, reading 0.7 to 1.1 TB/s of kv rows on three
decode shapes. Both indexer forms DeepGEMM ships for SM100 now exist for SM120. The einsum site's `bhr,hdr->bhd` has its v0
(`reports/fp8-einsum-v0-rtx5090-20261003.json`: within two bf16 half-ulps of the reference on six shapes, 709 GB/s at B 8 and
156 at B 128, the re-read of y being its cost), so every kernel family DeepGEMM routes to tcgen05 has a correct SM120 form
here. The tiled v1 (`reports/fp8-einsum-v1-rtx5090-20261003.json`) stages y and x in shared memory and is bit-identical to v0: 1.33
times faster at B 128 (192.5 against 256.0 us) and 7 to 9 percent slower at B 8 and 32, where its 64-block grid is the bound. The
einsum site, being FP8, stays last; the adoption items below come before its next version.

## Status

<!-- ci-status -->CI 2026-10-05: passed on 65ba8ae, 67 passed, 0 failed, 15 skipped in 12 s on NVIDIA GeForce RTX 5090 (driver 610.47, torch 2.13.0+cu130).

Stage 1 complete. Stage 2: the W4A16 decode layer is ahead of Marlin on seven of the eight measured rows of a real NVFP4
checkpoint on both the RTX 5090 and the RTX PRO 6000, behind on the same eighth row on both (16 tokens on the same 8
experts), and is deterministic; it runs inside vLLM 0.28 end to end as an opt-in backend (`SM120FP4_MOE=1`), where the
full model answers 300 of 300 retrieval items as stock does and the engine decodes 1.21 to 1.37 times faster at 1 to 16
concurrent sequences than on vLLM's own W4A4 path. Of the stage's gate, the RTX PRO 6000 reproduction and the engine
integration are met; the 8-to-16-token margin over FlashInfer's path at the layer level is not (the layer is level there)
and stays in `BACKLOG.md`. Stage 3 is in progress: the survey, the UE8M0 reference, the measured operand convention and
the first kernel at 2.1 times its byte floor on the widest shape, with the one-block planner as its default and the split-K reduce as the 2048-wide shapes' remaining cost; the MQA-logits kernel is correct against DeepGEMM's reference in both its forms (flat, 18 to 147 TFLOP/s; paged, 0.7 to 1.1 TB/s of kv) and, since v4, in the k format vLLM's indexer passes (one UE8M0 scale per 32 columns; bit-identical to v2 where the scales coincide, no slower), in both the flat and the paged form, and in the FP8-k form vLLM actually runs on SM120 (v5, since the engine refuses the MXFP4 indexer cache there), and the FP8 einsum has a correct v0 and a tiled v1 (faster only at B 128), as the section above states. See `PLAN.md` for the stage gates. Of the three adoption items (`BACKLOG.md`), the first is done: the stage-2 backend installs as a vLLM plugin from a wheel, with the kernels packaged, on a stock vLLM 0.28 (the Install section above). Next, in order: one stage-3 kernel wired into an engine (`docs/stage3-engine-wiring.md` names the five call sites in vLLM 0.28 and the order: the sparse-attention indexer first, unit-wired on the RTX 5090, then end to end on two RTX PRO 6000 with DeepSeek-V4-Flash NVFP4), and a minimal CI on an SM120 runner. The upstream issue or PR waits until the project is essentially complete. Adoption item 2 (one stage-3 kernel inside an engine) is at step 1: `sm120fp4/indexer.py` serves vLLM 0.28's three indexer entry points with the FP8-k kernel (v5, flat and paged, fp32 weights) behind `SM120FP4_INDEXER=1`, tested through the engine's own wrappers and its own top-k kernels (identical index sets to the reference); its paged form runs at 1.44 to 1.66 times the paged MXFP4 kernel's time after the staging moved to 16-byte loads; the adapter takes the V3.2 backend's 64-row blocks and V4's 256-row blocks (four pages each); the two-GPU DeepSeek-V4-Flash run remains.

## License

Apache-2.0.
