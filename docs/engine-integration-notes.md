# Engine integration notes: the W4A16 decode layer as an optional MoE backend in vLLM

Backlog item 2. These notes record what vLLM 0.28.0 (the version installed here, torch 2.13 cu130) exposes, what our
kernels need, and the design chosen before any code is written. Line references are to the installed package.

## 1. Where vLLM decides how an NVFP4 MoE layer runs

- A ModelOpt NVFP4 checkpoint (`hf_quant_config` algo `NVFP4`, or `W4A16_NVFP4` for weight-only) is claimed by
  `ModelOptNvFp4Config.override_quantization_method` (`quantization/modelopt.py:1045`), which returns the method name
  `modelopt_fp4`; the MoE layers then get `ModelOptNvFp4FusedMoE` (`modelopt.py:1371`), a `FusedMoEMethodBase`.
- Its constructor calls `select_nvfp4_moe_backend(config, weight_key=kNvfp4Static, activation_key=kNvfp4Dynamic or
  None)` (`fused_moe/oracle/nvfp4.py:166`), which walks an ordered list - FLASHINFER_TRTLLM, FLASHINFER_CUTEDSL,
  FLASHINFER_CUTEDSL_BATCHED, FLASHINFER_CUTLASS, VLLM_CUTLASS, MARLIN, HUMMING, EMULATION - and takes the first whose
  `FusedMoEExperts` class says it supports the device, the quant scheme, the activation and the parallel config.
  FLASHINFER_B12X exists in the enum but is excluded from auto-selection and only reachable with
  `moe_backend="flashinfer_b12x"`. For a W4A16 checkpoint `activation_key` is `None`, every W4A4 backend rejects
  itself, and Marlin is the one that survives (the comment at `modelopt.py:1386` says exactly this).
- The method's `apply(layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input)` (`modelopt.py:1626`)
  forwards to `self.moe_kernel.apply(x, layer.w13_weight, layer.w2_weight, topk_weights, topk_ids, activation=...,
  global_num_experts=..., expert_map=..., apply_router_weight_on_input=..., shared_experts=...)`. Routing (top-k
  weights and ids) is done by the engine before `apply`; the method only runs the experts and the weighted sum.
- `process_weights_after_loading` (`modelopt.py:~1524`) is where Marlin repacks `w13_weight`/`w2_weight` and their
  scales into Marlin's format and checks that the two halves of `w13_weight_scale_2` agree (the same check our
  `real_ckpt_layer.py` makes: 128 of 128 experts on Qwen3-30B-A3B-NVFP4). After this step the raw checkpoint layout
  is gone.
- A backend can be added without forking vLLM: `register_quantization_config(name)`
  (`quantization/__init__.py:58`) registers a `QuantizationConfig` subclass, and an installed package can run that
  registration at engine start through the `vllm.general_plugins` entry-point group (`plugins/__init__.py:18`).
  **A new name cannot claim the checkpoint, though** (found when writing the code): `_verify_quantization`
  (`config/model.py:1210`) walks a fixed list of override names, resolves a ModelOpt checkpoint to `modelopt_fp4`,
  and raises if `--quantization` names anything else. What works is re-registering `modelopt_fp4` itself:
  `register_quantization_config` accepts an existing name (it logs that the entry is overwritten) and
  `get_quantization_config` consults the customised registry first, so under the opt-in the subclass takes the
  name and vLLM's own override logic hands it the checkpoint. `sm120fp4/vllm_backend.py` does exactly this, and
  dispatches per layer inside `get_quant_method`: a routed-experts layer the kernels fit gets our method, every
  other layer the parent's, so vLLM's path remains the fallback layer by layer, not model by model.

## 2. What our kernels need, and where it differs from vLLM

- Weights in the checkpoint's own layout: FP4 codes `[E, 2I, H/2]` (uint8, two codes per byte) and `[E, H, I/2]`,
  block scales `[E, 2I, H/16]` and `[E, H, I/16]` as e4m3, one `weight_scale_2` per expert for FC1 and FC2. This is
  what `real_ckpt_layer.py` reads from `model-00001-of-00004.safetensors` and hands to `moe_layer.py` directly, with no
  repack. vLLM's `w13_weight` concatenates `[gate; up]`; our kernels take `[up; gate]` (the FlashInfer order), so the
  integration either rotates the two halves once at load (`gate_first` in `real_ckpt_layer.py` does the inverse for
  Marlin) or the FC1 kernel takes a flag. One rotation at load, once, is the cheaper choice.
- The two `weight_scale_2` halves of FC1 must be equal (our FC1 applies one alpha); vLLM already asserts this.
- The router output we need is per-token top-k `ids` and `weights` (the engine's `topk_ids`, `topk_weights`); our
  routing kernel (`mr.route`) consumes those and builds the expert-sorted pair buffers on device with no host sync.
- `activation` must be SwiGLU (`silu_and_mul`), `expert_map` None (no expert parallelism in v0),
  `apply_router_weight_on_input` False, `shared_experts` None (Qwen3-30B-A3B has none). Each is asserted at load and
  the method refuses the layer otherwise, so vLLM falls back to its own path.

## 3. The 16-token cap, and why the fallback must not keep a second copy of the weights

- The FC2 kernel takes at most 16 tokens per call (`MAXM = 16`). Decode batches of up to 16 sequences fit; prefill
  and larger batches do not.
- Qwen3-30B-A3B's routed experts are 48 layers x 128 experts x (2 x 768 x 2048 + 2048 x 768) parameters = 290 M
  parameters per layer, 14.5 GB at four bits for the model. Keeping a Marlin-repacked copy beside the raw codes for a
  fallback would double that, which does not fit a 32 GB RTX 5090 beside the activations and the KV cache. So the
  fallback cannot be "Marlin for the rest".
- v0 fallback: run the layer in slices of at most 16 tokens over the same resident weights. Correct by construction
  (each slice is a full MoE forward over its tokens), slow for prefill (the weights are re-read once per slice; a
  2,048-token prompt is 128 slices), stated in the README as a decode backend. Backlog item 5 (a prefill kernel or a
  documented hand-off) replaces this.
  Measured 2026-10-07 (`scripts/real_ckpt_layer.py --prefill`): over three runs (reports/real-ckpt-layer0-prefill-rtx5090-20261007-run1..3.json) the backend's sliced path costs 298.8 to 299.7 / 599.8 to 601.3 / 1154.8 to 1156.6 us at 32 / 64 / 128 randomly routed tokens against Marlin's 217.9 to 218.9 / 242.4 to 242.7 / 254.9 to 255.7 on the whole batch, 1.37 / 2.47 / 4.53 times, bit-identical over 50 calls and at the decode rows' error (2.0e-3 against Marlin's 3.7e-3); each slice re-reads every routed expert's weights, so the cost grows with the slice count while Marlin's barely moves.
- Alternative kept open: hand prefill to FlashInfer's CUTLASS W4A4 path, which reads the same codes but needs the
  128x4-swizzled scale layout (`sm120fp4.layouts.to_128x4`); that is a second copy of the scales only (1/16 of the
  codes), not of the codes, and is the direction if slicing proves too slow for first-token latency.

## 4. Validation plan (before any throughput number is reported)

- Correctness, layer level: the existing table - `real_ckpt_layer.py` against Marlin on the same codes and scales,
  bit-identical over 50 calls - already covers the kernels; the integration adds one check that the engine's
  `topk_ids`/`topk_weights` reach our routing kernel unchanged (compare the pair buffers with a torch routing of the
  same logits on one batch).
- Correctness, model level: greedy decoding of `nvidia/Qwen3-30B-A3B-NVFP4` under vLLM with the stock Marlin backend
  and with ours, over a fixed prompt set (the 300 retrieval items this repository's layer benches already use the
  shape of, plus 50 free-form prompts), 64 new tokens each; report the fraction of prompts with identical token
  sequences and, where they differ, the first differing position. The two backends compute the same function in
  different rounding order, so identity is not required; the report states the agreement rate the way the layer
  table states the normwise error.
- Throughput: vLLM's own benchmark (`vllm bench serve` or `benchmark_latency`) at 1, 2, 4, 8 and 16 concurrent
  decode streams, stock backend and ours, same model, same GPU, same session, median of repeats, with the layer-only
  table beside it so a reader can see how much of the per-layer gain survives the engine.

**Model-level result (2026-10-02, RTX 5090, vLLM 0.28.0, `reports/vllm-compare-20261002.json`).** The full `nvidia/Qwen3-30B-A3B-NVFP4` model, greedy, 64 new tokens, 350 prompts, the Triton attention backend in both runs so the MoE layers are the only difference. Stock vLLM runs the checkpoint on its `VLLM_CUTLASS` W4A4 path (activations quantized to FP4 per call); under `SM120FP4_MOE=1` every one of the 48 routed-experts layers runs on this repository's W4A16 kernels (bf16 activations). Both answer all 300 retrieval items (300 and 300 of 300; 0 items flip), and the first generated token, which on a retrieval item is the answer, agrees on 293 of the 300. Whole 64-token sequences are identical on 3 of 350 (3 retrieval, 0 free-form): the two paths compute different functions (W4A4 against W4A16), so after the answer the greedy continuations part - first differing position median 5, 90th percentile 20, 12 of 347 at the first token - the way two correct implementations of different numerics do, not the way a wrong one does (the layer-level test is bit-identical to Marlin W4A16 on the same codes). Generation of the 350 prompts took 19 s stock and 23 s on the plugin, prefill included; that number is not a throughput measurement (prefill runs through 16-token slices by design) and the decode-throughput table is the next item.

**Throughput result (2026-10-02, `reports/vllm-decode-compare-20261002.json`).** `scripts/vllm_decode_throughput.py`, same model, GPU, engine version and attention backend as the greedy comparison, one engine per run, prefix caching off, N in 1, 2, 4, 8, 16, decode throughput by differencing a 128-token and a 32-token budget (median of 3 after warm-up). Stock 150, 264, 522, 1097, 2098 decode tokens/s; this backend 206, 355, 671, 1435, 2534; ratios 1.37, 1.35, 1.28, 1.31, 1.21. The README's engine table carries the rows. The ratio falls with N as the layer table's margin over Marlin does, and at 16 the 32-token budget's wall time is 1.61 times stock's, the sliced prefill (section 3) that item 5 addresses.

## 5. Steps

1. Done (2026-10-02): `sm120fp4/vllm_backend.py` holds `SM120Fp4Config` (a `ModelOptNvFp4Config` subclass) and
   `SM120Fp4MoEMethod` (a `ModelOptNvFp4FusedMoE` subclass); the entry point
   `vllm.general_plugins: sm120fp4_moe = sm120fp4.vllm_backend:register` is in `pyproject.toml`; opt-in by
   `SM120FP4_MOE=1`, under which `register()` re-registers `modelopt_fp4` (section 1).
2. Done: `weights_from_vllm_layout` keeps the checkpoint layout, rotates `w13` and its block scales to `[up; gate]`
   once, refuses a layer whose gate and up global scales differ on any expert, and keeps the e4m3 block scales and
   the per-expert global scales as the kernels read them; `process_weights_after_loading` re-registers the
   parameters in that layout and drops the activation scales.
3. Done: `Weights.forward` slices to 16 tokens and runs route, FC1 and FC2 under `moe_layer.py`'s `choice` and
   `use_pdl` rules into a bf16 `[tokens, hidden]` output; `apply` refuses shared experts, an expert map and
   `apply_router_weight_on_input`.
4. Layer-level test done: `tests/test_vllm_backend.py` (13 tests, in the vLLM 0.28 venv on the RTX 5090) checks
   the rotation byte for byte against the loader's stacking, the forward bit-identical to the direct kernel calls
   at 1, 2, 4, 8 and 16 tokens on both routings with error under 1% against the fp32 reference, a 40-token batch
   equal to its three slices, and that the opt-in re-registration resolves `modelopt_fp4` to the subclass. The
   test builds the vLLM-layout tensors from the checkpoint shard itself rather than through vLLM's loader; the
   loader path is exercised by the model-level run.
5. Done at the correctness level (the paragraph above section 5): the model-level greedy comparison of section 4.
   Two defects the run found and the code now states: a class defined inside a function cannot be pickled into the
   engine-core process (the classes moved to `sm120fp4/vllm_classes.py`), and `RoutedExperts.weight_loader` picks
   its ModelOpt loading branch by the method's class name (`"ModelOpt" in quant_method_name`), so the method is
   named `ModelOptNvFp4FusedMoESM120`. `scripts/vllm_model_compare.py` runs the offline
   `LLM` API greedily over a fixed, seeded prompt set (300 retrieval items of the layer benches' shape plus 50
   free-form prompts, 64 new tokens, batches of 16), once per backend, each run to its own JSON (it refuses to
   overwrite), and `compare` reports the identical fraction, the first differing position per prompt and the
   retrieval items whose correctness flips. The package is installed into the vLLM 0.28 venv (`uv pip install -e`),
   so its `vllm.general_plugins` entry point is visible there (2026-10-02). What stands between the script and its
   first report was the other three checkpoint shards (18.1 GB in all), now in the WSL HF cache.
6. Done: the throughput table (the paragraph above section 5). Backlog item 2 closes with the install recipe in
   `README.md` (opt-in `SM120FP4_MOE=1`, the package installed into the vLLM venv) and the two measured tables. A run under the plugin is slow at prefill by design (section 3), so the comparison is a
   correctness measurement and the throughput table is decode-only.
