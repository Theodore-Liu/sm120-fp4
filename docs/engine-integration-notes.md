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
  (`quantization/__init__.py:58`) registers a `QuantizationConfig` subclass under a new method name, and an installed
  package can run that registration at engine start through the `vllm.general_plugins` entry-point group
  (`plugins/__init__.py:18`). A subclass of `ModelOptNvFp4Config` can set `FusedMoEMethodCls` to our method and
  override `override_quantization_method` to claim NVFP4 checkpoints when the device is SM120 and an opt-in is set.

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

## 5. Steps

1. Package: `sm120fp4.vllm` module with the `QuantizationConfig` subclass and the `FusedMoEMethodBase` subclass; an
   entry point `vllm.general_plugins = sm120fp4_moe = sm120fp4.vllm:register`; opt-in by `SM120FP4_MOE=1`.
2. Weights: keep the checkpoint layout, rotate `w13` to `[up; gate]` once in `process_weights_after_loading`, keep the
   e4m3 block scales and the two global scales as tensors our kernels read.
3. `apply`: slice to 16 tokens, call the routing, FC1 and FC2 kernels through `moe_layer.py`'s `choice` rules, write
   into the output tensor the engine expects (`[tokens, hidden]`, bf16).
4. Tests: a unit test that loads layer 0's weights through vLLM's loader and compares the method's output with
   `real_ckpt_layer.py`'s layer on the same inputs; then the model-level greedy comparison.
5. Measure, document, and move the gate clause.
