# Stage 3 into an engine: where vLLM calls the kernels this repository has, and what a PRO 6000 can run (read 2026-10-03)

Adoption item 2 asks for one stage-3 kernel wired into an engine on a model an SM120 card can hold, measured end to end. Before
any wiring, this note reads the engine as installed here (vLLM 0.28.0, `~/mlsys-5090-runtime/vllm028`, with its vendored
`vllm.third_party.deep_gemm`) and the public record of running DeepSeek-V4-Flash on consumer Blackwell, and names the call
sites, the model and the first step.

## 1. The call sites in vLLM 0.28.0

| site | file : line | what it calls | when | our kernel | the gap |
|---|---|---|---|---|---|
| sparse-attention indexer, prefill | `vllm/model_executor/layers/sparse_attn_indexer.py:500` | `fp8_fp4_mqa_logits((q, q_scale), kv, weights, start, end, ...)` per chunk, when `chunk.local_total_seq_lens > 0` and not XPU | every DeepSeek-V3.2 / V4 prefill chunk (the indexer picks the top-k keys) | `scripts/fp8_fp4_mqa_logits_sm120.py` v2 (flat, 18 to 147 TFLOP/s) | q is FP8 e4m3 or MXFP4 per token with a scale (`fused_indexer_q.py:289`, `use_fp4` selects); the paper's kv is FP8 or MXFP4 per token; our v2 takes FP8 q and FP4 kv with UE8M0 scales - the q-side dtype switch and the `(data, scale)` pair calling convention are the adaptation |
| sparse-attention indexer, decode | `sparse_attn_indexer.py:603` | `fp8_fp4_paged_mqa_logits((q, q_scale), kv_cache, weights, seq_lens (2-D), block_table, schedule_metadata, max_model_len, clean_logits=False, indices=...)` | every decode step of the same models | `fp8_fp4_mqa_logits_sm120.py` v3 (paged, 0.7 to 1.1 TB/s of kv) | `schedule_metadata` comes from `get_paged_mqa_logits_metadata` (`vllm/v1/attention/backends/mla/indexer.py:1030`), DeepGEMM's own scheduler; our v3 schedules one page per warp and would ignore it or need its own metadata; `clean_logits` and `indices` (the top-k output buffer) are convention |
| MoE experts, FP4 weights | `vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py:596, 616` (`DeepGemmFP4Experts`) | `m_grouped_fp8_fp4_gemm_nt_contiguous` for both GEMMs of the expert layer | `mxfp4_w4a8` quantization with DeepGEMM selected; the class docstring says "Requires Blackwell-family GPUs (SM100 datacenter or SM120 consumer)" | `scripts/fp8_fp4_gemm_sm120.py` v7 (dense, M <= 32) | grouped (one GEMM per expert over contiguous token groups), FP8 activations with per-128 block scales and MXFP4 weights with UE8M0 scales per 32: our dense GEMM needs the grouped driver (the stage-2 layer's routing kernel is the pattern) |
| DeepSeek-V4 MegaMoE | `vllm/models/deepseek_v4/nvidia/model.py:513` | `deep_gemm.fp8_fp4_mega_moe` | only when `_check_runtime_supported` passes, which raises `NotImplementedError("DeepGEMM MegaMoE requires SM100 GPUs.")` on any capability other than 10 (`model.py:315-318`) | none (a fused routing + grouped GEMM kernel) | not reachable on SM120 by construction; on SM120 the NVFP4 checkpoint's experts run on the default MoE backend (FlashInfer CUTLASS for NVFP4, or Marlin W4A16), per the public runs below |
| DeepSeek-V4 `o_proj` | `vllm/models/deepseek_v4/nvidia/ops/o_proj.py:13, 28` | `deep_gemm_fp8_o_proj` with a recipe from `compute_fp8_einsum_recipe()` read off the device capability | the V4 attention output projection as an FP8 einsum | `scripts/fp8_einsum_sm120.py` v0/v1 (`bhr,hdr->bhd`) | the recipe (block shape, UE8M0 or not) is chosen per capability at line 21; which expression the o_proj uses and its shapes are to be read next |

How the symbols reach these sites: `vllm/utils/deep_gemm.py` imports an external `deep_gemm` if installed, else the vendored
`vllm.third_party.deep_gemm` (`_import_deep_gemm`, lines 171-205), then binds `fp8_fp4_mqa_logits`, `fp8_fp4_paged_mqa_logits`
and `get_paged_mqa_logits_metadata` by `getattr` in `_lazy_init` (lines 279-285), each falling back to a `_missing` stub that
raises when called. Whether any of it runs is gated by `is_deep_gemm_supported()` = `VLLM_USE_DEEP_GEMM and has_deep_gemm() and
current_platform.support_deep_gemm()`; `should_auto_disable_deep_gemm` already knows capability family 120 (lines 42-43).
Attention on SM120 for V4 is FlashInfer's sparse MLA (`model.py:770-803` selects `DeepseekV4FlashInferSM120Attention` when the
capability major is 12; `flashinfer_sparse.py:130-160` requires `has_flashinfer_sparse_mla_sm120`), so the attention kernel itself
is not a DeepGEMM gap; the indexer that feeds it is.

## 2. The public record on consumer Blackwell

- vLLM issue #41063 (the tracking issue this stage started from) lists DeepGEMM's SM 12.x gaps as `csrc/apis/gemm.hpp:99`
  (`fp8_fp4_gemm_nt` routes to the SM100 kernel and fails on `tcgen05.fence`), `csrc/apis/attention.hpp:67, 177, 367` (the FP4
  attention / indexer kernels) and `csrc/apis/einsum.hpp:55` (an `sm120_fp4_einsum`; the existing `sm120_fp8_einsum` is FP8), plus
  three dispatch fixes (`csrc/utils/layout.hpp:76`, `csrc/apis/layout.hpp:48, 56, 106, 110`, the compiler include path) and vLLM-side
  gates (`support_deep_gemm` accepting family 120, `DeepGemmFP4Experts._supports_current_device`). The reporter runs
  DeepSeek-V4-Flash on two GB10 (SM121). These are exactly the three sites of `docs/stage3-survey.md`; our GEMM, both
  MQA-logits forms and the FP8 einsum are the SM120 kernels the issue says are missing (the einsum site is FP8 upstream, as the
  survey found).
- vLLM PR #41834 (SM12x support for DeepSeek V4 Flash, open at the time of reading) makes the FlashInfer sparse-MLA paths the
  SM120 default and notes that "the FP4 indexer cache path still depends on DeepGEMM kernels", failing at runtime on SM120
  without them; the NVFP4 experts run on the FlashInfer CUTLASS backend.
- A published two-GPU recipe (Infatoshi/dsv4-flash-2x-rtxpro6000s) serves DeepSeek-V4-Flash-0731-NVFP4 on two RTX PRO 6000 at
  TP=2 (about 87.8 GB of weights per GPU) on a nightly vLLM with 13 SM120 patches, Marlin W4A16 for the experts and a custom
  Triton kernel for the sparse-attention indexer in place of the fallback (9.5x end to end on a 504k-token prefill); it reports
  202.7 tokens per second at batch 1 with speculative decode. The indexer is what the public patches had to replace: that is
  the site where a kernel of ours changes something a user measures.

## 3. The model

DeepSeek-V4-Flash is 284B parameters, 13B active; its NVFP4 checkpoints (`nvidia/DeepSeek-V4-Flash-NVFP4`, `-0731-NVFP4`,
`DeepSeek-V4.1-Flash-NVFP4`) are about 160 to 168 GB, FP4 experts with FP8 attention, norm and router weights. One RTX PRO 6000
(96 GB) does not hold it; two do, at TP=2, which is the configuration the public record runs. There is no smaller shipped
checkpoint that exercises the indexer: the models that call `fp8_fp4_mqa_logits` in vLLM 0.28 are DeepSeek-V3.2 (671B) and
DeepSeek-V4 (284B and up); Kimi-K3 calls `fp8_fp4_mega_moe` (SM100 only). So "a model a PRO 6000 can hold" reads as "two PRO
6000 at TP=2 with the 0731 or V4.1 NVFP4 checkpoint", and the end-to-end measurement is a two-GPU pod.

## 3b. The operand formats, read against our kernels (2026-10-03, second pass)

vLLM's indexer hands the kernel one of two operand pairs (`vllm/utils/deep_gemm.py:510-577`, `sparse_attn_indexer.py:253-272`): the FP8
path is q `[M, H, D]` e4m3 with no q scale (the per-token scale is folded into `weights`) and k `[N, D]` e4m3 with one fp32 scale per
row; the MXFP4 path is q packed uint8 with a block-scale tensor and k `[N, D/2]` packed e2m1 with `[N, D/32]` UE8M0 scales, one per
32 along the head. Our `fp8_fp4_mqa_logits_sm120.py` v2 and v3 take q e4m3 with one UE8M0 scale per (row, head) and k packed e2m1 with
one UE8M0 scale per row over the whole 128-wide head (`quantize_inputs`, DeepGEMM's test recipe with `gran_k = D`). So neither of the
engine's pairs is our kernels' pair as they stand: the FP8-k path needs an e4m3 B operand with an fp32 row scale (the instruction form the
einsum kernel already uses), and the MXFP4-k path needs the scale fold at every 32 columns instead of once per row (four folds per k32
step group instead of one). The q side differs too: vLLM folds q's scale into `weights`, our v2 folds it per (row, head) after the MMA.
Both are kernel variants, not adapter work, and they come before any adapter can be checked bit for bit against the engine's call.

## 4. The order of work (adoption item 2)

0. **The kernel variants the engine's formats need** (found on the second reading, Section 3b): v4 of the MQA-logits kernel with k's
   UE8M0 scale per 32 along the head (the MXFP4 layout) and q's scale taken from `weights`; and an FP8-k form with the fp32 row scale.
   Each bit-identical to the current form on inputs where the layouts coincide, and checked against DeepGEMM's test reference.
1. **Unit wiring, on the RTX 5090, no model.** A `sm120fp4.indexer` module exposing `fp8_fp4_mqa_logits` and
   `fp8_fp4_paged_mqa_logits` with vLLM's calling convention (the `(data, scale)` q pair, FP8 q first, 2-D `seq_lens`,
   `block_table`, a `schedule_metadata` our scheduler ignores, `clean_logits`, `indices`), bound into
   `vllm.utils.deep_gemm._fp8_fp4_mqa_logits_impl` / `_fp8_fp4_paged_mqa_logits_impl` by the plugin's `register()` behind
   `SM120FP4_INDEXER=1`. Test: `sparse_attn_indexer.py`'s own code path on synthetic q, kv and block tables against the
   reference the paper's selftests use (DeepGEMM's test reference), bit for bit with our v2/v3, and vLLM's `is_deep_gemm_supported`
   left as the engine has it. Closes when the engine's indexer function, called as the engine calls it, returns our logits.
2. **The q-side formats.** vLLM's indexer quantises q to FP8 e4m3 per token (default) or MXFP4; our kernels take FP8 q. The
   MXFP4-q path (`use_fp4=True`) needs an FP4 x FP4 instruction form (`kind::f8f6f4` with e2m1 on both operands) that the
   repository has not measured; FP8 q is the default and is what step 1 wires.
3. **End to end, two RTX PRO 6000 (RunPod), DeepSeek-V4-Flash-0731-NVFP4, TP=2**, on the vLLM nightly the public recipe pins or
   on 0.29 if PR #41834 has merged by then: the engine's path (the Triton fallback or FlashInfer's) against ours behind the flag,
   same prompts, the 300-item slate for answers and decode tokens per second and prefill tokens per second at a long context
   where the indexer dominates (the public recipe's 500k-token request is the operating point where it mattered). Closes with
   the table in the README and the pod's log in `reports/`.
4. The grouped FP8 x FP4 GEMM (`DeepGemmFP4Experts`) is the second candidate, for an `mxfp4_w4a8` checkpoint that fits one card;
   it needs the grouped driver and is not started until step 3 has a number.

The upstream text (an issue or PR against vLLM or DeepGEMM) waits until the project is essentially complete (author,
2026-10-03); nothing here is drafted for posting.

## Sources

vLLM 0.28.0 as installed (`vllm/utils/deep_gemm.py`, `vllm/model_executor/layers/sparse_attn_indexer.py`,
`vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py`, `vllm/models/deepseek_v4/nvidia/{model.py, flashinfer_sparse.py,
ops/o_proj.py}`, `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`, `vllm/v1/attention/backends/mla/indexer.py`; line
numbers from that install); vLLM issue #41063 and PR #41834 (read 2026-10-03); github.com/Infatoshi/dsv4-flash-2x-rtxpro6000s
(read 2026-10-03); the NVFP4 checkpoint cards on Hugging Face (`nvidia/DeepSeek-V4-Flash-NVFP4`, `nvidia/DeepSeek-V4-Flash-0731-NVFP4`,
`nvidia/DeepSeek-V4.1-Flash-NVFP4`) and recipes.vllm.ai's DeepSeek-V4-Flash page for the parameter count and sizes.
