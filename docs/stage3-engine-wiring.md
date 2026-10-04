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

## 3c. Which k format the engine uses on SM120 (read 2026-10-04)

`vllm/v1/attention/backends/mla/indexer.py` (`dsa_indexer_uses_fp4`, lines 48 to 63): the indexer's k cache is fp8 by default and the
MXFP4 cache is refused outside datacenter Blackwell ("indexer_kv_dtype='mxfp4' requires Blackwell datacenter GPUs (sm_10x ...); sm_120
(consumer Blackwell) and earlier architectures are not supported"). So on an RTX 5090 or PRO 6000 the kernel the engine calls is the
FP8-k one: q e4m3 with its scale in `weights`, k `[N, 128]` e4m3 with one fp32 scale per row, the paged cache 132 bytes per entry
(128 e4m3 + the 4-byte scale). v4's MXFP4 k format is the datacenter path; the SM120 path needs the FP8-k form, which is
`scripts/fp8_mqa_logits_v5_sm120.py` (2026-10-04): v2's structure with 128-byte e4m3 k rows, the `e4m3.e4m3` instruction form the
einsum kernel uses, and the fp32 row scale folded once; within 4.5e-7 of the dequantised reference on six shapes, two of them with
q's scale folded into `weights` as the engine does. Timing, cold L2, median of 10 (`reports/fp8-mqa-logits-v5-rtx5090-20261004.json`):

| S | N | H | us | TFLOP/s |
|---|---|---|---|---|
| 32 | 4096 | 8 | 10.5 | 25.5 |
| 128 | 8192 | 8 | 33.5 | 64.0 |
| 32 | 32768 | 16 | 37.5 | 114.4 |

The paged v5 followed the same day (`fp8_paged_mqa_logits_sm120_v5`, same file, `reports/fp8-paged-mqa-logits-v5-rtx5090-20261004.json`): one 132-byte-entry page per warp per step in vLLM's own cache layout (`[num_blocks, 64, 1, 132]`), bit-identical to the flat v5 through random page permutations on four shapes, within 2.9e-7 of the reference. It is about three times slower than the paged v4 on the same shapes (62.4 / 234.2 / 240.7 us against 23.3 / 72.4 / 74.5): a 132-byte row stride has no 16-byte alignment, so the page is staged with 4-byte loads instead of `cp.async` 16-byte chunks, and the e4m3 rows are twice the bytes of the packed e2m1 ones. The staging is the next thing to fix (two 16-byte chunks per 33-word row with a 4-byte remainder, or a 16-byte-aligned copy of the page) once the adapter proves the layout is read right.

| S | N | H | pages | us | cache rows GB/s | TFLOP/s |
|---|---|---|---|---|---|---|
| 64 | 8192 | 8 | 128 | 62.4 | 1108 | 17.2 |
| 128 | 16384 | 8 | 256 | 234.2 | 1182 | 18.3 |
| 32 | 65536 | 16 | 1024 | 240.7 | 1150 | 35.7 |

The adapter therefore targets v5 (flat) and the paged v5 first; v4 serves the MXFP4 path if vLLM's gate is
ever relaxed, which is an upstream change this repository does not propose yet.

### 3d. The weights operand, and the adapter (step 1, 2026-10-04)

`sm120fp4/indexer.py` serves the three entry points vLLM binds in `vllm/utils/deep_gemm.py` (`fp8_fp4_mqa_logits`,
`fp8_fp4_paged_mqa_logits`, `get_paged_mqa_logits_metadata`) with the v5 kernel, behind `SM120FP4_INDEXER=1`; the plugin's
`register()` binds them, so a stock vLLM 0.28 with the plugin installed takes them in place of DeepGEMM's `_missing`. What the
adapter does beyond calling the kernel: it expands the flat kernel's compacted logits ([M, max(ke - ks)]) to the engine's [M, N] at
absolute columns with -inf outside each row's span (the engine's `top_k_per_row_prefill` reads [ks, ke) only); in decode it flattens
the engine's [B, next_n, H, 128] q, its 1-D or 2-D `seq_lens` and its one block-table row per request to the kernel's one row per
query, allocates the [B * next_n, max_model_len] output the engine sizes and lets the kernel write positions [0, ctx) of each row;
the schedule metadata is an empty int32 tensor (v5 schedules by (row, page) in its grid). The MXFP4-q pair and the varlen `indices`
form raise `NotImplementedError` by design (Section 3c: vLLM refuses the MXFP4 indexer cache on SM120; the varlen branch is taken
only with vLLM's own packing kernel).

**The weights operand.** The engine folds q's per-token scale into `weights` and passes them as fp32; the measured kernel reads
`weights` as bf16. Converting the engine's fp32 weights to bf16 costs 1.2e-3 to 2.8e-3 relative on the logits against an
fp32-weights reference over five shapes (M 7 to 64, N 300 to 8192, H 8 to 32), against the kernel's own bar of 1e-5
(`reports/indexer-adapter-weights-precision-rtx5090-20261004.json`; the first version of that measurement was vacuous because its
"fp32" weights were bf16 values times a power of two, which bf16 holds exactly, and was replaced by weights with full fp32
mantissas). So the kernel gained an fp32-weights variant (`fp8_mqa_logits_sm120_v5f`, `fp8_paged_mqa_logits_sm120_v5f`), derived
from the same source by text (the pointer type, the one read, the check, and every name suffixed `f`); it reads the engine's
operand as it is and lands within 3.0e-7 of the fp32 reference on the same five shapes. The adapter defaults to it
(`SM120FP4_INDEXER_WEIGHTS=fp32`); `bf16` selects the measured kernel. The variant's cost against the bf16 form is not yet timed
(the weights are read once per (row, head) and the difference is one conversion per read); it is timed with the paged staging work.

**Test** (`tests/test_indexer_adapter.py`, 9 tests on the RTX 5090, vLLM 0.28 venv): the flat call against the kernel module's
reference at absolute columns on four shapes (1e-5 bar); the two-mode precision record above; the paged call against the flat call
on the same rows through random page permutations, with 1-D and 2-D `seq_lens`, `next_n` of 1 and 2 and a block table with spare
columns, bit for bit; and both calls through `vllm.utils.deep_gemm`'s own wrappers after `register(force=True)`, equal to the direct
calls. `tests/test_indexer_topk.py` (6 tests) then runs the engine's own top-k kernels, `top_k_per_row_prefill` and
`top_k_per_row_decode` from `vllm._custom_ops`, over the adapter's logits and over the fp32 reference logits and compares the
selected index sets row by row (prefill: 2048 of spans up to 8192 and 512 of up to 3000, indices relative to each row's `ks`,
which is the kernel's convention; decode: 2048 and 512 over paged contexts with `next_n` of 1 and 2, absolute positions): the
sets are identical on every row. What the attention layer consumes is that index buffer, so the kernel-level path the indexer
runs on SM120 (our logits, its top-k) is closed. The function `sparse_attn_indexer()` itself is not called standalone: it reads
the engine's forward context and attention metadata, which exist only inside a running engine, so its call is exercised by
step 3's two-GPU run rather than by a unit test.

### 3e. The paged staging, and a page-geometry risk for DeepSeek-V4 (2026-10-04)

**Staging.** The first paged v5 read each 132-byte cache row as 33 4-byte words (2112 loads per page, 66 per lane) and ran at
62.4 / 234.2 / 240.7 us on the three decode shapes, three times the paged v4 (`reports/fp8-paged-mqa-logits-v5-rtx5090-20261004.json`).
Two 16-byte forms were tried. The first grouped four rows (528 bytes, 33 aligned chunks) and looped over 16 groups with a one-or-two
trip inner loop in which lane 0 alone fetched each group's 33rd chunk; it was bit-identical and slower, 78.6 / 296.3 / 308.0 us, because
that lone load serialised one memory latency per group on one lane while the other 31 waited. The second reads the page as 528
contiguous 16-byte chunks lane-strided (17 passes, the last one for 16 lanes), scattering each 4-byte word to its row; bit-identical to
the flat v5 through random page permutations on six shapes, and 33.5 / 115.6 / 123.7 us, 1.86 to 2.03 times faster than the 4-byte
form and 1.44 to 1.66 times the paged v4's 23.3 / 72.4 / 74.5 us (`reports/fp8-paged-mqa-logits-v5-stage16-rtx5090-20261004.json`,
GPU idle before and after, 3.0 GB in use by the desktop). The remaining gap to v4 is the row width: a 128-byte e4m3 row is twice the
bytes of v4's 64-byte e2m1 row, so the kernel reads twice the cache bytes per logit. The fp32-weights variant's cost against the bf16
form is not yet timed.

**A risk for the end-to-end run.** vLLM issue #53635 (open, read 2026-10-04) reports that on SM12x with DeepGEMM present the indexer's
decode paged MQA-logits path fails for DeepSeek-V4: the kernel is templated for pages of 32 or 64 states, while V4's compressed
indexer cache uses 2 states per page (128 tokens per state) after the KV-cache layout standardisation of PR #51718; the proposed fixes
are a kernel that takes the smaller geometry, the short-row decode path of PR #41834, or a non-DeepGEMM fallback. Our adapter checks
for the [num_blocks, 64, 1, 132] layout and refuses anything else, so on V4 it would refuse rather than corrupt; before step 3 the
V4 indexer cache's actual page geometry on the pinned vLLM has to be read from the source and, if it is the 2-state form, the paged
kernel needs a variant for it. PR #54929 (a portable Triton sparse-MLA fallback for SM12x) is the engine-side context for the same
machines.

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
   **State 2026-10-04: closed at the kernel level (Section 3d). The adapter, its binding, its test through
   `vllm.utils.deep_gemm`'s wrappers, the fp32-weights kernel variant the engine's operand needs, and the engine's own top-k
   kernels over our logits (identical index sets to the reference on every row) are in. `sparse_attn_indexer()` as a function is
   first called by step 3's engine run; it has no standalone form.**
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
