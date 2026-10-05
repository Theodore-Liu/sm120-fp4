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
the flat v5 through random page permutations on six shapes, and 33.6 / 115.5 / 123.7 us (the saved report; a first run printed 33.5 / 115.6 / 123.7), 1.86 to 2.03 times faster than the 4-byte
form and 1.44 to 1.66 times the paged v4's 23.3 / 72.4 / 74.5 us (`reports/fp8-paged-mqa-logits-v5-stage16-rtx5090-20261004.json`,
GPU idle before and after, 3.0 GB in use by the desktop). The remaining gap to v4 is the row width: a 128-byte e4m3 row is twice the
bytes of v4's 64-byte e2m1 row, so the kernel reads twice the cache bytes per logit. The fp32-weights variant's cost against the bf16
form is not yet timed.

**A risk for the end-to-end run.** vLLM issue #53635 (open, read 2026-10-04) reports that on SM12x with DeepGEMM present the indexer's
decode paged MQA-logits path fails for DeepSeek-V4: the kernel is templated for pages of 32 or 64 states, while V4's compressed
indexer cache uses 2 states per page (128 tokens per state) after the KV-cache layout standardisation of PR #51718; the proposed fixes
are a kernel that takes the smaller geometry, the short-row decode path of PR #41834, or a non-DeepGEMM fallback. What the pinned engine
declares (vLLM 0.28 as installed, `vllm/v1/attention/backends/mla/indexer.py`, read 2026-10-04): `DeepseekV32IndexerBackend`
supports a kernel block size of 64 on CUDA and `DeepseekV4IndexerBackend` one of 256; both give the cache the shape
(num_blocks, block_size, head_size) with one kv head, and `sparse_attn_indexer.kv_cache_as_quant_view` views it as
[num_blocks, block_size, 1, head_width] (132 bytes per row on the fp8 path). V4's positions reach the cache through
`compressor_utils.get_compressed_slot_mapping` (the model's `compress_ratios`), so the block table indexes blocks of 256 compressed
positions; the issue's "2 states per page" is DeepGEMM's kernel template against that geometry, not a different byte layout. Our
paged kernel reads 64-row pages, and a 256-row block is four consecutive 64-row pages of the same bytes, so the adapter now
accepts any block size that is a multiple of 64 and rewrites the block table (block b becomes pages 4b to 4b + 3) over the cache
viewed as [num_blocks * 4, 64, 1, 132]; `tests/test_indexer_adapter.py::test_paged_256_row_blocks` builds the 256-row cache in a
random block permutation and checks the result bit for bit against the 64-row form and the flat call on three shapes. Whether the
compressed positions' lengths are what `seq_lens` carries on V4 is read at step 3, on the engine. PR #54929 (a portable Triton
sparse-MLA fallback for SM12x) is the engine-side context for the same machines.

**The fp32-weights variant's cost.** Timed back to back on the idle GPU, the same three flat shapes, cold L2, median of 10
(`reports/fp8-mqa-logits-v5-bf16w-rerun-rtx5090-20261004.json`, `reports/fp8-mqa-logits-v5f-fp32w-rtx5090-20261004.json`): v5 with
bf16 weights 11.0 / 33.5 / 37.6 us, v5f with fp32 weights 9.9 / 33.5 / 37.6 us. The weights are read once per (row, head) and the
two kernels are the same bytes otherwise, so the fp32 operand costs nothing measurable; the adapter's default stands.

### 3f. The engine's operands against the adapter's assumptions (read from the pinned source, 2026-10-04)

The adapter assumes q arrives as e4m3 [M, H, 128] with a UE8M0 scale of one per (row, head) and `weights` as fp32 [M, H] that already
carries q's per-token scale. What vLLM 0.28 does (`vllm/model_executor/models/deepseek_v2.py`, class `Indexer`, and
`vllm/model_executor/layers/sparse_attn_indexer.py`):

- **q.** `per_token_group_quant_fp8(q.view(-1, head_dim), quant_block_size=128, use_ue8m0=scale_fmt is not None)`: one e4m3 group of
  128 per (token, head), one scale per group, so one scale per (row, head), which is the shape our UE8M0-of-one assumption needs. The
  fused CUDA path (`fused_indexer_q_rope_quant`) does the rotation, the quantisation and the fold in one kernel and returns `q_fp8`
  and `weights_out` of dtype fp32.
- **weights.** `weights = weights * q_scale * softmax_scale * n_head_scale` with `softmax_scale = head_dim ** -0.5` and
  `n_head_scale = n_head ** -0.5`; the fused kernel writes the same product as fp32. So the operand is fp32 and carries q's scale and
  both softmax factors; the adapter passes it to the fp32-weights kernel as it is (3d). Under `use_fp4_cache` a `q_scale` tensor
  travels separately and `weights` carries no q scale; that path raises `NotImplementedError` in the adapter, by design (3c).
- **k.** Quantised at cache insertion (`indexer_k_quant_and_cache`), e4m3 with an fp32 row scale in the 132-byte row the prefill
  gather (`cp_gather_indexer_k_quant_cache`) hands over as `k_quant` [N, 128] and `k_scale` viewed fp32 [N]: the flat call's `kv`
  pair as the adapter reads it.
- **decode lengths on V4.** The metadata builder carries `compress_ratio` from the MLA cache spec (1 for V3.2, the model's
  `compress_ratios` for V4), writes the compressed slot mapping, and fills an `expanded_seq_lens_buffer` of compressed lengths for the
  decode path; the block table indexes 256-row blocks of compressed positions. The adapter is agnostic to the unit (it takes the
  lengths and the table as given), which is what 3e's remap needs; whether `max_model_len` is passed in compressed or raw units on
  V4 is the one thing left to read on the engine at step 3 (it only sizes the output).

Nothing in the engine's operands contradicts the adapter; the two readings that remain (V4's `max_model_len` unit, and the
compressed lengths' arrival through `seq_lens`) are confirmed on the running engine, not from the source.

### 3g. Where the paged v5's time goes (Nsight Compute, 2026-10-04)

`reports/ncu-paged-v5-rtx5090-20261004.txt`: one launch of `k_paged_mqa_logits_v5` per bench shape (SpeedOfLight, MemoryWorkloadAnalysis,
WarpStateStats, Occupancy), the lane-strided 16-byte staging, the GPU otherwise idle.

| shape (S, N, H) | duration us | memory throughput | DRAM throughput | L2 hit | L1/TEX hit | compute throughput | warp cycles per issued instruction | occupancy, theoretical / achieved |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 64, 8192, 8 | 37.4 | 49.2 % | 5.4 % | 91.4 % | 80.8 % | 36.0 % | 7.9 | 16.67 / 16.36 % |
| 128, 16384, 8 | 133.5 | 54.6 % | 4.4 % | 94.4 % | 80.5 % | 40.0 % | 7.5 | 16.67 / 16.36 % |
| 32, 65536, 16 | 141.0 | 54.4 % | 3.8 % | 85.0 % | 76.8 % | 37.5 % | 8.0 | 16.67 / 16.24 % |

(The durations under the profiler are longer than the bench's 33.6 / 115.5 / 123.7 us; the profiler serialises and replays.) What the
numbers say: the cache rows come from L2 (hit 85 to 94 percent; DRAM under 6 percent of its bandwidth), so the kernel is not
DRAM-bound at these sizes; neither the memory pipes (49 to 55 percent) nor the SM (36 to 40 percent) is near its ceiling; and the
occupancy is pinned at 16.7 percent, one block of 8 warps per SM, by shared memory: **Block Limit Shared Mem = 1** against 3 by
registers and 6 by warps. The block stages 8 pages at once, 8 x (64 x 128 + 64 x 4) = 67.6 KB, and a second block does not fit. With
eight warps per SM and 7.5 to 8 cycles per issued instruction, the SM idles between loads it cannot overlap.

The lever this points at is shared memory per warp, not the load width: staging half a page per step (32 rows, 4.2 KB per warp, 33.8 KB
per block) would let two blocks share an SM, and a two-page double buffer per warp would cost the same as today while hiding the
next page's loads behind this page's MMAs. Either is a kernel change for a later turn; this section records the reading, not a fix.
The paged v4 reads 64-byte rows, half the bytes per page, and so stages 4.1 KB per warp: its two blocks per SM is where its 1.44 to
1.66 times advantage comes from, as much as from the bytes.

### 3h. The half-page staging, timed (2026-10-04)

`fp8_paged_mqa_logits_sm120_v5h` stages each page in two halves of 32 rows (4.2 KB per warp, 33.8 KB per block, so two blocks fit an SM where 3g found one). Bit-identical to the flat v5 on five shapes including a 70-row context (a partial second half). Timed back to back on the idle GPU, cold L2, median of 10 (`reports/fp8-paged-mqa-logits-v5h-rtx5090-20261004.json` and `reports/fp8-paged-mqa-logits-v5-full-rerun-rtx5090-20261004.json`):

| shape (S, N, H) | full page (v5), us | two halves (v5h), us | v5h / v5 | paged v4, us |
|---|---:|---:|---:|---:|
| 64, 8192, 8 | 33.6 | 33.5 | 1.00 | 23.3 |
| 128, 16384, 8 | 115.5 | 103.4 | 0.90 | 72.4 |
| 32, 65536, 16 | 124.9 | 113.4 | 0.91 | 74.5 |

Ten percent on the two large shapes, nothing on the small one (128 pages over 170 SMs: there is no second block to place). The gap to the paged v4 is now 1.44 to 1.52 times; the remaining lever 3g named, a two-page double buffer per warp so the next page's loads hide behind this page's MMAs, is the next kernel change. The adapter keeps the full-page kernel until the half-page form is the default after a wider shape sweep.

### 3i. The register-prefetch arm, and why it lost (2026-10-04)

`fp8_paged_mqa_logits_sm120_v5d` is the half-page kernel with the next half's nine 16-byte chunks loaded into registers while the current half is computed, shared memory unchanged at one half per warp. Bit-identical to the flat v5 on the five half-page shapes. Timed back to back with v5h on the idle GPU (`reports/fp8-paged-mqa-logits-v5d-rtx5090-20261004.json`, `reports/fp8-paged-mqa-logits-v5h-rerun-rtx5090-20261004.json`): 39.8 / 136.0 / 146.2 us against 33.5 / 103.2 / 113.4, slower on every shape. The cause, from Nsight Compute on the second shape (`reports/ncu-paged-v5d-v5h-rtx5090-20261004.txt`): the prefetch registers lift the kernel from 128 to 200 registers per thread, the block limit by registers falls from 2 to 1, and the occupancy goes back to 16.7 percent (one block per SM), which is exactly what the half-page form had bought back; the loads it hides cost more in residency than they save in latency. The arm stays in the file as a recorded negative; the adapter's paged call now uses v5h. A double buffer that keeps two blocks per SM would have to live in shared memory at 8.4 KB per warp, 67.6 KB per block, which is the full-page form's budget, so the next lever is not a buffer but the row width (two heads of q per MMA, or reading the 128-byte rows as two 64-byte halves to share the v4 path).

| shape (S, N, H) | v5h, us | v5d, us | registers per thread, v5h / v5d | occupancy, v5h / v5d |
|---|---:|---:|---|---|
| 64, 8192, 8 | 33.5 | 39.8 | | |
| 128, 16384, 8 | 103.2 | 136.0 | 128 / 200 | 33.3 / 16.7 % |
| 32, 65536, 16 | 113.4 | 146.2 | | |

### 3j. Raw-row staging (2026-10-04)

The paged kernel's remaining cost was the staging scatter, not the row width (`docs/stage3-paged-v5-row-width.md`): staging each 32-row half as raw 132-byte rows with straight 16-byte copies and reading at a 33-word stride (`fp8_paged_mqa_logits_sm120_v5r`) runs 21.2 / 62.4 / 72.5 us against the half-page scatter's 33.2 / 103.4 / 113.4 back to back, ahead of the paged v4's 23.3 / 72.4 / 74.5. Bit-identical to the flat v5; the adapter's paged call uses it, and the 18 adapter and top-k tests pass.

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
