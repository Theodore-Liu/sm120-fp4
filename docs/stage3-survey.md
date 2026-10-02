# Stage 3 survey: the three FP4 sites DeepGEMM routes to `tcgen05`, and what SM120 would need

Backlog item 3, opened 2026-10-02. This is a reading of the upstream sources as they stand at that date, written before any
stage-3 kernel exists here. Each claim names where it was read; what has only been read through a rendered page and not
in the checked-out source is marked "to confirm in source", and the first step of the work plan is to check those out.

## 1. The three sites, as the vLLM tracking issue names them

vLLM's tracking issue for DeepSeek-V4-Flash on consumer Blackwell ([vllm-project/vllm#41063]) lists three DeepGEMM
dispatch sites that have an SM100 kernel and no SM120 one:

| site | dispatch | SM100 kernel today | SM120 today |
|---|---|---|---|
| FP8xFP4 GEMM | `fp8_fp4_gemm_nt` in `csrc/apis/gemm.hpp` | `sm100_fp8_fp4_gemm_1d1d` | none; `DG_HOST_UNREACHABLE` |
| FP4 attention (MQA logits, the indexer) | three sites in `csrc/apis/attention.hpp` (around lines 67, 177, 367) | `sm100_mqa_logits`, `sm100_paged_mqa_logits`, `launch_sm100_sparse_mqa_logits` | none for FP4; SM90 has FP8-only `sm90_fp8_mqa_logits` / `sm90_fp8_paged_mqa_logits` |
| FP4 einsum | `csrc/apis/einsum.hpp` (around line 55) | `sm100_fp8_bmm` family | `sm120_fp8_einsum` covers FP8 only (to confirm in source: the einsum API as rendered shows `einsum` and `fp8_einsum` and no FP4 symbol) |

The ISA wall the issue quotes is exact: `ptxas: Instruction 'tcgen05.fence' not supported on .target 'sm_120f'` and
`Feature '.block32' not supported`. Everything an SM100 kernel does through `tcgen05` (MMA into tensor memory, TMEM
allocation, the `cta_group` pair, UTCCP scale-factor copies) has no SM120 counterpart; SM120 has `mma.sync` (and the
`mma.sync` FP4 shapes, which this repository's stage-2 kernels already use), TMA, mbarriers, 99 KB of shared memory per
block and 128 KB per SM, no TMEM, no 2-CTA MMA.

## 2. What each SM100 kernel does, and what has to change

### 2.1 `sm100_fp8_fp4_gemm_1d1d`

Read from `csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp` (rendered):

- Operands: A and B each an (FP8 or packed FP4) tensor with a scale-factor tensor; "an FP4xFP4 pair uses the MXF4 MMA;
  other combinations use the MXF8F6F4 MMA" (`kind::mxf8f6f4`, `is_mxf4_mma()`). FP4 is two codes per byte.
- Scales: the SM100 API takes scaling factors "in packed UE8M0 format, which packs 4 UE8M0 into a single `torch.int`"
  (README); the kernel builds TMA descriptors for A, B, SF_A, SF_B and C/D (`make_tma_sf_desc`, `cute::UMMA::Major::MN`),
  with "UTCCP-aligned SF blocks, with a per-128 K sub-block count" (`get_sf_block_config`). "1D1D" is one-dimensional
  (per-row, per-K-block) scaling on both operands. Granularity comes from the `recipe` tuple `(gran_k, gran_n,
  gran_k_detail)` of the API.
- Launch: a cluster of `get_cluster_size()` CTAs, grid = number of SMs (persistent), `num_stages` TMA stages plus
  `num_tma_store_stages`, shared memory from the pipeline config. The accumulator lives in TMEM; the epilogue is a
  pluggable `epilogue_class`.

For SM120 the MMA is `mma.sync.aligned.m16n8k64.kind::f8f6f4` on packed FP4/FP8 fragments, which is what this
repository's `scripts/fc1_mma.py` / `fc2_mma*.py` already issue for NVFP4xBF16 after decoding; the accumulator is in
registers, not TMEM, so the tile is bounded by the register file (two warps' worth of 128x128 is already past it) rather
than by TMEM columns. The scale format is the real difference from stage 2: UE8M0 (a power-of-two exponent per block,
MXFP-style, block 32 along K at the `kind::mxf8f6f4` MMA's native granularity) rather than NVFP4's E4M3 per 16 with a
per-tensor FP32. `mma.sync` has no scale-factor operand at all (the SM100 `tcgen05.mma` with `kind::mxf8f6f4` applies
block scales inside the MMA), so an SM120 kernel applies the UE8M0 scales outside the MMA, as a per-32-K shift on the
accumulator contribution - the same place stage 2 applies E4M3 block scales, and cheaper, since a UE8M0 scale is an
exponent add. The TMA/mbarrier pipeline and the persistent grid carry over; the cluster and multicast do not matter at
one CTA per tile.

### 2.2 `sm100_mqa_logits` (FP4 attention)

Read from `csrc/jit_kernels/impls/sm100_mqa_logits.hpp` and `csrc/apis/attention.hpp` (rendered):

- Inputs: `q` and `kv` each a (tensor, scale-factor) pair; the config checks `qk_dtype == kPackedFP4 or qk_dtype ==
  torch::kFloat8_e4m3fn`; for FP4 the head dimension is 64 or 128, for FP8 32, 64 or 128. The paged variant takes a fused
  KV cache whose row is `kv_head_dim + sf_bytes` with `sf_bytes` 4 (MX) or `sizeof(float)`. The output `weights` /
  `logits` is the score matrix the indexer ranks (this is DeepSeek's sparse-attention indexer, not the softmax attention
  itself).
- Tiling: a 128-row (token, head) tile with `block_q = 128 / num_heads`, 384 KV tokens per KV tile (`kMQALogitsSplitKV`);
  scale factors MX-style with `kNumUTCCPAlignedElems = 128`; TMA 2D/3D descriptors, mbarriers, `num_tmem_stages`.
- Variants: contiguous, paged (block table + context lengths), sparse (`sparse_block_kv` 8 or 16, up to 4096 blocks) and
  paged-sparse; the sparse pair is SM100-only even upstream.

For SM120 this is a batched small-K GEMM (head_dim 64/128) between an FP4 query tile and FP4 keys with a per-row score
output, which is close to stage 2's FC1 shape class (one activation tile against many weight rows, result per row), with
two differences: both operands are FP4 (so `kind::f8f6f4` on both fragments, no BF16 side), and the K dimension is tiny,
so the kernel is bound by key-stream bandwidth and by how many 128-row tiles a block keeps in flight, not by MMA. The
paged variant adds a gather through the block table, which the TMA 3D descriptor does on SM100 and a plain per-page
load does on SM120.

### 2.3 FP4 einsum

Read from `csrc/apis/einsum.hpp` (rendered): `einsum` (BF16) supports `"bmk,bnk->mn"`, `"bhr,hdr->bhd"`,
`"bhd,hdr->bhr"`, `"bhd,bhr->hdr"`; `fp8_einsum` takes quantized (tensor, scale) pairs for the last three, permutes to
`(batch, m, n, k)` and dispatches `sm90_fp8_bmm` / `sm100_fp8_bmm`. The rendered API shows no FP4-named einsum; the
tracking issue nonetheless lists `einsum.hpp:55` as an FP4 site, so either the quantized pairs accept `kPackedFP4` as the
attention API does, or the FP4 path lives in the kernel behind `fp8_einsum`. **To confirm in source before any design.**
If it is the former, the SM120 kernel is the batched-matmul form of 2.1 with the same UE8M0 handling and nothing new.

## 3. What this repository already has for each site

- The `mma.sync` FP4 fragment path, the E4M3 block-scale application, the TMA-free streaming of packed codes, PDL between
  dependent kernels, and the conformance suite for 128x4 scale layouts (stage 1) and for FC1/FC2 shapes (stage 2).
- Not yet: UE8M0 (MX) scales anywhere; FP4 on both MMA operands (stage 2 is FP4 weights x BF16 activations); any paged
  gather; any persistent-grid scheduler.

## 4. The order of work, and the first kernel

1. Check out DeepGEMM at a pinned commit and confirm the three "to confirm in source" items above; record the commit.
2. Write the UE8M0 (MX) block-scale path and its conformance test against `deep_gemm`'s own reference where one exists
   (the `tests/` of the repository), the way stage 1 tested the 128x4 layout against three implementations.
3. First kernel: **the FP8xFP4 dense GEMM (`sm120_fp8_fp4_gemm_1d1d`)**, NT layout, M <= 16 first (the decode shape this
   repository knows how to measure) and then the prefill shapes, because it is the site whose SM100 counterpart is one
   GEMM with a known reference, because the MoE path (`fp8_fp4_mega_moe`) is built from it, and because the attention and
   einsum sites reduce to batched forms of it. The attention indexer follows, the einsum last.
4. Each kernel closes with the conformance test, the single-kernel bench against the same byte floor the stage-2 design
   note uses, and a row in a report; the gate is DeepGEMM's own test passing through a `sm120_` dispatch branch in a
   fork, not a number here.

## Sources

- [vllm-project/vllm#41063](https://github.com/vllm-project/vllm/issues/41063), the tracking issue (dispatch sites, the
  ptxas errors, the linked PRs #41062, #41028, #40923).
- [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM): README (architectures, UE8M0 packing, API list);
  `csrc/apis/gemm.hpp`, `csrc/apis/attention.hpp`, `csrc/apis/einsum.hpp`; `csrc/jit_kernels/impls/` (the file list and
  the two SM100 kernels read above).
