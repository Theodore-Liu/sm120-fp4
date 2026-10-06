#!/usr/bin/env python3
"""MQA-logits v6: MXFP4 on both sides. vLLM's indexer can quantise q to MXFP4 (`use_fp4=True`: packed e2m1 [S, H, 64] with one
UE8M0 scale per 32 along the head) next to the MXFP4 k cache v4 takes (packed e2m1 [N, 64], UE8M0 [N, 4]). v6 is v4's staging and
tiling with the inner product done by `mma.sync.m16n8k64.kind::mxf4.block_scale.scale_vec::2X` (scripts/probe_f4f4.py): both
operands are loaded as raw packed 32-bit words (eight e2m1 values, low nibble first, no unpacking), a 128-wide head is two k64 MMAs,
and the four per-32 scales of each q row and each k column go to the MMA as scale operands, so no fold is done in software.

Scale operands (byte-id and thread-id 0, measured by the probe): row g of the A tile reads its two block scales from lane 4g's
scale-a register (bytes 0 and 1 for k 0..31 and 32..63 of the step), row g+8 from lane 4g+1's; column g of the B tile from lane 4g's
scale-b register. Every other lane's scale register is ignored by the instruction.

What the selftest checks: v6 against DeepGEMM's test reference form (torch.einsum on the dequantised operands, fp32) on five shapes
with independent block scales on both operands, relative error at most 1e-5 at the output's maximum (the bar the other indexer
kernels use), and the output outside each row's [ks, ke) span left untouched.

    ~/mlsys-5090-runtime/vllm028/.venv/bin/python sm120fp4/kernels/fp4_fp4_mqa_logits_v6_sm120.py --selftest
    ~/mlsys-5090-runtime/vllm028/.venv/bin/python sm120fp4/kernels/fp4_fp4_mqa_logits_v6_sm120.py --bench --out reports/fp4-fp4-mqa-logits-v6-rtx5090-<date>.json
    ~/mlsys-5090-runtime/vllm028/.venv/bin/python sm120fp4/kernels/fp4_fp4_mqa_logits_v6_sm120.py --bench-paged --out-paged reports/fp4-fp4-paged-mqa-logits-v6-rtx5090-<date>.json
    ncu --kernel-name regex:k_mqa_logits_v6 --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,smsp__inst_executed.sum <python> sm120fp4/kernels/fp4_fp4_mqa_logits_v6_sm120.py --ncu-shape
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fp8_fp4_mqa_logits_sm120 as base  # noqa: E402
import fp8_fp4_mqa_logits_v4_sm120 as v4  # noqa: E402
import ue8m0_reference as ref  # noqa: E402

HEAD_DIM = 128
BLOCK = 32
NBLK = HEAD_DIM // BLOCK

CPP_V6 = r"""
void fp4_fp4_mqa_logits_sm120_v6(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv, torch::Tensor sfkv, torch::Tensor w,
                                 torch::Tensor ks, torch::Tensor ke, torch::Tensor logits, int64_t kv_lo, int64_t kv_hi,
                                 int64_t rows, int64_t kvseg, int64_t group);
void fp4_fp4_mqa_logits_sm120_v6s(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv, torch::Tensor sfkv, torch::Tensor w,
                                  torch::Tensor ks, torch::Tensor ke, torch::Tensor logits, int64_t kv_lo, int64_t kv_hi,
                                  int64_t rows, int64_t kvseg, int64_t group);
void fp4_fp4_mqa_logits_sm120_v6e(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv, torch::Tensor sfkv, torch::Tensor w,
                                  torch::Tensor ks, torch::Tensor ke, torch::Tensor logits, int64_t kv_lo, int64_t kv_hi,
                                  int64_t rows, int64_t kvseg, int64_t group);
void fp4_fp4_paged_mqa_logits_sm120_v6(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv_cache, torch::Tensor sf_cache,
                                       torch::Tensor w, torch::Tensor context_lens, torch::Tensor block_table, torch::Tensor logits,
                                       int64_t max_pages);
void fp4_fp4_paged_mqa_logits_sm120_v6e(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv_cache, torch::Tensor sf_cache,
                                        torch::Tensor w, torch::Tensor context_lens, torch::Tensor block_table, torch::Tensor logits,
                                        int64_t max_pages);
void fp4_fp4_paged_mqa_logits_sm120_v6f(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv_cache, torch::Tensor sf_cache,
                                        torch::Tensor w, torch::Tensor context_lens, torch::Tensor block_table, torch::Tensor logits,
                                        int64_t max_pages, int64_t group);
void fp4_fp4_paged_mqa_logits_sm120_v6g(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv_cache, torch::Tensor sf_cache,
                                        torch::Tensor w, torch::Tensor context_lens, torch::Tensor block_table, torch::Tensor logits,
                                        int64_t max_pages);
void fp4_fp4_paged_mqa_logits_sm120_v6h(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv_cache, torch::Tensor sf_cache,
                                        torch::Tensor w, torch::Tensor context_lens, torch::Tensor block_table, torch::Tensor logits,
                                        int64_t max_pages);
void fp4_fp4_paged_mqa_logits_sm120_v6i(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv_cache, torch::Tensor sf_cache,
                                        torch::Tensor w, torch::Tensor context_lens, torch::Tensor block_table, torch::Tensor logits,
                                        int64_t max_pages);
void fp4_fp4_paged_mqa_logits_sm120_v6j(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv_cache, torch::Tensor sf_cache,
                                        torch::Tensor w, torch::Tensor context_lens, torch::Tensor block_table, torch::Tensor logits,
                                        int64_t max_pages);
void fp4_fp4_paged_mqa_logits_sm120_v6k(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv_cache, torch::Tensor sf_cache,
                                        torch::Tensor w, torch::Tensor context_lens, torch::Tensor block_table, torch::Tensor logits,
                                        int64_t max_pages, torch::Tensor page_offs);
torch::Tensor fp4_fp4_paged_mqa_logits_sm120_v6k_meta(torch::Tensor context_lens, int64_t max_pages);
"""

CUDA_V6 = r"""
__device__ __forceinline__ void mma_mxf4(float* c, const uint32_t* a, const uint32_t* b, uint32_t sa, uint32_t sb) {
  const uint16_t z = 0;
  asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%12, %12}, %11, {%12, %12};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb), "h"(z));
}

// v6: v4's staging (kv rows and their four UE8M0 bytes, double-buffered cp.async per KVSEG segment), q packed e2m1 [S, H, 64] with
// UE8M0 [S, H, 4]; two k64 block-scaled MMAs per n8 tile accumulate the whole head, then relu, weight, shuffle as in v2 and v4.
template <int ROWS, int KVSEG, int STRIDE, bool PACK>
__global__ void __launch_bounds__(V1_WARPS * 32)
k_mqa_logits_v6(const uint8_t* __restrict__ q, const uint8_t* __restrict__ sfq, const uint8_t* __restrict__ kv,
                const uint8_t* __restrict__ sfkv, const __nv_bfloat16* __restrict__ w, const int* __restrict__ ks,
                const int* __restrict__ ke, float* __restrict__ logits, int S, int H, int N, int max_k, int kv_lo, int kv_hi, int group) {
  constexpr int RPW = ROWS >= V1_WARPS ? ROWS / V1_WARPS : 1;
  __shared__ __align__(16) uint8_t s_kv[2][KVSEG * STRIDE];
  __shared__ __align__(16) uint8_t s_sf[2][KVSEG * 4];
  const int row0 = blockIdx.x * ROWS;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int first_seg = blockIdx.y * group;
  const int nsegs_total = (kv_hi - kv_lo + KVSEG - 1) / KVSEG;
  const int last_seg = min(nsegs_total, first_seg + group);
  if (first_seg >= last_seg) return;
  auto stage = [&](int seg, int buf) {
    const int seg0 = kv_lo + seg * KVSEG, seg1 = min(N, min(kv_hi, seg0 + KVSEG));
    for (int c = tid; c < KVSEG * 4; c += V1_WARPS * 32) {
      const int r = c >> 2, part = c & 3;
      if (seg0 + r < seg1) cp_async_16(s_kv[buf] + r * STRIDE + part * 16, kv + (size_t)(seg0 + r) * 64 + part * 16);
      else *reinterpret_cast<uint4*>(s_kv[buf] + r * STRIDE + part * 16) = make_uint4(0u, 0u, 0u, 0u);
    }
    for (int r = tid; r < KVSEG; r += V1_WARPS * 32)
      *reinterpret_cast<uint32_t*>(s_sf[buf] + r * 4) = (seg0 + r < seg1) ? *reinterpret_cast<const uint32_t*>(sfkv + (size_t)(seg0 + r) * 4) : 0x7F7F7F7Fu;
    asm volatile("cp.async.commit_group;\n");
  };
  stage(first_seg, 0);
  for (int seg = first_seg; seg < last_seg; ++seg) {
    const int buf = (seg - first_seg) & 1;
    if (seg + 1 < last_seg) { stage(seg + 1, buf ^ 1); asm volatile("cp.async.wait_group 1;\n"); }
    else { asm volatile("cp.async.wait_group 0;\n"); }
    __syncthreads();
    const int seg0 = kv_lo + seg * KVSEG, seg1 = min(N, min(kv_hi, seg0 + KVSEG));
    const uint8_t* skv = s_kv[buf];
    const uint8_t* ssf = s_sf[buf];
    for (int rr = 0; rr < RPW; ++rr) {
      const int i = row0 + (ROWS >= V1_WARPS ? warp * RPW + rr : warp);
      if (i >= S || (ROWS < V1_WARPS && warp >= ROWS)) continue;
      const int k_start = ks[i], k_end = ke[i];
      const int n_lo = max(k_start, seg0), n_hi = min(k_end, seg1);
      if (n_lo >= n_hi) continue;
      const uint8_t* qrow = q + (size_t)i * H * 64;
      float* out = logits + (size_t)i * max_k;
      for (int h0 = 0; h0 < H; h0 += 16) {
        const int ha = h0 + g, hb = h0 + g + 8;
        const bool has_a = ha < H, has_b = hb < H;
        const float wa = has_a ? __bfloat162float(w[(size_t)i * H + ha]) : 0.f;
        const float wb = has_b ? __bfloat162float(w[(size_t)i * H + hb]) : 0.f;
        uint32_t af[2][4];
        uint32_t sa[2];
        for (int st = 0; st < 2; ++st) {
          const int b0 = st * 32;   // byte offset of the step's 64 values in the packed row
          af[st][0] = has_a ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)ha * 64 + b0 + 4 * t) : 0u;
          af[st][1] = has_b ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)hb * 64 + b0 + 4 * t) : 0u;
          af[st][2] = has_a ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)ha * 64 + b0 + 16 + 4 * t) : 0u;
          af[st][3] = has_b ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)hb * 64 + b0 + 16 + 4 * t) : 0u;
          // scale-a: lane 4g carries row g (head ha), lane 4g+1 row g+8 (head hb); bytes 0 and 1 are the step's two 32-blocks
          const int hs = (t == 0) ? ha : hb;
          const bool hv = (t == 0) ? has_a : has_b;
          sa[st] = (t < 2 && hv) ? (uint32_t)sfq[((size_t)i * H + hs) * 4 + 2 * st] | ((uint32_t)sfq[((size_t)i * H + hs) * 4 + 2 * st + 1] << 8)
                                 : 0x7F7Fu;
        }
        const int tile0 = (n_lo - seg0) & ~7;
        for (int n0 = seg0 + tile0; n0 < n_hi; n0 += 8) {
          const int col = n0 + g;
          const bool has_col = col >= n_lo && col < n_hi;
          const uint8_t* brow = skv + (size_t)(has_col ? (col - seg0) : 0) * STRIDE;
          const uint8_t* bsf = ssf + (size_t)(has_col ? (col - seg0) : 0) * 4;
          const int c0 = n0 + 2 * t, c1 = c0 + 1;
          const bool in0 = c0 >= n_lo && c0 < n_hi, in1 = c1 >= n_lo && c1 < n_hi;
          float acc[4] = {0.f, 0.f, 0.f, 0.f};
          for (int st = 0; st < 2; ++st) {
            const int b0 = st * 32;
            uint32_t bf[2];
            bf[0] = has_col ? *reinterpret_cast<const uint32_t*>(brow + b0 + 4 * t) : 0u;
            bf[1] = has_col ? *reinterpret_cast<const uint32_t*>(brow + b0 + 16 + 4 * t) : 0u;
            // scale-b: lane 4g carries column g
            const uint32_t sb = (t == 0 && has_col) ? (uint32_t)bsf[2 * st] | ((uint32_t)bsf[2 * st + 1] << 8) : 0x7F7Fu;
            mma_mxf4(acc, af[st], bf, sa[st], sb);
          }
          float v0 = fmaxf(acc[0], 0.f) * wa + fmaxf(acc[2], 0.f) * wb;
          float v1 = fmaxf(acc[1], 0.f) * wa + fmaxf(acc[3], 0.f) * wb;
          if (PACK) {
            // one butterfly for both values: after the xor-4 exchange even g keeps the v0 sum, odd g the v1 sum
            const bool odd = (g & 1) != 0;
            const float send = odd ? v0 : v1;
            float keep = odd ? v1 : v0;
            keep += __shfl_xor_sync(0xffffffffu, send, 4);
            keep += __shfl_xor_sync(0xffffffffu, keep, 8);
            keep += __shfl_xor_sync(0xffffffffu, keep, 16);
            if (g < 2) {
              const int c = odd ? c1 : c0;
              if (odd ? in1 : in0) { if (h0 == 0) out[c - k_start] = keep; else out[c - k_start] += keep; }
            }
          } else {
            for (int m = 4; m < 32; m <<= 1) {
              v0 += __shfl_xor_sync(0xffffffffu, v0, m);
              v1 += __shfl_xor_sync(0xffffffffu, v1, m);
            }
            if (g == 0) {
              if (in0) { if (h0 == 0) out[c0 - k_start] = v0; else out[c0 - k_start] += v0; }
              if (in1) { if (h0 == 0) out[c1 - k_start] = v1; else out[c1 - k_start] += v1; }
            }
          }
        }
      }
    }
    __syncthreads();
  }
}

template <int ROWS, int KVSEG, int STRIDE, bool PACK>
static void launch_v6(const uint8_t* pq, const uint8_t* psq, const uint8_t* pkv, const uint8_t* psk, const __nv_bfloat16* pw,
                      const int* pks, const int* pke, float* pl, int S, int H, int N, int max_k, int kv_lo, int kv_hi, int group,
                      cudaStream_t st) {
  const int nsegs = (kv_hi - kv_lo + KVSEG - 1) / KVSEG;
  const dim3 grid((S + ROWS - 1) / ROWS, (nsegs + group - 1) / group);
  k_mqa_logits_v6<ROWS, KVSEG, STRIDE, PACK><<<grid, V1_WARPS * 32, 0, st>>>(pq, psq, pkv, psk, pw, pks, pke, pl, S, H, N, max_k, kv_lo, kv_hi, group);
}

template <int STR, bool PK>
static void mqa_logits_v6_impl(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv, torch::Tensor sfkv, torch::Tensor w,
                                 torch::Tensor ks, torch::Tensor ke, torch::Tensor logits, int64_t kv_lo, int64_t kv_hi,
                                 int64_t rows, int64_t kvseg, int64_t group) {
  const int S = (int)q.size(0), H = (int)q.size(1), N = (int)kv.size(0), max_k = (int)logits.size(1);
  TORCH_CHECK(q.scalar_type() == torch::kInt8 && q.is_contiguous() && q.dim() == 3 && q.size(2) == 64, "q: packed e2m1 int8 [S, H, 64]");
  TORCH_CHECK(sfq.scalar_type() == torch::kUInt8 && sfq.numel() == (int64_t)S * H * 4 && sfq.is_contiguous(), "sfq: uint8 UE8M0 [S, H, 4] (one per 32 columns)");
  TORCH_CHECK(kv.scalar_type() == torch::kInt8 && kv.is_contiguous() && kv.size(1) == 64, "kv: packed e2m1 int8 [N, 64]");
  TORCH_CHECK(sfkv.scalar_type() == torch::kUInt8 && sfkv.dim() == 2 && sfkv.size(0) == N && sfkv.size(1) == 4 && sfkv.is_contiguous(), "sfkv: uint8 UE8M0 [N, 4] (one per 32 columns)");
  TORCH_CHECK(w.scalar_type() == torch::kBFloat16 && w.size(0) == S && w.size(1) == H && w.is_contiguous(), "weights: bf16 [S, H]");
  TORCH_CHECK(ks.scalar_type() == torch::kInt && ke.scalar_type() == torch::kInt && ks.numel() == S && ke.numel() == S, "ks, ke: int32 [S]");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat && logits.size(0) == S && logits.is_contiguous(), "logits: fp32 [S, max_seqlen_k]");
  TORCH_CHECK(kv_lo >= 0 && kv_hi <= N && kv_lo < kv_hi, "kv_lo < kv_hi within [0, N]");
  TORCH_CHECK(group >= 1, "group >= 1");
  auto st = at::cuda::getCurrentCUDAStream();
  const uint8_t* pq = reinterpret_cast<const uint8_t*>(q.data_ptr<int8_t>());
  const uint8_t* psq = sfq.data_ptr<uint8_t>();
  const uint8_t* pkv = reinterpret_cast<const uint8_t*>(kv.data_ptr<int8_t>());
  const uint8_t* psk = sfkv.data_ptr<uint8_t>();
  const __nv_bfloat16* pw = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
  float* pl = logits.data_ptr<float>();
  const int* pks = ks.data_ptr<int>();
  const int* pke = ke.data_ptr<int>();
  if (rows == 16 && kvseg == 256) launch_v6<16, 256, STR, PK>(pq, psq, pkv, psk, pw, pks, pke, pl, S, H, N, max_k, (int)kv_lo, (int)kv_hi, (int)group, st);
  else if (rows == 16 && kvseg == 64) launch_v6<16, 64, STR, PK>(pq, psq, pkv, psk, pw, pks, pke, pl, S, H, N, max_k, (int)kv_lo, (int)kv_hi, (int)group, st);
  else if (rows == 8 && kvseg == 256) launch_v6<8, 256, STR, PK>(pq, psq, pkv, psk, pw, pks, pke, pl, S, H, N, max_k, (int)kv_lo, (int)kv_hi, (int)group, st);
  else if (rows == 8 && kvseg == 64) launch_v6<8, 64, STR, PK>(pq, psq, pkv, psk, pw, pks, pke, pl, S, H, N, max_k, (int)kv_lo, (int)kv_hi, (int)group, st);
  else TORCH_CHECK(false, "rows in {16, 8}, kvseg in {256, 64}");
}

void fp4_fp4_mqa_logits_sm120_v6(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv, torch::Tensor sfkv, torch::Tensor w,
                                 torch::Tensor ks, torch::Tensor ke, torch::Tensor logits, int64_t kv_lo, int64_t kv_hi,
                                 int64_t rows, int64_t kvseg, int64_t group) {
  mqa_logits_v6_impl<64, false>(q, sfq, kv, sfkv, w, ks, ke, logits, kv_lo, kv_hi, rows, kvseg, group);
}
// v6e: v6 with the packed epilogue (one butterfly for both column values, 3 shuffles per tile instead of 6)
void fp4_fp4_mqa_logits_sm120_v6e(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv, torch::Tensor sfkv, torch::Tensor w,
                                  torch::Tensor ks, torch::Tensor ke, torch::Tensor logits, int64_t kv_lo, int64_t kv_hi,
                                  int64_t rows, int64_t kvseg, int64_t group) {
  mqa_logits_v6_impl<64, true>(q, sfq, kv, sfkv, w, ks, ke, logits, kv_lo, kv_hi, rows, kvseg, group);
}
// v6s: the same kernel with the staged rows at an 80-byte stride (rows g and g+2 of a B-fragment load no longer share a bank)
void fp4_fp4_mqa_logits_sm120_v6s(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv, torch::Tensor sfkv, torch::Tensor w,
                                  torch::Tensor ks, torch::Tensor ke, torch::Tensor logits, int64_t kv_lo, int64_t kv_hi,
                                  int64_t rows, int64_t kvseg, int64_t group) {
  mqa_logits_v6_impl<80, false>(q, sfq, kv, sfkv, w, ks, ke, logits, kv_lo, kv_hi, rows, kvseg, group);
}

// paged v6: one page of V3_PAGE kv rows per warp per grid step (v3/v4's grid: x = row groups of V3_WARPS, y = page index). The page's rows
// are staged at an 80-byte stride: with 64 bytes (16 words) per row, lane (g, t)'s B-fragment word 16g + t + 8*st puts rows g and g+2 in one
// bank; at 20 words per row the eight rows land on eight disjoint 4-bank groups.
constexpr int V6P_STRIDE = 80;
template <bool PACK>
__global__ void __launch_bounds__(V3_WARPS * 32)
k_paged_mqa_logits_v6(const uint8_t* __restrict__ q, const uint8_t* __restrict__ sfq, const uint8_t* __restrict__ kv_cache,
                      const uint8_t* __restrict__ sf_cache, const __nv_bfloat16* __restrict__ w, const int* __restrict__ ctx,
                      const int* __restrict__ block_table, float* __restrict__ logits, int S, int H, int max_pages, int max_ctx) {
  __shared__ __align__(16) uint8_t s_kv[V3_WARPS][V3_PAGE * V6P_STRIDE];
  __shared__ __align__(16) uint8_t s_sf[V3_WARPS][V3_PAGE * 4];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int i = blockIdx.x * V3_WARPS + warp;
  const int p = blockIdx.y;
  if (i >= S) return;
  const int len = ctx[i];
  const int pos0 = p * V3_PAGE;
  if (pos0 >= len) return;
  const int n_valid = min(V3_PAGE, len - pos0);
  const int page = block_table[(size_t)i * max_pages + p];
  for (int c = lane; c < V3_PAGE * 4; c += 32) {
    const int r = c >> 2, part = c & 3;
    cp_async_16(s_kv[warp] + r * V6P_STRIDE + part * 16, kv_cache + ((size_t)page * V3_PAGE + r) * 64 + part * 16);
  }
  if (lane < 16) cp_async_16(s_sf[warp] + lane * 16, sf_cache + (size_t)page * V3_PAGE * 4 + lane * 16);
  asm volatile("cp.async.commit_group;\n");
  asm volatile("cp.async.wait_group 0;\n");
  __syncwarp();
  const uint8_t* qrow = q + (size_t)i * H * 64;
  float* out = logits + (size_t)i * max_ctx + pos0;
  const uint8_t* skv = s_kv[warp];
  const uint8_t* ssf = s_sf[warp];
  for (int h0 = 0; h0 < H; h0 += 16) {
    const int ha = h0 + g, hb = h0 + g + 8;
    const bool has_a = ha < H, has_b = hb < H;
    const float wa = has_a ? __bfloat162float(w[(size_t)i * H + ha]) : 0.f;
    const float wb = has_b ? __bfloat162float(w[(size_t)i * H + hb]) : 0.f;
    uint32_t af[2][4];
    uint32_t sa[2];
    for (int st = 0; st < 2; ++st) {
      const int b0 = st * 32;
      af[st][0] = has_a ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)ha * 64 + b0 + 4 * t) : 0u;
      af[st][1] = has_b ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)hb * 64 + b0 + 4 * t) : 0u;
      af[st][2] = has_a ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)ha * 64 + b0 + 16 + 4 * t) : 0u;
      af[st][3] = has_b ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)hb * 64 + b0 + 16 + 4 * t) : 0u;
      const int hs = (t == 0) ? ha : hb;
      const bool hv = (t == 0) ? has_a : has_b;
      sa[st] = (t < 2 && hv) ? (uint32_t)sfq[((size_t)i * H + hs) * 4 + 2 * st] | ((uint32_t)sfq[((size_t)i * H + hs) * 4 + 2 * st + 1] << 8)
                             : 0x7F7Fu;
    }
    for (int n0 = 0; n0 < n_valid; n0 += 8) {
      const int col = n0 + g;
      const bool has_col = col < n_valid;
      const uint8_t* brow = skv + (size_t)(has_col ? col : 0) * V6P_STRIDE;
      const uint8_t* bsf = ssf + (size_t)(has_col ? col : 0) * 4;
      const int c0 = n0 + 2 * t, c1 = c0 + 1;
      const bool in0 = c0 < n_valid, in1 = c1 < n_valid;
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      for (int st = 0; st < 2; ++st) {
        const int b0 = st * 32;
        uint32_t bf[2];
        bf[0] = has_col ? *reinterpret_cast<const uint32_t*>(brow + b0 + 4 * t) : 0u;
        bf[1] = has_col ? *reinterpret_cast<const uint32_t*>(brow + b0 + 16 + 4 * t) : 0u;
        const uint32_t sb = (t == 0 && has_col) ? (uint32_t)bsf[2 * st] | ((uint32_t)bsf[2 * st + 1] << 8) : 0x7F7Fu;
        mma_mxf4(acc, af[st], bf, sa[st], sb);
      }
      float v0 = fmaxf(acc[0], 0.f) * wa + fmaxf(acc[2], 0.f) * wb;
      float v1 = fmaxf(acc[1], 0.f) * wa + fmaxf(acc[3], 0.f) * wb;
      if (PACK) {
        // the flat v6e's epilogue: one butterfly for both column values, lanes g = 0 and 1 write c0 and c1
        const bool odd = (g & 1) != 0;
        const float send = odd ? v0 : v1;
        float keep = odd ? v1 : v0;
        keep += __shfl_xor_sync(0xffffffffu, send, 4);
        keep += __shfl_xor_sync(0xffffffffu, keep, 8);
        keep += __shfl_xor_sync(0xffffffffu, keep, 16);
        if (g < 2) {
          const int c = odd ? c1 : c0;
          if (odd ? in1 : in0) { if (h0 == 0) out[c] = keep; else out[c] += keep; }
        }
      } else {
        for (int m = 4; m < 32; m <<= 1) {
          v0 += __shfl_xor_sync(0xffffffffu, v0, m);
          v1 += __shfl_xor_sync(0xffffffffu, v1, m);
        }
        if (g == 0) {
          if (in0) { if (h0 == 0) out[c0] = v0; else out[c0] += v0; }
          if (in1) { if (h0 == 0) out[c1] = v1; else out[c1] += v1; }
        }
      }
    }
  }
}

template <bool PK>
static void paged_v6_impl(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv_cache, torch::Tensor sf_cache,
                                       torch::Tensor w, torch::Tensor context_lens, torch::Tensor block_table, torch::Tensor logits,
                                       int64_t max_pages) {
  const int S = (int)q.size(0), H = (int)q.size(1), max_ctx = (int)logits.size(1);
  TORCH_CHECK(q.scalar_type() == torch::kInt8 && q.is_contiguous() && q.dim() == 3 && q.size(2) == 64, "q: packed e2m1 int8 [S, H, 64]");
  TORCH_CHECK(sfq.scalar_type() == torch::kUInt8 && sfq.numel() == (int64_t)S * H * 4 && sfq.is_contiguous(), "sfq: uint8 UE8M0 [S, H, 4]");
  TORCH_CHECK(kv_cache.scalar_type() == torch::kInt8 && kv_cache.is_contiguous() && kv_cache.dim() == 3 && kv_cache.size(1) == 64 && kv_cache.size(2) == 64,
              "kv_cache: packed e2m1 int8 [num_blocks, 64, 64]");
  TORCH_CHECK(sf_cache.scalar_type() == torch::kUInt8 && sf_cache.is_contiguous() && sf_cache.dim() == 3 && sf_cache.size(0) == kv_cache.size(0) && sf_cache.size(1) == 64 && sf_cache.size(2) == 4,
              "sf_cache: uint8 UE8M0 [num_blocks, 64, 4]");
  TORCH_CHECK(w.scalar_type() == torch::kBFloat16 && w.size(0) == S && w.size(1) == H && w.is_contiguous(), "weights: bf16 [S, H]");
  TORCH_CHECK(context_lens.scalar_type() == torch::kInt && context_lens.numel() == S, "context_lens: int32 [S]");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt && block_table.is_contiguous() && block_table.size(0) == S && block_table.size(1) == max_pages, "block_table: int32 [S, max_pages]");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat && logits.size(0) == S && logits.is_contiguous(), "logits: fp32 [S, max_context_len]");
  auto st = at::cuda::getCurrentCUDAStream();
  k_paged_mqa_logits_v6<PK><<<dim3((S + V3_WARPS - 1) / V3_WARPS, (int)max_pages), V3_WARPS * 32, 0, st>>>(
      reinterpret_cast<const uint8_t*>(q.data_ptr<int8_t>()), sfq.data_ptr<uint8_t>(), reinterpret_cast<const uint8_t*>(kv_cache.data_ptr<int8_t>()),
      sf_cache.data_ptr<uint8_t>(), reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), context_lens.data_ptr<int>(),
      block_table.data_ptr<int>(), logits.data_ptr<float>(), S, H, (int)max_pages, max_ctx);
}

void fp4_fp4_paged_mqa_logits_sm120_v6(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv_cache, torch::Tensor sf_cache,
                                       torch::Tensor w, torch::Tensor context_lens, torch::Tensor block_table, torch::Tensor logits,
                                       int64_t max_pages) {
  paged_v6_impl<false>(q, sfq, kv_cache, sf_cache, w, context_lens, block_table, logits, max_pages);
}
// paged v6e: the paged v6 with the packed epilogue
void fp4_fp4_paged_mqa_logits_sm120_v6e(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv_cache, torch::Tensor sf_cache,
                                        torch::Tensor w, torch::Tensor context_lens, torch::Tensor block_table, torch::Tensor logits,
                                        int64_t max_pages) {
  paged_v6_impl<true>(q, sfq, kv_cache, sf_cache, w, context_lens, block_table, logits, max_pages);
}

// paged v6f: a warp walks G pages of its row, the next page's copy in flight while the current one is computed (two buffers per warp).
constexpr int V6F_WARPS = 4;
template <int G>
__global__ void __launch_bounds__(V6F_WARPS * 32)
k_paged_mqa_logits_v6f(const uint8_t* __restrict__ q, const uint8_t* __restrict__ sfq, const uint8_t* __restrict__ kv_cache,
                       const uint8_t* __restrict__ sf_cache, const __nv_bfloat16* __restrict__ w, const int* __restrict__ ctx,
                       const int* __restrict__ block_table, float* __restrict__ logits, int S, int H, int max_pages, int max_ctx) {
  __shared__ __align__(16) uint8_t s_kv[V6F_WARPS][2][V3_PAGE * V6P_STRIDE];
  __shared__ __align__(16) uint8_t s_sf[V6F_WARPS][2][V3_PAGE * 4];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int i = blockIdx.x * V6F_WARPS + warp;
  if (i >= S) return;
  const int len = ctx[i];
  const int p_first = blockIdx.y * G;
  if (p_first * V3_PAGE >= len) return;
  const int n_pages_row = (len + V3_PAGE - 1) / V3_PAGE;
  const int p_last = min(p_first + G, min(n_pages_row, max_pages));
  auto stage = [&](int p, int buf) {
    const int page = block_table[(size_t)i * max_pages + p];
    for (int c = lane; c < V3_PAGE * 4; c += 32) {
      const int r = c >> 2, part = c & 3;
      cp_async_16(s_kv[warp][buf] + r * V6P_STRIDE + part * 16, kv_cache + ((size_t)page * V3_PAGE + r) * 64 + part * 16);
    }
    if (lane < 16) cp_async_16(s_sf[warp][buf] + lane * 16, sf_cache + (size_t)page * V3_PAGE * 4 + lane * 16);
    asm volatile("cp.async.commit_group;\n");
  };
  const uint8_t* qrow = q + (size_t)i * H * 64;
  stage(p_first, 0);
  for (int p = p_first; p < p_last; ++p) {
    const int buf = (p - p_first) & 1;
    if (p + 1 < p_last) { stage(p + 1, buf ^ 1); asm volatile("cp.async.wait_group 1;\n"); }
    else { asm volatile("cp.async.wait_group 0;\n"); }
    __syncwarp();
    const int pos0 = p * V3_PAGE;
    const int n_valid = min(V3_PAGE, len - pos0);
    float* out = logits + (size_t)i * max_ctx + pos0;
    const uint8_t* skv = s_kv[warp][buf];
    const uint8_t* ssf = s_sf[warp][buf];
    for (int h0 = 0; h0 < H; h0 += 16) {
      const int ha = h0 + g, hb = h0 + g + 8;
      const bool has_a = ha < H, has_b = hb < H;
      const float wa = has_a ? __bfloat162float(w[(size_t)i * H + ha]) : 0.f;
      const float wb = has_b ? __bfloat162float(w[(size_t)i * H + hb]) : 0.f;
      uint32_t af[2][4];
      uint32_t sa[2];
      for (int st = 0; st < 2; ++st) {
        const int b0 = st * 32;
        af[st][0] = has_a ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)ha * 64 + b0 + 4 * t) : 0u;
        af[st][1] = has_b ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)hb * 64 + b0 + 4 * t) : 0u;
        af[st][2] = has_a ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)ha * 64 + b0 + 16 + 4 * t) : 0u;
        af[st][3] = has_b ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)hb * 64 + b0 + 16 + 4 * t) : 0u;
        const int hs = (t == 0) ? ha : hb;
        const bool hv = (t == 0) ? has_a : has_b;
        sa[st] = (t < 2 && hv) ? (uint32_t)sfq[((size_t)i * H + hs) * 4 + 2 * st] | ((uint32_t)sfq[((size_t)i * H + hs) * 4 + 2 * st + 1] << 8)
                               : 0x7F7Fu;
      }
      for (int n0 = 0; n0 < n_valid; n0 += 8) {
        const int col = n0 + g;
        const bool has_col = col < n_valid;
        const uint8_t* brow = skv + (size_t)(has_col ? col : 0) * V6P_STRIDE;
        const uint8_t* bsf = ssf + (size_t)(has_col ? col : 0) * 4;
        const int c0 = n0 + 2 * t, c1 = c0 + 1;
        const bool in0 = c0 < n_valid, in1 = c1 < n_valid;
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (int st = 0; st < 2; ++st) {
          const int b0 = st * 32;
          uint32_t bf[2];
          bf[0] = has_col ? *reinterpret_cast<const uint32_t*>(brow + b0 + 4 * t) : 0u;
          bf[1] = has_col ? *reinterpret_cast<const uint32_t*>(brow + b0 + 16 + 4 * t) : 0u;
          const uint32_t sb = (t == 0 && has_col) ? (uint32_t)bsf[2 * st] | ((uint32_t)bsf[2 * st + 1] << 8) : 0x7F7Fu;
          mma_mxf4(acc, af[st], bf, sa[st], sb);
        }
        float v0 = fmaxf(acc[0], 0.f) * wa + fmaxf(acc[2], 0.f) * wb;
        float v1 = fmaxf(acc[1], 0.f) * wa + fmaxf(acc[3], 0.f) * wb;
        const bool odd = (g & 1) != 0;
        const float send = odd ? v0 : v1;
        float keep = odd ? v1 : v0;
        keep += __shfl_xor_sync(0xffffffffu, send, 4);
        keep += __shfl_xor_sync(0xffffffffu, keep, 8);
        keep += __shfl_xor_sync(0xffffffffu, keep, 16);
        if (g < 2) {
          const int c = odd ? c1 : c0;
          if (odd ? in1 : in0) { if (h0 == 0) out[c] = keep; else out[c] += keep; }
        }
      }
    }
    __syncwarp();   // the buffer just read is refilled two pages from now; every lane is past it
  }
}

void fp4_fp4_paged_mqa_logits_sm120_v6f(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv_cache, torch::Tensor sf_cache,
                                        torch::Tensor w, torch::Tensor context_lens, torch::Tensor block_table, torch::Tensor logits,
                                        int64_t max_pages, int64_t group) {
  const int S = (int)q.size(0), H = (int)q.size(1), max_ctx = (int)logits.size(1);
  TORCH_CHECK(q.scalar_type() == torch::kInt8 && q.is_contiguous() && q.dim() == 3 && q.size(2) == 64, "q: packed e2m1 int8 [S, H, 64]");
  TORCH_CHECK(sfq.scalar_type() == torch::kUInt8 && sfq.numel() == (int64_t)S * H * 4 && sfq.is_contiguous(), "sfq: uint8 UE8M0 [S, H, 4]");
  TORCH_CHECK(kv_cache.scalar_type() == torch::kInt8 && kv_cache.is_contiguous() && kv_cache.dim() == 3 && kv_cache.size(1) == 64 && kv_cache.size(2) == 64,
              "kv_cache: packed e2m1 int8 [num_blocks, 64, 64]");
  TORCH_CHECK(sf_cache.scalar_type() == torch::kUInt8 && sf_cache.is_contiguous() && sf_cache.dim() == 3 && sf_cache.size(0) == kv_cache.size(0) && sf_cache.size(1) == 64 && sf_cache.size(2) == 4,
              "sf_cache: uint8 UE8M0 [num_blocks, 64, 4]");
  TORCH_CHECK(w.scalar_type() == torch::kBFloat16 && w.size(0) == S && w.size(1) == H && w.is_contiguous(), "weights: bf16 [S, H]");
  TORCH_CHECK(context_lens.scalar_type() == torch::kInt && context_lens.numel() == S, "context_lens: int32 [S]");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt && block_table.is_contiguous() && block_table.size(0) == S && block_table.size(1) == max_pages, "block_table: int32 [S, max_pages]");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat && logits.size(0) == S && logits.is_contiguous(), "logits: fp32 [S, max_context_len]");
  auto st = at::cuda::getCurrentCUDAStream();
  const dim3 block(V6F_WARPS * 32);
  auto args = [&](auto kern, int G) {
    kern<<<dim3((S + V6F_WARPS - 1) / V6F_WARPS, ((int)max_pages + G - 1) / G), block, 0, st>>>(
        reinterpret_cast<const uint8_t*>(q.data_ptr<int8_t>()), sfq.data_ptr<uint8_t>(), reinterpret_cast<const uint8_t*>(kv_cache.data_ptr<int8_t>()),
        sf_cache.data_ptr<uint8_t>(), reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), context_lens.data_ptr<int>(),
        block_table.data_ptr<int>(), logits.data_ptr<float>(), S, H, (int)max_pages, max_ctx);
  };
  if (group == 2) args(k_paged_mqa_logits_v6f<2>, 2);
  else if (group == 4) args(k_paged_mqa_logits_v6f<4>, 4);
  else if (group == 8) args(k_paged_mqa_logits_v6f<8>, 8);
  else TORCH_CHECK(false, "group in {2, 4, 8}");
}

// paged v6g: one page per warp as in v6e, staged as two 32-row halves; the second half's copy is in flight while the first is computed.
__global__ void __launch_bounds__(V3_WARPS * 32)
k_paged_mqa_logits_v6g(const uint8_t* __restrict__ q, const uint8_t* __restrict__ sfq, const uint8_t* __restrict__ kv_cache,
                       const uint8_t* __restrict__ sf_cache, const __nv_bfloat16* __restrict__ w, const int* __restrict__ ctx,
                       const int* __restrict__ block_table, float* __restrict__ logits, int S, int H, int max_pages, int max_ctx) {
  constexpr int HALF = V3_PAGE / 2;
  __shared__ __align__(16) uint8_t s_kv[V3_WARPS][2][HALF * V6P_STRIDE];
  __shared__ __align__(16) uint8_t s_sf[V3_WARPS][2][HALF * 4];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int i = blockIdx.x * V3_WARPS + warp;
  const int p = blockIdx.y;
  if (i >= S) return;
  const int len = ctx[i];
  const int pos0 = p * V3_PAGE;
  if (pos0 >= len) return;
  const int n_valid = min(V3_PAGE, len - pos0);
  const int page = block_table[(size_t)i * max_pages + p];
  auto stage = [&](int hf) {
    const uint8_t* src = kv_cache + ((size_t)page * V3_PAGE + hf * HALF) * 64;
    for (int c = lane; c < HALF * 4; c += 32) {
      const int r = c >> 2, part = c & 3;
      cp_async_16(s_kv[warp][hf] + r * V6P_STRIDE + part * 16, src + (size_t)r * 64 + part * 16);
    }
    if (lane < 8) cp_async_16(s_sf[warp][hf] + lane * 16, sf_cache + ((size_t)page * V3_PAGE + hf * HALF) * 4 + lane * 16);
    asm volatile("cp.async.commit_group;\n");
  };
  const int n_halves = n_valid > HALF ? 2 : 1;
  stage(0);
  const uint8_t* qrow = q + (size_t)i * H * 64;
  float* out = logits + (size_t)i * max_ctx + pos0;
  for (int hf = 0; hf < n_halves; ++hf) {
    if (hf + 1 < n_halves) { stage(hf + 1); asm volatile("cp.async.wait_group 1;\n"); }
    else { asm volatile("cp.async.wait_group 0;\n"); }
    __syncwarp();
    const uint8_t* skv = s_kv[warp][hf];
    const uint8_t* ssf = s_sf[warp][hf];
    const int base = hf * HALF;
    const int n_here = min(HALF, n_valid - base);
    for (int h0 = 0; h0 < H; h0 += 16) {
      const int ha = h0 + g, hb = h0 + g + 8;
      const bool has_a = ha < H, has_b = hb < H;
      const float wa = has_a ? __bfloat162float(w[(size_t)i * H + ha]) : 0.f;
      const float wb = has_b ? __bfloat162float(w[(size_t)i * H + hb]) : 0.f;
      uint32_t af[2][4];
      uint32_t sa[2];
      for (int st = 0; st < 2; ++st) {
        const int b0 = st * 32;
        af[st][0] = has_a ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)ha * 64 + b0 + 4 * t) : 0u;
        af[st][1] = has_b ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)hb * 64 + b0 + 4 * t) : 0u;
        af[st][2] = has_a ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)ha * 64 + b0 + 16 + 4 * t) : 0u;
        af[st][3] = has_b ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)hb * 64 + b0 + 16 + 4 * t) : 0u;
        const int hs = (t == 0) ? ha : hb;
        const bool hv = (t == 0) ? has_a : has_b;
        sa[st] = (t < 2 && hv) ? (uint32_t)sfq[((size_t)i * H + hs) * 4 + 2 * st] | ((uint32_t)sfq[((size_t)i * H + hs) * 4 + 2 * st + 1] << 8)
                               : 0x7F7Fu;
      }
      for (int n0 = 0; n0 < n_here; n0 += 8) {
        const int col = n0 + g;
        const bool has_col = col < n_here;
        const uint8_t* brow = skv + (size_t)(has_col ? col : 0) * V6P_STRIDE;
        const uint8_t* bsf = ssf + (size_t)(has_col ? col : 0) * 4;
        const int c0 = base + n0 + 2 * t, c1 = c0 + 1;
        const bool in0 = (n0 + 2 * t) < n_here, in1 = (n0 + 2 * t + 1) < n_here;
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (int st = 0; st < 2; ++st) {
          const int b0 = st * 32;
          uint32_t bf[2];
          bf[0] = has_col ? *reinterpret_cast<const uint32_t*>(brow + b0 + 4 * t) : 0u;
          bf[1] = has_col ? *reinterpret_cast<const uint32_t*>(brow + b0 + 16 + 4 * t) : 0u;
          const uint32_t sb = (t == 0 && has_col) ? (uint32_t)bsf[2 * st] | ((uint32_t)bsf[2 * st + 1] << 8) : 0x7F7Fu;
          mma_mxf4(acc, af[st], bf, sa[st], sb);
        }
        float v0 = fmaxf(acc[0], 0.f) * wa + fmaxf(acc[2], 0.f) * wb;
        float v1 = fmaxf(acc[1], 0.f) * wa + fmaxf(acc[3], 0.f) * wb;
        const bool odd = (g & 1) != 0;
        const float send = odd ? v0 : v1;
        float keep = odd ? v1 : v0;
        keep += __shfl_xor_sync(0xffffffffu, send, 4);
        keep += __shfl_xor_sync(0xffffffffu, keep, 8);
        keep += __shfl_xor_sync(0xffffffffu, keep, 16);
        if (g < 2) {
          const int c = odd ? c1 : c0;
          if (odd ? in1 : in0) { if (h0 == 0) out[c] = keep; else out[c] += keep; }
        }
      }
    }
  }
}

void fp4_fp4_paged_mqa_logits_sm120_v6g(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv_cache, torch::Tensor sf_cache,
                                        torch::Tensor w, torch::Tensor context_lens, torch::Tensor block_table, torch::Tensor logits,
                                        int64_t max_pages) {
  const int S = (int)q.size(0), H = (int)q.size(1), max_ctx = (int)logits.size(1);
  TORCH_CHECK(q.scalar_type() == torch::kInt8 && q.is_contiguous() && q.dim() == 3 && q.size(2) == 64, "q: packed e2m1 int8 [S, H, 64]");
  TORCH_CHECK(sfq.scalar_type() == torch::kUInt8 && sfq.numel() == (int64_t)S * H * 4 && sfq.is_contiguous(), "sfq: uint8 UE8M0 [S, H, 4]");
  TORCH_CHECK(kv_cache.scalar_type() == torch::kInt8 && kv_cache.is_contiguous() && kv_cache.dim() == 3 && kv_cache.size(1) == 64 && kv_cache.size(2) == 64,
              "kv_cache: packed e2m1 int8 [num_blocks, 64, 64]");
  TORCH_CHECK(sf_cache.scalar_type() == torch::kUInt8 && sf_cache.is_contiguous() && sf_cache.dim() == 3 && sf_cache.size(0) == kv_cache.size(0) && sf_cache.size(1) == 64 && sf_cache.size(2) == 4,
              "sf_cache: uint8 UE8M0 [num_blocks, 64, 4]");
  TORCH_CHECK(w.scalar_type() == torch::kBFloat16 && w.size(0) == S && w.size(1) == H && w.is_contiguous(), "weights: bf16 [S, H]");
  TORCH_CHECK(context_lens.scalar_type() == torch::kInt && context_lens.numel() == S, "context_lens: int32 [S]");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt && block_table.is_contiguous() && block_table.size(0) == S && block_table.size(1) == max_pages, "block_table: int32 [S, max_pages]");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat && logits.size(0) == S && logits.is_contiguous(), "logits: fp32 [S, max_context_len]");
  auto st = at::cuda::getCurrentCUDAStream();
  k_paged_mqa_logits_v6g<<<dim3((S + V3_WARPS - 1) / V3_WARPS, (int)max_pages), V3_WARPS * 32, 0, st>>>(
      reinterpret_cast<const uint8_t*>(q.data_ptr<int8_t>()), sfq.data_ptr<uint8_t>(), reinterpret_cast<const uint8_t*>(kv_cache.data_ptr<int8_t>()),
      sf_cache.data_ptr<uint8_t>(), reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), context_lens.data_ptr<int>(),
      block_table.data_ptr<int>(), logits.data_ptr<float>(), S, H, (int)max_pages, max_ctx);
}

// paged v6h: v6g's half buffers, both halves' copies issued first; q, weights and scales loaded once per head block for both halves.
__global__ void __launch_bounds__(V3_WARPS * 32)
k_paged_mqa_logits_v6h(const uint8_t* __restrict__ q, const uint8_t* __restrict__ sfq, const uint8_t* __restrict__ kv_cache,
                       const uint8_t* __restrict__ sf_cache, const __nv_bfloat16* __restrict__ w, const int* __restrict__ ctx,
                       const int* __restrict__ block_table, float* __restrict__ logits, int S, int H, int max_pages, int max_ctx) {
  constexpr int HALF = V3_PAGE / 2;
  __shared__ __align__(16) uint8_t s_kv[V3_WARPS][2][HALF * V6P_STRIDE];
  __shared__ __align__(16) uint8_t s_sf[V3_WARPS][2][HALF * 4];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int i = blockIdx.x * V3_WARPS + warp;
  const int p = blockIdx.y;
  if (i >= S) return;
  const int len = ctx[i];
  const int pos0 = p * V3_PAGE;
  if (pos0 >= len) return;
  const int n_valid = min(V3_PAGE, len - pos0);
  const int page = block_table[(size_t)i * max_pages + p];
  auto stage = [&](int hf) {
    const uint8_t* src = kv_cache + ((size_t)page * V3_PAGE + hf * HALF) * 64;
    for (int c = lane; c < HALF * 4; c += 32) {
      const int r = c >> 2, part = c & 3;
      cp_async_16(s_kv[warp][hf] + r * V6P_STRIDE + part * 16, src + (size_t)r * 64 + part * 16);
    }
    if (lane < 8) cp_async_16(s_sf[warp][hf] + lane * 16, sf_cache + ((size_t)page * V3_PAGE + hf * HALF) * 4 + lane * 16);
    asm volatile("cp.async.commit_group;\n");
  };
  const int n_halves = n_valid > HALF ? 2 : 1;
  stage(0);
  if (n_halves == 2) stage(1);
  const uint8_t* qrow = q + (size_t)i * H * 64;
  float* out = logits + (size_t)i * max_ctx + pos0;
  for (int h0 = 0; h0 < H; h0 += 16) {
    const int ha = h0 + g, hb = h0 + g + 8;
    const bool has_a = ha < H, has_b = hb < H;
    const float wa = has_a ? __bfloat162float(w[(size_t)i * H + ha]) : 0.f;
    const float wb = has_b ? __bfloat162float(w[(size_t)i * H + hb]) : 0.f;
    uint32_t af[2][4];
    uint32_t sa[2];
    for (int st = 0; st < 2; ++st) {
      const int b0 = st * 32;
      af[st][0] = has_a ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)ha * 64 + b0 + 4 * t) : 0u;
      af[st][1] = has_b ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)hb * 64 + b0 + 4 * t) : 0u;
      af[st][2] = has_a ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)ha * 64 + b0 + 16 + 4 * t) : 0u;
      af[st][3] = has_b ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)hb * 64 + b0 + 16 + 4 * t) : 0u;
      const int hs = (t == 0) ? ha : hb;
      const bool hv = (t == 0) ? has_a : has_b;
      sa[st] = (t < 2 && hv) ? (uint32_t)sfq[((size_t)i * H + hs) * 4 + 2 * st] | ((uint32_t)sfq[((size_t)i * H + hs) * 4 + 2 * st + 1] << 8)
                             : 0x7F7Fu;
    }
    for (int hf = 0; hf < n_halves; ++hf) {
      if (h0 == 0) {   // the first head block waits for each half as it is needed; later blocks find both resident
        if (hf == 0 && n_halves == 2) asm volatile("cp.async.wait_group 1;\n");
        else asm volatile("cp.async.wait_group 0;\n");
        __syncwarp();
      }
      const uint8_t* skv = s_kv[warp][hf];
      const uint8_t* ssf = s_sf[warp][hf];
      const int base = hf * HALF;
      const int n_here = min(HALF, n_valid - base);
      for (int n0 = 0; n0 < n_here; n0 += 8) {
        const int col = n0 + g;
        const bool has_col = col < n_here;
        const uint8_t* brow = skv + (size_t)(has_col ? col : 0) * V6P_STRIDE;
        const uint8_t* bsf = ssf + (size_t)(has_col ? col : 0) * 4;
        const int c0 = base + n0 + 2 * t, c1 = c0 + 1;
        const bool in0 = (n0 + 2 * t) < n_here, in1 = (n0 + 2 * t + 1) < n_here;
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (int st = 0; st < 2; ++st) {
          const int b0 = st * 32;
          uint32_t bf[2];
          bf[0] = has_col ? *reinterpret_cast<const uint32_t*>(brow + b0 + 4 * t) : 0u;
          bf[1] = has_col ? *reinterpret_cast<const uint32_t*>(brow + b0 + 16 + 4 * t) : 0u;
          const uint32_t sb = (t == 0 && has_col) ? (uint32_t)bsf[2 * st] | ((uint32_t)bsf[2 * st + 1] << 8) : 0x7F7Fu;
          mma_mxf4(acc, af[st], bf, sa[st], sb);
        }
        float v0 = fmaxf(acc[0], 0.f) * wa + fmaxf(acc[2], 0.f) * wb;
        float v1 = fmaxf(acc[1], 0.f) * wa + fmaxf(acc[3], 0.f) * wb;
        const bool odd = (g & 1) != 0;
        const float send = odd ? v0 : v1;
        float keep = odd ? v1 : v0;
        keep += __shfl_xor_sync(0xffffffffu, send, 4);
        keep += __shfl_xor_sync(0xffffffffu, keep, 8);
        keep += __shfl_xor_sync(0xffffffffu, keep, 16);
        if (g < 2) {
          const int c = odd ? c1 : c0;
          if (odd ? in1 : in0) { if (h0 == 0) out[c] = keep; else out[c] += keep; }
        }
      }
    }
  }
}

void fp4_fp4_paged_mqa_logits_sm120_v6h(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv_cache, torch::Tensor sf_cache,
                                        torch::Tensor w, torch::Tensor context_lens, torch::Tensor block_table, torch::Tensor logits,
                                        int64_t max_pages) {
  const int S = (int)q.size(0), H = (int)q.size(1), max_ctx = (int)logits.size(1);
  TORCH_CHECK(q.scalar_type() == torch::kInt8 && q.is_contiguous() && q.dim() == 3 && q.size(2) == 64, "q: packed e2m1 int8 [S, H, 64]");
  TORCH_CHECK(sfq.scalar_type() == torch::kUInt8 && sfq.numel() == (int64_t)S * H * 4 && sfq.is_contiguous(), "sfq: uint8 UE8M0 [S, H, 4]");
  TORCH_CHECK(kv_cache.scalar_type() == torch::kInt8 && kv_cache.is_contiguous() && kv_cache.dim() == 3 && kv_cache.size(1) == 64 && kv_cache.size(2) == 64,
              "kv_cache: packed e2m1 int8 [num_blocks, 64, 64]");
  TORCH_CHECK(sf_cache.scalar_type() == torch::kUInt8 && sf_cache.is_contiguous() && sf_cache.dim() == 3 && sf_cache.size(0) == kv_cache.size(0) && sf_cache.size(1) == 64 && sf_cache.size(2) == 4,
              "sf_cache: uint8 UE8M0 [num_blocks, 64, 4]");
  TORCH_CHECK(w.scalar_type() == torch::kBFloat16 && w.size(0) == S && w.size(1) == H && w.is_contiguous(), "weights: bf16 [S, H]");
  TORCH_CHECK(context_lens.scalar_type() == torch::kInt && context_lens.numel() == S, "context_lens: int32 [S]");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt && block_table.is_contiguous() && block_table.size(0) == S && block_table.size(1) == max_pages, "block_table: int32 [S, max_pages]");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat && logits.size(0) == S && logits.is_contiguous(), "logits: fp32 [S, max_context_len]");
  auto st = at::cuda::getCurrentCUDAStream();
  k_paged_mqa_logits_v6h<<<dim3((S + V3_WARPS - 1) / V3_WARPS, (int)max_pages), V3_WARPS * 32, 0, st>>>(
      reinterpret_cast<const uint8_t*>(q.data_ptr<int8_t>()), sfq.data_ptr<uint8_t>(), reinterpret_cast<const uint8_t*>(kv_cache.data_ptr<int8_t>()),
      sf_cache.data_ptr<uint8_t>(), reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), context_lens.data_ptr<int>(),
      block_table.data_ptr<int>(), logits.data_ptr<float>(), S, H, (int)max_pages, max_ctx);
}

// paged v6i: quarter-page double buffers (16 rows each), so four 8-warp blocks fit an SM; quarters streamed inside each head block.
__global__ void __launch_bounds__(V3_WARPS * 32)
k_paged_mqa_logits_v6i(const uint8_t* __restrict__ q, const uint8_t* __restrict__ sfq, const uint8_t* __restrict__ kv_cache,
                       const uint8_t* __restrict__ sf_cache, const __nv_bfloat16* __restrict__ w, const int* __restrict__ ctx,
                       const int* __restrict__ block_table, float* __restrict__ logits, int S, int H, int max_pages, int max_ctx) {
  constexpr int QR = V3_PAGE / 4;
  __shared__ __align__(16) uint8_t s_kv[V3_WARPS][2][QR * V6P_STRIDE];
  __shared__ __align__(16) uint8_t s_sf[V3_WARPS][2][QR * 4];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int i = blockIdx.x * V3_WARPS + warp;
  const int p = blockIdx.y;
  if (i >= S) return;
  const int len = ctx[i];
  const int pos0 = p * V3_PAGE;
  if (pos0 >= len) return;
  const int n_valid = min(V3_PAGE, len - pos0);
  const int page = block_table[(size_t)i * max_pages + p];
  const int nq = (n_valid + QR - 1) / QR;
  auto stage = [&](int qi, int buf) {
    const uint8_t* src = kv_cache + ((size_t)page * V3_PAGE + qi * QR) * 64;
    for (int c = lane; c < QR * 4; c += 32) {
      const int r = c >> 2, part = c & 3;
      cp_async_16(s_kv[warp][buf] + r * V6P_STRIDE + part * 16, src + (size_t)r * 64 + part * 16);
    }
    if (lane < 4) cp_async_16(s_sf[warp][buf] + lane * 16, sf_cache + ((size_t)page * V3_PAGE + qi * QR) * 4 + lane * 16);
    asm volatile("cp.async.commit_group;\n");
  };
  const uint8_t* qrow = q + (size_t)i * H * 64;
  float* out = logits + (size_t)i * max_ctx + pos0;
  for (int h0 = 0; h0 < H; h0 += 16) {
    const int ha = h0 + g, hb = h0 + g + 8;
    const bool has_a = ha < H, has_b = hb < H;
    const float wa = has_a ? __bfloat162float(w[(size_t)i * H + ha]) : 0.f;
    const float wb = has_b ? __bfloat162float(w[(size_t)i * H + hb]) : 0.f;
    uint32_t af[2][4];
    uint32_t sa[2];
    for (int st = 0; st < 2; ++st) {
      const int b0 = st * 32;
      af[st][0] = has_a ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)ha * 64 + b0 + 4 * t) : 0u;
      af[st][1] = has_b ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)hb * 64 + b0 + 4 * t) : 0u;
      af[st][2] = has_a ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)ha * 64 + b0 + 16 + 4 * t) : 0u;
      af[st][3] = has_b ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)hb * 64 + b0 + 16 + 4 * t) : 0u;
      const int hs = (t == 0) ? ha : hb;
      const bool hv = (t == 0) ? has_a : has_b;
      sa[st] = (t < 2 && hv) ? (uint32_t)sfq[((size_t)i * H + hs) * 4 + 2 * st] | ((uint32_t)sfq[((size_t)i * H + hs) * 4 + 2 * st + 1] << 8)
                             : 0x7F7Fu;
    }
    stage(0, 0);
    for (int qi = 0; qi < nq; ++qi) {
      const int buf = qi & 1;
      if (qi + 1 < nq) { stage(qi + 1, buf ^ 1); asm volatile("cp.async.wait_group 1;\n"); }
      else { asm volatile("cp.async.wait_group 0;\n"); }
      __syncwarp();
      const uint8_t* skv = s_kv[warp][buf];
      const uint8_t* ssf = s_sf[warp][buf];
      const int base = qi * QR;
      const int n_here = min(QR, n_valid - base);
      for (int n0 = 0; n0 < n_here; n0 += 8) {
        const int col = n0 + g;
        const bool has_col = col < n_here;
        const uint8_t* brow = skv + (size_t)(has_col ? col : 0) * V6P_STRIDE;
        const uint8_t* bsf = ssf + (size_t)(has_col ? col : 0) * 4;
        const int c0 = base + n0 + 2 * t, c1 = c0 + 1;
        const bool in0 = (n0 + 2 * t) < n_here, in1 = (n0 + 2 * t + 1) < n_here;
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (int st = 0; st < 2; ++st) {
          const int b0 = st * 32;
          uint32_t bf[2];
          bf[0] = has_col ? *reinterpret_cast<const uint32_t*>(brow + b0 + 4 * t) : 0u;
          bf[1] = has_col ? *reinterpret_cast<const uint32_t*>(brow + b0 + 16 + 4 * t) : 0u;
          const uint32_t sb = (t == 0 && has_col) ? (uint32_t)bsf[2 * st] | ((uint32_t)bsf[2 * st + 1] << 8) : 0x7F7Fu;
          mma_mxf4(acc, af[st], bf, sa[st], sb);
        }
        float v0 = fmaxf(acc[0], 0.f) * wa + fmaxf(acc[2], 0.f) * wb;
        float v1 = fmaxf(acc[1], 0.f) * wa + fmaxf(acc[3], 0.f) * wb;
        const bool odd = (g & 1) != 0;
        const float send = odd ? v0 : v1;
        float keep = odd ? v1 : v0;
        keep += __shfl_xor_sync(0xffffffffu, send, 4);
        keep += __shfl_xor_sync(0xffffffffu, keep, 8);
        keep += __shfl_xor_sync(0xffffffffu, keep, 16);
        if (g < 2) {
          const int c = odd ? c1 : c0;
          if (odd ? in1 : in0) { if (h0 == 0) out[c] = keep; else out[c] += keep; }
        }
      }
      __syncwarp();   // this buffer is refilled two quarters from now; every lane is past it
    }
  }
}

void fp4_fp4_paged_mqa_logits_sm120_v6i(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv_cache, torch::Tensor sf_cache,
                                        torch::Tensor w, torch::Tensor context_lens, torch::Tensor block_table, torch::Tensor logits,
                                        int64_t max_pages) {
  const int S = (int)q.size(0), H = (int)q.size(1), max_ctx = (int)logits.size(1);
  TORCH_CHECK(q.scalar_type() == torch::kInt8 && q.is_contiguous() && q.dim() == 3 && q.size(2) == 64, "q: packed e2m1 int8 [S, H, 64]");
  TORCH_CHECK(sfq.scalar_type() == torch::kUInt8 && sfq.numel() == (int64_t)S * H * 4 && sfq.is_contiguous(), "sfq: uint8 UE8M0 [S, H, 4]");
  TORCH_CHECK(kv_cache.scalar_type() == torch::kInt8 && kv_cache.is_contiguous() && kv_cache.dim() == 3 && kv_cache.size(1) == 64 && kv_cache.size(2) == 64,
              "kv_cache: packed e2m1 int8 [num_blocks, 64, 64]");
  TORCH_CHECK(sf_cache.scalar_type() == torch::kUInt8 && sf_cache.is_contiguous() && sf_cache.dim() == 3 && sf_cache.size(0) == kv_cache.size(0) && sf_cache.size(1) == 64 && sf_cache.size(2) == 4,
              "sf_cache: uint8 UE8M0 [num_blocks, 64, 4]");
  TORCH_CHECK(w.scalar_type() == torch::kBFloat16 && w.size(0) == S && w.size(1) == H && w.is_contiguous(), "weights: bf16 [S, H]");
  TORCH_CHECK(context_lens.scalar_type() == torch::kInt && context_lens.numel() == S, "context_lens: int32 [S]");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt && block_table.is_contiguous() && block_table.size(0) == S && block_table.size(1) == max_pages, "block_table: int32 [S, max_pages]");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat && logits.size(0) == S && logits.is_contiguous(), "logits: fp32 [S, max_context_len]");
  auto st = at::cuda::getCurrentCUDAStream();
  k_paged_mqa_logits_v6i<<<dim3((S + V3_WARPS - 1) / V3_WARPS, (int)max_pages), V3_WARPS * 32, 0, st>>>(
      reinterpret_cast<const uint8_t*>(q.data_ptr<int8_t>()), sfq.data_ptr<uint8_t>(), reinterpret_cast<const uint8_t*>(kv_cache.data_ptr<int8_t>()),
      sf_cache.data_ptr<uint8_t>(), reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), context_lens.data_ptr<int>(),
      block_table.data_ptr<int>(), logits.data_ptr<float>(), S, H, (int)max_pages, max_ctx);
}
// paged v6j: v6i's quarter-page double buffers with the per-quarter integer work cut: a full quarter (16 valid rows, every quarter but a short
// page's last) runs its two 8-column tiles unrolled with no column masks; only a partial quarter takes the masked path; staging is unrolled.
template <bool FULL>
__device__ __forceinline__ void v6j_tile(const uint8_t* skv, const uint8_t* ssf, int n0, int n_here, int base, int g, int t,
                                         const uint32_t (&af)[2][4], const uint32_t (&sa)[2], float wa, float wb, bool first, float* out) {
  const int col = n0 + g;
  const bool has_col = FULL || col < n_here;
  const uint8_t* brow = skv + (size_t)(has_col ? col : 0) * V6P_STRIDE;
  const uint8_t* bsf = ssf + (size_t)(has_col ? col : 0) * 4;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int st = 0; st < 2; ++st) {
    const int b0 = st * 32;
    uint32_t bf[2];
    bf[0] = has_col ? *reinterpret_cast<const uint32_t*>(brow + b0 + 4 * t) : 0u;
    bf[1] = has_col ? *reinterpret_cast<const uint32_t*>(brow + b0 + 16 + 4 * t) : 0u;
    const uint32_t sb = (t == 0 && has_col) ? (uint32_t)bsf[2 * st] | ((uint32_t)bsf[2 * st + 1] << 8) : 0x7F7Fu;
    mma_mxf4(acc, af[st], bf, sa[st], sb);
  }
  float v0 = fmaxf(acc[0], 0.f) * wa + fmaxf(acc[2], 0.f) * wb;
  float v1 = fmaxf(acc[1], 0.f) * wa + fmaxf(acc[3], 0.f) * wb;
  const bool odd = (g & 1) != 0;
  const float send = odd ? v0 : v1;
  float keep = odd ? v1 : v0;
  keep += __shfl_xor_sync(0xffffffffu, send, 4);
  keep += __shfl_xor_sync(0xffffffffu, keep, 8);
  keep += __shfl_xor_sync(0xffffffffu, keep, 16);
  if (g < 2) {
    const int cl = n0 + 2 * t + (odd ? 1 : 0);
    if (FULL || cl < n_here) { if (first) out[base + cl] = keep; else out[base + cl] += keep; }
  }
}

__global__ void __launch_bounds__(V3_WARPS * 32)
k_paged_mqa_logits_v6j(const uint8_t* __restrict__ q, const uint8_t* __restrict__ sfq, const uint8_t* __restrict__ kv_cache,
                       const uint8_t* __restrict__ sf_cache, const __nv_bfloat16* __restrict__ w, const int* __restrict__ ctx,
                       const int* __restrict__ block_table, float* __restrict__ logits, int S, int H, int max_pages, int max_ctx) {
  constexpr int QR = V3_PAGE / 4;
  static_assert(QR * 4 == 64, "staging assumes two 16-byte copies per lane per quarter");
  __shared__ __align__(16) uint8_t s_kv[V3_WARPS][2][QR * V6P_STRIDE];
  __shared__ __align__(16) uint8_t s_sf[V3_WARPS][2][QR * 4];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int i = blockIdx.x * V3_WARPS + warp;
  const int p = blockIdx.y;
  if (i >= S) return;
  const int len = ctx[i];
  const int pos0 = p * V3_PAGE;
  if (pos0 >= len) return;
  const int n_valid = min(V3_PAGE, len - pos0);
  const int page = block_table[(size_t)i * max_pages + p];
  const int nq = (n_valid + QR - 1) / QR;
  const uint8_t* kv_page = kv_cache + (size_t)page * V3_PAGE * 64;
  const uint8_t* sf_page = sf_cache + (size_t)page * V3_PAGE * 4;
  // lane's two 16-byte pieces of a quarter: rows lane>>2 and 8 + (lane>>2), part lane&3
  const int r0 = lane >> 2, part = lane & 3;
  const int dst0 = r0 * V6P_STRIDE + part * 16, dst1 = (r0 + 8) * V6P_STRIDE + part * 16;
  const int src0 = r0 * 64 + part * 16, src1 = (r0 + 8) * 64 + part * 16;
  auto stage = [&](int qi, int buf) {
    const uint8_t* src = kv_page + qi * QR * 64;
    cp_async_16(s_kv[warp][buf] + dst0, src + src0);
    cp_async_16(s_kv[warp][buf] + dst1, src + src1);
    if (lane < 4) cp_async_16(s_sf[warp][buf] + lane * 16, sf_page + qi * QR * 4 + lane * 16);
    asm volatile("cp.async.commit_group;\n");
  };
  const uint8_t* qrow = q + (size_t)i * H * 64;
  float* out = logits + (size_t)i * max_ctx + pos0;
  for (int h0 = 0; h0 < H; h0 += 16) {
    const int ha = h0 + g, hb = h0 + g + 8;
    const bool has_a = ha < H, has_b = hb < H;
    const float wa = has_a ? __bfloat162float(w[(size_t)i * H + ha]) : 0.f;
    const float wb = has_b ? __bfloat162float(w[(size_t)i * H + hb]) : 0.f;
    uint32_t af[2][4];
    uint32_t sa[2];
#pragma unroll
    for (int st = 0; st < 2; ++st) {
      const int b0 = st * 32;
      af[st][0] = has_a ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)ha * 64 + b0 + 4 * t) : 0u;
      af[st][1] = has_b ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)hb * 64 + b0 + 4 * t) : 0u;
      af[st][2] = has_a ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)ha * 64 + b0 + 16 + 4 * t) : 0u;
      af[st][3] = has_b ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)hb * 64 + b0 + 16 + 4 * t) : 0u;
      const int hs = (t == 0) ? ha : hb;
      const bool hv = (t == 0) ? has_a : has_b;
      sa[st] = (t < 2 && hv) ? (uint32_t)sfq[((size_t)i * H + hs) * 4 + 2 * st] | ((uint32_t)sfq[((size_t)i * H + hs) * 4 + 2 * st + 1] << 8)
                             : 0x7F7Fu;
    }
    const bool first = (h0 == 0);
    stage(0, 0);
    for (int qi = 0; qi < nq; ++qi) {
      const int buf = qi & 1;
      if (qi + 1 < nq) { stage(qi + 1, buf ^ 1); asm volatile("cp.async.wait_group 1;\n"); }
      else { asm volatile("cp.async.wait_group 0;\n"); }
      __syncwarp();
      const uint8_t* skv = s_kv[warp][buf];
      const uint8_t* ssf = s_sf[warp][buf];
      const int base = qi * QR;
      const int n_here = n_valid - base;
      if (n_here >= QR) {
        v6j_tile<true>(skv, ssf, 0, QR, base, g, t, af, sa, wa, wb, first, out);
        v6j_tile<true>(skv, ssf, 8, QR, base, g, t, af, sa, wa, wb, first, out);
      } else {
        for (int n0 = 0; n0 < n_here; n0 += 8) v6j_tile<false>(skv, ssf, n0, n_here, base, g, t, af, sa, wa, wb, first, out);
      }
      __syncwarp();   // this buffer is refilled two quarters from now; every lane is past it
    }
  }
}

void fp4_fp4_paged_mqa_logits_sm120_v6j(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv_cache, torch::Tensor sf_cache,
                                        torch::Tensor w, torch::Tensor context_lens, torch::Tensor block_table, torch::Tensor logits,
                                        int64_t max_pages) {
  const int S = (int)q.size(0), H = (int)q.size(1), max_ctx = (int)logits.size(1);
  TORCH_CHECK(q.scalar_type() == torch::kInt8 && q.is_contiguous() && q.dim() == 3 && q.size(2) == 64, "q: packed e2m1 int8 [S, H, 64]");
  TORCH_CHECK(sfq.scalar_type() == torch::kUInt8 && sfq.numel() == (int64_t)S * H * 4 && sfq.is_contiguous(), "sfq: uint8 UE8M0 [S, H, 4]");
  TORCH_CHECK(kv_cache.scalar_type() == torch::kInt8 && kv_cache.is_contiguous() && kv_cache.dim() == 3 && kv_cache.size(1) == 64 && kv_cache.size(2) == 64,
              "kv_cache: packed e2m1 int8 [num_blocks, 64, 64]");
  TORCH_CHECK(sf_cache.scalar_type() == torch::kUInt8 && sf_cache.is_contiguous() && sf_cache.dim() == 3 && sf_cache.size(0) == kv_cache.size(0) && sf_cache.size(1) == 64 && sf_cache.size(2) == 4,
              "sf_cache: uint8 UE8M0 [num_blocks, 64, 4]");
  TORCH_CHECK(w.scalar_type() == torch::kBFloat16 && w.size(0) == S && w.size(1) == H && w.is_contiguous(), "weights: bf16 [S, H]");
  TORCH_CHECK(context_lens.scalar_type() == torch::kInt && context_lens.numel() == S, "context_lens: int32 [S]");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt && block_table.is_contiguous() && block_table.size(0) == S && block_table.size(1) == max_pages, "block_table: int32 [S, max_pages]");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat && logits.size(0) == S && logits.is_contiguous(), "logits: fp32 [S, max_context_len]");
  auto st = at::cuda::getCurrentCUDAStream();
  k_paged_mqa_logits_v6j<<<dim3((S + V3_WARPS - 1) / V3_WARPS, (int)max_pages), V3_WARPS * 32, 0, st>>>(
      reinterpret_cast<const uint8_t*>(q.data_ptr<int8_t>()), sfq.data_ptr<uint8_t>(), reinterpret_cast<const uint8_t*>(kv_cache.data_ptr<int8_t>()),
      sf_cache.data_ptr<uint8_t>(), reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), context_lens.data_ptr<int>(),
      block_table.data_ptr<int>(), logits.data_ptr<float>(), S, H, (int)max_pages, max_ctx);
}
// paged v6k: v6j's body over a compacted work list. The (row, page) grid of v6j launches max_pages blocks per 8 rows, and a warp whose page starts past
// its row's context returns at once while its block keeps the SM slot; here warp w takes the w-th live (row, page) pair in row-major order, found by a
// binary search over the inclusive prefix sum of the rows' page counts, so consecutive warps share a row and every block but the last is full. The grid
// is sized by S x max_pages (no host sync); blocks past the live total return at once.
__global__ void __launch_bounds__(V3_WARPS * 32)
k_paged_mqa_logits_v6k(const uint8_t* __restrict__ q, const uint8_t* __restrict__ sfq, const uint8_t* __restrict__ kv_cache,
                       const uint8_t* __restrict__ sf_cache, const __nv_bfloat16* __restrict__ w, const int* __restrict__ ctx,
                       const int* __restrict__ block_table, const int* __restrict__ page_offs, float* __restrict__ logits,
                       int S, int H, int max_pages, int max_ctx) {
  constexpr int QR = V3_PAGE / 4;
  static_assert(QR * 4 == 64, "staging assumes two 16-byte copies per lane per quarter");
  __shared__ __align__(16) uint8_t s_kv[V3_WARPS][2][QR * V6P_STRIDE];
  __shared__ __align__(16) uint8_t s_sf[V3_WARPS][2][QR * 4];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wi = blockIdx.x * V3_WARPS + warp;
  if (wi >= page_offs[S - 1]) return;
  int lo = 0, hi = S - 1;                       // smallest i with page_offs[i] > wi
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (page_offs[mid] > wi) hi = mid; else lo = mid + 1;
  }
  const int i = lo;
  const int p = wi - (i > 0 ? page_offs[i - 1] : 0);
  const int len = ctx[i];
  const int pos0 = p * V3_PAGE;
  if (pos0 >= len) return;
  const int n_valid = min(V3_PAGE, len - pos0);
  const int page = block_table[(size_t)i * max_pages + p];
  const int nq = (n_valid + QR - 1) / QR;
  const uint8_t* kv_page = kv_cache + (size_t)page * V3_PAGE * 64;
  const uint8_t* sf_page = sf_cache + (size_t)page * V3_PAGE * 4;
  const int r0 = lane >> 2, part = lane & 3;
  const int dst0 = r0 * V6P_STRIDE + part * 16, dst1 = (r0 + 8) * V6P_STRIDE + part * 16;
  const int src0 = r0 * 64 + part * 16, src1 = (r0 + 8) * 64 + part * 16;
  auto stage = [&](int qi, int buf) {
    const uint8_t* src = kv_page + qi * QR * 64;
    cp_async_16(s_kv[warp][buf] + dst0, src + src0);
    cp_async_16(s_kv[warp][buf] + dst1, src + src1);
    if (lane < 4) cp_async_16(s_sf[warp][buf] + lane * 16, sf_page + qi * QR * 4 + lane * 16);
    asm volatile("cp.async.commit_group;\n");
  };
  const uint8_t* qrow = q + (size_t)i * H * 64;
  float* out = logits + (size_t)i * max_ctx + pos0;
  for (int h0 = 0; h0 < H; h0 += 16) {
    const int ha = h0 + g, hb = h0 + g + 8;
    const bool has_a = ha < H, has_b = hb < H;
    const float wa = has_a ? __bfloat162float(w[(size_t)i * H + ha]) : 0.f;
    const float wb = has_b ? __bfloat162float(w[(size_t)i * H + hb]) : 0.f;
    uint32_t af[2][4];
    uint32_t sa[2];
#pragma unroll
    for (int st = 0; st < 2; ++st) {
      const int b0 = st * 32;
      af[st][0] = has_a ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)ha * 64 + b0 + 4 * t) : 0u;
      af[st][1] = has_b ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)hb * 64 + b0 + 4 * t) : 0u;
      af[st][2] = has_a ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)ha * 64 + b0 + 16 + 4 * t) : 0u;
      af[st][3] = has_b ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)hb * 64 + b0 + 16 + 4 * t) : 0u;
      const int hs = (t == 0) ? ha : hb;
      const bool hv = (t == 0) ? has_a : has_b;
      sa[st] = (t < 2 && hv) ? (uint32_t)sfq[((size_t)i * H + hs) * 4 + 2 * st] | ((uint32_t)sfq[((size_t)i * H + hs) * 4 + 2 * st + 1] << 8)
                             : 0x7F7Fu;
    }
    const bool first = (h0 == 0);
    stage(0, 0);
    for (int qi = 0; qi < nq; ++qi) {
      const int buf = qi & 1;
      if (qi + 1 < nq) { stage(qi + 1, buf ^ 1); asm volatile("cp.async.wait_group 1;\n"); }
      else { asm volatile("cp.async.wait_group 0;\n"); }
      __syncwarp();
      const uint8_t* skv = s_kv[warp][buf];
      const uint8_t* ssf = s_sf[warp][buf];
      const int base = qi * QR;
      const int n_here = n_valid - base;
      if (n_here >= QR) {
        v6j_tile<true>(skv, ssf, 0, QR, base, g, t, af, sa, wa, wb, first, out);
        v6j_tile<true>(skv, ssf, 8, QR, base, g, t, af, sa, wa, wb, first, out);
      } else {
        for (int n0 = 0; n0 < n_here; n0 += 8) v6j_tile<false>(skv, ssf, n0, n_here, base, g, t, af, sa, wa, wb, first, out);
      }
      __syncwarp();   // this buffer is refilled two quarters from now; every lane is past it
    }
  }
}

void fp4_fp4_paged_mqa_logits_sm120_v6k(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv_cache, torch::Tensor sf_cache,
                                        torch::Tensor w, torch::Tensor context_lens, torch::Tensor block_table, torch::Tensor logits,
                                        int64_t max_pages, torch::Tensor page_offs) {
  const int S = (int)q.size(0), H = (int)q.size(1), max_ctx = (int)logits.size(1);
  TORCH_CHECK(q.scalar_type() == torch::kInt8 && q.is_contiguous() && q.dim() == 3 && q.size(2) == 64, "q: packed e2m1 int8 [S, H, 64]");
  TORCH_CHECK(sfq.scalar_type() == torch::kUInt8 && sfq.numel() == (int64_t)S * H * 4 && sfq.is_contiguous(), "sfq: uint8 UE8M0 [S, H, 4]");
  TORCH_CHECK(kv_cache.scalar_type() == torch::kInt8 && kv_cache.is_contiguous() && kv_cache.dim() == 3 && kv_cache.size(1) == 64 && kv_cache.size(2) == 64,
              "kv_cache: packed e2m1 int8 [num_blocks, 64, 64]");
  TORCH_CHECK(sf_cache.scalar_type() == torch::kUInt8 && sf_cache.is_contiguous() && sf_cache.dim() == 3 && sf_cache.size(0) == kv_cache.size(0) && sf_cache.size(1) == 64 && sf_cache.size(2) == 4,
              "sf_cache: uint8 UE8M0 [num_blocks, 64, 4]");
  TORCH_CHECK(w.scalar_type() == torch::kBFloat16 && w.size(0) == S && w.size(1) == H && w.is_contiguous(), "weights: bf16 [S, H]");
  TORCH_CHECK(context_lens.scalar_type() == torch::kInt && context_lens.numel() == S && context_lens.is_contiguous(), "context_lens: int32 [S]");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt && block_table.is_contiguous() && block_table.size(0) == S && block_table.size(1) == max_pages, "block_table: int32 [S, max_pages]");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat && logits.size(0) == S && logits.is_contiguous(), "logits: fp32 [S, max_context_len]");
  TORCH_CHECK(S >= 1, "at least one row");
  TORCH_CHECK(page_offs.scalar_type() == torch::kInt && page_offs.is_contiguous() && page_offs.numel() == S,
              "page_offs: int32 [S], from fp4_fp4_paged_mqa_logits_sm120_v6k_meta");
  const int64_t blocks = ((int64_t)S * max_pages + V3_WARPS - 1) / V3_WARPS;
  TORCH_CHECK(blocks < (int64_t)INT32_MAX, "grid too large");
  auto st = at::cuda::getCurrentCUDAStream();
  k_paged_mqa_logits_v6k<<<dim3((unsigned)blocks), V3_WARPS * 32, 0, st>>>(
      reinterpret_cast<const uint8_t*>(q.data_ptr<int8_t>()), sfq.data_ptr<uint8_t>(), reinterpret_cast<const uint8_t*>(kv_cache.data_ptr<int8_t>()),
      sf_cache.data_ptr<uint8_t>(), reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), context_lens.data_ptr<int>(),
      block_table.data_ptr<int>(), page_offs.data_ptr<int>(), logits.data_ptr<float>(), S, H, (int)max_pages, max_ctx);
}
// v6k's work list: the inclusive prefix sum of each row's page count (clamped to max_pages as v6j's grid clamps it). Computed once per batch and
// shared by every layer's call, as an engine computes its paged-attention schedule once per step.
torch::Tensor fp4_fp4_paged_mqa_logits_sm120_v6k_meta(torch::Tensor context_lens, int64_t max_pages) {
  TORCH_CHECK(context_lens.scalar_type() == torch::kInt && context_lens.is_contiguous(), "context_lens: int32 [S]");
  auto pages = at::clamp(at::floor_divide(context_lens + (V3_PAGE - 1), V3_PAGE), 0, max_pages);
  return at::cumsum(pages, 0, at::kInt).contiguous();
}
"""


def build(verbose: bool = False):
    return load_inline(name="sm120fp4_fp4_fp4_mqa_logits_v6", cpp_sources=base.CPP + v4.CPP_V4 + CPP_V6,
                       cuda_sources=base.CUDA + v4.CUDA_V4 + CUDA_V6,
                       functions=["fp8_fp4_mqa_logits_sm120_v4", "fp4_fp4_mqa_logits_sm120_v6", "fp4_fp4_mqa_logits_sm120_v6s", "fp4_fp4_mqa_logits_sm120_v6e", "fp8_fp4_paged_mqa_logits_sm120_v4",
                                  "fp4_fp4_paged_mqa_logits_sm120_v6", "fp4_fp4_paged_mqa_logits_sm120_v6e",
                                  "fp4_fp4_paged_mqa_logits_sm120_v6f", "fp4_fp4_paged_mqa_logits_sm120_v6g",
                                  "fp4_fp4_paged_mqa_logits_sm120_v6h", "fp4_fp4_paged_mqa_logits_sm120_v6i", "fp4_fp4_paged_mqa_logits_sm120_v6j", "fp4_fp4_paged_mqa_logits_sm120_v6k", "fp4_fp4_paged_mqa_logits_sm120_v6k_meta"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a"], verbose=verbose)


def quantize_q_blocked(q: torch.Tensor):
    """q [S, H, 128] -> packed e2m1 int8 [S, H, 64], UE8M0 [S, H, 4] (one per 32 columns), and the fp32 scales [S * H, 4]."""
    S, H, D = q.shape
    q4, sf = ref.per_token_cast_to_fp4(q.reshape(S * H, D), use_ue8m0=True, gran_k=BLOCK, use_packed_ue8m0=False)
    sf_u8 = (torch.round(torch.log2(sf.float())) + 127).clamp(0, 255).to(torch.uint8)
    return q4.reshape(S, H, D // 2).contiguous(), sf_u8.reshape(S, H, NBLK).contiguous(), sf.float()


def reference_v6(q4, sfq_f, kv4, sfkv_f, w, ks, ke, max_k):
    S, H, _ = q4.shape
    qf = ref.cast_back_from_fp4(q4.reshape(S * H, HEAD_DIM // 2), sfq_f, gran_k=BLOCK).reshape(S, H, HEAD_DIM)
    kf = ref.cast_back_from_fp4(kv4, sfkv_f, gran_k=BLOCK)
    score = torch.einsum("mhd,nd->hmn", qf, kf)
    logits_full = torch.einsum("hmn,mh->mn", score.relu(), w.float())
    out = torch.full((S, max_k), float("-inf"), device=q4.device, dtype=torch.float32)
    for i in range(S):
        a, b = int(ks[i]), int(ke[i])
        out[i, : b - a] = logits_full[i, a:b]
    return out


def make_case(S, N, H, seed, dev, full_span=False):
    q, kv, w, ks, ke = v4.make_case(S, N, H, seed, dev, full_span, block_spread=True)
    # spread q's four 32-column blocks too, so its block scales differ within a row
    g = torch.Generator().manual_seed(seed + 13)
    mult = torch.pow(2.0, torch.randint(-3, 4, (S, H, NBLK), generator=g).float()).to(dev).repeat_interleave(BLOCK, dim=2)
    return q * mult, kv, w, ks, ke


def launch_v6(mod, q4, sfq_u8, kv4, sfkv_u8, w, ks, ke, out, plan=None, span=None, padded=False, packed=False):
    lo, hi = span or base.span_of(ks, ke)
    rows, kvseg, group = plan or base.plan_v2(q4.shape[0], hi - lo, v4.sm_count())
    fn = mod.fp4_fp4_mqa_logits_sm120_v6e if packed else (mod.fp4_fp4_mqa_logits_sm120_v6s if padded else mod.fp4_fp4_mqa_logits_sm120_v6)
    fn(q4, sfq_u8, kv4, sfkv_u8, w, ks, ke, out, lo, hi, rows, kvseg, group)


def run_case(mod, S, N, H, seed, dev, full_span=False):
    q, kv, w, ks, ke = make_case(S, N, H, seed, dev, full_span)
    q4, sfq_u8, sfq_f = quantize_q_blocked(q)
    kv4, sfkv_u8, sfkv_f = v4.quantize_kv_blocked(kv)
    max_k = int((ke - ks).max())
    out = torch.full((S, max_k), float("-inf"), device=dev, dtype=torch.float32)
    launch_v6(mod, q4, sfq_u8, kv4, sfkv_u8, w, ks, ke, out)
    torch.cuda.synchronize()
    exact = reference_v6(q4, sfq_f, kv4, sfkv_f, w, ks, ke, max_k)
    valid = torch.isfinite(exact)
    diff = (out[valid] - exact[valid]).abs()
    scale = exact[valid].abs().max().clamp_min(1e-30)
    distinct_q_scales = int(sfq_u8.reshape(-1, NBLK).unique(dim=0).shape[0])
    res = {"S": S, "N": N, "H": H, "max_k": max_k, "max_abs_err": float(diff.max()), "ref_abs_max": float(scale),
           "rel_max_err": float(diff.max() / scale), "untouched_outside_span": bool(torch.equal(torch.isfinite(out), valid)),
           "q_rows_with_unequal_block_scales": int((sfq_u8.reshape(-1, NBLK) != sfq_u8.reshape(-1, NBLK)[:, :1]).any(dim=1).sum()),
           "k_rows_with_unequal_block_scales": int((sfkv_u8 != sfkv_u8[:, :1]).any(dim=1).sum()), "distinct_q_scale_patterns": distinct_q_scales}
    res["pass"] = res["rel_max_err"] < 1e-5 and res["untouched_outside_span"]
    return res, (q4, sfq_u8, kv4, sfkv_u8, w, ks, ke, out)


def selftest(mod, dev) -> int:
    ok = True
    print("v6 (MXFP4 q and k, block-scaled k64 MMA) against the dequantised reference, block scales independent on both operands")
    for (S, N, H, seed) in ((16, 512, 8, 1), (32, 1024, 16, 2), (64, 4096, 8, 3), (8, 256, 32, 4), (48, 2048, 8, 5)):
        r, _ = run_case(mod, S, N, H, seed, dev)
        ok &= r["pass"]
        print(f"  S={S} N={N} H={H}: rel max err {r['rel_max_err']:.2e}, outside span untouched {r['untouched_outside_span']}, "
              f"rows with unequal block scales q {r['q_rows_with_unequal_block_scales']} of {S * H}, k {r['k_rows_with_unequal_block_scales']} of {N} "
              f"-> {'ok' if r['pass'] else 'FAIL'}", flush=True)
    print("v6e (packed epilogue) against v6, bit for bit")
    for (S, N, H, seed) in ((16, 512, 8, 41), (64, 4096, 8, 42), (8, 256, 32, 43), (48, 2048, 8, 44), (32, 1000, 16, 45)):
        _, args = run_case(mod, S, N, H, seed, dev)
        q4_, sfq_, kv4_, sfkv_, w_, ks_, ke_, out_ = args
        o2 = torch.full_like(out_, float("-inf"))
        launch_v6(mod, q4_, sfq_, kv4_, sfkv_, w_, ks_, ke_, o2, packed=True)
        torch.cuda.synchronize()
        same = bool(torch.equal(out_, o2))
        ok &= same
        print(f"  S={S} N={N} H={H}: bit-identical {same}", flush=True)
    print("v6s (80-byte row stride) against v6, bit for bit")
    for (S, N, H, seed) in ((16, 512, 8, 31), (64, 4096, 8, 32), (8, 256, 32, 33)):
        _, args = run_case(mod, S, N, H, seed, dev)
        q4_, sfq_, kv4_, sfkv_, w_, ks_, ke_, out_ = args
        o2 = torch.full_like(out_, float("-inf"))
        launch_v6(mod, q4_, sfq_, kv4_, sfkv_, w_, ks_, ke_, o2, padded=True)
        torch.cuda.synchronize()
        same = bool(torch.equal(out_, o2))
        ok &= same
        print(f"  S={S} N={N} H={H}: bit-identical {same}", flush=True)
    # a fault check the selftest must be able to fail: swap the two scale bytes of every q block pair and expect a wrong answer
    q, kv, w, ks, ke = make_case(16, 512, 8, 21, dev)
    q4, sfq_u8, sfq_f = quantize_q_blocked(q)
    kv4, sfkv_u8, sfkv_f = v4.quantize_kv_blocked(kv)
    max_k = int((ke - ks).max())
    out = torch.full((16, max_k), float("-inf"), device=dev, dtype=torch.float32)
    launch_v6(mod, q4, sfq_u8[..., [1, 0, 3, 2]].contiguous(), kv4, sfkv_u8, w, ks, ke, out)
    torch.cuda.synchronize()
    exact = reference_v6(q4, sfq_f, kv4, sfkv_f, w, ks, ke, max_k)
    valid = torch.isfinite(exact)
    rel = float((out[valid] - exact[valid]).abs().max() / exact[valid].abs().max())
    caught = rel > 1e-3
    ok &= caught
    print(f"  fault check, q block scales swapped within each step: rel max err {rel:.2e} -> {'caught' if caught else 'NOT CAUGHT'}", flush=True)
    print("fp4_fp4_mqa_logits v6 selftest:", "ok" if ok else "FAIL")
    return 0 if ok else 1


def bench(mod, dev, out: Path | None) -> int:
    if out is not None and out.exists():
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 2
    props = torch.cuda.get_device_properties(0)
    flush_buf = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    rows = []
    for (S, N, H) in ((32, 4096, 8), (128, 8192, 8), (32, 32768, 16)):
        r, args = run_case(mod, S, N, H, 100 + S, dev, full_span=True)
        q4, sfq_u8, kv4, sfkv_u8, w, ks, ke, out_t = args
        # v4 on the same k codes with an e4m3 q holding the same values (e2m1 values times a power of two fit e4m3 exactly when the
        # row's block scales span less than e4m3's range; the q values are the dequantised MXFP4 q)
        S_, H_, _ = q4.shape
        sfq_f = torch.pow(2.0, sfq_u8.reshape(-1, NBLK).float() - 127)
        qf = ref.cast_back_from_fp4(q4.reshape(S_ * H_, HEAD_DIM // 2), sfq_f, gran_k=BLOCK)
        q8, sfq8_packed = ref.per_token_cast_to_fp8(qf, use_ue8m0=True, gran_k=HEAD_DIM, use_packed_ue8m0=True)
        sfq8 = ref.unpack_ue8m0_from_int(sfq8_packed)[:, :1]
        sfq8_u8 = (torch.round(torch.log2(sfq8)) + 127).clamp(0, 255).to(torch.uint8).reshape(S_, H_).contiguous()
        q8 = q8.reshape(S_, H_, HEAD_DIM).contiguous()
        span = base.span_of(ks, ke)
        plan = base.plan_v2(S, span[1] - span[0], v4.sm_count())
        for label, fn in (("v6 (MXFP4 q)", lambda: launch_v6(mod, q4, sfq_u8, kv4, sfkv_u8, w, ks, ke, out_t, plan, span)),
                          ("v6s (80-byte row stride)", lambda: launch_v6(mod, q4, sfq_u8, kv4, sfkv_u8, w, ks, ke, out_t, plan, span, padded=True)),
                          ("v6e (packed epilogue)", lambda: launch_v6(mod, q4, sfq_u8, kv4, sfkv_u8, w, ks, ke, out_t, plan, span, packed=True)),
                          ("v4 (e4m3 q, same k codes)", lambda: v4.launch_v4(mod, q8, sfq8_u8, kv4, sfkv_u8, w, ks, ke, out_t, plan, span))):
            times = []
            for _ in range(10):
                flush_buf.fill_(1)
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record(); fn(); e1.record()
                torch.cuda.synchronize()
                times.append(e0.elapsed_time(e1) * 1000)
            med = statistics.median(times)
            flops = 2.0 * S * H * N * HEAD_DIM
            rows.append({"kernel": label, "S": S, "N": N, "H": H, "plan": list(plan), "us_median": med, "us_min": min(times), "TFLOPs": flops / med / 1e6})
            print(f"{label:28s} S={S} N={N} H={H}: {med:.1f} us ({rows[-1]['TFLOPs']:.1f} TFLOP/s)", flush=True)
    report = {"kernel": "fp4_fp4_mqa_logits_sm120_v6: v4's staging with MXFP4 q and k, two kind::mxf4.block_scale k64 MMAs per n8 tile, scales applied by the MMA",
              "device": props.name, "note": "cold L2 (256 MB fill before each launch); median of 10; launch only in the timed region; v4 timed on the same k codes", "rows": rows}
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print("->", out)
    return 0


def run_paged_case(mod, S, N, H, seed, dev):
    """The paged v6 against the flat v6 on the same kv (ks = 0, ke = ctx), bit for bit, and against the reference."""
    q, kv, w, ks, ke = make_case(S, N, H, seed, dev, full_span=True)
    q4, sfq_u8, sfq_f = quantize_q_blocked(q)
    kv4, sfkv_u8, sfkv_f = v4.quantize_kv_blocked(kv)
    g = torch.Generator().manual_seed(seed + 7)
    ctx = torch.randint(1, N + 1, (S,), generator=g).to(torch.int32).to(dev)
    kv_cache, sf_cache, block_table, max_pages = v4.make_paged_blocked(kv4, sfkv_u8, S, ctx, seed, dev)
    max_ctx = int(ctx.max())
    out = torch.full((S, max_ctx), float("-inf"), device=dev, dtype=torch.float32)
    mod.fp4_fp4_paged_mqa_logits_sm120_v6(q4, sfq_u8, kv_cache, sf_cache, w, ctx, block_table, out, max_pages)
    ks0 = torch.zeros(S, dtype=torch.int32, device=dev)
    flat = torch.full((S, max_ctx), float("-inf"), device=dev, dtype=torch.float32)
    launch_v6(mod, q4, sfq_u8, kv4, sfkv_u8, w, ks0, ctx, flat, span=(0, max_ctx))
    torch.cuda.synchronize()
    valid = torch.isfinite(flat)
    same = bool(torch.equal(out, flat))
    untouched = bool(torch.equal(torch.isfinite(out), valid))
    exact = reference_v6(q4, sfq_f, kv4, sfkv_f, w, ks0, ctx, max_ctx)
    rel = float((out[valid] - exact[valid]).abs().max() / exact[valid].abs().max().clamp_min(1e-30))
    oe = torch.full((S, max_ctx), float("-inf"), device=dev, dtype=torch.float32)
    mod.fp4_fp4_paged_mqa_logits_sm120_v6e(q4, sfq_u8, kv_cache, sf_cache, w, ctx, block_table, oe, max_pages)
    torch.cuda.synchronize()
    same_e = bool(torch.equal(oe, out))
    same_f = True
    for G in (2, 4, 8):
        of = torch.full((S, max_ctx), float("-inf"), device=dev, dtype=torch.float32)
        mod.fp4_fp4_paged_mqa_logits_sm120_v6f(q4, sfq_u8, kv_cache, sf_cache, w, ctx, block_table, of, max_pages, G)
        torch.cuda.synchronize()
        same_f &= bool(torch.equal(of, oe))
    og = torch.full((S, max_ctx), float("-inf"), device=dev, dtype=torch.float32)
    mod.fp4_fp4_paged_mqa_logits_sm120_v6g(q4, sfq_u8, kv_cache, sf_cache, w, ctx, block_table, og, max_pages)
    torch.cuda.synchronize()
    same_g = bool(torch.equal(og, oe))
    oh = torch.full((S, max_ctx), float("-inf"), device=dev, dtype=torch.float32)
    mod.fp4_fp4_paged_mqa_logits_sm120_v6h(q4, sfq_u8, kv_cache, sf_cache, w, ctx, block_table, oh, max_pages)
    torch.cuda.synchronize()
    same_h = bool(torch.equal(oh, oe))
    oi = torch.full((S, max_ctx), float("-inf"), device=dev, dtype=torch.float32)
    mod.fp4_fp4_paged_mqa_logits_sm120_v6i(q4, sfq_u8, kv_cache, sf_cache, w, ctx, block_table, oi, max_pages)
    torch.cuda.synchronize()
    same_i = bool(torch.equal(oi, oe))
    oj = torch.full((S, max_ctx), float("-inf"), device=dev, dtype=torch.float32)
    mod.fp4_fp4_paged_mqa_logits_sm120_v6j(q4, sfq_u8, kv_cache, sf_cache, w, ctx, block_table, oj, max_pages)
    torch.cuda.synchronize()
    same_j = bool(torch.equal(oj, oe))
    ok_ = torch.full((S, max_ctx), float("-inf"), device=dev, dtype=torch.float32)
    mod.fp4_fp4_paged_mqa_logits_sm120_v6k(q4, sfq_u8, kv_cache, sf_cache, w, ctx, block_table, ok_, max_pages,
                                           mod.fp4_fp4_paged_mqa_logits_sm120_v6k_meta(ctx, max_pages))
    torch.cuda.synchronize()
    same_k = bool(torch.equal(ok_, oe))
    res = {"S": S, "N": N, "H": H, "pages": int(kv_cache.shape[0]), "max_pages": max_pages, "bit_identical_to_flat_v6": same, "paged_v6e_bit_identical": same_e, "paged_v6f_bit_identical_g248": same_f, "paged_v6g_bit_identical": same_g, "paged_v6h_bit_identical": same_h, "paged_v6i_bit_identical": same_i, "paged_v6j_bit_identical": same_j, "paged_v6k_bit_identical": same_k,
           "untouched_outside_context": untouched, "rel_max_err": rel, "pass": same and same_e and same_f and same_g and same_h and same_i and same_j and same_k and untouched and rel < 1e-5}
    return res, (q4, sfq_u8, kv_cache, sf_cache, w, ctx, block_table, out, max_pages, kv4, sfkv_u8)


def selftest_paged(mod, dev) -> int:
    ok = True
    print("paged v6 (80-byte row stride) against the flat v6 on the same kv through random page permutations, and against the reference")
    for (S, N, H, seed) in ((8, 512, 8, 21), (33, 2048, 8, 22), (64, 4096, 16, 23), (16, 1000, 8, 24)):
        r, _ = run_paged_case(mod, S, N, H, seed, dev)
        ok &= r["pass"]
        print(f"  S={S} N={N} H={H} pages={r['pages']}: bit-identical to flat v6 {r['bit_identical_to_flat_v6']}, outside context untouched "
              f"{r['untouched_outside_context']}, rel max err {r['rel_max_err']:.2e}, paged v6e bit-identical {r['paged_v6e_bit_identical']}, v6f (G 2/4/8) {r['paged_v6f_bit_identical_g248']}, v6g {r['paged_v6g_bit_identical']}, v6h {r['paged_v6h_bit_identical']}, v6i {r['paged_v6i_bit_identical']}, v6j {r['paged_v6j_bit_identical']}, v6k {r['paged_v6k_bit_identical']} -> {'ok' if r['pass'] else 'FAIL'}", flush=True)
    print("paged v6 selftest:", "ok" if ok else "FAIL")
    return 0 if ok else 1


def bench_paged(mod, dev, out: Path | None) -> int:
    if out is not None and out.exists():
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 2
    props = torch.cuda.get_device_properties(0)
    flush_buf = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    rows = []
    for (S, N, H) in ((32, 8192, 8), (64, 32768, 8), (32, 65536, 16)):
        r, args = run_paged_case(mod, S, N, H, 200 + S, dev)
        q4, sfq_u8, kv_cache, sf_cache, w, ctx, block_table, out_t, max_pages, kv4, sfkv_u8 = args
        S_, H_, _ = q4.shape
        qf = ref.cast_back_from_fp4(q4.reshape(S_ * H_, HEAD_DIM // 2), torch.pow(2.0, sfq_u8.reshape(-1, NBLK).float() - 127), gran_k=BLOCK)
        q8, sfq8_packed = ref.per_token_cast_to_fp8(qf, use_ue8m0=True, gran_k=HEAD_DIM, use_packed_ue8m0=True)
        sfq8_u8 = (torch.round(torch.log2(ref.unpack_ue8m0_from_int(sfq8_packed)[:, :1])) + 127).clamp(0, 255).to(torch.uint8).reshape(S_, H_).contiguous()
        q8 = q8.reshape(S_, H_, HEAD_DIM).contiguous()
        offs_k = mod.fp4_fp4_paged_mqa_logits_sm120_v6k_meta(ctx, max_pages)
        for label, fn in (("paged v6 (MXFP4 q)", lambda: mod.fp4_fp4_paged_mqa_logits_sm120_v6(q4, sfq_u8, kv_cache, sf_cache, w, ctx, block_table, out_t, max_pages)),
                          ("paged v6e (packed epilogue)", lambda: mod.fp4_fp4_paged_mqa_logits_sm120_v6e(q4, sfq_u8, kv_cache, sf_cache, w, ctx, block_table, out_t, max_pages)),
                          ("paged v6k (v6j over a compacted (row, page) list)", lambda: mod.fp4_fp4_paged_mqa_logits_sm120_v6k(q4, sfq_u8, kv_cache, sf_cache, w, ctx, block_table, out_t, max_pages, offs_k)),
                          ("paged v6k work list (once per batch)", lambda: mod.fp4_fp4_paged_mqa_logits_sm120_v6k_meta(ctx, max_pages)),
                          ("paged v6j (v6i, full quarters unmasked)", lambda: mod.fp4_fp4_paged_mqa_logits_sm120_v6j(q4, sfq_u8, kv_cache, sf_cache, w, ctx, block_table, out_t, max_pages)),
                          ("paged v6i (quarter-page double buffer)", lambda: mod.fp4_fp4_paged_mqa_logits_sm120_v6i(q4, sfq_u8, kv_cache, sf_cache, w, ctx, block_table, out_t, max_pages)),
                          ("paged v6h (half pages, q loaded once per head block)", lambda: mod.fp4_fp4_paged_mqa_logits_sm120_v6h(q4, sfq_u8, kv_cache, sf_cache, w, ctx, block_table, out_t, max_pages)),
                          ("paged v6g (half-page double buffer)", lambda: mod.fp4_fp4_paged_mqa_logits_sm120_v6g(q4, sfq_u8, kv_cache, sf_cache, w, ctx, block_table, out_t, max_pages)),
                          ("paged v6f G=2 (two pages per warp, prefetched)", lambda: mod.fp4_fp4_paged_mqa_logits_sm120_v6f(q4, sfq_u8, kv_cache, sf_cache, w, ctx, block_table, out_t, max_pages, 2)),
                          ("paged v6f G=4", lambda: mod.fp4_fp4_paged_mqa_logits_sm120_v6f(q4, sfq_u8, kv_cache, sf_cache, w, ctx, block_table, out_t, max_pages, 4)),
                          ("paged v6f G=8", lambda: mod.fp4_fp4_paged_mqa_logits_sm120_v6f(q4, sfq_u8, kv_cache, sf_cache, w, ctx, block_table, out_t, max_pages, 8)),
                          ("paged v4 (e4m3 q, same k codes)", lambda: mod.fp8_fp4_paged_mqa_logits_sm120_v4(q8, sfq8_u8, kv_cache, sf_cache, w, ctx, block_table, out_t, max_pages))):
            times = []
            for _ in range(10):
                flush_buf.fill_(1)
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record(); fn(); e1.record()
                torch.cuda.synchronize()
                times.append(e0.elapsed_time(e1) * 1000)
            med = statistics.median(times)
            rows.append({"kernel": label, "S": S, "N": N, "H": H, "ctx_sum": int(ctx.sum()), "us_median": med, "us_min": min(times)})
            print(f"{label:32s} S={S} N={N} H={H}: {med:.1f} us", flush=True)
    report = {"kernel": "fp4_fp4_paged_mqa_logits_sm120_v6: the paged v4's staging at an 80-byte row stride with v6's block-scaled k64 inner product",
              "device": props.name, "note": "cold L2 (256 MB fill before each launch); median of 10; random context lengths per row (seeded); paged v4 timed on the same k codes and pages", "rows": rows}
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print("->", out)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--bench-paged", action="store_true")
    ap.add_argument("--ncu-shape", action="store_true", help="one launch of v6 and one of v6s on S 128 N 8192 H 8, for Nsight Compute")
    ap.add_argument("--out-paged", type=Path)
    ap.add_argument("--ncu-paged", action="store_true", help="one launch of the paged v6e and one of the flat v6e on S 64 N 32768 H 8, for Nsight Compute")
    a = ap.parse_args(argv)
    dev = torch.device("cuda")
    base.SM_COUNT = v4.sm_count()
    mod = build()
    rc = 0
    if a.selftest:
        rc = selftest(mod, dev) or selftest_paged(mod, dev)
    if a.bench:
        rc = rc or bench(mod, dev, a.out)
    if a.bench_paged:
        rc = rc or bench_paged(mod, dev, a.out_paged)
    if a.ncu_paged:
        # one launch each of the paged v6e and the flat v6e on the same kv (S 64, N 32768, H 8), for Nsight Compute
        r_, args = run_paged_case(mod, 64, 32768, 8, 264, dev)
        q4, sfq_u8, kv_cache, sf_cache, w, ctx, block_table, out_t, max_pages, kv4, sfkv_u8 = args
        torch.cuda.synchronize()
        mod.fp4_fp4_paged_mqa_logits_sm120_v6e(q4, sfq_u8, kv_cache, sf_cache, w, ctx, block_table, out_t, max_pages)
        mod.fp4_fp4_paged_mqa_logits_sm120_v6f(q4, sfq_u8, kv_cache, sf_cache, w, ctx, block_table, out_t, max_pages, 4)
        flat = torch.full_like(out_t, float("-inf"))
        launch_v6(mod, q4, sfq_u8, kv4, sfkv_u8, w, torch.zeros_like(ctx), ctx, flat, span=(0, int(ctx.max())), packed=True)
        torch.cuda.synchronize()
    if a.ncu_shape:
        _, args = run_case(mod, 128, 8192, 8, 228, dev, full_span=True)
        q4, sfq_u8, kv4, sfkv_u8, w, ks, ke, out_t = args
        torch.cuda.synchronize()
        launch_v6(mod, q4, sfq_u8, kv4, sfkv_u8, w, ks, ke, out_t)
        launch_v6(mod, q4, sfq_u8, kv4, sfkv_u8, w, ks, ke, out_t, padded=True)
        launch_v6(mod, q4, sfq_u8, kv4, sfkv_u8, w, ks, ke, out_t, packed=True)
        # v4 on the same k codes with an e4m3 q holding the dequantised MXFP4 q, for a side-by-side profile
        S_, H_, _ = q4.shape
        qf = ref.cast_back_from_fp4(q4.reshape(S_ * H_, HEAD_DIM // 2), torch.pow(2.0, sfq_u8.reshape(-1, NBLK).float() - 127), gran_k=BLOCK)
        q8, sfq8_packed = ref.per_token_cast_to_fp8(qf, use_ue8m0=True, gran_k=HEAD_DIM, use_packed_ue8m0=True)
        sfq8_u8 = (torch.round(torch.log2(ref.unpack_ue8m0_from_int(sfq8_packed)[:, :1])) + 127).clamp(0, 255).to(torch.uint8).reshape(S_, H_).contiguous()
        v4.launch_v4(mod, q8.reshape(S_, H_, HEAD_DIM).contiguous(), sfq8_u8, kv4, sfkv_u8, w, ks, ke, out_t)
        torch.cuda.synchronize()
    return rc


if __name__ == "__main__":
    sys.exit(main())
