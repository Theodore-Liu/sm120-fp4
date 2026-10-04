#!/usr/bin/env python3
"""MQA-logits v5: the FP8-k form, which is the indexer path vLLM runs on SM120. vLLM's sparse indexer keeps its k cache in fp8 by
default and refuses the MXFP4 cache on consumer Blackwell (`vllm/v1/attention/backends/mla/indexer.py`, `dsa_indexer_uses_fp4`:
"indexer_kv_dtype='mxfp4' requires Blackwell datacenter GPUs"), so on an RTX 5090 or PRO 6000 the kernel the engine calls takes
q [M, H, 128] e4m3 (its per-token scale folded into `weights`) and k [N, 128] e4m3 with one fp32 scale per row (the paged cache
stores each row as 128 e4m3 bytes followed by the 4 scale bytes, 132 bytes per entry).

v5 is v2's kernel with the B operand read as e4m3 bytes (the instruction form `kind::f8f6f4 ... e4m3.e4m3` the FP8 einsum kernel
uses) from 128-byte rows, and the k scale an fp32 per row folded once after the four k32 steps. The q side keeps a UE8M0 scale per
(row, head); the engine's form is all ones (byte 127) with the scale in `weights`.

    ~/mlsys-5090-runtime/vllm028/.venv/bin/python scripts/fp8_mqa_logits_v5_sm120.py --selftest
    ~/mlsys-5090-runtime/vllm028/.venv/bin/python scripts/fp8_mqa_logits_v5_sm120.py --bench --out reports/fp8-mqa-logits-v5-rtx5090-20261004.json
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
import ue8m0_reference as ref  # noqa: E402

HEAD_DIM = 128
E4M3_MAX = 448.0

CPP_V5 = r"""
void fp8_mqa_logits_sm120_v5(torch::Tensor q, torch::Tensor sfq, torch::Tensor k, torch::Tensor k_scale, torch::Tensor w,
                             torch::Tensor ks, torch::Tensor ke, torch::Tensor logits, int64_t kv_lo, int64_t kv_hi,
                             int64_t rows, int64_t kvseg, int64_t group);
void fp8_paged_mqa_logits_sm120_v5(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv_cache, torch::Tensor w,
                                   torch::Tensor context_lens, torch::Tensor block_table, torch::Tensor logits, int64_t max_pages);
void fp8_paged_mqa_logits_sm120_v5h(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv_cache, torch::Tensor w,
                                    torch::Tensor context_lens, torch::Tensor block_table, torch::Tensor logits, int64_t max_pages);
void fp8_paged_mqa_logits_sm120_v5d(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv_cache, torch::Tensor w,
                                    torch::Tensor context_lens, torch::Tensor block_table, torch::Tensor logits, int64_t max_pages);
"""

CUDA_V5 = r"""
__device__ __forceinline__ void mma_e4m3_e4m3_v5(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// v5: v2's structure with k rows of 128 e4m3 bytes (staged 8 x 16 B per row) and an fp32 scale per row; the B fragment is the four
// e4m3 bytes of k 4t..4t+3 and 16+4t.. of the column, read directly. Shared memory per buffer: KVSEG x 128 B + KVSEG x 4 B.
template <int ROWS, int KVSEG>
__global__ void __launch_bounds__(V1_WARPS * 32)
k_mqa_logits_v5(const uint8_t* __restrict__ q, const uint8_t* __restrict__ sfq, const uint8_t* __restrict__ k,
                const float* __restrict__ k_scale, const __nv_bfloat16* __restrict__ w, const int* __restrict__ ks,
                const int* __restrict__ ke, float* __restrict__ logits, int S, int H, int N, int max_k, int kv_lo, int kv_hi, int group) {
  constexpr int RPW = ROWS >= V1_WARPS ? ROWS / V1_WARPS : 1;
  __shared__ __align__(16) uint8_t s_k[2][KVSEG * 128];
  __shared__ __align__(16) float s_sf[2][KVSEG];
  const int row0 = blockIdx.x * ROWS;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int first_seg = blockIdx.y * group;
  const int nsegs_total = (kv_hi - kv_lo + KVSEG - 1) / KVSEG;
  const int last_seg = min(nsegs_total, first_seg + group);
  if (first_seg >= last_seg) return;
  auto stage = [&](int seg, int buf) {
    const int seg0 = kv_lo + seg * KVSEG, seg1 = min(N, min(kv_hi, seg0 + KVSEG));
    for (int c = tid; c < KVSEG * 8; c += V1_WARPS * 32) {
      const int r = c >> 3, part = c & 7;
      if (seg0 + r < seg1) cp_async_16(s_k[buf] + r * 128 + part * 16, k + (size_t)(seg0 + r) * 128 + part * 16);
      else *reinterpret_cast<uint4*>(s_k[buf] + r * 128 + part * 16) = make_uint4(0u, 0u, 0u, 0u);
    }
    for (int r = tid; r < KVSEG; r += V1_WARPS * 32) s_sf[buf][r] = (seg0 + r < seg1) ? k_scale[seg0 + r] : 0.f;
    asm volatile("cp.async.commit_group;\n");
  };
  stage(first_seg, 0);
  for (int seg = first_seg; seg < last_seg; ++seg) {
    const int buf = (seg - first_seg) & 1;
    if (seg + 1 < last_seg) { stage(seg + 1, buf ^ 1); asm volatile("cp.async.wait_group 1;\n"); }
    else { asm volatile("cp.async.wait_group 0;\n"); }
    __syncthreads();
    const int seg0 = kv_lo + seg * KVSEG, seg1 = min(N, min(kv_hi, seg0 + KVSEG));
    const uint8_t* sk = s_k[buf];
    const float* ssf = s_sf[buf];
    for (int rr = 0; rr < RPW; ++rr) {
      const int i = row0 + (ROWS >= V1_WARPS ? warp * RPW + rr : warp);
      if (i >= S || (ROWS < V1_WARPS && warp >= ROWS)) continue;
      const int k_start = ks[i], k_end = ke[i];
      const int n_lo = max(k_start, seg0), n_hi = min(k_end, seg1);
      if (n_lo >= n_hi) continue;
      const uint8_t* qrow = q + (size_t)i * H * 128;
      float* out = logits + (size_t)i * max_k;
      for (int h0 = 0; h0 < H; h0 += 16) {
        const int ha = h0 + g, hb = h0 + g + 8;
        const bool has_a = ha < H, has_b = hb < H;
        const float sa = has_a ? ue8m0_to_float(sfq[(size_t)i * H + ha]) : 0.f;
        const float sb = has_b ? ue8m0_to_float(sfq[(size_t)i * H + hb]) : 0.f;
        const float wa = has_a ? __bfloat162float(w[(size_t)i * H + ha]) : 0.f;
        const float wb = has_b ? __bfloat162float(w[(size_t)i * H + hb]) : 0.f;
        uint32_t af[4][4];
        for (int st = 0; st < 4; ++st) {
          const int k0 = st * 32;
          af[st][0] = has_a ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)ha * 128 + k0 + 4 * t) : 0u;
          af[st][1] = has_b ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)hb * 128 + k0 + 4 * t) : 0u;
          af[st][2] = has_a ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)ha * 128 + k0 + 16 + 4 * t) : 0u;
          af[st][3] = has_b ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)hb * 128 + k0 + 16 + 4 * t) : 0u;
        }
        const int tile0 = (n_lo - seg0) & ~7;
        for (int n0 = seg0 + tile0; n0 < n_hi; n0 += 8) {
          const int col = n0 + g;
          const bool has_col = col >= n_lo && col < n_hi;
          const uint8_t* brow = sk + (size_t)(has_col ? (col - seg0) : 0) * 128;
          float part[4] = {0.f, 0.f, 0.f, 0.f};
          for (int st = 0; st < 4; ++st) {
            const int k0 = st * 32;
            uint32_t bf[2];
            bf[0] = has_col ? *reinterpret_cast<const uint32_t*>(brow + k0 + 4 * t) : 0u;
            bf[1] = has_col ? *reinterpret_cast<const uint32_t*>(brow + k0 + 16 + 4 * t) : 0u;
            mma_e4m3_e4m3_v5(part, af[st], bf);
          }
          const int c0 = n0 + 2 * t, c1 = c0 + 1;
          const bool in0 = c0 >= n_lo && c0 < n_hi, in1 = c1 >= n_lo && c1 < n_hi;
          const float sk0 = in0 ? ssf[c0 - seg0] : 0.f;
          const float sk1 = in1 ? ssf[c1 - seg0] : 0.f;
          float v0 = fmaxf(part[0] * (sa * sk0), 0.f) * wa + fmaxf(part[2] * (sb * sk0), 0.f) * wb;
          float v1 = fmaxf(part[1] * (sa * sk1), 0.f) * wa + fmaxf(part[3] * (sb * sk1), 0.f) * wb;
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
    __syncthreads();
  }
}

template <int ROWS, int KVSEG>
static void launch_v5(const uint8_t* pq, const uint8_t* psq, const uint8_t* pk, const float* psk, const __nv_bfloat16* pw,
                      const int* pks, const int* pke, float* pl, int S, int H, int N, int max_k, int kv_lo, int kv_hi, int group,
                      cudaStream_t st) {
  const int nsegs = (kv_hi - kv_lo + KVSEG - 1) / KVSEG;
  const dim3 grid((S + ROWS - 1) / ROWS, (nsegs + group - 1) / group);
  k_mqa_logits_v5<ROWS, KVSEG><<<grid, V1_WARPS * 32, 0, st>>>(pq, psq, pk, psk, pw, pks, pke, pl, S, H, N, max_k, kv_lo, kv_hi, group);
}

void fp8_mqa_logits_sm120_v5(torch::Tensor q, torch::Tensor sfq, torch::Tensor k, torch::Tensor k_scale, torch::Tensor w,
                             torch::Tensor ks, torch::Tensor ke, torch::Tensor logits, int64_t kv_lo, int64_t kv_hi,
                             int64_t rows, int64_t kvseg, int64_t group) {
  const int S = (int)q.size(0), H = (int)q.size(1), N = (int)k.size(0), max_k = (int)logits.size(1);
  TORCH_CHECK(q.scalar_type() == torch::kFloat8_e4m3fn && q.is_contiguous() && q.size(2) == 128, "q: e4m3 [S, H, 128]");
  TORCH_CHECK(sfq.scalar_type() == torch::kUInt8 && sfq.numel() == (int64_t)S * H && sfq.is_contiguous(), "sfq: uint8 UE8M0 [S, H]");
  TORCH_CHECK(k.scalar_type() == torch::kFloat8_e4m3fn && k.is_contiguous() && k.size(1) == 128, "k: e4m3 [N, 128]");
  TORCH_CHECK(k_scale.scalar_type() == torch::kFloat && k_scale.numel() == N && k_scale.is_contiguous(), "k_scale: fp32 [N]");
  TORCH_CHECK(w.scalar_type() == torch::kBFloat16 && w.size(0) == S && w.size(1) == H && w.is_contiguous(), "weights: bf16 [S, H]");
  TORCH_CHECK(ks.scalar_type() == torch::kInt && ke.scalar_type() == torch::kInt && ks.numel() == S && ke.numel() == S, "ks, ke: int32 [S]");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat && logits.size(0) == S && logits.is_contiguous(), "logits: fp32 [S, max_seqlen_k]");
  TORCH_CHECK(kv_lo >= 0 && kv_hi <= N && kv_lo < kv_hi, "kv_lo < kv_hi within [0, N]");
  TORCH_CHECK(group >= 1, "group >= 1");
  auto st = at::cuda::getCurrentCUDAStream();
  const uint8_t* pq = static_cast<const uint8_t*>(q.data_ptr());
  const uint8_t* psq = sfq.data_ptr<uint8_t>();
  const uint8_t* pk = static_cast<const uint8_t*>(k.data_ptr());
  const float* psk = k_scale.data_ptr<float>();
  const __nv_bfloat16* pw = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
  float* pl = logits.data_ptr<float>();
  const int* pks = ks.data_ptr<int>();
  const int* pke = ke.data_ptr<int>();
  if (rows == 16 && kvseg == 128) launch_v5<16, 128>(pq, psq, pk, psk, pw, pks, pke, pl, S, H, N, max_k, (int)kv_lo, (int)kv_hi, (int)group, st);
  else if (rows == 16 && kvseg == 64) launch_v5<16, 64>(pq, psq, pk, psk, pw, pks, pke, pl, S, H, N, max_k, (int)kv_lo, (int)kv_hi, (int)group, st);
  else if (rows == 8 && kvseg == 128) launch_v5<8, 128>(pq, psq, pk, psk, pw, pks, pke, pl, S, H, N, max_k, (int)kv_lo, (int)kv_hi, (int)group, st);
  else if (rows == 8 && kvseg == 64) launch_v5<8, 64>(pq, psq, pk, psk, pw, pks, pke, pl, S, H, N, max_k, (int)kv_lo, (int)kv_hi, (int)group, st);
  else TORCH_CHECK(false, "rows in {16, 8}, kvseg in {128, 64}");
}

// paged v5: one page of 64 cache rows per warp per step, the page in vLLM's fp8 indexer layout (132 bytes per row: 128 e4m3 bytes
// then the fp32 scale). The rows are staged into a 128-byte-stride shared buffer with the scales beside it; the first version
// read the page as 2112 4-byte words (66 per lane), this one as 528 16-byte chunks lane-strided (17 passes), see the loop.
constexpr int V5_PAGE = 64, V5_WARPS = 8, V5_ENTRY = 132, V5_HALF = 32;

__global__ void __launch_bounds__(V5_WARPS * 32)
k_paged_mqa_logits_v5(const uint8_t* __restrict__ q, const uint8_t* __restrict__ sfq, const uint8_t* __restrict__ kv_cache,
                      const __nv_bfloat16* __restrict__ w, const int* __restrict__ ctx, const int* __restrict__ block_table,
                      float* __restrict__ logits, int S, int H, int max_pages, int max_ctx) {
  __shared__ __align__(16) uint8_t s_k[V5_WARPS][V5_PAGE * 128];
  __shared__ __align__(16) float s_sf[V5_WARPS][V5_PAGE];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int i = blockIdx.x * V5_WARPS + warp;
  const int p = blockIdx.y;
  if (i >= S) return;
  const int len = ctx[i];
  const int pos0 = p * V5_PAGE;
  if (pos0 >= len) return;
  const int n_valid = min(V5_PAGE, len - pos0);
  const int page = block_table[(size_t)i * max_pages + p];
  // A 132-byte row is 4-byte aligned only, but the page (64 rows, 8448 bytes, base a multiple of 8448 = 528 * 16) is 528 contiguous
  // 16-byte chunks. Each lane takes chunks lane, lane + 32, ... (16 full passes and a 17th for lanes 0 to 15); each 4-byte word of a
  // chunk goes to its row (word w of the page lands in row w / 33, part w % 33; 132 is a multiple of 4, so no word straddles rows).
  const uint4* src4 = reinterpret_cast<const uint4*>(kv_cache + (size_t)page * V5_PAGE * V5_ENTRY);
#pragma unroll
  for (int it = 0; it < 17; ++it) {
    const int c = it * 32 + lane;
    if (c < 528) {
      const uint4 v = src4[c];
      const uint32_t wv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int word = c * 4 + j;
        const int r = word / 33, part = word - r * 33;
        if (part < 32) *reinterpret_cast<uint32_t*>(s_k[warp] + r * 128 + part * 4) = wv[j];
        else s_sf[warp][r] = __uint_as_float(wv[j]);
      }
    }
  }
  __syncwarp();
  const uint8_t* qrow = q + (size_t)i * H * 128;
  float* out = logits + (size_t)i * max_ctx + pos0;
  const uint8_t* sk = s_k[warp];
  const float* ssf = s_sf[warp];
  for (int h0 = 0; h0 < H; h0 += 16) {
    const int ha = h0 + g, hb = h0 + g + 8;
    const bool has_a = ha < H, has_b = hb < H;
    const float sa = has_a ? ue8m0_to_float(sfq[(size_t)i * H + ha]) : 0.f;
    const float sb = has_b ? ue8m0_to_float(sfq[(size_t)i * H + hb]) : 0.f;
    const float wa = has_a ? __bfloat162float(w[(size_t)i * H + ha]) : 0.f;
    const float wb = has_b ? __bfloat162float(w[(size_t)i * H + hb]) : 0.f;
    uint32_t af[4][4];
    for (int st = 0; st < 4; ++st) {
      const int k0 = st * 32;
      af[st][0] = has_a ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)ha * 128 + k0 + 4 * t) : 0u;
      af[st][1] = has_b ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)hb * 128 + k0 + 4 * t) : 0u;
      af[st][2] = has_a ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)ha * 128 + k0 + 16 + 4 * t) : 0u;
      af[st][3] = has_b ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)hb * 128 + k0 + 16 + 4 * t) : 0u;
    }
    for (int n0 = 0; n0 < n_valid; n0 += 8) {
      const int col = n0 + g;
      const bool has_col = col < n_valid;
      const uint8_t* brow = sk + (size_t)(has_col ? col : 0) * 128;
      float part[4] = {0.f, 0.f, 0.f, 0.f};
      for (int st = 0; st < 4; ++st) {
        const int k0 = st * 32;
        uint32_t bf[2];
        bf[0] = has_col ? *reinterpret_cast<const uint32_t*>(brow + k0 + 4 * t) : 0u;
        bf[1] = has_col ? *reinterpret_cast<const uint32_t*>(brow + k0 + 16 + 4 * t) : 0u;
        mma_e4m3_e4m3_v5(part, af[st], bf);
      }
      const int c0 = n0 + 2 * t, c1 = c0 + 1;
      const bool in0 = c0 < n_valid, in1 = c1 < n_valid;
      const float sk0 = in0 ? ssf[c0] : 0.f;
      const float sk1 = in1 ? ssf[c1] : 0.f;
      float v0 = fmaxf(part[0] * (sa * sk0), 0.f) * wa + fmaxf(part[2] * (sb * sk0), 0.f) * wb;
      float v1 = fmaxf(part[1] * (sa * sk1), 0.f) * wa + fmaxf(part[3] * (sb * sk1), 0.f) * wb;
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

void fp8_paged_mqa_logits_sm120_v5(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv_cache, torch::Tensor w,
                                   torch::Tensor context_lens, torch::Tensor block_table, torch::Tensor logits, int64_t max_pages) {
  const int S = (int)q.size(0), H = (int)q.size(1), max_ctx = (int)logits.size(1);
  TORCH_CHECK(q.scalar_type() == torch::kFloat8_e4m3fn && q.is_contiguous() && q.size(2) == 128, "q: e4m3 [S, H, 128]");
  TORCH_CHECK(sfq.scalar_type() == torch::kUInt8 && sfq.numel() == (int64_t)S * H && sfq.is_contiguous(), "sfq: uint8 UE8M0 [S, H]");
  TORCH_CHECK(kv_cache.scalar_type() == torch::kUInt8 && kv_cache.is_contiguous() && kv_cache.dim() == 4 && kv_cache.size(1) == 64 && kv_cache.size(2) == 1 && kv_cache.size(3) == 132,
              "kv_cache: uint8 [num_blocks, 64, 1, 132] (128 e4m3 bytes then the fp32 scale per row)");
  TORCH_CHECK(w.scalar_type() == torch::kBFloat16 && w.size(0) == S && w.size(1) == H && w.is_contiguous(), "weights: bf16 [S, H]");
  TORCH_CHECK(context_lens.scalar_type() == torch::kInt && context_lens.numel() == S, "context_lens: int32 [S]");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt && block_table.is_contiguous() && block_table.size(0) == S && block_table.size(1) == max_pages, "block_table: int32 [S, max_pages]");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat && logits.size(0) == S && logits.is_contiguous(), "logits: fp32 [S, max_context_len]");
  auto st = at::cuda::getCurrentCUDAStream();
  k_paged_mqa_logits_v5<<<dim3((S + V5_WARPS - 1) / V5_WARPS, (int)max_pages), V5_WARPS * 32, 0, st>>>(
      static_cast<const uint8_t*>(q.data_ptr()), sfq.data_ptr<uint8_t>(), kv_cache.data_ptr<uint8_t>(),
      reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), context_lens.data_ptr<int>(), block_table.data_ptr<int>(),
      logits.data_ptr<float>(), S, H, (int)max_pages, max_ctx);
}
__global__ void __launch_bounds__(V5_WARPS * 32)
k_paged_mqa_logits_v5h(const uint8_t* __restrict__ q, const uint8_t* __restrict__ sfq, const uint8_t* __restrict__ kv_cache,
                      const __nv_bfloat16* __restrict__ w, const int* __restrict__ ctx, const int* __restrict__ block_table,
                      float* __restrict__ logits, int S, int H, int max_pages, int max_ctx) {
  __shared__ __align__(16) uint8_t s_k[V5_WARPS][V5_HALF * 128];
  __shared__ __align__(16) float s_sf[V5_WARPS][V5_HALF];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int i = blockIdx.x * V5_WARPS + warp;
  const int p = blockIdx.y;
  if (i >= S) return;
  const int len = ctx[i];
  const int pos0 = p * V5_PAGE;
  if (pos0 >= len) return;
  const int n_valid = min(V5_PAGE, len - pos0);
  const int page = block_table[(size_t)i * max_pages + p];
  // The page in two halves of 32 rows (4224 bytes = 264 chunks of 16 bytes each, both halves 16-byte aligned): each half is staged
  // lane-strided (8 full passes and a 9th for lanes 0 to 7) into a 4.2 KB buffer, then every head's scores for its rows are formed;
  // 33.8 KB per block lets two blocks share an SM where the full-page form fits one.
  const uint4* src4 = reinterpret_cast<const uint4*>(kv_cache + (size_t)page * V5_PAGE * V5_ENTRY);
  const uint8_t* qrow = q + (size_t)i * H * 128;
  float* out = logits + (size_t)i * max_ctx + pos0;
  const uint8_t* sk = s_k[warp];
  const float* ssf = s_sf[warp];
  for (int half = 0; half < 2; ++half) {
    const int row0 = half * V5_HALF;
    if (row0 >= n_valid) break;
    const int nv = min(V5_HALF, n_valid - row0);
    __syncwarp();
    const uint4* src4h = src4 + half * 264;
#pragma unroll
    for (int it = 0; it < 9; ++it) {
      const int c = it * 32 + lane;
      if (c < 264) {
        const uint4 v = src4h[c];
        const uint32_t wv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int word = c * 4 + j;
          const int r = word / 33, part = word - r * 33;
          if (part < 32) *reinterpret_cast<uint32_t*>(s_k[warp] + r * 128 + part * 4) = wv[j];
          else s_sf[warp][r] = __uint_as_float(wv[j]);
        }
      }
    }
    __syncwarp();
    for (int h0 = 0; h0 < H; h0 += 16) {
      const int ha = h0 + g, hb = h0 + g + 8;
      const bool has_a = ha < H, has_b = hb < H;
      const float sa = has_a ? ue8m0_to_float(sfq[(size_t)i * H + ha]) : 0.f;
      const float sb = has_b ? ue8m0_to_float(sfq[(size_t)i * H + hb]) : 0.f;
      const float wa = has_a ? __bfloat162float(w[(size_t)i * H + ha]) : 0.f;
      const float wb = has_b ? __bfloat162float(w[(size_t)i * H + hb]) : 0.f;
      uint32_t af[4][4];
      for (int st = 0; st < 4; ++st) {
        const int k0 = st * 32;
        af[st][0] = has_a ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)ha * 128 + k0 + 4 * t) : 0u;
        af[st][1] = has_b ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)hb * 128 + k0 + 4 * t) : 0u;
        af[st][2] = has_a ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)ha * 128 + k0 + 16 + 4 * t) : 0u;
        af[st][3] = has_b ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)hb * 128 + k0 + 16 + 4 * t) : 0u;
      }
      for (int n0 = 0; n0 < nv; n0 += 8) {
        const int col = n0 + g;
        const bool has_col = col < nv;
        const uint8_t* brow = sk + (size_t)(has_col ? col : 0) * 128;
        float part[4] = {0.f, 0.f, 0.f, 0.f};
        for (int st = 0; st < 4; ++st) {
          const int k0 = st * 32;
          uint32_t bf[2];
          bf[0] = has_col ? *reinterpret_cast<const uint32_t*>(brow + k0 + 4 * t) : 0u;
          bf[1] = has_col ? *reinterpret_cast<const uint32_t*>(brow + k0 + 16 + 4 * t) : 0u;
          mma_e4m3_e4m3_v5(part, af[st], bf);
        }
        const int l0 = n0 + 2 * t, l1 = l0 + 1;
        const bool in0 = l0 < nv, in1 = l1 < nv;
        const float sk0 = in0 ? ssf[l0] : 0.f;
        const float sk1 = in1 ? ssf[l1] : 0.f;
        const int c0 = row0 + l0, c1 = row0 + l1;
        float v0 = fmaxf(part[0] * (sa * sk0), 0.f) * wa + fmaxf(part[2] * (sb * sk0), 0.f) * wb;
        float v1 = fmaxf(part[1] * (sa * sk1), 0.f) * wa + fmaxf(part[3] * (sb * sk1), 0.f) * wb;
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
void fp8_paged_mqa_logits_sm120_v5h(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv_cache, torch::Tensor w,
                                   torch::Tensor context_lens, torch::Tensor block_table, torch::Tensor logits, int64_t max_pages) {
  const int S = (int)q.size(0), H = (int)q.size(1), max_ctx = (int)logits.size(1);
  TORCH_CHECK(q.scalar_type() == torch::kFloat8_e4m3fn && q.is_contiguous() && q.size(2) == 128, "q: e4m3 [S, H, 128]");
  TORCH_CHECK(sfq.scalar_type() == torch::kUInt8 && sfq.numel() == (int64_t)S * H && sfq.is_contiguous(), "sfq: uint8 UE8M0 [S, H]");
  TORCH_CHECK(kv_cache.scalar_type() == torch::kUInt8 && kv_cache.is_contiguous() && kv_cache.dim() == 4 && kv_cache.size(1) == 64 && kv_cache.size(2) == 1 && kv_cache.size(3) == 132,
              "kv_cache: uint8 [num_blocks, 64, 1, 132] (128 e4m3 bytes then the fp32 scale per row)");
  TORCH_CHECK(w.scalar_type() == torch::kBFloat16 && w.size(0) == S && w.size(1) == H && w.is_contiguous(), "weights: bf16 [S, H]");
  TORCH_CHECK(context_lens.scalar_type() == torch::kInt && context_lens.numel() == S, "context_lens: int32 [S]");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt && block_table.is_contiguous() && block_table.size(0) == S && block_table.size(1) == max_pages, "block_table: int32 [S, max_pages]");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat && logits.size(0) == S && logits.is_contiguous(), "logits: fp32 [S, max_context_len]");
  auto st = at::cuda::getCurrentCUDAStream();
  k_paged_mqa_logits_v5h<<<dim3((S + V5_WARPS - 1) / V5_WARPS, (int)max_pages), V5_WARPS * 32, 0, st>>>(
      static_cast<const uint8_t*>(q.data_ptr()), sfq.data_ptr<uint8_t>(), kv_cache.data_ptr<uint8_t>(),
      reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), context_lens.data_ptr<int>(), block_table.data_ptr<int>(),
      logits.data_ptr<float>(), S, H, (int)max_pages, max_ctx);
}
__global__ void __launch_bounds__(V5_WARPS * 32)
k_paged_mqa_logits_v5d(const uint8_t* __restrict__ q, const uint8_t* __restrict__ sfq, const uint8_t* __restrict__ kv_cache,
                      const __nv_bfloat16* __restrict__ w, const int* __restrict__ ctx, const int* __restrict__ block_table,
                      float* __restrict__ logits, int S, int H, int max_pages, int max_ctx) {
  __shared__ __align__(16) uint8_t s_k[V5_WARPS][V5_HALF * 128];
  __shared__ __align__(16) float s_sf[V5_WARPS][V5_HALF];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int i = blockIdx.x * V5_WARPS + warp;
  const int p = blockIdx.y;
  if (i >= S) return;
  const int len = ctx[i];
  const int pos0 = p * V5_PAGE;
  if (pos0 >= len) return;
  const int n_valid = min(V5_PAGE, len - pos0);
  const int page = block_table[(size_t)i * max_pages + p];
  // The page in two halves of 32 rows (4224 bytes = 264 chunks of 16 bytes each, both halves 16-byte aligned): each half is staged
  // lane-strided (8 full passes and a 9th for lanes 0 to 7) into a 4.2 KB buffer, then every head's scores for its rows are formed;
  // 33.8 KB per block lets two blocks share an SM where the full-page form fits one.
  const uint4* src4 = reinterpret_cast<const uint4*>(kv_cache + (size_t)page * V5_PAGE * V5_ENTRY);
  const uint8_t* qrow = q + (size_t)i * H * 128;
  float* out = logits + (size_t)i * max_ctx + pos0;
  const uint8_t* sk = s_k[warp];
  const float* ssf = s_sf[warp];
  // Double buffering through registers: each lane holds its 9 chunks (36 words) of the half being staged; the next half's chunks are
  // loaded into the same registers after this half's are written to shared memory, so their latency overlaps this half's MMAs.
  const int halves = n_valid > V5_HALF ? 2 : 1;
  uint4 pre[9];
#pragma unroll
  for (int it = 0; it < 9; ++it) {
    const int c = it * 32 + lane;
    pre[it] = (c < 264) ? src4[c] : make_uint4(0u, 0u, 0u, 0u);
  }
  for (int half = 0; half < halves; ++half) {
    const int row0 = half * V5_HALF;
    const int nv = min(V5_HALF, n_valid - row0);
    __syncwarp();
#pragma unroll
    for (int it = 0; it < 9; ++it) {
      const int c = it * 32 + lane;
      if (c < 264) {
        const uint32_t wv[4] = {pre[it].x, pre[it].y, pre[it].z, pre[it].w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int word = c * 4 + j;
          const int r = word / 33, part = word - r * 33;
          if (part < 32) *reinterpret_cast<uint32_t*>(s_k[warp] + r * 128 + part * 4) = wv[j];
          else s_sf[warp][r] = __uint_as_float(wv[j]);
        }
      }
    }
    __syncwarp();
    if (half + 1 < halves) {
      const uint4* src4n = src4 + 264;
#pragma unroll
      for (int it = 0; it < 9; ++it) {
        const int c = it * 32 + lane;
        pre[it] = (c < 264) ? src4n[c] : make_uint4(0u, 0u, 0u, 0u);
      }
    }
    for (int h0 = 0; h0 < H; h0 += 16) {
      const int ha = h0 + g, hb = h0 + g + 8;
      const bool has_a = ha < H, has_b = hb < H;
      const float sa = has_a ? ue8m0_to_float(sfq[(size_t)i * H + ha]) : 0.f;
      const float sb = has_b ? ue8m0_to_float(sfq[(size_t)i * H + hb]) : 0.f;
      const float wa = has_a ? __bfloat162float(w[(size_t)i * H + ha]) : 0.f;
      const float wb = has_b ? __bfloat162float(w[(size_t)i * H + hb]) : 0.f;
      uint32_t af[4][4];
      for (int st = 0; st < 4; ++st) {
        const int k0 = st * 32;
        af[st][0] = has_a ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)ha * 128 + k0 + 4 * t) : 0u;
        af[st][1] = has_b ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)hb * 128 + k0 + 4 * t) : 0u;
        af[st][2] = has_a ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)ha * 128 + k0 + 16 + 4 * t) : 0u;
        af[st][3] = has_b ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)hb * 128 + k0 + 16 + 4 * t) : 0u;
      }
      for (int n0 = 0; n0 < nv; n0 += 8) {
        const int col = n0 + g;
        const bool has_col = col < nv;
        const uint8_t* brow = sk + (size_t)(has_col ? col : 0) * 128;
        float part[4] = {0.f, 0.f, 0.f, 0.f};
        for (int st = 0; st < 4; ++st) {
          const int k0 = st * 32;
          uint32_t bf[2];
          bf[0] = has_col ? *reinterpret_cast<const uint32_t*>(brow + k0 + 4 * t) : 0u;
          bf[1] = has_col ? *reinterpret_cast<const uint32_t*>(brow + k0 + 16 + 4 * t) : 0u;
          mma_e4m3_e4m3_v5(part, af[st], bf);
        }
        const int l0 = n0 + 2 * t, l1 = l0 + 1;
        const bool in0 = l0 < nv, in1 = l1 < nv;
        const float sk0 = in0 ? ssf[l0] : 0.f;
        const float sk1 = in1 ? ssf[l1] : 0.f;
        const int c0 = row0 + l0, c1 = row0 + l1;
        float v0 = fmaxf(part[0] * (sa * sk0), 0.f) * wa + fmaxf(part[2] * (sb * sk0), 0.f) * wb;
        float v1 = fmaxf(part[1] * (sa * sk1), 0.f) * wa + fmaxf(part[3] * (sb * sk1), 0.f) * wb;
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
void fp8_paged_mqa_logits_sm120_v5d(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv_cache, torch::Tensor w,
                                   torch::Tensor context_lens, torch::Tensor block_table, torch::Tensor logits, int64_t max_pages) {
  const int S = (int)q.size(0), H = (int)q.size(1), max_ctx = (int)logits.size(1);
  TORCH_CHECK(q.scalar_type() == torch::kFloat8_e4m3fn && q.is_contiguous() && q.size(2) == 128, "q: e4m3 [S, H, 128]");
  TORCH_CHECK(sfq.scalar_type() == torch::kUInt8 && sfq.numel() == (int64_t)S * H && sfq.is_contiguous(), "sfq: uint8 UE8M0 [S, H]");
  TORCH_CHECK(kv_cache.scalar_type() == torch::kUInt8 && kv_cache.is_contiguous() && kv_cache.dim() == 4 && kv_cache.size(1) == 64 && kv_cache.size(2) == 1 && kv_cache.size(3) == 132,
              "kv_cache: uint8 [num_blocks, 64, 1, 132] (128 e4m3 bytes then the fp32 scale per row)");
  TORCH_CHECK(w.scalar_type() == torch::kBFloat16 && w.size(0) == S && w.size(1) == H && w.is_contiguous(), "weights: bf16 [S, H]");
  TORCH_CHECK(context_lens.scalar_type() == torch::kInt && context_lens.numel() == S, "context_lens: int32 [S]");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt && block_table.is_contiguous() && block_table.size(0) == S && block_table.size(1) == max_pages, "block_table: int32 [S, max_pages]");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat && logits.size(0) == S && logits.is_contiguous(), "logits: fp32 [S, max_context_len]");
  auto st = at::cuda::getCurrentCUDAStream();
  k_paged_mqa_logits_v5d<<<dim3((S + V5_WARPS - 1) / V5_WARPS, (int)max_pages), V5_WARPS * 32, 0, st>>>(
      static_cast<const uint8_t*>(q.data_ptr()), sfq.data_ptr<uint8_t>(), kv_cache.data_ptr<uint8_t>(),
      reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), context_lens.data_ptr<int>(), block_table.data_ptr<int>(),
      logits.data_ptr<float>(), S, H, (int)max_pages, max_ctx);
}
"""


def _fp32_weights_variant(cpp: str, cuda: str) -> tuple[str, str]:
    """The same two kernels reading ``weights`` as fp32, the operand vLLM's indexer hands over (q's per-token scale folded in).
    Converting those weights to bf16 costs up to 2.8e-3 relative on the logits against an fp32 reference
    (reports/indexer-adapter-weights-precision-rtx5090-20261004.json), so the engine path reads them as they are. The variant is
    derived from the bf16 source by text: the weights pointer type, the one read of it, its TORCH_CHECK, and every v5 name
    suffixed ``f``; the constants are shared and defined once."""
    import re

    def conv(t: str) -> str:
        t = t.replace("const __nv_bfloat16* __restrict__ w", "const float* __restrict__ w")
        t = t.replace("const __nv_bfloat16* pw", "const float* pw")
        t = t.replace("__bfloat162float(w[", "(w[")
        t = t.replace("w.scalar_type() == torch::kBFloat16", "w.scalar_type() == torch::kFloat")
        t = t.replace('"weights: bf16 [S, H]"', '"weights: fp32 [S, H]"')
        t = t.replace("reinterpret_cast<const __nv_bfloat16*>(w.data_ptr())", "w.data_ptr<float>()")
        for name in ("mma_e4m3_e4m3_v5", "k_mqa_logits_v5", "launch_v5", "fp8_mqa_logits_sm120_v5", "k_paged_mqa_logits_v5",
                     "fp8_paged_mqa_logits_sm120_v5", "k_paged_mqa_logits_v5h", "fp8_paged_mqa_logits_sm120_v5h",
                     "k_paged_mqa_logits_v5d", "fp8_paged_mqa_logits_sm120_v5d"):
            t = re.sub(r"\b" + name + r"\b", name + "f", t)
        return t

    cuda_f = conv(cuda).replace("constexpr int V5_PAGE = 64, V5_WARPS = 8, V5_ENTRY = 132, V5_HALF = 32;", "")
    assert cuda_f.count("const float* __restrict__ w") == 4 and "__bfloat162float(w[" not in cuda_f
    assert cuda_f.count('"weights: fp32 [S, H]"') == 4
    return conv(cpp), cuda_f


CPP_V5F, CUDA_V5F = _fp32_weights_variant(CPP_V5, CUDA_V5)


def build(verbose: bool = False):
    return load_inline(name="sm120fp4_fp8_mqa_logits_v5i", cpp_sources=base.CPP + CPP_V5 + CPP_V5F,
                       cuda_sources=base.CUDA + CUDA_V5 + CUDA_V5F,
                       functions=["fp8_mqa_logits_sm120_v5", "fp8_paged_mqa_logits_sm120_v5", "fp8_paged_mqa_logits_sm120_v5h", "fp8_paged_mqa_logits_sm120_v5d",
                                  "fp8_mqa_logits_sm120_v5f", "fp8_paged_mqa_logits_sm120_v5f", "fp8_paged_mqa_logits_sm120_v5hf", "fp8_paged_mqa_logits_sm120_v5df"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a"], verbose=verbose)


def flat_fn(mod, w: torch.Tensor):
    """The flat entry point for the weights' dtype (bf16: the measured kernel; fp32: the engine-operand variant)."""
    return mod.fp8_mqa_logits_sm120_v5f if w.dtype == torch.float32 else mod.fp8_mqa_logits_sm120_v5


def paged_fn(mod, w: torch.Tensor, half: bool = False, double: bool = False):
    """The paged entry point: full-page staging (v5), two halves of 32 rows (v5h), or the halves with the next one prefetched into
    registers (v5d), for the weights' dtype."""
    if double:
        return mod.fp8_paged_mqa_logits_sm120_v5df if w.dtype == torch.float32 else mod.fp8_paged_mqa_logits_sm120_v5d
    if half:
        return mod.fp8_paged_mqa_logits_sm120_v5hf if w.dtype == torch.float32 else mod.fp8_paged_mqa_logits_sm120_v5h
    return mod.fp8_paged_mqa_logits_sm120_v5f if w.dtype == torch.float32 else mod.fp8_paged_mqa_logits_sm120_v5


def sm_count() -> int:
    return torch.cuda.get_device_properties(0).multi_processor_count


def plan_v5(S: int, span: int) -> tuple[int, int, int]:
    """v2's planner with the 128-row segment in place of 256 (a 128 B row doubles the segment's bytes)."""
    rows, kvseg, group = base.plan_v2(S, span, sm_count())
    return rows, (128 if kvseg == 256 else kvseg), group


def quantize_k_fp8(k: torch.Tensor):
    """k [N, 128] fp32 -> e4m3 [N, 128] with one fp32 scale per row (amax / 448), the engine's fp8 indexer cache recipe."""
    amax = k.abs().float().amax(dim=1, keepdim=True).clamp_min(1e-4)
    scale = amax / E4M3_MAX
    k8 = (k.float() / scale).to(torch.float8_e4m3fn).contiguous()
    return k8, scale.squeeze(1).contiguous()


def reference(q8, sfq_f, k8, k_scale, w, ks, ke, max_k):
    S, H, D = q8.shape
    qf = base.dequant_fp8(q8.reshape(S * H, D), sfq_f).reshape(S, H, D)
    kf = k8.float() * k_scale.float().unsqueeze(1)
    score = torch.einsum("mhd,nd->hmn", qf, kf)
    logits_full = torch.einsum("hmn,mh->mn", score.relu(), w.float())
    out = torch.full((S, max_k), float("-inf"), device=q8.device, dtype=torch.float32)
    for i in range(S):
        a, b = int(ks[i]), int(ke[i])
        out[i, : b - a] = logits_full[i, a:b]
    return out


def launch_v5(mod, q8, sfq_u8, k8, k_scale, w, ks, ke, out, plan=None, span=None):
    lo, hi = span or base.span_of(ks, ke)
    rows, kvseg, group = plan or plan_v5(q8.shape[0], hi - lo)
    flat_fn(mod, w)(q8, sfq_u8, k8, k_scale, w, ks, ke, out, lo, hi, rows, kvseg, group)


def run_case(mod, S, N, H, seed, dev, full_span=False, unit_q_scale=False, weights_fp32=False):
    q, kv, w, ks, ke = base.make_case(S, N, H, seed, dev, full_span)
    q8, sfq_packed = ref.per_token_cast_to_fp8(q.reshape(S * H, HEAD_DIM), use_ue8m0=True, gran_k=HEAD_DIM, use_packed_ue8m0=True)
    sfq_f = ref.unpack_ue8m0_from_int(sfq_packed)[:, :1]
    if unit_q_scale:
        # the engine's form: q's scale lives in weights; the kernel sees a scale of one
        w = (w.float() * sfq_f.reshape(S, H)).to(torch.bfloat16)
        sfq_f = torch.ones_like(sfq_f)
    if weights_fp32:
        # the engine's operand: fp32 with full mantissas (the reference below reads w as given, so the f variant is held to 1e-5)
        gw = torch.Generator().manual_seed(seed + 77)
        w = ((torch.rand(S, H, generator=gw) + 0.05).to(dev) * (w.float() / w.float().clamp_min(1e-30))).float().contiguous()
    sfq_u8 = (torch.round(torch.log2(sfq_f)) + 127).clamp(0, 255).to(torch.uint8).reshape(S, H).contiguous()
    q8 = q8.reshape(S, H, HEAD_DIM).contiguous()
    k8, k_scale = quantize_k_fp8(kv)
    max_k = int((ke - ks).max())
    out = torch.full((S, max_k), float("-inf"), device=dev, dtype=torch.float32)
    launch_v5(mod, q8, sfq_u8, k8, k_scale, w, ks, ke, out)
    torch.cuda.synchronize()
    exact = reference(q8, sfq_f, k8, k_scale, w, ks, ke, max_k)
    valid = torch.isfinite(exact)
    diff = (out[valid] - exact[valid]).abs()
    scale = exact[valid].abs().max().clamp_min(1e-30)
    res = {"S": S, "N": N, "H": H, "max_k": max_k, "unit_q_scale": unit_q_scale, "weights_fp32": weights_fp32, "max_abs_err": float(diff.max()), "ref_abs_max": float(scale),
           "rel_max_err": float(diff.max() / scale), "untouched_outside_span": bool(torch.equal(torch.isfinite(out), valid))}
    res["pass"] = res["rel_max_err"] < 1e-5 and res["untouched_outside_span"]
    return res, (q8, sfq_u8, k8, k_scale, w, ks, ke, out)


PAGE = 64


def make_paged_fp8(k8, k_scale, S, ctx_lens, seed, dev):
    """vLLM's fp8 indexer cache layout: [pages, 64, 1, 132] uint8, each row 128 e4m3 bytes then its fp32 scale (little endian);
    every row of the batch reads positions [0, ctx_lens[i]) of the same flat k through a random page permutation."""
    g = torch.Generator().manual_seed(seed)
    N = k8.shape[0]
    n_pages = -(-N // PAGE)
    pad = n_pages * PAGE - N
    k_flat = torch.cat([k8, torch.zeros(pad, 128, dtype=k8.dtype, device=dev)]) if pad else k8
    s_flat = torch.cat([k_scale, torch.zeros(pad, dtype=k_scale.dtype, device=dev)]) if pad else k_scale
    rows = torch.cat([k_flat.view(torch.uint8), s_flat.contiguous().view(torch.uint8).view(-1, 4)], dim=1)   # [n_pages*64, 132]
    perm = torch.randperm(n_pages, generator=g).to(dev)
    kv_cache = torch.empty(n_pages, PAGE, 1, 132, dtype=torch.uint8, device=dev)
    kv_cache[perm] = rows.view(n_pages, PAGE, 1, 132)
    max_pages = int(-(-int(ctx_lens.max()) // PAGE))
    block_table = perm[:max_pages].unsqueeze(0).repeat(S, 1).to(torch.int32).contiguous()
    return kv_cache.contiguous(), block_table, max_pages


def run_paged_case(mod, S, N, H, seed, dev, weights_fp32=False, half=False, double=False):
    q, kv, w, ks, ke = base.make_case(S, N, H, seed, dev, full_span=True)
    q8, sfq_packed = ref.per_token_cast_to_fp8(q.reshape(S * H, HEAD_DIM), use_ue8m0=True, gran_k=HEAD_DIM, use_packed_ue8m0=True)
    sfq_f = ref.unpack_ue8m0_from_int(sfq_packed)[:, :1]
    w = (w.float() * sfq_f.reshape(S, H)).to(torch.bfloat16)      # the engine's form: q's scale in weights
    sfq_f = torch.ones_like(sfq_f)
    if weights_fp32:
        gw = torch.Generator().manual_seed(seed + 77)
        w = ((torch.rand(S, H, generator=gw) + 0.05).to(dev) * (w.float() / w.float().clamp_min(1e-30))).float().contiguous()
    sfq_u8 = torch.full((S, H), 127, dtype=torch.uint8, device=dev)
    q8 = q8.reshape(S, H, HEAD_DIM).contiguous()
    k8, k_scale = quantize_k_fp8(kv)
    g = torch.Generator().manual_seed(seed + 7)
    ctx = torch.randint(1, N + 1, (S,), generator=g).to(torch.int32).to(dev)
    kv_cache, block_table, max_pages = make_paged_fp8(k8, k_scale, S, ctx, seed, dev)
    max_ctx = int(ctx.max())
    out = torch.full((S, max_ctx), float("-inf"), device=dev, dtype=torch.float32)
    paged_fn(mod, w, half, double)(q8, sfq_u8, kv_cache, w, ctx, block_table, out, max_pages)
    ks0 = torch.zeros(S, dtype=torch.int32, device=dev)
    flat = torch.full((S, max_ctx), float("-inf"), device=dev, dtype=torch.float32)
    launch_v5(mod, q8, sfq_u8, k8, k_scale, w, ks0, ctx, flat, span=(0, max_ctx))
    torch.cuda.synchronize()
    valid = torch.isfinite(flat)
    same = bool(torch.equal(out, flat))
    untouched = bool(torch.equal(torch.isfinite(out), valid))
    exact = reference(q8, sfq_f, k8, k_scale, w, ks0, ctx, max_ctx)
    rel = float((out[valid] - exact[valid]).abs().max() / exact[valid].abs().max().clamp_min(1e-30))
    res = {"S": S, "N": N, "H": H, "weights_fp32": weights_fp32, "half": half, "double": double, "pages": int(kv_cache.shape[0]), "max_pages": max_pages, "bit_identical_to_flat_v5": same,
           "untouched_outside_context": untouched, "rel_max_err": rel, "pass": same and untouched and rel < 1e-5}
    return res, (q8, sfq_u8, kv_cache, w, ctx, block_table, out, max_pages)


def selftest_paged(mod, dev) -> int:
    ok = True
    print("paged v5 (vLLM's [pages, 64, 1, 132] fp8 cache layout) against the flat v5 on the same k through random page permutations, and against the reference")
    for (S, N, H, seed, f32, half) in ((8, 512, 8, 31, False, False), (33, 2048, 8, 32, False, False), (64, 4096, 16, 33, False, False), (16, 1000, 8, 34, False, False), (33, 2048, 8, 35, True, False), (16, 1000, 8, 36, True, False),
                                        (8, 512, 8, 37, False, True), (33, 2048, 8, 38, False, True), (64, 4096, 16, 39, False, True), (16, 1000, 8, 40, True, True), (5, 70, 8, 41, False, True)):
        r, _ = run_paged_case(mod, S, N, H, seed, dev, weights_fp32=f32, half=half)
        ok &= r["pass"]
        if half:
            r2, _ = run_paged_case(mod, S, N, H, seed, dev, weights_fp32=f32, half=False, double=True)
            ok &= r2["pass"]
            print(f"  double-buffered (v5d) S={S} N={N} H={H}: bit-identical to flat v5 {r2['bit_identical_to_flat_v5']} -> {'ok' if r2['pass'] else 'FAIL'}", flush=True)
        ok &= r["pass"]
        print(f"  S={S} N={N} H={H} pages={r['pages']}: bit-identical to flat v5 {r['bit_identical_to_flat_v5']}, outside context untouched {r['untouched_outside_context']}, rel max err {r['rel_max_err']:.2e} -> {'ok' if r['pass'] else 'FAIL'}", flush=True)
    print("paged v5 selftest:", "ok" if ok else "FAIL")
    return 0 if ok else 1


def bench_paged(mod, dev, out: Path | None, half: bool = False, double: bool = False) -> int:
    if out is not None and out.exists():
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 2
    props = torch.cuda.get_device_properties(0)
    flush_buf = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    rows = []
    for (S, N, H) in ((64, 8192, 8), (128, 16384, 8), (32, 65536, 16)):
        q, kv, w, ks, ke = base.make_case(S, N, H, 400 + S, dev, full_span=True)
        q8, sfq_packed = ref.per_token_cast_to_fp8(q.reshape(S * H, HEAD_DIM), use_ue8m0=True, gran_k=HEAD_DIM, use_packed_ue8m0=True)
        sfq_f = ref.unpack_ue8m0_from_int(sfq_packed)[:, :1]
        w = (w.float() * sfq_f.reshape(S, H)).to(torch.bfloat16)
        sfq_u8 = torch.full((S, H), 127, dtype=torch.uint8, device=dev)
        q8 = q8.reshape(S, H, HEAD_DIM).contiguous()
        k8, k_scale = quantize_k_fp8(kv)
        ctx = torch.full((S,), N, dtype=torch.int32, device=dev)
        kv_cache, bt, max_pages = make_paged_fp8(k8, k_scale, S, ctx, 400 + S, dev)
        out_full = torch.full((S, N), float("-inf"), device=dev, dtype=torch.float32)
        times = []
        for _ in range(10):
            flush_buf.fill_(1)
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); paged_fn(mod, w, half, double)(q8, sfq_u8, kv_cache, w, ctx, bt, out_full, max_pages); e1.record()
            torch.cuda.synchronize()
            times.append(e0.elapsed_time(e1) * 1000)
        med = statistics.median(times)
        rows.append({"S": S, "N": N, "H": H, "pages": int(kv_cache.shape[0]), "us_median": med, "us_min": min(times), "kv_rows_read": S * N,
                     "kv_GBps": S * N * 132 / med / 1e3, "TFLOPs": 2.0 * S * H * N * HEAD_DIM / med / 1e6})
        print(f"paged v5 S={S} N={N} H={H}: {med:.1f} us ({rows[-1]['kv_GBps']:.0f} GB/s of cache rows, {rows[-1]['TFLOPs']:.1f} TFLOP/s)", flush=True)
    report = {"staging": "two halves with the next prefetched into registers (v5d)" if double else ("two halves of 32 rows (v5h)" if half else "the full page (v5)"), "kernel": "fp8_paged_mqa_logits_sm120_v5: one 132-byte-entry page per warp per step (vLLM's fp8 indexer cache layout), staged with 4-byte loads, v5's fold",
              "device": props.name, "note": "cold L2 (256 MB fill before each launch); median of 10; every row reads the whole cache; the cache rate counts logical rows read (S x N) as the v3 and v4 reports did; the timed region holds the launch only", "rows": rows}
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print("->", out)
    return 0


def selftest(mod, dev) -> int:
    ok = True
    print("v5 (e4m3 k with an fp32 scale per row) against torch.einsum on the dequantised operands")
    for (S, N, H, seed, unit, f32) in ((16, 512, 8, 1, False, False), (32, 1024, 16, 2, False, False), (64, 4096, 8, 3, False, False), (8, 256, 32, 4, False, False), (48, 2048, 8, 5, True, False), (24, 777, 8, 6, True, False),
                                      (48, 2048, 8, 7, True, True), (64, 4096, 16, 8, True, True), (7, 300, 32, 9, True, True)):
        r, _ = run_case(mod, S, N, H, seed, dev, unit_q_scale=unit, weights_fp32=f32)
        ok &= r["pass"]
        print(f"  S={S} N={N} H={H}{' (q scale in weights)' if unit else ''}{' fp32 weights' if f32 else ''}: rel max err {r['rel_max_err']:.2e}, outside span untouched {r['untouched_outside_span']} -> {'ok' if r['pass'] else 'FAIL'}", flush=True)
    print("fp8_mqa_logits v5 selftest:", "ok" if ok else "FAIL")
    return 0 if ok else 1


def bench(mod, dev, out: Path | None, weights_fp32: bool = False) -> int:
    if out is not None and out.exists():
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 2
    props = torch.cuda.get_device_properties(0)
    flush_buf = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    rows = []
    for (S, N, H) in ((32, 4096, 8), (128, 8192, 8), (32, 32768, 16)):
        r, args = run_case(mod, S, N, H, 100 + S, dev, full_span=True, unit_q_scale=True, weights_fp32=weights_fp32)
        q8, sfq_u8, k8, k_scale, w, ks, ke, out_t = args
        span = base.span_of(ks, ke)
        plan = plan_v5(S, span[1] - span[0])
        times = []
        for _ in range(10):
            flush_buf.fill_(1)
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); launch_v5(mod, q8, sfq_u8, k8, k_scale, w, ks, ke, out_t, plan, span); e1.record()
            torch.cuda.synchronize()
            times.append(e0.elapsed_time(e1) * 1000)
        med = statistics.median(times)
        kbytes = k8.numel() + k_scale.numel() * 4
        rows.append({"S": S, "N": N, "H": H, "plan": list(plan), "us_median": med, "us_min": min(times), "TFLOPs": 2.0 * S * H * N * HEAD_DIM / med / 1e6,
                     "k_bytes": kbytes, "k_GBps": kbytes / med / 1e3})
        print(f"v5 S={S} N={N} H={H}: {med:.1f} us ({rows[-1]['TFLOPs']:.1f} TFLOP/s, k read at {rows[-1]['k_GBps']:.0f} GB/s)", flush=True)
    report = {"weights": "fp32 (the v5f variant)" if weights_fp32 else "bf16 (v5)", "kernel": "fp8_mqa_logits_sm120_v5: v2's structure with e4m3 k rows (128 B) and an fp32 scale per row, the fp8 indexer cache vLLM runs on SM120",
              "device": props.name, "note": "cold L2 (256 MB fill before each launch); median of 10; q's scale folded into weights (the engine's form); the timed region holds the launch only", "rows": rows}
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
    ap.add_argument("--fp32-weights", action="store_true", help="time the fp32-weights variant (v5f) in --bench")
    ap.add_argument("--half", action="store_true", help="--bench-paged times the half-page staging (v5h)")
    ap.add_argument("--double", action="store_true", help="--bench-paged times the register-prefetched halves (v5d)")
    ap.add_argument("--out-paged", type=Path)
    a = ap.parse_args(argv)
    dev = torch.device("cuda")
    base.SM_COUNT = sm_count()
    mod = build()
    rc = 0
    if a.selftest:
        rc = selftest(mod, dev) or selftest_paged(mod, dev)
    if a.bench:
        rc = rc or bench(mod, dev, a.out, weights_fp32=a.fp32_weights)
    if a.bench_paged:
        rc = rc or bench_paged(mod, dev, a.out_paged, half=a.half, double=a.double)
    return rc


if __name__ == "__main__":
    sys.exit(main())
