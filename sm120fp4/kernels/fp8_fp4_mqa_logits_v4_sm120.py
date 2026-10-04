#!/usr/bin/env python3
"""MQA-logits v4: the engine's k operand format. vLLM's sparse-attention indexer hands DeepGEMM an MXFP4 k with one UE8M0 scale per
32 columns along the head ([N, D/32] = [N, 4] for D = 128) and a q whose per-token scale is folded into `weights`
(docs/stage3-engine-wiring.md, Section 3b). v2 takes one UE8M0 scale per k row over the whole head. v4 is v2's kernel with the
k scale folded per 32-column block: each of the four k32 MMA steps is one block, so each step's partial product is scaled by that
block's UE8M0 before it joins the accumulator (four folds per n8 tile instead of one). The q side keeps v2's per-(row, head) UE8M0
scale; an all-ones scale (byte 127) is the engine's form, where the scale has gone into `weights`.

What the selftest checks:
  (i)  with the four block scales of every k row equal, v4 against v2 on the same inputs: the counts of differing elements and the
       relative difference are recorded (multiplication by a power of two is exact, but v2 accumulates the four steps inside the MMA
       and v4 adds four scaled partials in fp32, so bit identity is not asserted, it is measured);
  (ii) with the block scales independent, v4 against DeepGEMM's test reference (torch.einsum on the dequantised operands, fp32):
       relative error at most 1e-5 at the output's maximum, the bar the base script uses.
Reuses scripts/fp8_fp4_mqa_logits_sm120.py unchanged: its CUDA source (v0 to v3) is compiled together with the v4 kernel so the
same module exposes v2 for the comparison.

    ~/mlsys-5090-runtime/vllm028/.venv/bin/python scripts/fp8_fp4_mqa_logits_v4_sm120.py --selftest
    ~/mlsys-5090-runtime/vllm028/.venv/bin/python scripts/fp8_fp4_mqa_logits_v4_sm120.py --bench --out reports/fp8-fp4-mqa-logits-v4-rtx5090-20261003.json
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
BLOCK = 32
NBLK = HEAD_DIM // BLOCK

CPP_V4 = r"""
void fp8_fp4_mqa_logits_sm120_v4(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv, torch::Tensor sfkv, torch::Tensor w,
                                 torch::Tensor ks, torch::Tensor ke, torch::Tensor logits, int64_t kv_lo, int64_t kv_hi,
                                 int64_t rows, int64_t kvseg, int64_t group);
void fp8_fp4_paged_mqa_logits_sm120_v4(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv_cache, torch::Tensor sf_cache,
                                       torch::Tensor w, torch::Tensor context_lens, torch::Tensor block_table, torch::Tensor logits,
                                       int64_t max_pages);
"""

CUDA_V4 = r"""
// v4: v2 with the k scale per 32-column block ([N, 4] UE8M0 bytes). The four k32 steps of a tile are the four blocks; each step's
// partial product is scaled by its block's UE8M0 and added to a per-column fp32 accumulator, then relu, weight, shuffle as in v2.
template <int ROWS, int KVSEG>
__global__ void __launch_bounds__(V1_WARPS * 32)
k_mqa_logits_v4(const uint8_t* __restrict__ q, const uint8_t* __restrict__ sfq, const uint8_t* __restrict__ kv,
                const uint8_t* __restrict__ sfkv, const __nv_bfloat16* __restrict__ w, const int* __restrict__ ks,
                const int* __restrict__ ke, float* __restrict__ logits, int S, int H, int N, int max_k, int kv_lo, int kv_hi, int group) {
  constexpr int RPW = ROWS >= V1_WARPS ? ROWS / V1_WARPS : 1;
  __shared__ __align__(16) uint8_t s_kv[2][KVSEG * 64];
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
      if (seg0 + r < seg1) cp_async_16(s_kv[buf] + r * 64 + part * 16, kv + (size_t)(seg0 + r) * 64 + part * 16);
      else *reinterpret_cast<uint4*>(s_kv[buf] + r * 64 + part * 16) = make_uint4(0u, 0u, 0u, 0u);
    }
    for (int r = tid; r < KVSEG; r += V1_WARPS * 32)
      *reinterpret_cast<uint32_t*>(s_sf[buf] + r * 4) = (seg0 + r < seg1) ? *reinterpret_cast<const uint32_t*>(sfkv + (size_t)(seg0 + r) * 4) : 0u;
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
          const uint8_t* brow = skv + (size_t)(has_col ? (col - seg0) : 0) * 64;
          const int c0 = n0 + 2 * t, c1 = c0 + 1;
          const bool in0 = c0 >= n_lo && c0 < n_hi, in1 = c1 >= n_lo && c1 < n_hi;
          const uint8_t* sf0 = ssf + (size_t)(in0 ? (c0 - seg0) : 0) * 4;
          const uint8_t* sf1 = ssf + (size_t)(in1 ? (c1 - seg0) : 0) * 4;
          float acc[4] = {0.f, 0.f, 0.f, 0.f};
          for (int st = 0; st < 4; ++st) {
            const int k0 = st * 32;
            uint32_t bf[2];
            bf[0] = has_col ? unpack_e2m1_x4(*reinterpret_cast<const uint16_t*>(brow + (k0 + 4 * t) / 2)) : 0u;
            bf[1] = has_col ? unpack_e2m1_x4(*reinterpret_cast<const uint16_t*>(brow + (k0 + 16 + 4 * t) / 2)) : 0u;
            float part[4] = {0.f, 0.f, 0.f, 0.f};
            mma_f8f6f4(part, af[st], bf);
            const float sk0 = in0 ? ue8m0_to_float(sf0[st]) : 0.f;
            const float sk1 = in1 ? ue8m0_to_float(sf1[st]) : 0.f;
            acc[0] += part[0] * sk0;
            acc[1] += part[1] * sk1;
            acc[2] += part[2] * sk0;
            acc[3] += part[3] * sk1;
          }
          float v0 = fmaxf(acc[0] * sa, 0.f) * wa + fmaxf(acc[2] * sb, 0.f) * wb;
          float v1 = fmaxf(acc[1] * sa, 0.f) * wa + fmaxf(acc[3] * sb, 0.f) * wb;
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
static void launch_v4(const uint8_t* pq, const uint8_t* psq, const uint8_t* pkv, const uint8_t* psk, const __nv_bfloat16* pw,
                      const int* pks, const int* pke, float* pl, int S, int H, int N, int max_k, int kv_lo, int kv_hi, int group,
                      cudaStream_t st) {
  const int nsegs = (kv_hi - kv_lo + KVSEG - 1) / KVSEG;
  const dim3 grid((S + ROWS - 1) / ROWS, (nsegs + group - 1) / group);
  k_mqa_logits_v4<ROWS, KVSEG><<<grid, V1_WARPS * 32, 0, st>>>(pq, psq, pkv, psk, pw, pks, pke, pl, S, H, N, max_k, kv_lo, kv_hi, group);
}

void fp8_fp4_mqa_logits_sm120_v4(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv, torch::Tensor sfkv, torch::Tensor w,
                                 torch::Tensor ks, torch::Tensor ke, torch::Tensor logits, int64_t kv_lo, int64_t kv_hi,
                                 int64_t rows, int64_t kvseg, int64_t group) {
  const int S = (int)q.size(0), H = (int)q.size(1), N = (int)kv.size(0), max_k = (int)logits.size(1);
  TORCH_CHECK(q.scalar_type() == torch::kFloat8_e4m3fn && q.is_contiguous() && q.size(2) == 128, "q: e4m3 [S, H, 128]");
  TORCH_CHECK(sfq.scalar_type() == torch::kUInt8 && sfq.numel() == (int64_t)S * H && sfq.is_contiguous(), "sfq: uint8 UE8M0 [S, H]");
  TORCH_CHECK(kv.scalar_type() == torch::kInt8 && kv.is_contiguous() && kv.size(1) == 64, "kv: packed e2m1 int8 [N, 64]");
  TORCH_CHECK(sfkv.scalar_type() == torch::kUInt8 && sfkv.dim() == 2 && sfkv.size(0) == N && sfkv.size(1) == 4 && sfkv.is_contiguous(), "sfkv: uint8 UE8M0 [N, 4] (one per 32 columns)");
  TORCH_CHECK(w.scalar_type() == torch::kBFloat16 && w.size(0) == S && w.size(1) == H && w.is_contiguous(), "weights: bf16 [S, H]");
  TORCH_CHECK(ks.scalar_type() == torch::kInt && ke.scalar_type() == torch::kInt && ks.numel() == S && ke.numel() == S, "ks, ke: int32 [S]");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat && logits.size(0) == S && logits.is_contiguous(), "logits: fp32 [S, max_seqlen_k]");
  TORCH_CHECK(kv_lo >= 0 && kv_hi <= N && kv_lo < kv_hi, "kv_lo < kv_hi within [0, N]");
  TORCH_CHECK(group >= 1, "group >= 1");
  auto st = at::cuda::getCurrentCUDAStream();
  const uint8_t* pq = static_cast<const uint8_t*>(q.data_ptr());
  const uint8_t* psq = sfq.data_ptr<uint8_t>();
  const uint8_t* pkv = reinterpret_cast<const uint8_t*>(kv.data_ptr<int8_t>());
  const uint8_t* psk = sfkv.data_ptr<uint8_t>();
  const __nv_bfloat16* pw = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
  float* pl = logits.data_ptr<float>();
  const int* pks = ks.data_ptr<int>();
  const int* pke = ke.data_ptr<int>();
  if (rows == 16 && kvseg == 256) launch_v4<16, 256>(pq, psq, pkv, psk, pw, pks, pke, pl, S, H, N, max_k, (int)kv_lo, (int)kv_hi, (int)group, st);
  else if (rows == 16 && kvseg == 64) launch_v4<16, 64>(pq, psq, pkv, psk, pw, pks, pke, pl, S, H, N, max_k, (int)kv_lo, (int)kv_hi, (int)group, st);
  else if (rows == 8 && kvseg == 256) launch_v4<8, 256>(pq, psq, pkv, psk, pw, pks, pke, pl, S, H, N, max_k, (int)kv_lo, (int)kv_hi, (int)group, st);
  else if (rows == 8 && kvseg == 64) launch_v4<8, 64>(pq, psq, pkv, psk, pw, pks, pke, pl, S, H, N, max_k, (int)kv_lo, (int)kv_hi, (int)group, st);
  else TORCH_CHECK(false, "rows in {16, 8}, kvseg in {256, 64}");
}

// paged v4: v3's page staging (one page of 64 kv rows per warp per step) with the four UE8M0 bytes per row of the page staged
// beside it (sf_cache [num_blocks, 64, 4]) and v4's per-block fold.
__global__ void __launch_bounds__(V3_WARPS * 32)
k_paged_mqa_logits_v4(const uint8_t* __restrict__ q, const uint8_t* __restrict__ sfq, const uint8_t* __restrict__ kv_cache,
                      const uint8_t* __restrict__ sf_cache, const __nv_bfloat16* __restrict__ w, const int* __restrict__ ctx,
                      const int* __restrict__ block_table, float* __restrict__ logits, int S, int H, int max_pages, int max_ctx) {
  __shared__ __align__(16) uint8_t s_kv[V3_WARPS][V3_PAGE * 64];
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
    cp_async_16(s_kv[warp] + r * 64 + part * 16, kv_cache + ((size_t)page * V3_PAGE + r) * 64 + part * 16);
  }
  // 64 rows x 4 bytes = 256 bytes of scales: 16 bytes per lane for 16 lanes
  if (lane < 16) cp_async_16(s_sf[warp] + lane * 16, sf_cache + (size_t)page * V3_PAGE * 4 + lane * 16);
  asm volatile("cp.async.commit_group;\n");
  asm volatile("cp.async.wait_group 0;\n");
  __syncwarp();
  const uint8_t* qrow = q + (size_t)i * H * 128;
  float* out = logits + (size_t)i * max_ctx + pos0;
  const uint8_t* skv = s_kv[warp];
  const uint8_t* ssf = s_sf[warp];
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
      const uint8_t* brow = skv + (size_t)(has_col ? col : 0) * 64;
      const int c0 = n0 + 2 * t, c1 = c0 + 1;
      const bool in0 = c0 < n_valid, in1 = c1 < n_valid;
      const uint8_t* sf0 = ssf + (size_t)(in0 ? c0 : 0) * 4;
      const uint8_t* sf1 = ssf + (size_t)(in1 ? c1 : 0) * 4;
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      for (int st = 0; st < 4; ++st) {
        const int k0 = st * 32;
        uint32_t bf[2];
        bf[0] = has_col ? unpack_e2m1_x4(*reinterpret_cast<const uint16_t*>(brow + (k0 + 4 * t) / 2)) : 0u;
        bf[1] = has_col ? unpack_e2m1_x4(*reinterpret_cast<const uint16_t*>(brow + (k0 + 16 + 4 * t) / 2)) : 0u;
        float part[4] = {0.f, 0.f, 0.f, 0.f};
        mma_f8f6f4(part, af[st], bf);
        const float sk0 = in0 ? ue8m0_to_float(sf0[st]) : 0.f;
        const float sk1 = in1 ? ue8m0_to_float(sf1[st]) : 0.f;
        acc[0] += part[0] * sk0;
        acc[1] += part[1] * sk1;
        acc[2] += part[2] * sk0;
        acc[3] += part[3] * sk1;
      }
      float v0 = fmaxf(acc[0] * sa, 0.f) * wa + fmaxf(acc[2] * sb, 0.f) * wb;
      float v1 = fmaxf(acc[1] * sa, 0.f) * wa + fmaxf(acc[3] * sb, 0.f) * wb;
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

void fp8_fp4_paged_mqa_logits_sm120_v4(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv_cache, torch::Tensor sf_cache,
                                       torch::Tensor w, torch::Tensor context_lens, torch::Tensor block_table, torch::Tensor logits,
                                       int64_t max_pages) {
  const int S = (int)q.size(0), H = (int)q.size(1), max_ctx = (int)logits.size(1);
  TORCH_CHECK(q.scalar_type() == torch::kFloat8_e4m3fn && q.is_contiguous() && q.size(2) == 128, "q: e4m3 [S, H, 128]");
  TORCH_CHECK(sfq.scalar_type() == torch::kUInt8 && sfq.numel() == (int64_t)S * H && sfq.is_contiguous(), "sfq: uint8 UE8M0 [S, H]");
  TORCH_CHECK(kv_cache.scalar_type() == torch::kInt8 && kv_cache.is_contiguous() && kv_cache.dim() == 3 && kv_cache.size(1) == 64 && kv_cache.size(2) == 64,
              "kv_cache: packed e2m1 int8 [num_blocks, 64, 64]");
  TORCH_CHECK(sf_cache.scalar_type() == torch::kUInt8 && sf_cache.is_contiguous() && sf_cache.dim() == 3 && sf_cache.size(0) == kv_cache.size(0) && sf_cache.size(1) == 64 && sf_cache.size(2) == 4,
              "sf_cache: uint8 UE8M0 [num_blocks, 64, 4]");
  TORCH_CHECK(w.scalar_type() == torch::kBFloat16 && w.size(0) == S && w.size(1) == H && w.is_contiguous(), "weights: bf16 [S, H]");
  TORCH_CHECK(context_lens.scalar_type() == torch::kInt && context_lens.numel() == S, "context_lens: int32 [S]");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt && block_table.is_contiguous() && block_table.size(0) == S && block_table.size(1) == max_pages, "block_table: int32 [S, max_pages]");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat && logits.size(0) == S && logits.is_contiguous(), "logits: fp32 [S, max_context_len]");
  auto st = at::cuda::getCurrentCUDAStream();
  k_paged_mqa_logits_v4<<<dim3((S + V3_WARPS - 1) / V3_WARPS, (int)max_pages), V3_WARPS * 32, 0, st>>>(
      static_cast<const uint8_t*>(q.data_ptr()), sfq.data_ptr<uint8_t>(), reinterpret_cast<const uint8_t*>(kv_cache.data_ptr<int8_t>()),
      sf_cache.data_ptr<uint8_t>(), reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), context_lens.data_ptr<int>(),
      block_table.data_ptr<int>(), logits.data_ptr<float>(), S, H, (int)max_pages, max_ctx);
}
"""


def build(verbose: bool = False):
    return load_inline(name="sm120fp4_fp8_fp4_mqa_logits_v4b", cpp_sources=base.CPP + CPP_V4, cuda_sources=base.CUDA + CUDA_V4,
                       functions=["fp8_fp4_mqa_logits_sm120_v0", "fp8_fp4_mqa_logits_sm120_v2", "fp8_fp4_mqa_logits_sm120_v4", "fp8_fp4_paged_mqa_logits_sm120_v4"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a"], verbose=verbose)


def quantize_kv_blocked(kv: torch.Tensor, equal_blocks: bool = False):
    """kv [N, 128] -> packed e2m1 int8 [N, 64] and UE8M0 scales [N, 4] (one per 32 columns), plus the fp32 scales.
    equal_blocks: quantise with one scale per row (v2's recipe) and repeat it over the four blocks, so v2 and v4 see the same codes."""
    if equal_blocks:
        kv4, sf_packed = ref.per_token_cast_to_fp4(kv, use_ue8m0=True, gran_k=HEAD_DIM, use_packed_ue8m0=True)
        sf_row = ref.unpack_ue8m0_from_int(sf_packed)[:, :1]                 # [N, 1] fp32
        sf = sf_row.repeat(1, NBLK).contiguous()                              # [N, 4]
    else:
        kv4, sf = ref.per_token_cast_to_fp4(kv, use_ue8m0=True, gran_k=BLOCK, use_packed_ue8m0=False)   # sf [N, 4] fp32
    sf_u8 = (torch.round(torch.log2(sf.float())) + 127).clamp(0, 255).to(torch.uint8).contiguous()
    return kv4.contiguous(), sf_u8, sf.float()


def dequant_fp4_blocked(kv4: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    return ref.cast_back_from_fp4(kv4, sf.float(), gran_k=BLOCK)


def reference_blocked(q8, sfq_f, kv4, sfkv_f, w, ks, ke, max_k):
    S, H, D = q8.shape
    qf = base.dequant_fp8(q8.reshape(S * H, D), sfq_f).reshape(S, H, D)
    kf = dequant_fp4_blocked(kv4, sfkv_f)
    score = torch.einsum("mhd,nd->hmn", qf, kf)
    logits_full = torch.einsum("hmn,mh->mn", score.relu(), w.float())
    out = torch.full((S, max_k), float("-inf"), device=q8.device, dtype=torch.float32)
    for i in range(S):
        a, b = int(ks[i]), int(ke[i])
        out[i, : b - a] = logits_full[i, a:b]
    return out


def make_case(S, N, H, seed, dev, full_span=False, block_spread=True):
    q, kv, w, ks, ke = base.make_case(S, N, H, seed, dev, full_span)
    if block_spread:
        # give the four 32-column blocks of each k row different magnitudes, so the per-block scales differ
        g = torch.Generator().manual_seed(seed + 7)
        mult = torch.pow(2.0, torch.randint(-3, 4, (N, NBLK), generator=g).float()).to(dev).repeat_interleave(BLOCK, dim=1)
        kv = kv * mult
    return q, kv, w, ks, ke


def sm_count() -> int:
    return base.SM_COUNT or torch.cuda.get_device_properties(0).multi_processor_count


def launch_v4(mod, q8, sfq_u8, kv4, sfkv_u8, w, ks, ke, out, plan=None, span=None):
    lo, hi = span or base.span_of(ks, ke)
    rows, kvseg, group = plan or base.plan_v2(q8.shape[0], hi - lo, sm_count())
    mod.fp8_fp4_mqa_logits_sm120_v4(q8, sfq_u8, kv4, sfkv_u8, w, ks, ke, out, lo, hi, rows, kvseg, group)


def run_case(mod, S, N, H, seed, dev, full_span=False, equal_blocks=False):
    q, kv, w, ks, ke = make_case(S, N, H, seed, dev, full_span, block_spread=not equal_blocks)
    q8, sfq_packed = ref.per_token_cast_to_fp8(q.reshape(S * H, HEAD_DIM), use_ue8m0=True, gran_k=HEAD_DIM, use_packed_ue8m0=True)
    sfq_f = ref.unpack_ue8m0_from_int(sfq_packed)[:, :1]
    sfq_u8 = (torch.round(torch.log2(sfq_f)) + 127).clamp(0, 255).to(torch.uint8).reshape(S, H).contiguous()
    q8 = q8.reshape(S, H, HEAD_DIM).contiguous()
    kv4, sfkv_u8, sfkv_f = quantize_kv_blocked(kv, equal_blocks)
    max_k = int((ke - ks).max())
    out = torch.full((S, max_k), float("-inf"), device=dev, dtype=torch.float32)
    launch_v4(mod, q8, sfq_u8, kv4, sfkv_u8, w, ks, ke, out)
    torch.cuda.synchronize()
    exact = reference_blocked(q8, sfq_f, kv4, sfkv_f, w, ks, ke, max_k)
    valid = torch.isfinite(exact)
    diff = (out[valid] - exact[valid]).abs()
    scale = exact[valid].abs().max().clamp_min(1e-30)
    res = {"S": S, "N": N, "H": H, "max_k": max_k, "equal_blocks": equal_blocks, "max_abs_err": float(diff.max()), "ref_abs_max": float(scale),
           "rel_max_err": float(diff.max() / scale), "untouched_outside_span": bool(torch.equal(torch.isfinite(out), valid))}
    res["pass"] = res["rel_max_err"] < 1e-5 and res["untouched_outside_span"]
    if equal_blocks:
        out2 = torch.full((S, max_k), float("-inf"), device=dev, dtype=torch.float32)
        base.launch(mod, 2, q8, sfq_u8, kv4, sfkv_u8[:, :1].contiguous(), w, ks, ke, out2)
        torch.cuda.synchronize()
        d = (out[valid] - out2[valid]).abs()
        res["rel_max_err_vs_v2"] = float(d.max() / scale)
        res["elements_differing_from_v2"] = int((out[valid] != out2[valid]).sum())
        res["elements"] = int(valid.sum())
        res["pass"] = res["pass"] and res["rel_max_err_vs_v2"] < 1e-6
    return res, (q8, sfq_u8, kv4, sfkv_u8, w, ks, ke, out)


PAGE = 64


def make_paged_blocked(kv4, sfkv_u8, S, ctx_lens, seed, dev):
    """The base script's make_paged with four scale bytes per row: returns (kv_cache [pages, 64, 64] int8, sf_cache [pages, 64, 4] uint8,
    block_table [S, max_pages] int32, max_pages); every row uses positions [0, ctx_lens[i]) of the same flat kv through a random page
    permutation."""
    g = torch.Generator().manual_seed(seed)
    N = kv4.shape[0]
    n_pages = -(-N // PAGE)
    pad = n_pages * PAGE - N
    kv_flat = torch.cat([kv4, torch.zeros(pad, 64, dtype=kv4.dtype, device=dev)]) if pad else kv4
    sf_flat = torch.cat([sfkv_u8, torch.zeros(pad, NBLK, dtype=sfkv_u8.dtype, device=dev)]) if pad else sfkv_u8
    perm = torch.randperm(n_pages, generator=g).to(dev)
    kv_cache = torch.empty(n_pages, PAGE, 64, dtype=kv4.dtype, device=dev)
    sf_cache = torch.empty(n_pages, PAGE, NBLK, dtype=sfkv_u8.dtype, device=dev)
    kv_cache[perm] = kv_flat.view(n_pages, PAGE, 64)
    sf_cache[perm] = sf_flat.view(n_pages, PAGE, NBLK)
    max_pages = int(-(-int(ctx_lens.max()) // PAGE))
    block_table = perm[:max_pages].unsqueeze(0).repeat(S, 1).to(torch.int32).contiguous()
    return kv_cache, sf_cache.contiguous(), block_table, max_pages


def run_paged_case(mod, S, N, H, seed, dev):
    """The paged v4 against the flat v4 on the same kv (ks = 0, ke = ctx), bit for bit, and against the reference."""
    q, kv, w, ks, ke = make_case(S, N, H, seed, dev, full_span=True, block_spread=True)
    q8, sfq_packed = ref.per_token_cast_to_fp8(q.reshape(S * H, HEAD_DIM), use_ue8m0=True, gran_k=HEAD_DIM, use_packed_ue8m0=True)
    sfq_f = ref.unpack_ue8m0_from_int(sfq_packed)[:, :1]
    sfq_u8 = (torch.round(torch.log2(sfq_f)) + 127).clamp(0, 255).to(torch.uint8).reshape(S, H).contiguous()
    q8 = q8.reshape(S, H, HEAD_DIM).contiguous()
    kv4, sfkv_u8, sfkv_f = quantize_kv_blocked(kv)
    g = torch.Generator().manual_seed(seed + 7)
    ctx = torch.randint(1, N + 1, (S,), generator=g).to(torch.int32).to(dev)
    kv_cache, sf_cache, block_table, max_pages = make_paged_blocked(kv4, sfkv_u8, S, ctx, seed, dev)
    max_ctx = int(ctx.max())
    out = torch.full((S, max_ctx), float("-inf"), device=dev, dtype=torch.float32)
    mod.fp8_fp4_paged_mqa_logits_sm120_v4(q8, sfq_u8, kv_cache, sf_cache, w, ctx, block_table, out, max_pages)
    ks0 = torch.zeros(S, dtype=torch.int32, device=dev)
    flat = torch.full((S, max_ctx), float("-inf"), device=dev, dtype=torch.float32)
    launch_v4(mod, q8, sfq_u8, kv4, sfkv_u8, w, ks0, ctx, flat, span=(0, max_ctx))
    torch.cuda.synchronize()
    valid = torch.isfinite(flat)
    same = bool(torch.equal(out, flat))
    untouched = bool(torch.equal(torch.isfinite(out), valid))
    exact = reference_blocked(q8, sfq_f, kv4, sfkv_f, w, ks0, ctx, max_ctx)
    rel = float((out[valid] - exact[valid]).abs().max() / exact[valid].abs().max().clamp_min(1e-30))
    res = {"S": S, "N": N, "H": H, "pages": int(kv_cache.shape[0]), "max_pages": max_pages, "bit_identical_to_flat_v4": same,
           "untouched_outside_context": untouched, "rel_max_err": rel, "pass": same and untouched and rel < 1e-5}
    return res, (q8, sfq_u8, kv_cache, sf_cache, w, ctx, block_table, out, max_pages)


def selftest_paged(mod, dev) -> int:
    ok = True
    print("paged v4 (sf_cache [pages, 64, 4]) against the flat v4 on the same kv through random page permutations, and against the reference")
    for (S, N, H, seed) in ((8, 512, 8, 21), (33, 2048, 8, 22), (64, 4096, 16, 23), (16, 1000, 8, 24)):
        r, _ = run_paged_case(mod, S, N, H, seed, dev)
        ok &= r["pass"]
        print(f"  S={S} N={N} H={H} pages={r['pages']}: bit-identical to flat v4 {r['bit_identical_to_flat_v4']}, outside context untouched {r['untouched_outside_context']}, rel max err {r['rel_max_err']:.2e} -> {'ok' if r['pass'] else 'FAIL'}", flush=True)
    print("paged v4 selftest:", "ok" if ok else "FAIL")
    return 0 if ok else 1


def bench_paged(mod, dev, out: Path | None) -> int:
    if out is not None and out.exists():
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 2
    props = torch.cuda.get_device_properties(0)
    flush_buf = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    rows = []
    for (S, N, H) in ((64, 8192, 8), (128, 16384, 8), (32, 65536, 16)):
        # every row reads the whole kv, the decode case the bench reports
        q, kv, w, ks, ke = make_case(S, N, H, 300 + S, dev, full_span=True, block_spread=True)
        q8, sfq_packed = ref.per_token_cast_to_fp8(q.reshape(S * H, HEAD_DIM), use_ue8m0=True, gran_k=HEAD_DIM, use_packed_ue8m0=True)
        sfq_f = ref.unpack_ue8m0_from_int(sfq_packed)[:, :1]
        sfq_u8 = (torch.round(torch.log2(sfq_f)) + 127).clamp(0, 255).to(torch.uint8).reshape(S, H).contiguous()
        q8 = q8.reshape(S, H, HEAD_DIM).contiguous()
        kv4, sfkv_u8, _ = quantize_kv_blocked(kv)
        ctx = torch.full((S,), N, dtype=torch.int32, device=dev)
        kv_cache, sf_cache, bt, max_pages_full = make_paged_blocked(kv4, sfkv_u8, S, ctx, 300 + S, dev)
        out_full = torch.full((S, N), float("-inf"), device=dev, dtype=torch.float32)
        times = []
        for _ in range(10):
            flush_buf.fill_(1)
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); mod.fp8_fp4_paged_mqa_logits_sm120_v4(q8, sfq_u8, kv_cache, sf_cache, w, ctx, bt, out_full, max_pages_full); e1.record()
            torch.cuda.synchronize()
            times.append(e0.elapsed_time(e1) * 1000)
        med = statistics.median(times)
        kv_bytes = S * N * (64 + 4)   # every row reads every kv row's 64 packed bytes and 4 scale bytes
        rows.append({"S": S, "N": N, "H": H, "pages": int(kv_cache.shape[0]), "us_median": med, "us_min": min(times), "kv_rows_read": S * N, "kv_GBps": kv_bytes / med / 1e3, "TFLOPs": 2.0 * S * H * N * HEAD_DIM / med / 1e6})
        print(f"paged v4 S={S} N={N} H={H}: {med:.1f} us ({rows[-1]['kv_GBps']:.0f} GB/s of kv rows, {rows[-1]['TFLOPs']:.1f} TFLOP/s)", flush=True)
    report = {"kernel": "fp8_fp4_paged_mqa_logits_sm120_v4: v3's page staging with four UE8M0 bytes per cached row (sf_cache [pages, 64, 4]) and v4's per-block fold",
              "device": props.name, "note": "cold L2 (256 MB fill before each launch); median of 10; every row reads the whole kv; the timed region holds the launch only", "rows": rows}
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print("->", out)
    return 0


def selftest(mod, dev) -> int:
    ok = True
    print("v4 (k scale per 32 columns) against the dequantised reference, block scales independent")
    for (S, N, H, seed) in ((16, 512, 8, 1), (32, 1024, 16, 2), (64, 4096, 8, 3), (8, 256, 32, 4), (48, 2048, 8, 5)):
        r, _ = run_case(mod, S, N, H, seed, dev)
        ok &= r["pass"]
        print(f"  S={S} N={N} H={H}: rel max err {r['rel_max_err']:.2e}, outside span untouched {r['untouched_outside_span']} -> {'ok' if r['pass'] else 'FAIL'}", flush=True)
    print("v4 against v2 with the four block scales equal (v2's recipe repeated), measured not asserted")
    for (S, N, H, seed) in ((16, 512, 8, 11), (64, 4096, 8, 12), (32, 2048, 16, 13)):
        r, _ = run_case(mod, S, N, H, seed, dev, equal_blocks=True)
        ok &= r["pass"]
        print(f"  S={S} N={N} H={H}: rel max err vs reference {r['rel_max_err']:.2e}, vs v2 {r['rel_max_err_vs_v2']:.2e}, {r['elements_differing_from_v2']} of {r['elements']} elements differ -> {'ok' if r['pass'] else 'FAIL'}", flush=True)
    print("fp8_fp4_mqa_logits v4 selftest:", "ok" if ok else "FAIL")
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
        q8, sfq_u8, kv4, sfkv_u8, w, ks, ke, out_t = args
        span = base.span_of(ks, ke)
        plan = base.plan_v2(S, span[1] - span[0], sm_count())
        for label, fn in (("v4", lambda: launch_v4(mod, q8, sfq_u8, kv4, sfkv_u8, w, ks, ke, out_t, plan, span)),
                          ("v2 (one scale per row, same codes)", lambda: base.launch(mod, 2, q8, sfq_u8, kv4, sfkv_u8[:, :1].contiguous(), w, ks, ke, out_t, plan, span))):
            times = []
            for _ in range(10):
                flush_buf.fill_(1)
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record(); fn(); e1.record()
                torch.cuda.synchronize()
                times.append(e0.elapsed_time(e1) * 1000)
            med = statistics.median(times)
            flops = 2.0 * S * H * N * HEAD_DIM
            rows.append({"kernel": label, "S": S, "N": N, "H": H, "plan": list(plan), "us_median": med, "us_min": min(times), "TFLOPs": flops / med / 1e6,
                         "kv_bytes": kv4.numel() + sfkv_u8.numel(), "kv_GBps": (kv4.numel() + sfkv_u8.numel()) / med / 1e3})
            print(f"{label:38s} S={S} N={N} H={H}: {med:.1f} us ({rows[-1]['TFLOPs']:.1f} TFLOP/s)", flush=True)
    report = {"kernel": "fp8_fp4_mqa_logits_sm120_v4: v2's kernel with the k UE8M0 scale per 32 columns ([N, 4]), four folds per n8 tile; the engine's MXFP4 k format",
              "device": props.name, "note": "cold L2 (256 MB fill before each launch); median of 10; the timed region holds the launch only (span and plan computed outside); v2 timed on the same codes with the row scale", "rows": rows}
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
    ap.add_argument("--out-paged", type=Path)
    a = ap.parse_args(argv)
    dev = torch.device("cuda")
    base.SM_COUNT = sm_count()  # the base module sets this in its own main(); its launch() for the v2 comparison reads it
    mod = build()
    rc = 0
    if a.selftest:
        rc = selftest(mod, dev) or selftest_paged(mod, dev)
    if a.bench:
        rc = rc or bench(mod, dev, a.out)
    if a.bench_paged:
        rc = rc or bench_paged(mod, dev, a.out_paged)
    return rc


if __name__ == "__main__":
    sys.exit(main())
