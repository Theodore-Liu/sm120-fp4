"""Stage 2, FC2 on tensor cores with the next expert's weights prefetched, optionally split over G blocks per tile.

scripts/fc2_mma.py (v1) loads an expert's FC2 weights and only then computes on them, and each warp walks its experts
one after another, so at 16 random tokens (about ten experts per warp) the kernel sat at 2.1x a read of its bytes.
Two changes, measured separately against v1:

- prefetch: while a warp computes expert u it has already issued the loads for its next expert (register double
  buffer). The k loop is specialised on the chunk count (I / 128) so both buffers fit in registers.
- split (G > 1): the grid is (H / 16) x G; group g's warps take experts g*8 + warp, g*8 + warp + 8G, ... With H = 2048
  there are only 128 column tiles for the RTX 5090's 170 SMs; G = 2 gives 256 blocks. Each group reduces its warps in
  warp order and writes an fp32 partial; the last group to finish a tile (an atomic counter) adds the G partials in
  group order 0..G-1 and writes the output, so the summation order is fixed and the result deterministic.

Same layout, decode, fragment mapping and per-token accumulation as v1.

    PYTHONPATH=. python scripts/fc2_mma_pf.py --out reports/fc2-mma-pf-<device>-<date>.json
    PYTHONPATH=. python scripts/fc2_mma_pf.py --check-only      # correctness and determinism, no timing
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_here = Path(__file__).resolve().parent
for _name in ("bench_moe_baseline", "micro_floor", "fc1_w4a16", "fc2_w4a16", "fc1_mma", "fc2_mma"):
    _spec = importlib.util.spec_from_file_location(_name, _here / f"{_name}.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)
bench = sys.modules["bench_moe_baseline"]
floor = sys.modules["micro_floor"]
fc1 = sys.modules["fc1_w4a16"]
fc2m = sys.modules["fc2_mma"]

CPP = r"""
#include <torch/extension.h>
void fc2_pf(torch::Tensor q2, torch::Tensor s2, torch::Tensor act, torch::Tensor experts, torch::Tensor offsets,
            torch::Tensor pairs, torch::Tensor weights, torch::Tensor alpha, torch::Tensor out, torch::Tensor scratch,
            torch::Tensor counters, int64_t top_k, int64_t groups);
void fc2_pf_set_pdl(bool on);
"""

# The helpers (e4m3, fp4x2, pack_bf16, decode_pairs, mma) are v1's, reused verbatim.
_V1 = fc2m.CUDA
_HELPERS = _V1[: _V1.index("template <int NT>")]

CUDA = _HELPERS + r"""
#if defined(PF_DECODE_BF16) || defined(PF_DECODE_AUTO)
// e2m1 pair -> bf16x2 in one conversion, scaled by one bf16x2 multiply. e2m1 carries one significand bit and e4m3 three, so the
// product has at most four and the bf16 multiply is exact: the same bits the shipped path reaches through fp32.
__device__ __forceinline__ unsigned fp4x2_bf16(unsigned byte) {
  unsigned o;
  unsigned short in = (unsigned short)byte;
  asm("{ .reg .b8 lo, hi;\n mov.b16 {lo, hi}, %1;\n cvt.rn.bf16x2.e2m1x2 %0, lo; }\n" : "=r"(o) : "h"(in));
  return o;
}
__device__ __forceinline__ unsigned bf16x2_broadcast(float s) {
  __nv_bfloat162 v = __floats2bfloat162_rn(s, s);
  return *reinterpret_cast<unsigned*>(&v);
}
__device__ __forceinline__ unsigned hmul2_bf16(unsigned a, unsigned b) {
  __nv_bfloat162 x = *reinterpret_cast<__nv_bfloat162*>(&a), y = *reinterpret_cast<__nv_bfloat162*>(&b);
  __nv_bfloat162 r = __hmul2(x, y);
  return *reinterpret_cast<unsigned*>(&r);
}
__device__ __forceinline__ void decode_pairs_bf16(uint4 q, unsigned s01, unsigned* dst) {
  const unsigned s0 = bf16x2_broadcast(e4m3(s01 & 0xff)), s1 = bf16x2_broadcast(e4m3((s01 >> 8) & 0xff));
  const unsigned w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
  for (int j = 0; j < 16; ++j) dst[j] = hmul2_bf16(fp4x2_bf16((w[j >> 2] >> (8 * (j & 3))) & 0xff), j < 8 ? s0 : s1);
}
#endif
template <int CH>
__device__ __forceinline__ void load_expert(const unsigned char* __restrict__ q2, const unsigned char* __restrict__ s2,
                                            long long r0, int I, int tig, uint4 (&wq)[CH][2], unsigned (&ws)[CH][2]) {
  const long long r1 = r0 + 8;
#pragma unroll
  for (int ch = 0; ch < CH; ++ch) {
    const int k = ch * 128 + tig * 32;
#ifdef PF_MATH_ONLY
    // timing variant: the codes come from the row index, so the decode and MMA run on data no load delivered
    const unsigned h0 = (unsigned)(r0 * 2654435761u) ^ (unsigned)(ch * 40503u + tig), h1 = h0 * 747796405u + 1u;
    wq[ch][0] = make_uint4(h0, h0 ^ 0x5bd1e995u, h0 + 0x27d4eb2fu, ~h0);
    wq[ch][1] = make_uint4(h1, h1 ^ 0x5bd1e995u, h1 + 0x27d4eb2fu, ~h1);
    ws[ch][0] = 0x3838u; ws[ch][1] = 0x3838u;
    (void)q2; (void)s2; (void)r1;
#elif defined(PF_LOADS_CONTIG)
    // timing variant: the same bytes (the tile's 16 rows, contiguous in memory) read at lane stride, 512 bytes of
    // codes per warp instruction; used only with PF_LOADS_ONLY, since the fragments no longer match the MMA layout
    const int lane = threadIdx.x & 31;
    const long long base = r0 - (lane >> 2);
    const uint4* qb = reinterpret_cast<const uint4*>(q2 + base * (I / 2));
    const unsigned short* sb = reinterpret_cast<const unsigned short*>(s2 + base * (I / 16));
    wq[ch][0] = __ldcs(qb + lane + 32 * (2 * ch));
    wq[ch][1] = __ldcs(qb + lane + 32 * (2 * ch + 1));
    ws[ch][0] = sb[lane + 32 * (2 * ch)];
    ws[ch][1] = sb[lane + 32 * (2 * ch + 1)];
    (void)k; (void)r1;
#else
    if (k < I) {
      wq[ch][0] = __ldcs(reinterpret_cast<const uint4*>(q2 + r0 * (I / 2) + k / 2));
      wq[ch][1] = __ldcs(reinterpret_cast<const uint4*>(q2 + r1 * (I / 2) + k / 2));
      ws[ch][0] = *reinterpret_cast<const unsigned short*>(s2 + r0 * (I / 16) + k / 16);
      ws[ch][1] = *reinterpret_cast<const unsigned short*>(s2 + r1 * (I / 16) + k / 16);
    } else {                                   // a lane past I in the last chunk (I % 128 != 0): zero codes and scales, so its fragments add nothing
      wq[ch][0] = wq[ch][1] = make_uint4(0u, 0u, 0u, 0u);
      ws[ch][0] = ws[ch][1] = 0u;
    }
#endif
  }
}

// PF_CHAIN: everything an expert's routing needs, loaded one expert ahead with its weights (offsets, the pair
// indices of this lane's tile rows, and for its accumulator slots the pair, the token and weight x alpha)
template <int NT>
struct Meta {
  int cnt;
  int pr[NT];
  bool valid[NT];
  bool ok[NT][2];
  int tok[NT][2];
  float wt[NT][2];
};

template <int NT>
__device__ __forceinline__ void load_meta(const int* __restrict__ offsets, const int* __restrict__ pairs,
                                          const float* __restrict__ weights, const float* __restrict__ alpha, int u,
                                          int e, int gid, int tig, int top_k, Meta<NT>& m) {
  const int p0 = offsets[u];
  m.cnt = min(offsets[u + 1] - p0, 8 * NT);
  const float al = alpha[e];
#pragma unroll
  for (int t = 0; t < NT; ++t) {
    const int slot = t * 8 + gid;
    m.valid[t] = slot < m.cnt;
    m.pr[t] = m.valid[t] ? pairs[p0 + slot] : 0;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int s2 = t * 8 + 2 * tig + h;
      m.ok[t][h] = s2 < m.cnt;
      const int pp = m.ok[t][h] ? pairs[p0 + s2] : 0;
      m.wt[t][h] = m.ok[t][h] ? weights[pp] * al : 0.f;
      m.tok[t][h] = pp / top_k;
    }
  }
}

template <int NT, int CH>
__global__ void __launch_bounds__(WARPS * 32)
k_fc2_pf(const unsigned char* __restrict__ q2, const unsigned char* __restrict__ s2, const __nv_bfloat16* __restrict__ act,
         const int* __restrict__ experts, const int* __restrict__ offsets, const int* __restrict__ pairs,
         const float* __restrict__ weights, const float* __restrict__ alpha, __nv_bfloat16* __restrict__ out,
         float* __restrict__ scratch, int* __restrict__ counters, int U, int M, int H, int I, int top_k) {
  __shared__ float part[WARPS][COLS][MAXM];
  __shared__ int last;
  cudaGridDependencySynchronize();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
  const int n0 = blockIdx.x * COLS, g = blockIdx.y, G = gridDim.y;
  for (int z = threadIdx.x; z < WARPS * COLS * MAXM; z += blockDim.x) (&part[0][0][0])[z] = 0.f;
  __syncthreads();

  const int stride = WARPS * G;
  int u = g * WARPS + warp;
  int e = u < U ? experts[u] : -1;
  uint4 wq[CH][2];
  unsigned ws[CH][2];
  if (e >= 0) load_expert<CH>(q2, s2, (long long)e * H + n0 + gid, I, tig, wq, ws);
#ifdef PF_INFLIGHT2
  // two experts ahead: wq2 holds the next expert's weights while nq (below) receives the one after it
  uint4 wq2[CH][2];
  unsigned ws2[CH][2];
  {
    const int u1 = u + stride;
    const int e1 = u1 < U ? experts[u1] : -1;
    if (e1 >= 0) {
      load_expert<CH>(q2, s2, (long long)e1 * H + n0 + gid, I, tig, wq2, ws2);
    } else {
#pragma unroll
      for (int ch = 0; ch < CH; ++ch) { wq2[ch][0] = wq2[ch][1] = make_uint4(0, 0, 0, 0); ws2[ch][0] = ws2[ch][1] = 0u; }
    }
  }
#endif
#ifdef PF_CHAIN
  Meta<NT> cur, nxt;
  if (e >= 0) load_meta<NT>(offsets, pairs, weights, alpha, u, e, gid, tig, top_k, cur);
#endif
  for (; u < U; u += stride) {
    // issue the next expert's loads before this expert's arithmetic (two ahead under PF_INFLIGHT2)
    const int un = u + stride;
    const int en = un < U ? experts[un] : -1;
    uint4 nq[CH][2];
    unsigned ns[CH][2];
#ifdef PF_INFLIGHT2
    const int u2 = u + 2 * stride;
    const int eload = u2 < U ? experts[u2] : -1;
#else
    const int eload = en;
#endif
    if (eload >= 0) {
      load_expert<CH>(q2, s2, (long long)eload * H + n0 + gid, I, tig, nq, ns);
    } else {
#pragma unroll
      for (int ch = 0; ch < CH; ++ch) { nq[ch][0] = nq[ch][1] = make_uint4(0, 0, 0, 0); ns[ch][0] = ns[ch][1] = 0u; }
    }
#ifdef PF_CHAIN
    if (en >= 0) load_meta<NT>(offsets, pairs, weights, alpha, un, en, gid, tig, top_k, nxt);
#endif
    if (e >= 0) {                                   // a negative id is a padding slot from the GPU router
#ifdef PF_CHAIN
      const int cnt = cur.cnt;
      int pr[NT];
      bool valid[NT];
#pragma unroll
      for (int t = 0; t < NT; ++t) { pr[t] = cur.pr[t]; valid[t] = cur.valid[t]; }
#else
      const int p0 = offsets[u];
      const int cnt = min(offsets[u + 1] - p0, 8 * NT);
      int pr[NT];
      bool valid[NT];
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        const int slot = t * 8 + gid;
        valid[t] = slot < cnt;
        pr[t] = valid[t] ? pairs[p0 + slot] : 0;
      }
#endif
      float c[NT][4];
#pragma unroll
      for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int i = 0; i < 4; ++i) c[t][i] = 0.f;
#pragma unroll
      for (int ch = 0; ch < CH; ++ch) {
        const int k = ch * 128 + tig * 32;
        unsigned xb[NT][16];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
          const uint4* av = reinterpret_cast<const uint4*>(act + (long long)pr[t] * I + k);
#pragma unroll
          for (int v = 0; v < 4; ++v) {
#ifdef PF_NO_ACT
            // timing variant: activations made from the pair index, never read; finite bf16 in [1, 2) (random bits
            // would include NaN, which no bit-identity check can pass)
            const unsigned hh = ((unsigned)pr[t] * 2654435761u + (unsigned)(k + v)) & 0x007f007fu;
            const uint4 q = valid[t] ? make_uint4(hh | 0x3f803f80u, (hh ^ 0x00550055u) | 0x3f803f80u,
                                                  ((hh + 0x00030003u) & 0x007f007fu) | 0x3f803f80u, 0x3f803f80u)
                                     : make_uint4(0, 0, 0, 0);
            (void)av;
#else
            const uint4 q = (valid[t] && k < I) ? __ldg(av + v) : make_uint4(0, 0, 0, 0);
#endif
            xb[t][4 * v] = q.x; xb[t][4 * v + 1] = q.y; xb[t][4 * v + 2] = q.z; xb[t][4 * v + 3] = q.w;
          }
        }
#ifdef PF_LOADS_ONLY
        // timing variant: every weight, scale and activation load stays; the arithmetic is one fold of them
        unsigned acc = wq[ch][0].x ^ wq[ch][0].y ^ wq[ch][0].z ^ wq[ch][0].w ^ wq[ch][1].x ^ wq[ch][1].y ^
                       wq[ch][1].z ^ wq[ch][1].w ^ ws[ch][0] ^ ws[ch][1];
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
          for (int j = 0; j < 16; ++j) acc ^= xb[t][j];
        c[0][0] += (acc == 0x9e3779b9u) ? 1.f : 0.f;
#else
        unsigned a[2][16];
#if defined(PF_DECODE_BF16)
        decode_pairs_bf16(wq[ch][0], ws[ch][0], a[0]);
        decode_pairs_bf16(wq[ch][1], ws[ch][1], a[1]);
#elif defined(PF_DECODE_AUTO)
        if constexpr (NT == 1) {                       // the bf16 decode wins at one pair tile (up to 8 tokens) and loses at two
          decode_pairs_bf16(wq[ch][0], ws[ch][0], a[0]);
          decode_pairs_bf16(wq[ch][1], ws[ch][1], a[1]);
        } else {
          decode_pairs(wq[ch][0], ws[ch][0], a[0]);
          decode_pairs(wq[ch][1], ws[ch][1], a[1]);
        }
#else
        decode_pairs(wq[ch][0], ws[ch][0], a[0]);
        decode_pairs(wq[ch][1], ws[ch][1], a[1]);
#endif
#pragma unroll
        for (int s = 0; s < 8; ++s)
#pragma unroll
          for (int t = 0; t < NT; ++t) {
#ifdef PF_SKIP_EMPTY
            if (t * 8 >= cnt) continue;              // warp-uniform: a tile with no pair of this expert adds nothing
#endif
            mma(c[t], a[0][2 * s], a[1][2 * s], a[0][2 * s + 1], a[1][2 * s + 1], xb[t][2 * s], xb[t][2 * s + 1]);
          }
#endif
      }
#ifdef PF_CHAIN
#pragma unroll
      for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          if (cur.ok[t][h]) {
            const float wt = cur.wt[t][h];
            const int tok = cur.tok[t][h];
            part[warp][gid][tok] += wt * c[t][h];
            part[warp][gid + 8][tok] += wt * c[t][2 + h];
          }
        }
#else
      const float al = alpha[e];
#pragma unroll
      for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int slot = t * 8 + 2 * tig + h;
          if (slot < cnt) {
            const int p = pairs[p0 + slot];
            const float wt = weights[p] * al;
            const int tok = p / top_k;
            part[warp][gid][tok] += wt * c[t][h];
            part[warp][gid + 8][tok] += wt * c[t][2 + h];
          }
        }
#endif
    }
    __syncwarp();                                   // the next expert may put a token in another lane
#ifdef PF_INFLIGHT2
#pragma unroll
    for (int ch = 0; ch < CH; ++ch) {
      wq[ch][0] = wq2[ch][0]; wq[ch][1] = wq2[ch][1]; ws[ch][0] = ws2[ch][0]; ws[ch][1] = ws2[ch][1];
      wq2[ch][0] = nq[ch][0]; wq2[ch][1] = nq[ch][1]; ws2[ch][0] = ns[ch][0]; ws2[ch][1] = ns[ch][1];
    }
#else
#pragma unroll
    for (int ch = 0; ch < CH; ++ch) { wq[ch][0] = nq[ch][0]; wq[ch][1] = nq[ch][1]; ws[ch][0] = ns[ch][0]; ws[ch][1] = ns[ch][1]; }
#endif
    e = en;
#ifdef PF_CHAIN
    cur = nxt;
#endif
  }
  __syncthreads();
  if (G == 1) {
    for (int idx = threadIdx.x; idx < COLS * M; idx += blockDim.x) {
      const int r = idx / M, t = idx % M;
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < WARPS; ++w) v += part[w][r][t];   // fixed warp order
      out[(long long)t * H + n0 + r] = __float2bfloat16(v);
    }
    return;
  }
  // G > 1: this group's partial, then the last group to finish the tile adds all partials in group order.
  for (int idx = threadIdx.x; idx < COLS * M; idx += blockDim.x) {
    const int r = idx / M, t = idx % M;
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) v += part[w][r][t];
    scratch[((long long)g * MAXM + t) * H + n0 + r] = v;
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = (atomicAdd(&counters[blockIdx.x], 1) == G - 1);
  __syncthreads();
  if (!last) return;
  __threadfence();
  for (int idx = threadIdx.x; idx < COLS * M; idx += blockDim.x) {
    const int r = idx / M, t = idx % M;
    float v = 0.f;
    for (int gg = 0; gg < G; ++gg) v += __ldcg(&scratch[((long long)gg * MAXM + t) * H + n0 + r]);   // fixed group order
    out[(long long)t * H + n0 + r] = __float2bfloat16(v);
  }
  if (threadIdx.x == 0) counters[blockIdx.x] = 0;   // ready for the next launch
}


static bool g_pdl = false;   // launch with programmatic stream serialization (set by the module's set_pdl)
template <typename... KArgs, typename... Args>
static void pdl_launch(void (*kernel)(KArgs...), dim3 grid, dim3 block, cudaStream_t st, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = g_pdl ? 1 : 0;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, kernel, args...) == cudaSuccess, "launch");
}
void fc2_pf_set_pdl(bool on) { g_pdl = on; }

template <int NT, int CH>
static void launch(dim3 grid, cudaStream_t st, const unsigned char* q2, const unsigned char* s2, const __nv_bfloat16* act,
                   const int* experts, const int* offsets, const int* pairs, const float* weights, const float* alpha,
                   __nv_bfloat16* out, float* scratch, int* counters, int U, int M, int H, int I, int top_k) {
  pdl_launch(k_fc2_pf<NT, CH>, grid, dim3(WARPS * 32), st, q2, s2, act, experts, offsets, pairs, weights, alpha, out,
             scratch, counters, U, M, H, I, top_k);
}

void fc2_pf(torch::Tensor q2, torch::Tensor s2, torch::Tensor act, torch::Tensor experts, torch::Tensor offsets,
            torch::Tensor pairs, torch::Tensor weights, torch::Tensor alpha, torch::Tensor out, torch::Tensor scratch,
            torch::Tensor counters, int64_t top_k, int64_t groups) {
  const int M = (int)out.size(0), H = (int)out.size(1), I = (int)act.size(1), U = (int)experts.numel();
  TORCH_CHECK(M <= MAXM && H % COLS == 0 && I % 32 == 0 && I >= 128 && I <= 1024, "shape: intermediate a multiple of 32 between 128 and 1024");
  TORCH_CHECK(groups >= 1 && groups <= 4 && scratch.numel() >= groups * MAXM * H && counters.numel() >= H / COLS, "scratch");
  auto st = at::cuda::getCurrentCUDAStream();
  dim3 grid(H / COLS, (unsigned)groups);
#define PF_ARGS grid, st, q2.data_ptr<uint8_t>(), s2.data_ptr<uint8_t>(), reinterpret_cast<const __nv_bfloat16*>(act.data_ptr()), \
    experts.data_ptr<int>(), offsets.data_ptr<int>(), pairs.data_ptr<int>(), weights.data_ptr<float>(), alpha.data_ptr<float>(), \
    reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), scratch.data_ptr<float>(), counters.data_ptr<int>(), U, M, H, I, (int)top_k
  const int ch = (I + 127) / 128;            // chunks of 128 (four quad lanes x 32); the last may be partial
#define PF_CH(NT, C) if (ch == C) { launch<NT, C>(PF_ARGS); launched = true; }
  bool launched = false;
  if (M <= 8) { PF_CH(1, 1) PF_CH(1, 2) PF_CH(1, 3) PF_CH(1, 4) PF_CH(1, 5) PF_CH(1, 6) PF_CH(1, 7) PF_CH(1, 8) }
  else { PF_CH(2, 1) PF_CH(2, 2) PF_CH(2, 3) PF_CH(2, 4) PF_CH(2, 5) PF_CH(2, 6) PF_CH(2, 7) PF_CH(2, 8) }
  TORCH_CHECK(launched, "chunk count");
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "launch");
}
"""


def build(verbose: bool = False, skip_empty: bool = False, warps: int = 8, min_blocks: int = 0, mode: str = "full",
          chain: bool = False, no_act: bool = False, inflight: int = 1, decode: str = "f32"):
    """skip_empty: skip the tensor-core work of a pair tile an expert does not fill (M > 8, fewer than 9 pairs).
    warps: warps per block (8 by default); min_blocks: __launch_bounds__'s minimum resident blocks per SM (0: unset).
    no_act: timing only, activations made from the pair index instead of read.
    chain: load the next expert's routing (offsets, pair indices, weight x alpha, token) with its weights.
    mode: "full", or a timing variant that gives wrong answers by design: "loads" (every load, no decode or MMA) or
    "math" (decode and MMA on codes made from the row index, no weight load).
    inflight: experts whose weights a warp holds ahead of the one it computes (1, the register double buffer; 2, a three-buffer ring)."""
    assert mode in ("full", "loads", "math", "loads_contig")
    assert inflight in (1, 2), inflight
    assert decode in ("f32", "bf16", "auto"), decode   # bf16: PF_DECODE_BF16, the e2m1 pair converted straight to bf16x2 and scaled with one bf16x2 multiply; auto: bf16 at NT 1, f32 at NT 2
    src = CUDA
    if warps != 8:
        assert src.count("constexpr int WARPS = 8;") == 1
        src = src.replace("constexpr int WARPS = 8;", f"constexpr int WARPS = {warps};")
    if min_blocks:
        assert src.count("__launch_bounds__(WARPS * 32)") == 1
        src = src.replace("__launch_bounds__(WARPS * 32)", f"__launch_bounds__(WARPS * 32, {min_blocks})")
    name = "sm120fp4_fc2_pf" + ("_skip" if skip_empty else "") + (f"_w{warps}" if warps != 8 else "") +         (f"_mb{min_blocks}" if min_blocks else "") + ("" if mode == "full" else f"_{mode}") + ("_chain" if chain else "") + ("_noact" if no_act else "") + ("_if2" if inflight == 2 else "") + ({"f32": "", "bf16": "_dbf", "auto": "_dauto"}[decode])
    return load_inline(name=name, cpp_sources=CPP, cuda_sources=src,
                       functions=["fc2_pf", "fc2_pf_set_pdl"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a"]
                       + (["-DPF_SKIP_EMPTY"] if skip_empty else []) + (["-DPF_CHAIN"] if chain else []) + (["-DPF_NO_ACT"] if no_act else []) + (["-DPF_INFLIGHT2"] if inflight == 2 else []) + ({"f32": [], "bf16": ["-DPF_DECODE_BF16"], "auto": ["-DPF_DECODE_AUTO"]}[decode])
                       + ({"full": [], "loads": ["-DPF_LOADS_ONLY"], "math": ["-DPF_MATH_ONLY"],
                           "loads_contig": ["-DPF_LOADS_ONLY", "-DPF_LOADS_CONTIG"]}[mode])
                       + (["-Xptxas=-v"] if verbose else []),
                       verbose=verbose)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, help="JSON report, written into the repository")
    ap.add_argument("--check-only", action="store_true", help="correctness and determinism only, no timing")
    ap.add_argument("--ptxas", action="store_true", help="print register use and spills")
    ap.add_argument("--hidden", type=int, default=2048, help="hidden size (a multiple of 16); 2816 for the Gemma-4-26B-A4B shape")
    ap.add_argument("--inter", type=int, default=768, help="expert intermediate size (a multiple of 32, at most 1024); 704 for the Gemma-4-26B-A4B shape")
    ap.add_argument("--inflight2", action="store_true", help="also build and run the two-ahead prefetch (PF_INFLIGHT2) as the inflight2 and inflight2_split2 variants")
    ap.add_argument("--decode-bf16", action="store_true", help="also build and run the bf16 decode (PF_DECODE_BF16) as the decode_bf16 and decode_bf16_split2 variants")
    a = ap.parse_args(argv)
    if not a.check_only and a.out is None:
        ap.error("--out is required unless --check-only")
    pf, v1, m1, fl = build(a.ptxas), fc2m.build(), fc1.build(), floor.build()
    pf2 = build(a.ptxas, inflight=2) if a.inflight2 else None
    pfd = build(a.ptxas, decode="bf16") if a.decode_bf16 else None
    dev = torch.device("cuda")
    e, k, h, i = 128, 8, a.hidden, a.inter
    print(f"shape: {e} experts, top-{k}, hidden {h}, intermediate {i}")
    w = bench.build(e, h, i, dev)
    q1, s1, q2, s2 = (w[n].contiguous() for n in ("q1", "s1", "q2", "s2"))
    alpha = torch.ones(e, device=dev)
    sink = torch.zeros(4, dtype=torch.int32, device=dev)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    scratch = torch.zeros(4 * 16 * h, device=dev)
    counters = torch.zeros(h // 16, dtype=torch.int32, device=dev)
    rows = []
    print("routing | tokens | variant | normwise vs fp32 MoE | bit-identical x50 | equals v1 | us")

    def case(label, m, ids, wts, x):
        experts, offsets, pairs = fc1.route(ids)
        act = torch.empty(ids.numel(), i, device=dev, dtype=torch.bfloat16)
        m1.fc1_w4a16(q1, s1, x, experts, offsets, pairs, alpha, act, i, k)
        wf = wts.reshape(-1).contiguous()
        ref = bench.reference(x, w, ids, wts, i, act_quant=False)
        row = {"routing": label, "tokens": m}
        o1 = torch.empty(m, h, device=dev, dtype=torch.bfloat16)
        variants = {"v1": lambda o: v1.fc2_mma(q2, s2, act, experts, offsets, pairs, wf, alpha, o, k),
                    "prefetch": lambda o: pf.fc2_pf(q2, s2, act, experts, offsets, pairs, wf, alpha, o, scratch, counters, k, 1),
                    "prefetch_split2": lambda o: pf.fc2_pf(q2, s2, act, experts, offsets, pairs, wf, alpha, o, scratch, counters, k, 2)}
        if pf2 is not None:
            variants["inflight2"] = lambda o: pf2.fc2_pf(q2, s2, act, experts, offsets, pairs, wf, alpha, o, scratch, counters, k, 1)
            variants["inflight2_split2"] = lambda o: pf2.fc2_pf(q2, s2, act, experts, offsets, pairs, wf, alpha, o, scratch, counters, k, 2)
        if pfd is not None:
            variants["decode_bf16"] = lambda o: pfd.fc2_pf(q2, s2, act, experts, offsets, pairs, wf, alpha, o, scratch, counters, k, 1)
            variants["decode_bf16_split2"] = lambda o: pfd.fc2_pf(q2, s2, act, experts, offsets, pairs, wf, alpha, o, scratch, counters, k, 2)
        if i % 128 != 0:
            del variants["v1"]                 # v1 takes I in multiples of 128 only; the comparison column then reads against the prefetch kernel's own output
        (variants.get("v1") or variants["prefetch"])(o1)
        torch.cuda.synchronize()
        for name, fn in variants.items():
            out = torch.empty(m, h, device=dev, dtype=torch.bfloat16)
            go = lambda: fn(out)  # noqa: E731
            go()
            torch.cuda.synchronize()
            rel = float((out.float() - ref).norm() / ref.norm())
            first = out.clone()
            stable = True
            for _ in range(50):
                go()
                stable = stable and bool(torch.equal(out, first))
            same_v1 = bool(torch.equal(first, o1))
            t = None if a.check_only else floor.graph_time(go)
            row[name] = {"rel_err_vs_fp32_moe": rel, "bit_identical_50": stable, "equals_v1": same_v1, "us": t}
            print(f"{label:7s} | {m:6d} | {name:15s} | {rel:20.5f} | {str(stable):17s} | {str(same_v1):9s} | "
                  + ("-" if t is None else f"{t:.1f}"))
        if not a.check_only:
            ptrs = torch.tensor([q2.data_ptr() + int(x_) * h * i // 2 for x_ in experts.tolist()], dtype=torch.int64, device=dev)
            row["read_fc2_codes_us"] = floor.graph_time(lambda: fl.stream_read(ptrs, h * i // 2, 0, h * i // 2, sms * 4, 256, sink))
            print(f"{'':7s} | {'':6s} | {'read floor':15s} | {'':20s} | {'':17s} | {'':9s} | {row['read_fc2_codes_us']:.1f}")
        rows.append(row)

    for m in (1, 2, 4, 8, 16):
        g = torch.Generator().manual_seed(1000 + m)
        x = torch.randn(m, h, generator=g).to(device=dev, dtype=torch.bfloat16)
        wts, ids = torch.topk(F.softmax(torch.randn(m, e, generator=g), dim=-1), k, dim=-1)
        wts = (wts / wts.sum(-1, keepdim=True)).float().to(dev).contiguous()
        case("random", m, ids.to(torch.int32).to(dev).contiguous(), wts, x)
    fixed = torch.arange(8, dtype=torch.int32, device=dev) * 16
    for m in (1, 4, 8, 16):
        x = torch.randn(m, h, generator=torch.Generator().manual_seed(7)).to(device=dev, dtype=torch.bfloat16)
        wts = torch.full((m, k), 1.0 / k, device=dev)
        case("fixed8", m, fixed.repeat(m, 1).contiguous(), wts, x)
    if a.out is not None:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"device": torch.cuda.get_device_name(0), "check_only": a.check_only, "rows": rows},
                                    indent=1) + "\n", encoding="utf-8")
        print(f"written {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
