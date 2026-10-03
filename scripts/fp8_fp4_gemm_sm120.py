"""Stage 3, the first kernel: SM120 FP8xFP4 GEMM with UE8M0 block scales (DeepGEMM's `fp8_fp4_gemm_nt` contract, 1D1D,
dense, NT, K-major), in three versions.

    D[m, n] = sum_k A[m, k] * B[n, k],   A e4m3 [M, K] with packed UE8M0 scales sfa [M, K/gran_k] (int32, 4 per word),
                                        B packed e2m1 [N, K/2] with sfb [N, K/gran_k]; D bf16 [M, N].

v0 (correctness first, 2026-10-02): one warp per 16 x 8 tile over the whole K with direct global loads; M <= 16, N a
multiple of 8. It settled three things against `scripts/ue8m0_reference.py` (itself bit for bit with DeepGEMM's helpers):
the `mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e2m1.f32` fragment layouts, the e2m1 container (the code
left-aligned in the low six bits, bits 5:2, measured by scripts/probe_f8f6f4_onehot.py), and the UE8M0 fold outside the
MMA (four k32 steps share one scale pair at gran_k 128; the block partial is multiplied by 2^(ea-127) * 2^(eb-127) and
added to the running fp32 accumulator).

v1 (the tiled kernel of docs/stage3-survey.md section 2.1): a block of 8 warps owns a 32 x 128 output tile; each warp
owns 16 columns (two n8 tiles) for both m16 tiles; A (32 x 128 bytes) and B (128 x 64 packed bytes) tiles for one 128-K
block stream through a 4-stage cp.async pipeline in shared memory, fragments are loaded from shared memory (the e2m1
nibbles unpacked to containers on the way), and the UE8M0 fold happens once per 128-K block exactly as in v0, so v1's
arithmetic order per output element is v0's and the two agree bit for bit. M <= 32, N a multiple of 128.

v2 (split-K, 2026-10-02): v1's block, templated on BN (128 or 64 columns, 8 or 4 warps), with the grid (N/BN, splits);
each block folds its slice of the 128-K blocks into an fp32 partial and writes it to a workspace [splits, M, N]; a
reduce kernel sums the partials in a fixed order and stores bf16. The fold inside a slice is v1's, so each slice's
partial is exact in v1's sense; only the order in which slices are summed differs from v1's single running sum, which
moves an output by at most a few fp32 ulps before the bf16 round. `splits` and `BN` default to the smallest that give
at least two blocks per SM (the N = 2048 case that left 16 blocks on 170 SMs in v1 gets BN 64 and 11 splits, 352 blocks).

    PYTHONPATH=. python scripts/fp8_fp4_gemm_sm120.py --selftest                       # all versions against the reference
    PYTHONPATH=. python scripts/fp8_fp4_gemm_sm120.py --bench --out reports/fp8-fp4-gemm-v2-<device>-<date>.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ue8m0_reference", _here / "ue8m0_reference.py")
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)

CPP = r"""
#include <torch/extension.h>
void fp8_fp4_gemm_nt_sm120(torch::Tensor a, torch::Tensor sfa, torch::Tensor b, torch::Tensor sfb, torch::Tensor d, int64_t gran_k);
void fp8_fp4_gemm_nt_sm120_v1(torch::Tensor a, torch::Tensor sfa, torch::Tensor b, torch::Tensor sfb, torch::Tensor d, int64_t gran_k);
void fp8_fp4_gemm_nt_sm120_v2(torch::Tensor a, torch::Tensor sfa, torch::Tensor b, torch::Tensor sfb, torch::Tensor d, torch::Tensor ws,
                              int64_t gran_k, int64_t bn, int64_t splits, int64_t arms);
"""

CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cstdint>

// Fragment layout of mma.m16n8k32 with 8-bit containers (PTX ISA; confirmed by scripts/probe_f8f6f4.py): lane = 4g + t,
//   A: a0 = row g, k 4t..4t+3   a1 = row g+8, same k   a2 = row g, k 16+4t..   a3 = row g+8, k 16+4t..   (k ascending in the register)
//   B: b0 = col g, k 4t..4t+3   b1 = col g, k 16+4t..                                                     (4 e2m1 codes, one per byte)
//   C/D: c0, c1 = row g, cols 2t, 2t+1;   c2, c3 = row g+8, cols 2t, 2t+1
__device__ __forceinline__ void mma_f8f6f4(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e2m1.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Four e2m1 codes (k..k+3) packed two per byte (even code low) -> four 8-bit containers. Measured on the RTX 5090
// (scripts/probe_f8f6f4_onehot.py, 2026-10-02): under .kind::f8f6f4 the hardware reads each container as a 6-bit field in
// bits 5:0 (sign at bit 5, two exponent bits, three mantissa bits), so an e2m1 code goes in bits 5:2 (code << 2) with bits
// 1:0 and 7:6 zero; 0x08 reads as 1.0, 0x3C as -6.0, and a code left in the low nibble reads as a wrong, smaller number.
__device__ __forceinline__ uint32_t unpack_e2m1_x4(uint32_t packed2) {
  const uint32_t p = packed2 & 0xFFFFu;
  return ((p & 0x000Fu) << 2) | ((p & 0x00F0u) << 6) | ((p & 0x0F00u) << 10) | ((p & 0xF000u) << 14);
}

__device__ __forceinline__ float ue8m0_to_float(uint32_t e) { return __uint_as_float(e << 23); }

// ------------------------------------------------------------------------------------------------------------ v0
__global__ void __launch_bounds__(32)
k_fp8_fp4_gemm_nt_v0(const uint8_t* __restrict__ a, const uint8_t* __restrict__ sfa,
                     const uint8_t* __restrict__ b, const uint8_t* __restrict__ sfb,
                     __nv_bfloat16* __restrict__ d, int M, int N, int K, int gran_k, int sf_stride) {
  const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
  const int n0 = blockIdx.x * 8;
  const int row0 = g, row1 = g + 8, col = n0 + g;
  const bool has0 = row0 < M, has1 = row1 < M;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  const int steps_per_block = gran_k / 32;
  for (int kb = 0; kb < K; kb += gran_k) {
    float part[4] = {0.f, 0.f, 0.f, 0.f};
    for (int s = 0; s < steps_per_block; ++s) {
      const int k0 = kb + s * 32;
      uint32_t af[4], bf[2];
      af[0] = has0 ? *reinterpret_cast<const uint32_t*>(a + (size_t)row0 * K + k0 + 4 * t) : 0u;
      af[1] = has1 ? *reinterpret_cast<const uint32_t*>(a + (size_t)row1 * K + k0 + 4 * t) : 0u;
      af[2] = has0 ? *reinterpret_cast<const uint32_t*>(a + (size_t)row0 * K + k0 + 16 + 4 * t) : 0u;
      af[3] = has1 ? *reinterpret_cast<const uint32_t*>(a + (size_t)row1 * K + k0 + 16 + 4 * t) : 0u;
      const uint8_t* brow = b + (size_t)col * (K / 2);
      bf[0] = unpack_e2m1_x4(*reinterpret_cast<const uint16_t*>(brow + (k0 + 4 * t) / 2));
      bf[1] = unpack_e2m1_x4(*reinterpret_cast<const uint16_t*>(brow + (k0 + 16 + 4 * t) / 2));
      mma_f8f6f4(part, af, bf);
    }
    const int blk = kb / gran_k;
    const float sa0 = has0 ? ue8m0_to_float(sfa[(size_t)row0 * sf_stride + blk]) : 0.f;
    const float sa1 = has1 ? ue8m0_to_float(sfa[(size_t)row1 * sf_stride + blk]) : 0.f;
    const float sb0 = ue8m0_to_float(sfb[(size_t)(n0 + 2 * t) * sf_stride + blk]);
    const float sb1 = ue8m0_to_float(sfb[(size_t)(n0 + 2 * t + 1) * sf_stride + blk]);
    acc[0] += part[0] * (sa0 * sb0);
    acc[1] += part[1] * (sa0 * sb1);
    acc[2] += part[2] * (sa1 * sb0);
    acc[3] += part[3] * (sa1 * sb1);
  }
  if (has0) {
    d[(size_t)row0 * N + n0 + 2 * t] = __float2bfloat16(acc[0]);
    d[(size_t)row0 * N + n0 + 2 * t + 1] = __float2bfloat16(acc[1]);
  }
  if (has1) {
    d[(size_t)row1 * N + n0 + 2 * t] = __float2bfloat16(acc[2]);
    d[(size_t)row1 * N + n0 + 2 * t + 1] = __float2bfloat16(acc[3]);
  }
}

static void check_common(const torch::Tensor& a, const torch::Tensor& sfa, const torch::Tensor& b, const torch::Tensor& sfb,
                         int64_t gran_k, int M, int N, int K) {
  TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn && a.is_contiguous(), "A: e4m3 [M, K], contiguous");
  TORCH_CHECK(b.scalar_type() == torch::kInt8 && b.is_contiguous() && b.size(1) * 2 == K, "B: packed e2m1 int8 [N, K/2]");
  TORCH_CHECK(sfa.scalar_type() == torch::kInt && sfb.scalar_type() == torch::kInt, "scales: packed UE8M0 int32");
  TORCH_CHECK(gran_k == 128 && K % gran_k == 0, "gran_k 128, K a multiple of 128");
  TORCH_CHECK(sfa.size(0) == M && sfb.size(0) == N && sfa.size(1) == sfb.size(1) && sfa.size(1) * 4 >= K / gran_k
              && sfa.is_contiguous() && sfb.is_contiguous(), "scale shapes: [rows, ceil((K/gran_k)/4)] int32, padded to a multiple of 4 scales");
}

void fp8_fp4_gemm_nt_sm120(torch::Tensor a, torch::Tensor sfa, torch::Tensor b, torch::Tensor sfb, torch::Tensor d, int64_t gran_k) {
  const int M = (int)a.size(0), K = (int)a.size(1), N = (int)b.size(0);
  check_common(a, sfa, b, sfb, gran_k, M, N, K);
  TORCH_CHECK(M >= 1 && M <= 16 && N % 8 == 0, "v0: M <= 16, N a multiple of 8");
  TORCH_CHECK(d.scalar_type() == torch::kBFloat16 && d.size(0) == M && d.size(1) == N && d.is_contiguous(), "D: bf16 [M, N]");
  auto st = at::cuda::getCurrentCUDAStream();
  k_fp8_fp4_gemm_nt_v0<<<N / 8, 32, 0, st>>>(static_cast<const uint8_t*>(a.data_ptr()), reinterpret_cast<const uint8_t*>(sfa.data_ptr<int>()),
                                             reinterpret_cast<const uint8_t*>(b.data_ptr<int8_t>()),
                                             reinterpret_cast<const uint8_t*>(sfb.data_ptr<int>()),
                                             reinterpret_cast<__nv_bfloat16*>(d.data_ptr()), M, N, K, (int)gran_k, (int)sfa.size(1) * 4);
}

// ------------------------------------------------------------------------------------------------------------ v1 / v2 core
// Block: BN/16 warps, a 32 (M) x BN (N) output tile; warp w owns columns 16w..16w+15 (two n8 tiles) for both m16 tiles.
// Per 128-K block (one UE8M0 block): A tile 32 x 128 bytes (4 KB), B tile BN cols x 64 packed bytes, staged by cp.async in
// STAGES stages. Fragments come from shared memory; the fold per 128-K block is v0's.
constexpr int BM = 32, BK = 128, STAGES = 4;

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
  const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

// Computes the fold-accumulated fp32 tile for 128-K blocks [kb0, kb1) and hands it to `store` (per-thread fragment values).
// Arms (scripts/fp8_fp4_gemm_sm120.py --bench, 2026-10-02): SWZ swizzles the A tile's 16-byte chunks by (chunk ^ (row & 7)) so the
// eight lanes of equal t that read one chunk column hit eight banks; KPERM permutes k inside each k32 step so that lane t's A bytes
// (k 4t..4t+3 and 16+4t..16+4t+3) sit at physical bytes 8t..8t+7 (one 8-byte load) and its B nibbles at physical codes 8t..8t+7
// (one 4-byte load) - a permutation of the summation index that the MMA sums over, applied to A and B alike; ONESYNC drops the
// trailing barrier of each 128-K block (the barrier after the wait already orders every thread's previous compute before the refill).
template <int BN, int SWZ, int KPERM, int ONESYNC, typename Store>
__device__ __forceinline__ void gemm_tile(const uint8_t* __restrict__ a, const uint8_t* __restrict__ sfa,
                                          const uint8_t* __restrict__ b, const uint8_t* __restrict__ sfb,
                                          int M, int N, int K, int sf_stride, int n_base, int m_base, int kb0, int kb1,
                                          uint8_t* smem, Store store) {
  constexpr int THREADS = (BN / 16) * 32;
  constexpr int A_BYTES = BM * BK, B_BYTES = BN * (BK / 2), STAGE_BYTES = A_BYTES + B_BYTES;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int nb = kb1 - kb0;

  auto issue = [&](int kb, int stage) {
    uint8_t* sA = smem + stage * STAGE_BYTES;
    uint8_t* sB = sA + A_BYTES;
    for (int c = threadIdx.x; c < BM * 8; c += THREADS) {            // A: 32 rows x 8 chunks of 16 bytes
      const int row = c >> 3, ch = c & 7;
      const int chs = SWZ ? (ch ^ (row & 7)) : ch;   // where the chunk lands in shared memory
      if (m_base + row < M) cp_async_16(sA + row * BK + chs * 16, a + (size_t)(m_base + row) * K + (size_t)kb * BK + ch * 16);
      else *reinterpret_cast<uint4*>(sA + row * BK + chs * 16) = make_uint4(0, 0, 0, 0);
    }
    for (int c = threadIdx.x; c < BN * 4; c += THREADS) {            // B: BN cols x 4 chunks of 16 bytes
      const int col = c >> 2, ch = c & 3;
      cp_async_16(sB + col * 64 + ch * 16, b + (size_t)(n_base + col) * (K / 2) + (size_t)kb * (BK / 2) + ch * 16);
    }
  };

  float acc[2][2][4];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[mt][nt][i] = 0.f;

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < nb) issue(kb0 + s, s);
    cp_async_commit();
  }
  const int row_g = g, row_g8 = g + 8;
  const bool has[2][2] = {{m_base + row_g < M, m_base + row_g8 < M}, {m_base + 16 + row_g < M, m_base + 16 + row_g8 < M}};
  for (int i = 0; i < nb; ++i) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    const int nxt = i + STAGES - 1;
    if (nxt < nb) issue(kb0 + nxt, nxt % STAGES);
    cp_async_commit();
    const uint8_t* sA = smem + (i % STAGES) * STAGE_BYTES;
    const uint8_t* sB = sA + A_BYTES;
    const int kb = kb0 + i;

    float part[2][2][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int q = 0; q < 4; ++q) part[mt][nt][q] = 0.f;

#pragma unroll
    for (int s = 0; s < BK / 32; ++s) {
      const int k0 = s * 32;
      uint32_t bf[2][2];
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const uint8_t* brow = sB + (warp * 16 + nt * 8 + g) * 64;
        if (KPERM) {   // physical codes 8t..8t+7 of this k32 step: one 4-byte load, low 16 bits -> b0, high 16 bits -> b1
          const uint32_t w = *reinterpret_cast<const uint32_t*>(brow + (k0 + 8 * t) / 2);
          bf[nt][0] = unpack_e2m1_x4(w & 0xFFFFu);
          bf[nt][1] = unpack_e2m1_x4(w >> 16);
        } else {
          bf[nt][0] = unpack_e2m1_x4(*reinterpret_cast<const uint16_t*>(brow + (k0 + 4 * t) / 2));
          bf[nt][1] = unpack_e2m1_x4(*reinterpret_cast<const uint16_t*>(brow + (k0 + 16 + 4 * t) / 2));
        }
      }
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        uint32_t af[4];
        const int ra = mt * 16 + row_g, rb = mt * 16 + row_g8;
        const uint8_t* r0 = sA + ra * BK;
        const uint8_t* r1 = sA + rb * BK;
        if (KPERM) {   // physical bytes 8t..8t+7 of this step: one 8-byte load per row (the swizzle moves whole 16-byte chunks)
          const int off = k0 + 8 * t;
          const int c0a = SWZ ? ((off >> 4) ^ (ra & 7)) : (off >> 4), c0b = SWZ ? ((off >> 4) ^ (rb & 7)) : (off >> 4);
          const uint2 wa = *reinterpret_cast<const uint2*>(r0 + c0a * 16 + (off & 15));
          const uint2 wb = *reinterpret_cast<const uint2*>(r1 + c0b * 16 + (off & 15));
          af[0] = wa.x; af[2] = wa.y; af[1] = wb.x; af[3] = wb.y;
        } else {
          const int o0 = k0 + 4 * t, o2 = k0 + 16 + 4 * t;
          const int ca0 = SWZ ? ((o0 >> 4) ^ (ra & 7)) : (o0 >> 4), ca2 = SWZ ? ((o2 >> 4) ^ (ra & 7)) : (o2 >> 4);
          const int cb0 = SWZ ? ((o0 >> 4) ^ (rb & 7)) : (o0 >> 4), cb2 = SWZ ? ((o2 >> 4) ^ (rb & 7)) : (o2 >> 4);
          af[0] = *reinterpret_cast<const uint32_t*>(r0 + ca0 * 16 + (o0 & 15));
          af[1] = *reinterpret_cast<const uint32_t*>(r1 + cb0 * 16 + (o0 & 15));
          af[2] = *reinterpret_cast<const uint32_t*>(r0 + ca2 * 16 + (o2 & 15));
          af[3] = *reinterpret_cast<const uint32_t*>(r1 + cb2 * 16 + (o2 & 15));
        }
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) mma_f8f6f4(part[mt][nt], af, bf[nt]);
      }
    }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      const int r0 = m_base + mt * 16 + row_g, r1 = m_base + mt * 16 + row_g8;
      const float sa0 = has[mt][0] ? ue8m0_to_float(sfa[(size_t)r0 * sf_stride + kb]) : 0.f;
      const float sa1 = has[mt][1] ? ue8m0_to_float(sfa[(size_t)r1 * sf_stride + kb]) : 0.f;
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const int c0 = n_base + warp * 16 + nt * 8 + 2 * t;
        const float sb0 = ue8m0_to_float(sfb[(size_t)c0 * sf_stride + kb]);
        const float sb1 = ue8m0_to_float(sfb[(size_t)(c0 + 1) * sf_stride + kb]);
        acc[mt][nt][0] += part[mt][nt][0] * (sa0 * sb0);
        acc[mt][nt][1] += part[mt][nt][1] * (sa0 * sb1);
        acc[mt][nt][2] += part[mt][nt][2] * (sa1 * sb0);
        acc[mt][nt][3] += part[mt][nt][3] * (sa1 * sb1);
      }
    }
    if (!ONESYNC) __syncthreads();
  }
  cp_async_wait<0>();

#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
    const int r0 = m_base + mt * 16 + row_g, r1 = m_base + mt * 16 + row_g8;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int c0 = n_base + warp * 16 + nt * 8 + 2 * t;
      if (has[mt][0]) { store(r0, c0, acc[mt][nt][0]); store(r0, c0 + 1, acc[mt][nt][1]); }
      if (has[mt][1]) { store(r1, c0, acc[mt][nt][2]); store(r1, c0 + 1, acc[mt][nt][3]); }
    }
  }
}

// v1: one block over the whole K, bf16 store
__global__ void __launch_bounds__(256)
k_fp8_fp4_gemm_nt_v1(const uint8_t* __restrict__ a, const uint8_t* __restrict__ sfa,
                     const uint8_t* __restrict__ b, const uint8_t* __restrict__ sfb,
                     __nv_bfloat16* __restrict__ d, int M, int N, int K, int sf_stride) {
  extern __shared__ __align__(16) uint8_t smem[];
  gemm_tile<128, 0, 0, 0>(a, sfa, b, sfb, M, N, K, sf_stride, blockIdx.x * 128, blockIdx.y * BM, 0, K / BK, smem,
                 [&](int r, int c, float v) { d[(size_t)r * N + c] = __float2bfloat16(v); });
}

// v2: block (n, split) over its slice of the 128-K blocks, fp32 partial to the workspace
template <int BN, int SWZ, int KPERM, int ONESYNC>
__global__ void __launch_bounds__((BN / 16) * 32)
k_fp8_fp4_gemm_nt_v2(const uint8_t* __restrict__ a, const uint8_t* __restrict__ sfa,
                     const uint8_t* __restrict__ b, const uint8_t* __restrict__ sfb,
                     float* __restrict__ ws, int M, int N, int K, int sf_stride, int chunk) {
  extern __shared__ __align__(16) uint8_t smem[];
  const int nblocks = K / BK;
  const int kb0 = blockIdx.y * chunk, kb1 = min(nblocks, kb0 + chunk);
  float* out = ws + (size_t)blockIdx.y * M * N;
  gemm_tile<BN, SWZ, KPERM, ONESYNC>(a, sfa, b, sfb, M, N, K, sf_stride, blockIdx.x * BN, 0, kb0, kb1, smem,
                [&](int r, int c, float v) { out[(size_t)r * N + c] = v; });
}

// v2 reduce: sum the splits in a fixed order, store bf16
__global__ void k_split_reduce(const float* __restrict__ ws, __nv_bfloat16* __restrict__ d, int M, int N, int splits) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)M * N) return;
  float s = 0.f;
  for (int p = 0; p < splits; ++p) s += ws[(size_t)p * M * N + i];
  d[i] = __float2bfloat16(s);
}

void fp8_fp4_gemm_nt_sm120_v1(torch::Tensor a, torch::Tensor sfa, torch::Tensor b, torch::Tensor sfb, torch::Tensor d, int64_t gran_k) {
  const int M = (int)a.size(0), K = (int)a.size(1), N = (int)b.size(0);
  check_common(a, sfa, b, sfb, gran_k, M, N, K);
  TORCH_CHECK(M >= 1 && M <= BM && N % 128 == 0, "v1: M <= 32, N a multiple of 128");
  TORCH_CHECK(d.scalar_type() == torch::kBFloat16 && d.size(0) == M && d.size(1) == N && d.is_contiguous(), "D: bf16 [M, N]");
  const int smem = STAGES * (BM * BK + 128 * (BK / 2));
  static bool attr_set = false;
  if (!attr_set) { cudaFuncSetAttribute(k_fp8_fp4_gemm_nt_v1, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr_set = true; }
  auto st = at::cuda::getCurrentCUDAStream();
  k_fp8_fp4_gemm_nt_v1<<<dim3(N / 128, 1), 256, smem, st>>>(static_cast<const uint8_t*>(a.data_ptr()), reinterpret_cast<const uint8_t*>(sfa.data_ptr<int>()),
                                                           reinterpret_cast<const uint8_t*>(b.data_ptr<int8_t>()),
                                                           reinterpret_cast<const uint8_t*>(sfb.data_ptr<int>()),
                                                           reinterpret_cast<__nv_bfloat16*>(d.data_ptr()), M, N, K, (int)sfa.size(1) * 4);
}

template <int BN, int SWZ, int KPERM, int ONESYNC>
static void launch_v2(const uint8_t* pa, const uint8_t* psa, const uint8_t* pb, const uint8_t* psb, float* ws, int M, int N, int K,
                      int sfs, int chunk, int used, cudaStream_t st) {
  const int smem = STAGES * (BM * BK + BN * (BK / 2));
  static bool set = false;
  if (!set) { cudaFuncSetAttribute(k_fp8_fp4_gemm_nt_v2<BN, SWZ, KPERM, ONESYNC>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); set = true; }
  k_fp8_fp4_gemm_nt_v2<BN, SWZ, KPERM, ONESYNC><<<dim3(N / BN, used), (BN / 16) * 32, smem, st>>>(pa, psa, pb, psb, ws, M, N, K, sfs, chunk);
}

template <int BN>
static void launch_v2_arms(int arms, const uint8_t* pa, const uint8_t* psa, const uint8_t* pb, const uint8_t* psb, float* ws, int M, int N,
                           int K, int sfs, int chunk, int used, cudaStream_t st) {
  switch (arms & 7) {
    case 0: launch_v2<BN, 0, 0, 0>(pa, psa, pb, psb, ws, M, N, K, sfs, chunk, used, st); break;
    case 1: launch_v2<BN, 1, 0, 0>(pa, psa, pb, psb, ws, M, N, K, sfs, chunk, used, st); break;
    case 2: launch_v2<BN, 0, 1, 0>(pa, psa, pb, psb, ws, M, N, K, sfs, chunk, used, st); break;
    case 3: launch_v2<BN, 1, 1, 0>(pa, psa, pb, psb, ws, M, N, K, sfs, chunk, used, st); break;
    case 4: launch_v2<BN, 0, 0, 1>(pa, psa, pb, psb, ws, M, N, K, sfs, chunk, used, st); break;
    case 5: launch_v2<BN, 1, 0, 1>(pa, psa, pb, psb, ws, M, N, K, sfs, chunk, used, st); break;
    case 6: launch_v2<BN, 0, 1, 1>(pa, psa, pb, psb, ws, M, N, K, sfs, chunk, used, st); break;
    default: launch_v2<BN, 1, 1, 1>(pa, psa, pb, psb, ws, M, N, K, sfs, chunk, used, st); break;
  }
}

void fp8_fp4_gemm_nt_sm120_v2(torch::Tensor a, torch::Tensor sfa, torch::Tensor b, torch::Tensor sfb, torch::Tensor d, torch::Tensor ws,
                              int64_t gran_k, int64_t bn, int64_t splits, int64_t arms) {
  const int M = (int)a.size(0), K = (int)a.size(1), N = (int)b.size(0);
  check_common(a, sfa, b, sfb, gran_k, M, N, K);
  TORCH_CHECK(M >= 1 && M <= BM, "v2: M <= 32");
  TORCH_CHECK(bn == 64 || bn == 128, "v2: BN 64 or 128");
  TORCH_CHECK(N % bn == 0, "v2: N a multiple of BN");
  const int nblocks = K / BK;
  TORCH_CHECK(splits >= 1 && splits <= nblocks, "v2: 1 <= splits <= K/128");
  const int chunk = (nblocks + (int)splits - 1) / (int)splits;
  const int used = (nblocks + chunk - 1) / chunk;              // splits that get at least one block
  TORCH_CHECK(ws.scalar_type() == torch::kFloat && ws.numel() >= (int64_t)used * M * N && ws.is_contiguous(), "workspace: fp32 [splits, M, N]");
  TORCH_CHECK(d.scalar_type() == torch::kBFloat16 && d.size(0) == M && d.size(1) == N && d.is_contiguous(), "D: bf16 [M, N]");
  auto st = at::cuda::getCurrentCUDAStream();
  const uint8_t* pa = static_cast<const uint8_t*>(a.data_ptr());
  const uint8_t* psa = reinterpret_cast<const uint8_t*>(sfa.data_ptr<int>());
  const uint8_t* pb = reinterpret_cast<const uint8_t*>(b.data_ptr<int8_t>());
  const uint8_t* psb = reinterpret_cast<const uint8_t*>(sfb.data_ptr<int>());
  const int sfs = (int)sfa.size(1) * 4;
  if (bn == 128) launch_v2_arms<128>((int)arms, pa, psa, pb, psb, ws.data_ptr<float>(), M, N, K, sfs, chunk, used, st);
  else launch_v2_arms<64>((int)arms, pa, psa, pb, psb, ws.data_ptr<float>(), M, N, K, sfs, chunk, used, st);
  const int total = M * N;
  k_split_reduce<<<(total + 255) / 256, 256, 0, st>>>(ws.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(d.data_ptr()), M, N, used);
}
"""


def build(verbose: bool = False):
    return load_inline(name="sm120fp4_fp8_fp4_gemm_nt_v3a", cpp_sources=CPP, cuda_sources=CUDA,
                       functions=["fp8_fp4_gemm_nt_sm120", "fp8_fp4_gemm_nt_sm120_v1", "fp8_fp4_gemm_nt_sm120_v2"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a"], verbose=verbose)


SM_COUNT = None


def plan_v2(n: int, k: int, sm_count: int) -> tuple[int, int]:
    """(BN, splits): the smallest split count that gives at least two blocks per SM, BN 64 when 128 would need more than
    half the K blocks as splits, else 128."""
    nblocks = k // 128
    for bn in (128, 64):
        cols = n // bn
        splits = max(1, min(nblocks, math.ceil(2 * sm_count / cols)))
        if splits <= max(1, nblocks // 2) or bn == 64:
            return bn, splits
    return 64, 1


def v2(mod, a8, sfa, b4, sfb, d, bn=None, splits=None, arms=0):
    global SM_COUNT
    if SM_COUNT is None:
        SM_COUNT = torch.cuda.get_device_properties(0).multi_processor_count
    m, n, k = a8.shape[0], b4.shape[0], a8.shape[1]
    pbn, psp = plan_v2(n, k, SM_COUNT)
    bn, splits = bn or pbn, splits or psp
    ws = torch.empty(splits, m, n, device=a8.device, dtype=torch.float32)
    mod.fp8_fp4_gemm_nt_sm120_v2(a8, sfa, b4, sfb, d, ws, 128, bn, splits, arms)
    return bn, splits


def make_inputs(m: int, n: int, k: int, gran_k: int, seed: int, dev):
    g = torch.Generator().manual_seed(seed)
    a = (torch.randn(m, k, generator=g) * torch.pow(2.0, torch.randint(-6, 8, (m, 1), generator=g).float())).to(dev)
    b = (torch.randn(n, k, generator=g) * torch.pow(2.0, torch.randint(-6, 8, (n, 1), generator=g).float())).to(dev)
    a8, sfa = ref.per_token_cast_to_fp8(a, use_ue8m0=True, gran_k=gran_k, use_packed_ue8m0=True)
    b4, sfb = ref.per_token_cast_to_fp4(b, use_ue8m0=True, gran_k=gran_k, use_packed_ue8m0=True)
    return a8, sfa, b4, sfb


def check(d, a8, sfa, b4, sfb, gran_k):
    exact = ref.mx_gemm_reference(a8, ref.unpack_ue8m0_from_int(sfa), b4, ref.unpack_ue8m0_from_int(sfb), gran_k)
    diff = (d.float() - exact).abs()
    out = {"max_abs_err": float(diff.max()), "rel_fro_err": float((d.float() - exact).norm() / exact.norm()),
           "ref_abs_max": float(exact.abs().max()), "bf16_half_ulp_at_max": float(exact.abs().max()) * 2 ** -9}
    out["pass"] = out["max_abs_err"] <= 2 * out["bf16_half_ulp_at_max"] + 1e-6 and out["rel_fro_err"] < 4e-3
    return out


def bf16_ulps_apart(x: torch.Tensor, y: torch.Tensor) -> int:
    """Largest distance in bf16 ulps between two bf16 tensors (0 when equal)."""
    xi = x.view(torch.int16).to(torch.int32)
    yi = y.view(torch.int16).to(torch.int32)
    return int((xi - yi).abs().max())


def selftest(mod, dev) -> int:
    ok = True
    print("v0 against the reference")
    for (m, n, k, seed) in ((16, 128, 256, 1), (16, 8, 128, 2), (5, 64, 512, 3), (16, 512, 2048, 4), (1, 256, 1024, 5)):
        a8, sfa, b4, sfb = make_inputs(m, n, k, 128, seed, dev)
        d = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
        mod.fp8_fp4_gemm_nt_sm120(a8, sfa, b4, sfb, d, 128)
        torch.cuda.synchronize()
        r = check(d, a8, sfa, b4, sfb, 128)
        ok &= r["pass"]
        print(f"  m={m} n={n} k={k}: max|err| {r['max_abs_err']:.4g} (half-ulp at max {r['bf16_half_ulp_at_max']:.4g}), rel Frobenius {r['rel_fro_err']:.3e} -> {'ok' if r['pass'] else 'FAIL'}", flush=True)
    print("v1 against the reference and bit for bit against v0 (M <= 16); v2 against the reference and against v1 in bf16 ulps")
    for (m, n, k, seed) in ((16, 128, 256, 11), (32, 128, 2048, 12), (16, 512, 2048, 13), (32, 2048, 7168, 14), (7, 256, 1024, 15), (32, 128, 128, 16), (16, 7168, 7168, 17)):
        a8, sfa, b4, sfb = make_inputs(m, n, k, 128, seed, dev)
        d1 = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
        mod.fp8_fp4_gemm_nt_sm120_v1(a8, sfa, b4, sfb, d1, 128)
        torch.cuda.synchronize()
        r1 = check(d1, a8, sfa, b4, sfb, 128)
        ok &= r1["pass"]
        same0 = None
        if m <= 16:
            d0 = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
            mod.fp8_fp4_gemm_nt_sm120(a8, sfa, b4, sfb, d0, 128)
            torch.cuda.synchronize()
            same0 = bool(torch.equal(d0, d1))
            ok &= same0
        d2 = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
        bn, splits = v2(mod, a8, sfa, b4, sfb, d2)
        torch.cuda.synchronize()
        r2 = check(d2, a8, sfa, b4, sfb, 128)
        ulps = bf16_ulps_apart(d1, d2)
        ndiff = int((d1 != d2).sum())
        ok &= r2["pass"] and ulps <= 1          # fp32 summation-order differences before the bf16 round: at most one bf16 ulp
        tail0 = "" if same0 is None else f", v1==v0 {same0}"
        print(f"  m={m} n={n} k={k}: v1 rel {r1['rel_fro_err']:.2e}{tail0}; v2 (BN {bn}, splits {splits}) rel {r2['rel_fro_err']:.2e}, "
              f"max|err| {r2['max_abs_err']:.4g} vs half-ulp {r2['bf16_half_ulp_at_max']:.4g}, items differing from v1 {ndiff}/{m * n}, "
              f"max {ulps} bf16 ulp -> {'ok' if (r1['pass'] and r2['pass'] and ulps <= 1 and same0 is not False) else 'FAIL'}", flush=True)
    # explicit BN 64 and BN 128 at the same shape agree with each other up to the same ulp bound
    a8, sfa, b4, sfb = make_inputs(16, 2048, 7168, 128, 21, dev)
    da = torch.empty(16, 2048, device=dev, dtype=torch.bfloat16)
    db = torch.empty(16, 2048, device=dev, dtype=torch.bfloat16)
    v2(mod, a8, sfa, b4, sfb, da, bn=64, splits=11)
    v2(mod, a8, sfa, b4, sfb, db, bn=128, splits=7)
    torch.cuda.synchronize()
    u = bf16_ulps_apart(da, db)
    ok &= u <= 1
    print(f"  v2 BN 64/11 splits vs BN 128/7 splits: max {u} bf16 ulp apart -> {'ok' if u <= 1 else 'FAIL'}")
    # the three arms, alone and together, against arm 0 on three shapes: SWZ and ONESYNC must be bit-identical (same arithmetic);
    # KPERM permutes the summation index inside the MMA and is allowed one ulp, and the count of differing elements is reported
    print("arms against v2 (arm 0): 1 = swizzled A rows, 2 = K permutation (8-byte A, 4-byte B loads), 4 = one barrier per stage")
    for (m, n, k, seed) in ((16, 2048, 7168, 31), (32, 7168, 7168, 32), (16, 4096, 2048, 33)):
        a8, sfa, b4, sfb = make_inputs(m, n, k, 128, seed, dev)
        base = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
        v2(mod, a8, sfa, b4, sfb, base, arms=0)
        torch.cuda.synchronize()
        rb = check(base, a8, sfa, b4, sfb, 128)
        for arms in (1, 2, 4, 7):
            dd = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
            v2(mod, a8, sfa, b4, sfb, dd, arms=arms)
            torch.cuda.synchronize()
            r = check(dd, a8, sfa, b4, sfb, 128)
            u = bf16_ulps_apart(base, dd)
            nd = int((base != dd).sum())
            bound = 1 if (arms & 2) else 0
            good = r["pass"] and u <= bound
            ok &= good
            print(f"  m={m} n={n} k={k} arms={arms}: rel {r['rel_fro_err']:.2e}, differing from arm 0 {nd}/{m * n}, max {u} ulp (bound {bound}) -> {'ok' if good else 'FAIL'}", flush=True)
    print("fp8_fp4_gemm_sm120 selftest:", "ok" if ok else "FAIL")
    return 0 if ok else 1


def bench(mod, dev, out):
    """v1 and v2 timed on decode shapes after an L2 flush, median of 20 single launches (v2 includes its reduce kernel)."""
    if out is not None and out.exists():
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 2
    props = torch.cuda.get_device_properties(0)
    flush_buf = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    rows = []
    for (m, n, k) in ((16, 2048, 7168), (16, 7168, 7168), (32, 2048, 7168), (32, 7168, 7168), (16, 4096, 2048)):
        a8, sfa, b4, sfb = make_inputs(m, n, k, 128, 100 + m, dev)
        d = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
        bn, splits = plan_v2(n, k, props.multi_processor_count)
        ws = torch.empty(splits, m, n, device=dev, dtype=torch.float32)
        res = {"m": m, "n": n, "k": k, "bn": bn, "splits": splits, "blocks_v2": (n // bn) * splits, "blocks_v1": n // 128}
        variants = [("v1", lambda: mod.fp8_fp4_gemm_nt_sm120_v1(a8, sfa, b4, sfb, d, 128))]
        for arms, label in ((0, "v2"), (1, "v2_swz"), (2, "v2_kperm"), (4, "v2_onesync"), (7, "v2_all")):
            variants.append((label, (lambda arms=arms: mod.fp8_fp4_gemm_nt_sm120_v2(a8, sfa, b4, sfb, d, ws, 128, bn, splits, arms))))
        for name, fn in variants:
            fn()
            torch.cuda.synchronize()
            times = []
            for _ in range(20):
                flush_buf.fill_(1)
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                fn()
                e1.record()
                torch.cuda.synchronize()
                times.append(e0.elapsed_time(e1) * 1000)
            res[f"{name}_us_median"] = statistics.median(times)
            res[f"{name}_us_min"] = min(times)
        nbytes = a8.numel() + b4.numel() + sfa.numel() * 4 + sfb.numel() * 4 + d.numel() * 2
        res["bytes"] = nbytes
        res["v2_achieved_GBps"] = nbytes / res["v2_us_median"] / 1e3
        for label in ("v2_swz", "v2_kperm", "v2_onesync", "v2_all"):
            res[f"{label}_achieved_GBps"] = nbytes / res[f"{label}_us_median"] / 1e3
            res[f"{label}_gain_vs_v2"] = res["v2_us_median"] / res[f"{label}_us_median"] - 1.0
        res["v1_achieved_GBps"] = nbytes / res["v1_us_median"] / 1e3
        res["floor_us_at_1792"] = nbytes / 1792.0 / 1e3
        rows.append(res)
        print(f"m={m} n={n} k={k}: v1 {res['v1_us_median']:.1f} us; v2 {res['v2_us_median']:.1f} us ({res['v2_achieved_GBps']:.0f} GB/s); "
              f"swz {res['v2_swz_us_median']:.1f} ({res['v2_swz_gain_vs_v2']:+.1%}), kperm {res['v2_kperm_us_median']:.1f} ({res['v2_kperm_gain_vs_v2']:+.1%}), "
              f"onesync {res['v2_onesync_us_median']:.1f} ({res['v2_onesync_gain_vs_v2']:+.1%}), all {res['v2_all_us_median']:.1f} ({res['v2_all_gain_vs_v2']:+.1%}); "
              f"floor {res['floor_us_at_1792']:.1f} us", flush=True)
    report = {"kernel": "fp8_fp4_gemm_nt_sm120_v2 with arms (and v1)", "device": props.name, "sm_count": props.multi_processor_count,
              "note": "cold L2 (256 MB fill before each launch); one launch per timing (v2 = tile kernel + reduce); bytes = A e4m3 + B packed e2m1 + packed scales + bf16 D; floor at 1792 GB/s",
              "tile": {"BM": 32, "BK": 128, "stages": 4}, "rows": rows}
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
    a = ap.parse_args(argv)
    if not a.selftest and not a.bench:
        ap.error("pass --selftest and/or --bench")
    dev = torch.device("cuda")
    mod = build()
    rc = selftest(mod, dev) if a.selftest else 0
    if a.bench and rc == 0:
        rc = bench(mod, dev, a.out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
