"""Stage 3, the first kernel: SM120 FP8xFP4 GEMM with UE8M0 block scales (DeepGEMM's `fp8_fp4_gemm_nt` contract, 1D1D,
dense, NT, K-major), in two versions.

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
block stream through a 4-stage cp.async pipeline in shared memory (12 KB a stage, 48 KB), fragments are loaded from
shared memory (the e2m1 nibbles unpacked to containers on the way), and the UE8M0 fold happens once per 128-K block
exactly as in v0, so v1's arithmetic order per output element is v0's and the two agree bit for bit. M <= 32, N a
multiple of 128, K a multiple of 128, gran_k 128, no epilogue.

    PYTHONPATH=. python scripts/fp8_fp4_gemm_sm120.py --selftest                       # both versions against the reference
    PYTHONPATH=. python scripts/fp8_fp4_gemm_sm120.py --bench --out reports/fp8-fp4-gemm-v1-<device>-<date>.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
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

void fp8_fp4_gemm_nt_sm120(torch::Tensor a, torch::Tensor sfa, torch::Tensor b, torch::Tensor sfb, torch::Tensor d, int64_t gran_k) {
  const int M = (int)a.size(0), K = (int)a.size(1), N = (int)b.size(0);
  TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn && a.is_contiguous(), "A: e4m3 [M, K], contiguous");
  TORCH_CHECK(b.scalar_type() == torch::kInt8 && b.is_contiguous() && b.size(1) * 2 == K, "B: packed e2m1 int8 [N, K/2]");
  TORCH_CHECK(sfa.scalar_type() == torch::kInt && sfb.scalar_type() == torch::kInt, "scales: packed UE8M0 int32");
  TORCH_CHECK(gran_k == 128 && K % gran_k == 0, "this version: gran_k 128, K a multiple of 128");
  TORCH_CHECK(sfa.size(0) == M && sfb.size(0) == N && sfa.size(1) == sfb.size(1) && sfa.size(1) * 4 >= K / gran_k
              && sfa.is_contiguous() && sfb.is_contiguous(), "scale shapes: [rows, ceil((K/gran_k)/4)] int32, padded to a multiple of 4 scales");
  TORCH_CHECK(M >= 1 && M <= 16 && N % 8 == 0, "v0: M <= 16, N a multiple of 8");
  TORCH_CHECK(d.scalar_type() == torch::kBFloat16 && d.size(0) == M && d.size(1) == N && d.is_contiguous(), "D: bf16 [M, N]");
  auto st = at::cuda::getCurrentCUDAStream();
  k_fp8_fp4_gemm_nt_v0<<<N / 8, 32, 0, st>>>(static_cast<const uint8_t*>(a.data_ptr()), reinterpret_cast<const uint8_t*>(sfa.data_ptr<int>()),
                                             reinterpret_cast<const uint8_t*>(b.data_ptr<int8_t>()),
                                             reinterpret_cast<const uint8_t*>(sfb.data_ptr<int>()),
                                             reinterpret_cast<__nv_bfloat16*>(d.data_ptr()), M, N, K, (int)gran_k, (int)sfa.size(1) * 4);
}

// ------------------------------------------------------------------------------------------------------------ v1
// Block: 8 warps, a 32 (M) x 128 (N) output tile; warp w owns columns 16w..16w+15 (two n8 tiles) for both m16 tiles.
// Per 128-K block (one UE8M0 block): A tile 32 x 128 bytes (4 KB), B tile 128 cols x 64 packed bytes (8 KB), staged by
// cp.async in STAGES stages. Fragments come from shared memory; the fold per 128-K block is v0's, so v1 == v0 bit for bit.
constexpr int V1_BM = 32, V1_BN = 128, V1_BK = 128, V1_STAGES = 4, V1_THREADS = 256;
constexpr int V1_A_BYTES = V1_BM * V1_BK;          // 4096
constexpr int V1_B_BYTES = V1_BN * (V1_BK / 2);    // 8192
constexpr int V1_STAGE_BYTES = V1_A_BYTES + V1_B_BYTES;

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
  const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__global__ void __launch_bounds__(V1_THREADS)
k_fp8_fp4_gemm_nt_v1(const uint8_t* __restrict__ a, const uint8_t* __restrict__ sfa,
                     const uint8_t* __restrict__ b, const uint8_t* __restrict__ sfb,
                     __nv_bfloat16* __restrict__ d, int M, int N, int K, int sf_stride) {
  extern __shared__ __align__(16) uint8_t smem[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int n_base = blockIdx.x * V1_BN, m_base = blockIdx.y * V1_BM;
  const int nblocks = K / V1_BK;

  // copy duties per stage: A has 256 16-byte chunks (one per thread), B has 512 (two per thread)
  const int a_row = threadIdx.x >> 3, a_chunk = threadIdx.x & 7;
  const bool a_row_ok = (m_base + a_row) < M;
  const int b_col0 = threadIdx.x >> 2, b_chunk0 = threadIdx.x & 3;
  const int b_col1 = b_col0 + 64;
  auto issue = [&](int kb, int stage) {
    uint8_t* sA = smem + stage * V1_STAGE_BYTES;
    uint8_t* sB = sA + V1_A_BYTES;
    if (a_row_ok) cp_async_16(sA + a_row * V1_BK + a_chunk * 16, a + (size_t)(m_base + a_row) * K + (size_t)kb * V1_BK + a_chunk * 16);
    else *reinterpret_cast<uint4*>(sA + a_row * V1_BK + a_chunk * 16) = make_uint4(0, 0, 0, 0);
    cp_async_16(sB + b_col0 * 64 + b_chunk0 * 16, b + (size_t)(n_base + b_col0) * (K / 2) + (size_t)kb * (V1_BK / 2) + b_chunk0 * 16);
    cp_async_16(sB + b_col1 * 64 + b_chunk0 * 16, b + (size_t)(n_base + b_col1) * (K / 2) + (size_t)kb * (V1_BK / 2) + b_chunk0 * 16);
  };

  float acc[2][2][4];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[mt][nt][i] = 0.f;

#pragma unroll
  for (int s = 0; s < V1_STAGES - 1; ++s) {
    if (s < nblocks) issue(s, s);
    cp_async_commit();
  }

  const int row_g = g, row_g8 = g + 8;
  const bool has[2][2] = {{m_base + row_g < M, m_base + row_g8 < M}, {m_base + 16 + row_g < M, m_base + 16 + row_g8 < M}};
  for (int kb = 0; kb < nblocks; ++kb) {
    cp_async_wait<V1_STAGES - 2>();
    __syncthreads();
    const int nxt = kb + V1_STAGES - 1;
    if (nxt < nblocks) issue(nxt, nxt % V1_STAGES);
    cp_async_commit();
    const uint8_t* sA = smem + (kb % V1_STAGES) * V1_STAGE_BYTES;
    const uint8_t* sB = sA + V1_A_BYTES;

    float part[2][2][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int i = 0; i < 4; ++i) part[mt][nt][i] = 0.f;

#pragma unroll
    for (int s = 0; s < V1_BK / 32; ++s) {
      const int k0 = s * 32;
      uint32_t bf[2][2];
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const uint8_t* brow = sB + (warp * 16 + nt * 8 + g) * 64;
        bf[nt][0] = unpack_e2m1_x4(*reinterpret_cast<const uint16_t*>(brow + (k0 + 4 * t) / 2));
        bf[nt][1] = unpack_e2m1_x4(*reinterpret_cast<const uint16_t*>(brow + (k0 + 16 + 4 * t) / 2));
      }
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        uint32_t af[4];
        const uint8_t* r0 = sA + (mt * 16 + row_g) * V1_BK;
        const uint8_t* r1 = sA + (mt * 16 + row_g8) * V1_BK;
        af[0] = *reinterpret_cast<const uint32_t*>(r0 + k0 + 4 * t);
        af[1] = *reinterpret_cast<const uint32_t*>(r1 + k0 + 4 * t);
        af[2] = *reinterpret_cast<const uint32_t*>(r0 + k0 + 16 + 4 * t);
        af[3] = *reinterpret_cast<const uint32_t*>(r1 + k0 + 16 + 4 * t);
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
    __syncthreads();
  }
  cp_async_wait<0>();

#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
    const int r0 = m_base + mt * 16 + row_g, r1 = m_base + mt * 16 + row_g8;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int c0 = n_base + warp * 16 + nt * 8 + 2 * t;
      if (has[mt][0]) {
        d[(size_t)r0 * N + c0] = __float2bfloat16(acc[mt][nt][0]);
        d[(size_t)r0 * N + c0 + 1] = __float2bfloat16(acc[mt][nt][1]);
      }
      if (has[mt][1]) {
        d[(size_t)r1 * N + c0] = __float2bfloat16(acc[mt][nt][2]);
        d[(size_t)r1 * N + c0 + 1] = __float2bfloat16(acc[mt][nt][3]);
      }
    }
  }
}

void fp8_fp4_gemm_nt_sm120_v1(torch::Tensor a, torch::Tensor sfa, torch::Tensor b, torch::Tensor sfb, torch::Tensor d, int64_t gran_k) {
  const int M = (int)a.size(0), K = (int)a.size(1), N = (int)b.size(0);
  TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn && a.is_contiguous(), "A: e4m3 [M, K], contiguous");
  TORCH_CHECK(b.scalar_type() == torch::kInt8 && b.is_contiguous() && b.size(1) * 2 == K, "B: packed e2m1 int8 [N, K/2]");
  TORCH_CHECK(sfa.scalar_type() == torch::kInt && sfb.scalar_type() == torch::kInt, "scales: packed UE8M0 int32");
  TORCH_CHECK(gran_k == V1_BK && K % V1_BK == 0, "v1: gran_k 128, K a multiple of 128");
  TORCH_CHECK(sfa.size(0) == M && sfb.size(0) == N && sfa.size(1) == sfb.size(1) && sfa.size(1) * 4 >= K / gran_k
              && sfa.is_contiguous() && sfb.is_contiguous(), "scale shapes: [rows, ceil((K/gran_k)/4)] int32, padded to a multiple of 4 scales");
  TORCH_CHECK(M >= 1 && M <= V1_BM && N % V1_BN == 0, "v1: M <= 32, N a multiple of 128");
  TORCH_CHECK(d.scalar_type() == torch::kBFloat16 && d.size(0) == M && d.size(1) == N && d.is_contiguous(), "D: bf16 [M, N]");
  const int smem = V1_STAGES * V1_STAGE_BYTES;
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(k_fp8_fp4_gemm_nt_v1, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    attr_set = true;
  }
  auto st = at::cuda::getCurrentCUDAStream();
  const dim3 grid(N / V1_BN, 1);
  k_fp8_fp4_gemm_nt_v1<<<grid, V1_THREADS, smem, st>>>(static_cast<const uint8_t*>(a.data_ptr()), reinterpret_cast<const uint8_t*>(sfa.data_ptr<int>()),
                                                      reinterpret_cast<const uint8_t*>(b.data_ptr<int8_t>()),
                                                      reinterpret_cast<const uint8_t*>(sfb.data_ptr<int>()),
                                                      reinterpret_cast<__nv_bfloat16*>(d.data_ptr()), M, N, K, (int)sfa.size(1) * 4);
}
"""


def build(verbose: bool = False):
    return load_inline(name="sm120fp4_fp8_fp4_gemm_nt_v1a", cpp_sources=CPP, cuda_sources=CUDA,
                       functions=["fp8_fp4_gemm_nt_sm120", "fp8_fp4_gemm_nt_sm120_v1"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a"], verbose=verbose)


def make_inputs(m: int, n: int, k: int, gran_k: int, seed: int, dev):
    g = torch.Generator().manual_seed(seed)
    a = (torch.randn(m, k, generator=g) * torch.pow(2.0, torch.randint(-6, 8, (m, 1), generator=g).float())).to(dev)
    b = (torch.randn(n, k, generator=g) * torch.pow(2.0, torch.randint(-6, 8, (n, 1), generator=g).float())).to(dev)
    a8, sfa = ref.per_token_cast_to_fp8(a, use_ue8m0=True, gran_k=gran_k, use_packed_ue8m0=True)
    b4, sfb = ref.per_token_cast_to_fp4(b, use_ue8m0=True, gran_k=gran_k, use_packed_ue8m0=True)
    return a8, sfa, b4, sfb


def run_case(mod, fn, m: int, n: int, k: int, gran_k: int, seed: int, dev):
    a8, sfa, b4, sfb = make_inputs(m, n, k, gran_k, seed, dev)
    d = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
    fn(a8, sfa, b4, sfb, d, gran_k)
    torch.cuda.synchronize()
    exact = ref.mx_gemm_reference(a8, ref.unpack_ue8m0_from_int(sfa), b4, ref.unpack_ue8m0_from_int(sfb), gran_k)
    diff = (d.float() - exact).abs()
    out = {"m": m, "n": n, "k": k, "gran_k": gran_k, "seed": seed, "max_abs_err": float(diff.max()),
           "rel_fro_err": float((d.float() - exact).norm() / exact.norm()), "ref_abs_max": float(exact.abs().max()),
           "bf16_half_ulp_at_max": float(exact.abs().max()) * 2 ** -9}
    out["pass"] = out["max_abs_err"] <= 2 * out["bf16_half_ulp_at_max"] + 1e-6 and out["rel_fro_err"] < 4e-3
    return out, d


def selftest(mod, dev) -> int:
    # The reference is the exact dequantized product, so the only admissible error is bf16 output rounding (half an ulp at
    # each element's magnitude) plus fp32 summation order; DeepGEMM's own tolerance for a mixed FP8xFP4 configuration is
    # max_diff 0.01 against an fp32 reference of the *original* values (tests/generators.py), a different and looser bar.
    ok = True
    print("v0 against the reference")
    for (m, n, k, seed) in ((16, 128, 256, 1), (16, 8, 128, 2), (5, 64, 512, 3), (16, 512, 2048, 4), (1, 256, 1024, 5)):
        r, _ = run_case(mod, mod.fp8_fp4_gemm_nt_sm120, m, n, k, 128, seed, dev)
        ok &= r["pass"]
        print(f"  m={m} n={n} k={k}: max|err| {r['max_abs_err']:.4g} (half-ulp at max {r['bf16_half_ulp_at_max']:.4g}), rel Frobenius {r['rel_fro_err']:.3e} -> {'ok' if r['pass'] else 'FAIL'}", flush=True)
    print("v1 against the reference, and bit for bit against v0 where v0 applies (M <= 16)")
    for (m, n, k, seed) in ((16, 128, 256, 11), (32, 128, 2048, 12), (16, 512, 2048, 13), (32, 2048, 7168, 14), (7, 256, 1024, 15), (32, 128, 128, 16)):
        r, d1 = run_case(mod, mod.fp8_fp4_gemm_nt_sm120_v1, m, n, k, 128, seed, dev)
        same = None
        if m <= 16:
            _, d0 = run_case(mod, mod.fp8_fp4_gemm_nt_sm120, m, n, k, 128, seed, dev)
            same = bool(torch.equal(d0, d1))
            ok &= same
        ok &= r["pass"]
        tail = "" if same is None else f", v0 bit-identical {same}"
        print(f"  m={m} n={n} k={k}: max|err| {r['max_abs_err']:.4g} (half-ulp at max {r['bf16_half_ulp_at_max']:.4g}), rel Frobenius {r['rel_fro_err']:.3e}{tail} -> {'ok' if r['pass'] and same is not False else 'FAIL'}", flush=True)
    print("fp8_fp4_gemm_sm120 selftest:", "ok" if ok else "FAIL")
    return 0 if ok else 1


def bench(mod, dev, out):
    """v1 timed on decode shapes after an L2 flush, median of 20 single launches; the achieved GB/s is operand bytes over
    the measured time, to be read against the card's peak bandwidth stated in the report."""
    if out is not None and out.exists():
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 2
    props = torch.cuda.get_device_properties(0)
    flush_buf = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    rows = []
    for (m, n, k) in ((16, 2048, 7168), (16, 7168, 7168), (32, 2048, 7168), (32, 7168, 7168), (16, 4096, 2048)):
        a8, sfa, b4, sfb = make_inputs(m, n, k, 128, 100 + m, dev)
        d = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
        mod.fp8_fp4_gemm_nt_sm120_v1(a8, sfa, b4, sfb, d, 128)
        torch.cuda.synchronize()
        times = []
        for _ in range(20):
            flush_buf.fill_(1)
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            mod.fp8_fp4_gemm_nt_sm120_v1(a8, sfa, b4, sfb, d, 128)
            e1.record()
            torch.cuda.synchronize()
            times.append(e0.elapsed_time(e1) * 1000)
        med = statistics.median(times)
        nbytes = a8.numel() + b4.numel() + sfa.numel() * 4 + sfb.numel() * 4 + d.numel() * 2
        rows.append({"m": m, "n": n, "k": k, "us_median": med, "us_min": min(times), "repeats": 20, "bytes": nbytes,
                     "achieved_GBps": nbytes / med / 1e3, "weight_bytes": b4.numel()})
        print(f"m={m} n={n} k={k}: {med:.1f} us median ({min(times):.1f} min), {nbytes / 1e6:.2f} MB moved, {nbytes / med / 1e3:.0f} GB/s achieved", flush=True)
    report = {"kernel": "fp8_fp4_gemm_nt_sm120_v1", "device": props.name, "sm_count": props.multi_processor_count,
              "note": "cold L2 (256 MB fill before each launch); one launch per timing; bytes = A e4m3 + B packed e2m1 + packed scales + bf16 D",
              "tile": {"BM": 32, "BN": 128, "BK": 128, "stages": 4, "threads": 256}, "rows": rows}
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
