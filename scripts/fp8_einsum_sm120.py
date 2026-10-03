#!/usr/bin/env python3
"""FP8 einsum 'bhr,hdr->bhd' on SM120 (RTX 5090): DeepGEMM's fp8_einsum expression that runs on SM90 and SM100 (`sm100_fp8_bmm`),
written against the mma.sync kind::f8f6f4 path this repository measured for the FP8 x FP4 GEMM.

What it computes: z[b, h, d] = sum_r x[b, h, r] * y[h, d, r], per head h a GEMM of x's [B, R] slice against y's [D, R] slice.
Operands follow DeepGEMM's tests: x is e4m3 with one UE8M0 scale per (b, h) row and 128-wide R block (`per_token_cast_to_fp8`);
y is e4m3 with one UE8M0 scale per 128 x 128 block of its [D, R] slice (`per_block_cast_to_fp8`: blocks over d and r, scale layout
[D/128, R/128]); z is bf16. The fold of the two scales is outside the MMA per 128-R block, indexed by (b, kblock) for x and by
(dblock, kblock) for y, as the GEMM folds its UE8M0 pairs.

v0 is for correctness: one warp per (h, 16 rows of b, 8 columns of d), the FP8 x FP4 GEMM's v0 shape with the B fragment read
as e4m3 bytes (no e2m1 container shift). The scale of y is per 128 d-columns, so the fold differs between d tiles only at the
128 boundary, which the n8 tile never straddles.

v1 is the tiled form: one block of 8 warps per (h, 128 columns of d, up to 128 rows of b); per 128-R block the y tile
(128 d x 128 r bytes) and the x tile (128 b x 128 r bytes) are staged in shared memory with cp.async, double-buffered, and
each warp owns 16 columns (two n8 tiles) for every 16-row b tile of the block. y is read from global memory once per 128-row
b chunk, where v0 re-read it once per 16-row b tile. Same accumulation order per element as v0, so the two are bit-identical.

    ~/mlsys-5090-runtime/vllm028/.venv/bin/python scripts/fp8_einsum_sm120.py --selftest
    ~/mlsys-5090-runtime/vllm028/.venv/bin/python scripts/fp8_einsum_sm120.py --bench --out reports/fp8-einsum-v1-rtx5090-20261003.json
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
import ue8m0_reference as ref  # noqa: E402

GRAN = 128
E4M3_MAX = 448.0

CPP = r"""
#include <torch/extension.h>
void fp8_einsum_bhr_hdr_bhd_sm120_v0(torch::Tensor x, torch::Tensor sfx, torch::Tensor y, torch::Tensor sfy, torch::Tensor z);
void fp8_einsum_bhr_hdr_bhd_sm120_v1(torch::Tensor x, torch::Tensor sfx, torch::Tensor y, torch::Tensor sfy, torch::Tensor z);
"""

CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cstdint>

// e4m3 x e4m3 through the same instruction family the FP8 x FP4 GEMM uses; both operands are 8-bit containers, so the B fragment
// is the four e4m3 bytes of k 4t..4t+3 (and 16+4t..) of column g, read directly (scripts/fp8_fp4_gemm_sm120.py's layout).
__device__ __forceinline__ void mma_e4m3_e4m3(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ float ue8m0_to_float(uint32_t e) { return __uint_as_float(e << 23); }

// v0: one warp per (h, 16-row b tile, 8-column d tile). x: [B, H, R] e4m3; sfx: [B, H, R/128] UE8M0 bytes; y: [H, D, R] e4m3;
// sfy: [H, D/128, R/128] UE8M0 bytes; z: [B, H, D] bf16.
__global__ void __launch_bounds__(32)
k_einsum_v0(const uint8_t* __restrict__ x, const uint8_t* __restrict__ sfx, const uint8_t* __restrict__ y,
            const uint8_t* __restrict__ sfy, __nv_bfloat16* __restrict__ z, int B, int H, int D, int R) {
  const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
  const int h = blockIdx.z, b0 = blockIdx.y * 16, d0 = blockIdx.x * 8;
  const int row0 = b0 + g, row1 = b0 + g + 8, col = d0 + g;
  const bool has0 = row0 < B, has1 = row1 < B;
  const int nkb = R / 128, ndb = D / 128;
  const uint8_t* yh = y + (size_t)h * D * R;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int kb = 0; kb < nkb; ++kb) {
    float part[4] = {0.f, 0.f, 0.f, 0.f};
    for (int s = 0; s < 4; ++s) {
      const int k0 = kb * 128 + s * 32;
      uint32_t af[4], bf[2];
      const uint8_t* xr0 = x + ((size_t)row0 * H + h) * R;
      const uint8_t* xr1 = x + ((size_t)row1 * H + h) * R;
      af[0] = has0 ? *reinterpret_cast<const uint32_t*>(xr0 + k0 + 4 * t) : 0u;
      af[1] = has1 ? *reinterpret_cast<const uint32_t*>(xr1 + k0 + 4 * t) : 0u;
      af[2] = has0 ? *reinterpret_cast<const uint32_t*>(xr0 + k0 + 16 + 4 * t) : 0u;
      af[3] = has1 ? *reinterpret_cast<const uint32_t*>(xr1 + k0 + 16 + 4 * t) : 0u;
      const uint8_t* yrow = yh + (size_t)col * R;
      bf[0] = *reinterpret_cast<const uint32_t*>(yrow + k0 + 4 * t);
      bf[1] = *reinterpret_cast<const uint32_t*>(yrow + k0 + 16 + 4 * t);
      mma_e4m3_e4m3(part, af, bf);
    }
    const float sx0 = has0 ? ue8m0_to_float(sfx[((size_t)row0 * H + h) * nkb + kb]) : 0.f;
    const float sx1 = has1 ? ue8m0_to_float(sfx[((size_t)row1 * H + h) * nkb + kb]) : 0.f;
    // C columns 2t, 2t+1 of this d tile share one 128-wide d block
    const float sy = ue8m0_to_float(sfy[((size_t)h * ndb + (d0 / 128)) * nkb + kb]);
    acc[0] += part[0] * (sx0 * sy);
    acc[1] += part[1] * (sx0 * sy);
    acc[2] += part[2] * (sx1 * sy);
    acc[3] += part[3] * (sx1 * sy);
  }
  if (has0) {
    z[((size_t)row0 * H + h) * D + d0 + 2 * t] = __float2bfloat16(acc[0]);
    z[((size_t)row0 * H + h) * D + d0 + 2 * t + 1] = __float2bfloat16(acc[1]);
  }
  if (has1) {
    z[((size_t)row1 * H + h) * D + d0 + 2 * t] = __float2bfloat16(acc[2]);
    z[((size_t)row1 * H + h) * D + d0 + 2 * t + 1] = __float2bfloat16(acc[3]);
  }
}

// v1: one block (8 warps) per (h, 128-column d tile, 128-row b chunk). Per 128-R block the y tile [128 d][128 r] and the
// x tile [128 b][128 r] are staged in shared memory (row stride 144 bytes: 36 words, so the eight g-rows of a fragment
// read land in distinct banks), double-buffered with cp.async. Warp w owns columns 16w..16w+15 (two n8 tiles) for every
// 16-row b tile of the chunk; accumulators are per (b tile, n tile). The per-element accumulation order (kb outer, four
// k32 steps inner, then the scale fold) is v0's, so the result is bit-identical to v0.
constexpr int V1_STRIDE = 144;
constexpr int V1_TILE_BYTES = 128 * V1_STRIDE;  // 18432

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem, int src_bytes) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__device__ __forceinline__ void v1_stage(uint8_t* ys, uint8_t* xs, const uint8_t* __restrict__ x, const uint8_t* __restrict__ y,
                                         int h, int d0, int bbase, int nb, int kb, int H, int D, int R) {
  // 256 threads; each copies 64 bytes (4 x 16) of one y row and 64 bytes of one x row.
  const int tid = threadIdx.x, row = tid >> 1, half = (tid & 1) * 64;
  const uint8_t* ysrc = y + ((size_t)h * D + d0 + row) * R + (size_t)kb * 128 + half;
  uint8_t* ydst = ys + row * V1_STRIDE + half;
  #pragma unroll
  for (int i = 0; i < 4; ++i) cp_async_16(ydst + 16 * i, ysrc + 16 * i, 16);
  const bool has = row < nb;
  const int brow = has ? bbase + row : bbase;  // a valid address; src_bytes 0 zero-fills
  const uint8_t* xsrc = x + ((size_t)brow * H + h) * R + (size_t)kb * 128 + half;
  uint8_t* xdst = xs + row * V1_STRIDE + half;
  #pragma unroll
  for (int i = 0; i < 4; ++i) cp_async_16(xdst + 16 * i, xsrc + 16 * i, has ? 16 : 0);
}

__global__ void __launch_bounds__(256)
k_einsum_v1(const uint8_t* __restrict__ x, const uint8_t* __restrict__ sfx, const uint8_t* __restrict__ y,
            const uint8_t* __restrict__ sfy, __nv_bfloat16* __restrict__ z, int B, int H, int D, int R) {
  extern __shared__ __align__(16) uint8_t smem[];
  uint8_t* ys[2] = {smem, smem + V1_TILE_BYTES};
  uint8_t* xs[2] = {smem + 2 * V1_TILE_BYTES, smem + 3 * V1_TILE_BYTES};
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int h = blockIdx.z, d0 = blockIdx.x * 128, bbase = blockIdx.y * 128;
  const int nb = min(128, B - bbase), nmt = (nb + 15) >> 4;
  const int nkb = R / 128, ndb = D / 128;
  float acc[8][2][4];
  #pragma unroll
  for (int m = 0; m < 8; ++m)
    #pragma unroll
    for (int j = 0; j < 2; ++j)
      #pragma unroll
      for (int i = 0; i < 4; ++i) acc[m][j][i] = 0.f;

  v1_stage(ys[0], xs[0], x, y, h, d0, bbase, nb, 0, H, D, R);
  cp_async_commit();
  for (int kb = 0; kb < nkb; ++kb) {
    const int buf = kb & 1;
    if (kb + 1 < nkb) {
      v1_stage(ys[buf ^ 1], xs[buf ^ 1], x, y, h, d0, bbase, nb, kb + 1, H, D, R);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
    const uint8_t* yt = ys[buf];
    const uint8_t* xt = xs[buf];
    const float sy = ue8m0_to_float(sfy[((size_t)h * ndb + (d0 >> 7)) * nkb + kb]);
    for (int m = 0; m < nmt; ++m) {
      float part[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
      const uint8_t* xr0 = xt + (m * 16 + g) * V1_STRIDE;
      const uint8_t* xr1 = xt + (m * 16 + g + 8) * V1_STRIDE;
      #pragma unroll
      for (int s = 0; s < 4; ++s) {
        const int k0 = s * 32;
        uint32_t af[4];
        af[0] = *reinterpret_cast<const uint32_t*>(xr0 + k0 + 4 * t);
        af[1] = *reinterpret_cast<const uint32_t*>(xr1 + k0 + 4 * t);
        af[2] = *reinterpret_cast<const uint32_t*>(xr0 + k0 + 16 + 4 * t);
        af[3] = *reinterpret_cast<const uint32_t*>(xr1 + k0 + 16 + 4 * t);
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
          const uint8_t* yrow = yt + (w * 16 + j * 8 + g) * V1_STRIDE;
          uint32_t bf[2];
          bf[0] = *reinterpret_cast<const uint32_t*>(yrow + k0 + 4 * t);
          bf[1] = *reinterpret_cast<const uint32_t*>(yrow + k0 + 16 + 4 * t);
          mma_e4m3_e4m3(part[j], af, bf);
        }
      }
      const int row0 = bbase + m * 16 + g, row1 = row0 + 8;
      const float sx0 = row0 < B ? ue8m0_to_float(sfx[((size_t)row0 * H + h) * nkb + kb]) : 0.f;
      const float sx1 = row1 < B ? ue8m0_to_float(sfx[((size_t)row1 * H + h) * nkb + kb]) : 0.f;
      #pragma unroll
      for (int j = 0; j < 2; ++j) {
        acc[m][j][0] += part[j][0] * (sx0 * sy);
        acc[m][j][1] += part[j][1] * (sx0 * sy);
        acc[m][j][2] += part[j][2] * (sx1 * sy);
        acc[m][j][3] += part[j][3] * (sx1 * sy);
      }
    }
    __syncthreads();
  }
  for (int m = 0; m < nmt; ++m) {
    const int row0 = bbase + m * 16 + g, row1 = row0 + 8;
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int c = d0 + w * 16 + j * 8 + 2 * t;
      if (row0 < B) {
        z[((size_t)row0 * H + h) * D + c] = __float2bfloat16(acc[m][j][0]);
        z[((size_t)row0 * H + h) * D + c + 1] = __float2bfloat16(acc[m][j][1]);
      }
      if (row1 < B) {
        z[((size_t)row1 * H + h) * D + c] = __float2bfloat16(acc[m][j][2]);
        z[((size_t)row1 * H + h) * D + c + 1] = __float2bfloat16(acc[m][j][3]);
      }
    }
  }
}

static void check_args(torch::Tensor x, torch::Tensor sfx, torch::Tensor y, torch::Tensor sfy, torch::Tensor z) {
  const int B = (int)x.size(0), H = (int)x.size(1), R = (int)x.size(2), D = (int)y.size(1);
  TORCH_CHECK(x.scalar_type() == torch::kFloat8_e4m3fn && x.is_contiguous(), "x: e4m3 [B, H, R]");
  TORCH_CHECK(y.scalar_type() == torch::kFloat8_e4m3fn && y.is_contiguous() && y.size(0) == H && y.size(2) == R, "y: e4m3 [H, D, R]");
  TORCH_CHECK(R % 128 == 0 && D % 128 == 0, "R and D multiples of 128 (one UE8M0 block)");
  TORCH_CHECK(sfx.scalar_type() == torch::kUInt8 && sfx.is_contiguous() && sfx.numel() == (int64_t)B * H * (R / 128), "sfx: uint8 UE8M0 [B, H, R/128]");
  TORCH_CHECK(sfy.scalar_type() == torch::kUInt8 && sfy.is_contiguous() && sfy.numel() == (int64_t)H * (D / 128) * (R / 128), "sfy: uint8 UE8M0 [H, D/128, R/128]");
  TORCH_CHECK(z.scalar_type() == torch::kBFloat16 && z.is_contiguous() && z.size(0) == B && z.size(1) == H && z.size(2) == D, "z: bf16 [B, H, D]");
}

void fp8_einsum_bhr_hdr_bhd_sm120_v0(torch::Tensor x, torch::Tensor sfx, torch::Tensor y, torch::Tensor sfy, torch::Tensor z) {
  check_args(x, sfx, y, sfy, z);
  const int B = (int)x.size(0), H = (int)x.size(1), R = (int)x.size(2), D = (int)y.size(1);
  auto st = at::cuda::getCurrentCUDAStream();
  const dim3 grid(D / 8, (B + 15) / 16, H);
  k_einsum_v0<<<grid, 32, 0, st>>>(static_cast<const uint8_t*>(x.data_ptr()), sfx.data_ptr<uint8_t>(), static_cast<const uint8_t*>(y.data_ptr()),
                                   sfy.data_ptr<uint8_t>(), reinterpret_cast<__nv_bfloat16*>(z.data_ptr()), B, H, D, R);
}

void fp8_einsum_bhr_hdr_bhd_sm120_v1(torch::Tensor x, torch::Tensor sfx, torch::Tensor y, torch::Tensor sfy, torch::Tensor z) {
  check_args(x, sfx, y, sfy, z);
  const int B = (int)x.size(0), H = (int)x.size(1), R = (int)x.size(2), D = (int)y.size(1);
  auto st = at::cuda::getCurrentCUDAStream();
  static bool attr_set = false;
  const int smem = 4 * V1_TILE_BYTES;  // 73728
  if (!attr_set) { cudaFuncSetAttribute(k_einsum_v1, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr_set = true; }
  const dim3 grid(D / 128, (B + 127) / 128, H);
  k_einsum_v1<<<grid, 256, smem, st>>>(static_cast<const uint8_t*>(x.data_ptr()), sfx.data_ptr<uint8_t>(), static_cast<const uint8_t*>(y.data_ptr()),
                                       sfy.data_ptr<uint8_t>(), reinterpret_cast<__nv_bfloat16*>(z.data_ptr()), B, H, D, R);
}
"""

KERNELS = ("v0", "v1")


def build(verbose: bool = False):
    return load_inline(name="sm120fp4_fp8_einsum_v1a", cpp_sources=CPP, cuda_sources=CUDA,
                       functions=["fp8_einsum_bhr_hdr_bhd_sm120_v0", "fp8_einsum_bhr_hdr_bhd_sm120_v1"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a"], verbose=verbose)


def fn(mod, kernel: str):
    return getattr(mod, f"fp8_einsum_bhr_hdr_bhd_sm120_{kernel}")


def per_block_cast_to_fp8(x: torch.Tensor, use_ue8m0: bool = True, gran: int = GRAN):
    """DeepGEMM's per_block_cast_to_fp8 (deep_gemm/utils/math.py, read 2026-10-03): [m, n] -> e4m3 [m, n] and scales
    [ceil(m/128), ceil(n/128)] = amax / 448 per 128 x 128 block, ceil_to_ue8m0 when asked. m and n here are multiples of 128."""
    m, n = x.shape
    assert m % gran == 0 and n % gran == 0
    xv = x.view(m // gran, gran, n // gran, gran).permute(0, 2, 1, 3)          # [mb, nb, 128, 128]
    amax = xv.abs().float().amax(dim=(2, 3)).clamp_min(1e-4)
    sf = amax / E4M3_MAX
    if use_ue8m0:
        sf = ref.ceil_to_ue8m0(sf)
    x8 = (xv.float() / sf[:, :, None, None]).to(torch.float8_e4m3fn).permute(0, 2, 1, 3).reshape(m, n).contiguous()
    return x8, sf


def to_u8(sf: torch.Tensor) -> torch.Tensor:
    """UE8M0 byte = biased exponent of an exact power of two."""
    return (torch.round(torch.log2(sf.float())) + 127).clamp(0, 255).to(torch.uint8)


def quantize_inputs(x: torch.Tensor, y: torch.Tensor):
    """x [B, H, R] bf16/fp32, y [H, D, R] -> (x8, sfx_u8, y8, sfy_u8, sfx_f, sfy_f)."""
    B, H, R = x.shape
    D = y.shape[1]
    x8, sfx = ref.per_token_cast_to_fp8(x.reshape(B * H, R).float(), use_ue8m0=True, gran_k=GRAN)   # sfx [B*H, R/128] fp32
    y8 = torch.empty(H, D, R, dtype=torch.float8_e4m3fn, device=y.device)
    sfy = torch.empty(H, D // GRAN, R // GRAN, dtype=torch.float32, device=y.device)
    for h in range(H):
        y8[h], sfy[h] = per_block_cast_to_fp8(y[h].float(), use_ue8m0=True)
    return (x8.reshape(B, H, R).contiguous(), to_u8(sfx).reshape(B, H, R // GRAN).contiguous(), y8.contiguous(), to_u8(sfy).contiguous(),
            sfx.reshape(B, H, R // GRAN), sfy)


def reference(x8, sfx_f, y8, sfy_f) -> torch.Tensor:
    """torch.einsum on the dequantised operands in fp32 (DeepGEMM's test reference, with its scales applied)."""
    B, H, R = x8.shape
    D = y8.shape[1]
    xf = x8.float() * sfx_f.repeat_interleave(GRAN, dim=2)
    yf = y8.float() * sfy_f.repeat_interleave(GRAN, dim=1).repeat_interleave(GRAN, dim=2)
    return torch.einsum("bhr,hdr->bhd", xf, yf)


def make_case(B, H, D, R, seed, dev):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(B, H, R, generator=g) * torch.pow(2.0, torch.randint(-4, 5, (B, H, 1), generator=g).float())).to(dev)
    y = (torch.randn(H, D, R, generator=g) * torch.pow(2.0, torch.randint(-4, 5, (H, D // GRAN, 1), generator=g).float()).repeat_interleave(GRAN, dim=1)).to(dev)
    return x, y


def run_case(mod, B, H, D, R, seed, dev, kernel: str = "v0"):
    x, y = make_case(B, H, D, R, seed, dev)
    x8, sfx_u8, y8, sfy_u8, sfx_f, sfy_f = quantize_inputs(x, y)
    z = torch.empty(B, H, D, device=dev, dtype=torch.bfloat16)
    fn(mod, kernel)(x8, sfx_u8, y8, sfy_u8, z)
    torch.cuda.synchronize()
    exact = reference(x8, sfx_f, y8, sfy_f)
    diff = (z.float() - exact).abs()
    scale = exact.abs().max().clamp_min(1e-30)
    res = {"kernel": kernel, "B": B, "H": H, "D": D, "R": R, "max_abs_err": float(diff.max()), "ref_abs_max": float(scale), "rel_max_err": float(diff.max() / scale),
           "rel_fro_err": float((z.float() - exact).norm() / exact.norm()), "bf16_half_ulp_at_max": float(scale) * 2 ** -9}
    # bf16 output rounding: within two half-ulps at the output's maximum, and a tiny Frobenius error
    res["pass"] = res["max_abs_err"] <= 2 * res["bf16_half_ulp_at_max"] + 1e-6 and res["rel_fro_err"] < 4e-3
    if kernel != "v0":
        z0 = torch.empty_like(z)
        fn(mod, "v0")(x8, sfx_u8, y8, sfy_u8, z0)
        torch.cuda.synchronize()
        res["elements_differing_from_v0"] = int((z0.view(torch.int16) != z.view(torch.int16)).sum())
        res["pass"] = res["pass"] and res["elements_differing_from_v0"] == 0
    return res, (x8, sfx_u8, y8, sfy_u8, z)


def selftest(mod, dev) -> int:
    ok = True
    for kernel in KERNELS:
        print(f"fp8 einsum 'bhr,hdr->bhd' {kernel} against torch.einsum on the dequantised operands" + ("" if kernel == "v0" else " and bit for bit against v0"))
        for (B, H, D, R, seed) in ((1, 4, 128, 256, 1), (5, 8, 256, 512, 2), (32, 8, 1024, 4096, 3), (16, 4, 128, 128, 4), (33, 8, 512, 2048, 5), (8, 16, 1024, 4096, 6), (200, 4, 256, 1024, 7)):
            r, _ = run_case(mod, B, H, D, R, seed, dev, kernel)
            ok &= r["pass"]
            extra = "" if kernel == "v0" else f", {r['elements_differing_from_v0']} elements differ from v0"
            print(f"  B={B} H={H} D={D} R={R}: max|err| {r['max_abs_err']:.4g} (half-ulp at max {r['bf16_half_ulp_at_max']:.4g}), rel Frobenius {r['rel_fro_err']:.3e}{extra} -> {'ok' if r['pass'] else 'FAIL'}", flush=True)
    print("fp8_einsum_sm120 selftest:", "ok" if ok else "FAIL")
    return 0 if ok else 1


def bench(mod, dev, out: Path | None, kernels=KERNELS) -> int:
    if out is not None and out.exists():
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 2
    props = torch.cuda.get_device_properties(0)
    flush_buf = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    rows = []
    for (B, H, D, R) in ((8, 8, 1024, 4096), (32, 8, 1024, 4096), (128, 8, 1024, 4096)):
        for kernel in kernels:
            r, args = run_case(mod, B, H, D, R, 100 + B, dev, kernel)
            x8, sfx_u8, y8, sfy_u8, z = args
            f = fn(mod, kernel)
            times = []
            for _ in range(10):
                flush_buf.fill_(1)
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record(); f(x8, sfx_u8, y8, sfy_u8, z); e1.record()
                torch.cuda.synchronize()
                times.append(e0.elapsed_time(e1) * 1000)
            med = statistics.median(times)
            nbytes = x8.numel() + y8.numel() + sfx_u8.numel() + sfy_u8.numel() + z.numel() * 2
            r.update({"us_median": med, "us_min": min(times), "bytes": nbytes, "GBps": nbytes / med / 1e3, "TFLOPs": 2.0 * B * H * D * R / med / 1e6,
                      "floor_us_at_1792": nbytes / 1792.0 / 1e3})
            rows.append(r)
            print(f"{kernel} B={B} H={H} D={D} R={R}: {med:.1f} us ({r['GBps']:.0f} GB/s, {r['TFLOPs']:.2f} TFLOP/s; floor {r['floor_us_at_1792']:.1f} us); rel Frobenius {r['rel_fro_err']:.1e}", flush=True)
    report = {"kernel": "fp8_einsum_bhr_hdr_bhd_sm120 v0 (one warp per (h, 16 b rows, 8 d columns)) and v1 (one 8-warp block per (h, 128 d columns, 128 b rows); y and x tiles staged in shared memory per 128-R block, double-buffered cp.async)",
              "device": props.name,
              "note": "cold L2 (256 MB fill before each launch); median of 10; bytes = x e4m3 + y e4m3 + scales + bf16 z; floor at 1792 GB/s; y (H x D x R bytes) dominates the bytes; timed region holds the launch only (no host sync inside)", "rows": rows}
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
    dev = torch.device("cuda")
    mod = build()
    rc = 0
    if a.selftest:
        rc = selftest(mod, dev)
    if a.bench:
        rc = rc or bench(mod, dev, a.out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
