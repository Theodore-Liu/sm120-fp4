"""Stage 2, FC2 phase: W4A16 on CUDA cores, one warp per output column, touched experts in a fixed order.

Each warp owns one output (hidden) column n. It walks the experts the batch touches in ascending id order, reads that
expert's FC2 row n once (16-byte streaming loads, FP4 decoded with SM120's cvt.rn.f16x2.e2m1x2 and the E4M3 block
scales), and for each (token, expert) pair routed to the expert dots the row with the pair's FC1 activation, reduces
across the warp with a fixed butterfly, and adds routing weight x alpha x that value into the token's accumulator in
shared memory. Every expert row is read by exactly one warp, and every output element has one writer and one
summation order, so the result is deterministic without atomics.

Layout: codes [E, H, I/2] (even element in the low nibble), block scales [E * H, I/16] row-major E4M3 bytes, per-expert
fp32 alpha; activations from scripts/fc1_w4a16.py, [pairs, I] bf16 with pair p = token * top_k + j.

    PYTHONPATH=. python scripts/fc2_w4a16.py      (checks FC1 + FC2 against the fp32 MoE reference and times both)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_here = Path(__file__).resolve().parent
for _name in ("bench_moe_baseline", "micro_floor", "fc1_w4a16"):
    _spec = importlib.util.spec_from_file_location(_name, _here / f"{_name}.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)
bench = sys.modules["bench_moe_baseline"]
floor = sys.modules["micro_floor"]
fc1 = sys.modules["fc1_w4a16"]

CPP = r"""
#include <torch/extension.h>
void fc2_w4a16(torch::Tensor q2, torch::Tensor s2, torch::Tensor act, torch::Tensor experts, torch::Tensor offsets,
               torch::Tensor pairs, torch::Tensor weights, torch::Tensor alpha, torch::Tensor out, int64_t top_k);
void fc2_w4a16_v1(torch::Tensor q2, torch::Tensor s2, torch::Tensor act, torch::Tensor experts, torch::Tensor offsets,
                  torch::Tensor pairs, torch::Tensor weights, torch::Tensor alpha, torch::Tensor out, int64_t top_k,
                  int64_t cols);
void fc2_w4a16_v0(torch::Tensor q2, torch::Tensor s2, torch::Tensor act, torch::Tensor experts, torch::Tensor offsets,
                  torch::Tensor pairs, torch::Tensor weights, torch::Tensor alpha, torch::Tensor out, int64_t top_k);
void fc2_set_pdl(bool on);
"""

CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <ATen/cuda/CUDAContext.h>

constexpr int WARPS = 4;
constexpr int MAXM = 16;        // decode batch: at most 16 tokens
constexpr int PREFETCH = 4;     // expert rows loaded ahead per lane

__device__ __forceinline__ float e4m3(unsigned char b) {
  __nv_fp8_e4m3 v;
  v.__x = b;
  return float(v);
}

__device__ __forceinline__ __half2 fp4x2_to_half2(unsigned int byte) {
  unsigned int out;
  unsigned short in = (unsigned short)byte;
  asm("{ .reg .b8 lo, hi;\n"
      " mov.b16 {lo, hi}, %1;\n"
      " cvt.rn.f16x2.e2m1x2 %0, lo; }\n" : "=r"(out) : "h"(in));
  return *reinterpret_cast<__half2*>(&out);
}

__device__ __forceinline__ void decode32(uint4 q, unsigned short s, float* w) {
  const float s0 = e4m3(s & 0xff), s1 = e4m3(s >> 8);
  const unsigned words[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
  for (int wd = 0; wd < 4; ++wd) {
#pragma unroll
    for (int by = 0; by < 4; ++by) {
      const int idx = wd * 8 + by * 2;
      const float2 f = __half22float2(fp4x2_to_half2((words[wd] >> (8 * by)) & 0xff));
      const float sc = idx < 16 ? s0 : s1;
      w[idx] = f.x * sc;
      w[idx + 1] = f.y * sc;
    }
  }
}

// I must be a multiple of 32 and at most 32 * 32 = 1024 (one 32-value chunk per lane).
__global__ void __launch_bounds__(WARPS * 32)
k_fc2(const unsigned char* __restrict__ q2, const unsigned char* __restrict__ s2, const __nv_bfloat16* __restrict__ act,
      const int* __restrict__ experts, const int* __restrict__ offsets, const int* __restrict__ pairs,
      const float* __restrict__ weights, const float* __restrict__ alpha, __nv_bfloat16* __restrict__ out,
      int U, int M, int H, int I, int top_k) {
  __shared__ float acc_s[WARPS][MAXM];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = blockIdx.x * WARPS + warp;
  const int chunks = I / 32;
  const bool active = lane < chunks;
  if (lane < MAXM) acc_s[warp][lane] = 0.f;
  __syncwarp();

  for (int u0 = 0; u0 < U; u0 += PREFETCH) {
    uint4 wq[PREFETCH];
    unsigned short ws[PREFETCH];
#pragma unroll
    for (int d = 0; d < PREFETCH; ++d) {
      if (u0 + d < U && active && experts[u0 + d] >= 0) {
        const long long row = (long long)experts[u0 + d] * H + n;
        wq[d] = __ldcs(reinterpret_cast<const uint4*>(q2 + row * (I / 2)) + lane);
        ws[d] = reinterpret_cast<const unsigned short*>(s2 + row * (I / 16))[lane];
      } else {
        wq[d] = make_uint4(0, 0, 0, 0);
        ws[d] = 0;
      }
    }
#pragma unroll
    for (int d = 0; d < PREFETCH; ++d) {
      const int u = u0 + d;
      if (u < U && experts[u] >= 0) {
        float w[32];
        decode32(wq[d], ws[d], w);
        const float a = alpha[experts[u]];
        for (int q = offsets[u]; q < offsets[u + 1]; ++q) {      // the expert's pairs, in a fixed order
          const int p = pairs[q];
          float sum = 0.f;
          if (active) {
            const uint4* av = reinterpret_cast<const uint4*>(act + (long long)p * I + lane * 32);
#pragma unroll
            for (int v4 = 0; v4 < 4; ++v4) {
              const uint4 v = __ldg(av + v4);
              const __nv_bfloat162* bb = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
              for (int h2 = 0; h2 < 4; ++h2) {
                const float2 f = __bfloat1622float2(bb[h2]);
                sum += w[v4 * 8 + h2 * 2] * f.x + w[v4 * 8 + h2 * 2 + 1] * f.y;
              }
            }
          }
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
          if (lane == 0) acc_s[warp][p / top_k] += weights[p] * a * sum;
        }
      }
    }
  }
  __syncwarp();
  if (lane < M) out[(long long)lane * H + n] = __float2bfloat16(acc_s[warp][lane]);
}

// v1: one block per tile of COLS output columns; its WARPS_B warps split the touched experts round-robin, each keeps
// its partial sums in shared memory, and the block adds the warps' partials in warp order at the end. The order of every
// sum is fixed by the routing, so the result is still deterministic; more warps and more rows are in flight per SM.
constexpr int WARPS_B = 8;

template <int COLS>
__global__ void __launch_bounds__(WARPS_B * 32)
k_fc2_v1(const unsigned char* __restrict__ q2, const unsigned char* __restrict__ s2, const __nv_bfloat16* __restrict__ act,
         const int* __restrict__ experts, const int* __restrict__ offsets, const int* __restrict__ pairs,
         const float* __restrict__ weights, const float* __restrict__ alpha, __nv_bfloat16* __restrict__ out,
         int U, int M, int H, int I, int top_k) {
  __shared__ float part[WARPS_B][MAXM][COLS];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n0 = blockIdx.x * COLS;
  const int chunks = I / 32;
  const bool active = lane < chunks;
  for (int z = threadIdx.x; z < WARPS_B * MAXM * COLS; z += blockDim.x) (&part[0][0][0])[z] = 0.f;
  __syncthreads();

  bool waited = false;
  for (int u0 = warp; u0 < U; u0 += 2 * WARPS_B) {          // two experts per warp in flight
    uint4 wq[2][COLS];
    unsigned short ws[2][COLS];
#pragma unroll
    for (int d = 0; d < 2; ++d) {
      const int u = u0 + d * WARPS_B;
#pragma unroll
      for (int c = 0; c < COLS; ++c) {
        if (u < U && active && experts[u] >= 0) {
          const long long row = (long long)experts[u] * H + n0 + c;
          wq[d][c] = __ldcs(reinterpret_cast<const uint4*>(q2 + row * (I / 2)) + lane);
          ws[d][c] = reinterpret_cast<const unsigned short*>(s2 + row * (I / 16))[lane];
        } else {
          wq[d][c] = make_uint4(0, 0, 0, 0);
          ws[d][c] = 0;
        }
      }
    }
    if (!waited) {          // PDL: the first weight loads are in flight; now wait for FC1's activations
      cudaGridDependencySynchronize();
      waited = true;
    }
#pragma unroll
    for (int d = 0; d < 2; ++d) {
      const int u = u0 + d * WARPS_B;
      if (u < U && experts[u] >= 0) {     // -1 marks a padding slot from the GPU router
        const float a = alpha[experts[u]];
#pragma unroll
        for (int c = 0; c < COLS; ++c) {
          float w[32];
          decode32(wq[d][c], ws[d][c], w);
          for (int q = offsets[u]; q < offsets[u + 1]; ++q) {
            const int p = pairs[q];
            float sum = 0.f;
            if (active) {
              const uint4* av = reinterpret_cast<const uint4*>(act + (long long)p * I + lane * 32);
#pragma unroll
              for (int v4 = 0; v4 < 4; ++v4) {
                const uint4 v = __ldg(av + v4);
                const __nv_bfloat162* bb = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
                for (int h2 = 0; h2 < 4; ++h2) {
                  const float2 f = __bfloat1622float2(bb[h2]);
                  sum += w[v4 * 8 + h2 * 2] * f.x + w[v4 * 8 + h2 * 2 + 1] * f.y;
                }
              }
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            if (lane == 0) part[warp][p / top_k][c] += weights[p] * a * sum;
          }
        }
      }
    }
  }
  __syncthreads();
  for (int z = threadIdx.x; z < M * COLS; z += blockDim.x) {
    const int t = z / COLS, c = z % COLS;
    float v = 0.f;
#pragma unroll
    for (int wq_ = 0; wq_ < WARPS_B; ++wq_) v += part[wq_][t][c];      // fixed warp order
    out[(long long)t * H + n0 + c] = __float2bfloat16(v);
  }
}

static bool g_pdl2 = false;   // launch with programmatic stream serialization (set by fc2_set_pdl)
void fc2_set_pdl(bool on) { g_pdl2 = on; }

void fc2_w4a16_v1(torch::Tensor q2, torch::Tensor s2, torch::Tensor act, torch::Tensor experts, torch::Tensor offsets,
                  torch::Tensor pairs, torch::Tensor weights, torch::Tensor alpha, torch::Tensor out, int64_t top_k,
                  int64_t cols) {
  const int M = (int)out.size(0), H = (int)out.size(1), I = (int)act.size(1), U = (int)experts.numel();
  TORCH_CHECK(M <= MAXM && I % 32 == 0 && I <= 1024 && H % 4 == 0, "shape");
  auto st = at::cuda::getCurrentCUDAStream();
#define FC2_ARGS q2.data_ptr<uint8_t>(), s2.data_ptr<uint8_t>(), reinterpret_cast<const __nv_bfloat16*>(act.data_ptr()), \
    experts.data_ptr<int>(), offsets.data_ptr<int>(), pairs.data_ptr<int>(), weights.data_ptr<float>(),                 \
    alpha.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), U, M, H, I, (int)top_k
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(H / (int)cols);
  cfg.blockDim = dim3(WARPS_B * 32);
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = g_pdl2 ? 1 : 0;
  cudaError_t err;
  if (cols == 1) err = cudaLaunchKernelEx(&cfg, k_fc2_v1<1>, FC2_ARGS);
  else if (cols == 2) err = cudaLaunchKernelEx(&cfg, k_fc2_v1<2>, FC2_ARGS);
  else err = cudaLaunchKernelEx(&cfg, k_fc2_v1<4>, FC2_ARGS);
  TORCH_CHECK(err == cudaSuccess, "launch");
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "launch");
}

// the default entry point: v1 with the column count the sweep chose on the RTX 5090 (2 at one token, 1 otherwise)
void fc2_w4a16(torch::Tensor q2, torch::Tensor s2, torch::Tensor act, torch::Tensor experts, torch::Tensor offsets,
               torch::Tensor pairs, torch::Tensor weights, torch::Tensor alpha, torch::Tensor out, int64_t top_k) {
  fc2_w4a16_v1(q2, s2, act, experts, offsets, pairs, weights, alpha, out, top_k, out.size(0) <= 1 ? 2 : 1);
}

// v0, kept for comparison: one warp per output column walking every touched expert
void fc2_w4a16_v0(torch::Tensor q2, torch::Tensor s2, torch::Tensor act, torch::Tensor experts, torch::Tensor offsets,
                  torch::Tensor pairs, torch::Tensor weights, torch::Tensor alpha, torch::Tensor out, int64_t top_k) {
  const int M = (int)out.size(0), H = (int)out.size(1), I = (int)act.size(1), U = (int)experts.numel();
  TORCH_CHECK(M <= MAXM && I % 32 == 0 && I <= 1024 && H % WARPS == 0, "shape");
  auto st = at::cuda::getCurrentCUDAStream();
  k_fc2<<<H / WARPS, WARPS * 32, 0, st>>>(
      q2.data_ptr<uint8_t>(), s2.data_ptr<uint8_t>(), reinterpret_cast<const __nv_bfloat16*>(act.data_ptr()),
      experts.data_ptr<int>(), offsets.data_ptr<int>(), pairs.data_ptr<int>(), weights.data_ptr<float>(),
      alpha.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), U, M, H, I, (int)top_k);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "launch");
}
"""


def build():
    return load_inline(name="sm120fp4_fc2_w4a16", cpp_sources=CPP, cuda_sources=CUDA, functions=["fc2_w4a16", "fc2_w4a16_v1", "fc2_w4a16_v0", "fc2_set_pdl"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a"], verbose=False)


def main() -> int:
    m1 = fc1.build()
    m2 = build()
    dev = torch.device("cuda")
    e, k, h, i = 128, 8, 2048, 768
    w = bench.build(e, h, i, dev)
    q1, s1, q2, s2 = w["q1"].contiguous(), w["s1"].contiguous(), w["q2"].contiguous(), w["s2"].contiguous()
    alpha = torch.ones(e, device=dev)
    print("NOTE: routing preparation (fc1.route: argsort/unique on the GPU, with a host sync) is outside every timing here")
    print("tokens | normwise vs fp32 MoE (FC2 v0) | bit-identical x50 | FC1 us | FC2 v0 us | FC1+FC2 v0 us | best existing us")
    best_existing = {1: 37.6, 2: 49.9, 4: 78.8, 8: 113.9, 16: 166.5}
    for m in (1, 2, 4, 8, 16):
        g = torch.Generator().manual_seed(1000 + m)
        x = torch.randn(m, h, generator=g).to(device=dev, dtype=torch.bfloat16)
        wts, ids = torch.topk(F.softmax(torch.randn(m, e, generator=g), dim=-1), k, dim=-1)
        wts = (wts / wts.sum(-1, keepdim=True)).float().to(dev).contiguous()
        ids = ids.to(torch.int32).to(dev)
        experts, offsets, pairs = fc1.route(ids)
        act = torch.empty(m * k, i, device=dev, dtype=torch.bfloat16)
        out = torch.empty(m, h, device=dev, dtype=torch.bfloat16)
        wflat = wts.view(-1)
        run1 = lambda: m1.fc1_w4a16(q1, s1, x, experts, offsets, pairs, alpha, act, i, k)  # noqa: E731
        run2 = lambda: m2.fc2_w4a16_v0(q2, s2, act, experts, offsets, pairs, wflat, alpha, out, k)  # noqa: E731

        def both():
            run1()
            run2()
        both()
        torch.cuda.synchronize()
        ref = bench.reference(x, w, ids, wts, i, act_quant=False)
        rel = float((out.float() - ref).norm() / ref.norm())
        first = out.clone()
        same = True
        for _ in range(50):
            both()
            same = same and bool(torch.equal(out, first))
        t1, t2, t12 = floor.graph_time(run1), floor.graph_time(run2), floor.graph_time(both)
        v1 = {}
        for cols in (1, 2, 4):
            r2 = lambda: m2.fc2_w4a16_v1(q2, s2, act, experts, offsets, pairs, wflat, alpha, out, k, cols)  # noqa: E731
            run1()
            r2()
            torch.cuda.synchronize()
            rel1 = float((out.float() - ref).norm() / ref.norm())
            f1 = out.clone()
            st1 = True
            for _ in range(50):
                r2()
                st1 = st1 and bool(torch.equal(out, f1))

            def both1():
                run1()
                r2()
            v1[cols] = (round(floor.graph_time(r2), 1), round(floor.graph_time(both1), 1), round(rel1, 5), st1)
        print(f"{m:6d} | {rel:20.5f} | {str(same):17s} | {t1:6.1f} | {t2:6.1f} | {t12:10.1f} | {best_existing[m]:6.1f}"
              f" || v1 by cols (FC2 us, FC1+FC2 us, normwise, stable): {v1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
