"""Stage 2, FC1 phase: W4A16 on CUDA cores, one warp per (touched expert, group of intermediate columns).

At 1 to 16 decode tokens the FC1 arithmetic is small next to reading the weights (about 0.4 G multiply-adds at 16
tokens for the Qwen3-30B-A3B layer, a few microseconds on CUDA cores), so this version skips tensor cores: each warp
streams one up row and the matching gate row of one expert exactly once (16-byte streaming loads), dequantizes the FP4
codes with their E4M3 block scales in registers, dots them with every token routed to that expert, reduces across the
warp in a fixed order and writes SiLU(gate) * up for each (token, expert) pair. Every output element has one writer and
one summation order, so the result is deterministic.

Layout it consumes (this repository's reference layout, not FlashInfer's): codes [E, 2I, H/2] with rows [up ; gate],
even element in the low nibble; block scales [E * 2I, H/16] row-major E4M3 bytes; per-expert fp32 alpha. Output: the
activation for pair p = token * top_k + j at act[p, :] in bf16.

    PYTHONPATH=. python scripts/fc1_w4a16.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_here = Path(__file__).resolve().parent
for _name in ("bench_moe_baseline", "micro_floor"):
    _spec = importlib.util.spec_from_file_location(_name, _here / f"{_name}.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)
bench = sys.modules["bench_moe_baseline"]
floor = sys.modules["micro_floor"]

CPP = r"""
#include <torch/extension.h>
void fc1_w4a16(torch::Tensor q1, torch::Tensor s1, torch::Tensor x, torch::Tensor experts, torch::Tensor offsets,
               torch::Tensor pairs, torch::Tensor alpha, torch::Tensor act, int64_t inter, int64_t top_k);
void fc1_w4a16_cols(torch::Tensor q1, torch::Tensor s1, torch::Tensor x, torch::Tensor experts, torch::Tensor offsets,
                    torch::Tensor pairs, torch::Tensor alpha, torch::Tensor act, int64_t inter, int64_t top_k, int64_t cols);
"""

CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <ATen/cuda/CUDAContext.h>

constexpr int WARPS = 4;
constexpr int MAXT = 16;   // tokens per expert: a decode batch of at most 16 tokens

#include <cuda_fp16.h>

__device__ __forceinline__ float e4m3(unsigned char b) {
  __nv_fp8_e4m3 v;
  v.__x = b;
  return float(v);
}

// one byte = two E2M1 codes -> half2 (low nibble in .x), with SM120's hardware conversion
__device__ __forceinline__ __half2 fp4x2_to_half2(unsigned int byte) {
  unsigned int out;
  unsigned short in = (unsigned short)byte;
  asm("{ .reg .b8 lo, hi;\n"
      " mov.b16 {lo, hi}, %1;\n"
      " cvt.rn.f16x2.e2m1x2 %0, lo; }\n" : "=r"(out) : "h"(in));
  return *reinterpret_cast<__half2*>(&out);
}

// 16 bytes of codes (32 values, element 2j in the low nibble of byte j) and their two block scales -> 32 floats
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

// MAXT: upper bound on tokens per expert (the batch size, rounded up to 1/2/4/8/16), a template parameter so the
// accumulators stay in registers. COLS: intermediate columns per warp; every weight load of the warp is issued before
// any is decoded, so COLS x 2 rows x 2 x 16 bytes are in flight per lane.
template <int MAXT, int COLS>
__global__ void __launch_bounds__(WARPS * 32)
k_fc1(const unsigned char* __restrict__ q1, const unsigned char* __restrict__ s1, const __nv_bfloat16* __restrict__ x,
      const int* __restrict__ experts, const int* __restrict__ offsets, const int* __restrict__ pairs,
      const float* __restrict__ alpha, __nv_bfloat16* __restrict__ act, int H, int I, int top_k) {
  constexpr int ITERS = 2;                 // H / 32 / 32 lanes for H = 2048; checked on the host
  const int tiles = I / (WARPS * COLS);
  const int u = blockIdx.x / tiles;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int c0 = ((blockIdx.x % tiles) * WARPS + warp) * COLS;
  const int e = experts[u];
  const int p0 = offsets[u];
  const int n = min(offsets[u + 1] - p0, MAXT);
  const long long base = (long long)e * 2 * I;

  uint4 wq[COLS][2][ITERS];
  unsigned short ws[COLS][2][ITERS];
#pragma unroll
  for (int c = 0; c < COLS; ++c)
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const long long row = base + c0 + c + r * I;
      const uint4* wr = reinterpret_cast<const uint4*>(q1 + row * (H / 2));
      const unsigned short* sr = reinterpret_cast<const unsigned short*>(s1 + row * (H / 16));
#pragma unroll
      for (int j = 0; j < ITERS; ++j) {
        wq[c][r][j] = __ldcs(wr + lane + 32 * j);
        ws[c][r][j] = sr[lane + 32 * j];
      }
    }

  int tok[MAXT];
#pragma unroll
  for (int t = 0; t < MAXT; ++t) tok[t] = t < n ? pairs[p0 + t] / top_k : 0;
  float acc[COLS][2][MAXT];
#pragma unroll
  for (int c = 0; c < COLS; ++c)
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int t = 0; t < MAXT; ++t) acc[c][r][t] = 0.f;

#pragma unroll
  for (int j = 0; j < ITERS; ++j) {
    const int k0 = (lane + 32 * j) * 32;
#pragma unroll
    for (int t = 0; t < MAXT; ++t) {
      if (t < n) {
        float xf[32];
        const uint4* xv = reinterpret_cast<const uint4*>(x + (long long)tok[t] * H + k0);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const uint4 v = __ldg(xv + q);
          const __nv_bfloat162* bb = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
          for (int h2 = 0; h2 < 4; ++h2) {
            const float2 f = __bfloat1622float2(bb[h2]);
            xf[q * 8 + h2 * 2] = f.x;
            xf[q * 8 + h2 * 2 + 1] = f.y;
          }
        }
#pragma unroll
        for (int c = 0; c < COLS; ++c)
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            float w[32];
            decode32(wq[c][r][j], ws[c][r][j], w);
            float sum = 0.f;
#pragma unroll
            for (int z = 0; z < 32; ++z) sum += w[z] * xf[z];
            acc[c][r][t] += sum;
          }
      }
    }
  }
  const float a = alpha[e];
#pragma unroll
  for (int c = 0; c < COLS; ++c)
#pragma unroll
    for (int t = 0; t < MAXT; ++t) {
      if (t < n) {
        float uu = acc[c][0][t], gg = acc[c][1][t];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {         // fixed butterfly: same order every call
          uu += __shfl_xor_sync(0xffffffffu, uu, o);
          gg += __shfl_xor_sync(0xffffffffu, gg, o);
        }
        if (lane == 0) {
          uu *= a; gg *= a;
          const float silu = gg / (1.f + __expf(-gg));
          act[(long long)pairs[p0 + t] * I + c0 + c] = __float2bfloat16(silu * uu);
        }
      }
    }
}

template <int MAXT, int COLS>
static void launch(const torch::Tensor& q1, const torch::Tensor& s1, const torch::Tensor& x, const torch::Tensor& experts,
                   const torch::Tensor& offsets, const torch::Tensor& pairs, const torch::Tensor& alpha, torch::Tensor& act,
                   int H, int I, int top_k) {
  TORCH_CHECK(I % (WARPS * COLS) == 0, "intermediate size must be a multiple of WARPS * COLS");
  const int U = (int)experts.numel();
  auto st = at::cuda::getCurrentCUDAStream();
  k_fc1<MAXT, COLS><<<U * (I / (WARPS * COLS)), WARPS * 32, 0, st>>>(
      q1.data_ptr<uint8_t>(), s1.data_ptr<uint8_t>(), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      experts.data_ptr<int>(), offsets.data_ptr<int>(), pairs.data_ptr<int>(), alpha.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(act.data_ptr()), H, I, top_k);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "launch");
}

#define FC1_DISPATCH(MT)                                                                                   if (cols == 1) launch<MT, 1>(q1, s1, x, experts, offsets, pairs, alpha, act, H, I, (int)top_k);           else if (cols == 2) launch<MT, 2>(q1, s1, x, experts, offsets, pairs, alpha, act, H, I, (int)top_k);      else launch<MT, 4>(q1, s1, x, experts, offsets, pairs, alpha, act, H, I, (int)top_k);

void fc1_w4a16_cols(torch::Tensor q1, torch::Tensor s1, torch::Tensor x, torch::Tensor experts, torch::Tensor offsets,
                    torch::Tensor pairs, torch::Tensor alpha, torch::Tensor act, int64_t inter, int64_t top_k, int64_t cols) {
  const int H = (int)x.size(1), I = (int)inter, M = (int)x.size(0);
  TORCH_CHECK(H == 2048 && M <= 16, "shape");
  if (M <= 1) { FC1_DISPATCH(1) } else if (M <= 2) { FC1_DISPATCH(2) } else if (M <= 4) { FC1_DISPATCH(4) }
  else if (M <= 8) { FC1_DISPATCH(8) } else { FC1_DISPATCH(16) }
}

void fc1_w4a16(torch::Tensor q1, torch::Tensor s1, torch::Tensor x, torch::Tensor experts, torch::Tensor offsets,
               torch::Tensor pairs, torch::Tensor alpha, torch::Tensor act, int64_t inter, int64_t top_k) {
  const int H = (int)x.size(1), I = (int)inter, M = (int)x.size(0);
  TORCH_CHECK(H == 2048, "this version is specialised to hidden 2048 (2 iterations of 32 lanes x 32 values)");
  TORCH_CHECK(M <= 16, "decode kernel: at most 16 tokens");
  // columns per warp chosen by the sweep in docs/stage2-design.md (RTX 5090): 2 at one token, 1 otherwise
  if (M <= 1) launch<1, 2>(q1, s1, x, experts, offsets, pairs, alpha, act, H, I, (int)top_k);
  else if (M <= 2) launch<2, 1>(q1, s1, x, experts, offsets, pairs, alpha, act, H, I, (int)top_k);
  else if (M <= 4) launch<4, 1>(q1, s1, x, experts, offsets, pairs, alpha, act, H, I, (int)top_k);
  else if (M <= 8) launch<8, 1>(q1, s1, x, experts, offsets, pairs, alpha, act, H, I, (int)top_k);
  else launch<16, 1>(q1, s1, x, experts, offsets, pairs, alpha, act, H, I, (int)top_k);
}
"""


def build(verbose=False):
    return load_inline(name="sm120fp4_fc1_w4a16", cpp_sources=CPP, cuda_sources=CUDA, functions=["fc1_w4a16", "fc1_w4a16_cols"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a", "-Xptxas=-v"],
                       verbose=verbose)


def route(ids):
    """(experts touched, offsets, pair indices grouped by expert): the host-side routing the fused kernel will absorb."""
    flat = ids.flatten().long()
    order = torch.argsort(flat, stable=True)
    experts, counts = torch.unique_consecutive(flat[order], return_counts=True)
    offsets = torch.zeros(experts.numel() + 1, dtype=torch.int32, device=ids.device)
    offsets[1:] = torch.cumsum(counts, 0).to(torch.int32)
    return experts.to(torch.int32), offsets, order.to(torch.int32)


def main() -> int:
    mod = build()
    fl = floor.build()
    dev = torch.device("cuda")
    e, k, h, i = 128, 8, 2048, 768
    w = bench.build(e, h, i, dev)
    q1 = w["q1"].contiguous()
    s1 = w["s1"].contiguous()
    alpha = torch.ones(e, device=dev)
    sink = torch.zeros(4, dtype=torch.int32, device=dev)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    print("tokens | experts | max rel err vs fp32 | normwise | bit-identical over 50 calls | FC1 kernel us | FC1-bytes read-only us")
    for m in (1, 2, 4, 8, 16):
        g = torch.Generator().manual_seed(1000 + m)
        x = torch.randn(m, h, generator=g).to(device=dev, dtype=torch.bfloat16)
        _, ids = torch.topk(F.softmax(torch.randn(m, e, generator=g), dim=-1), k, dim=-1)
        ids = ids.to(torch.int32).to(dev)
        experts, offsets, pairs = route(ids)
        act = torch.empty(m * k, i, device=dev, dtype=torch.bfloat16)
        run = lambda: mod.fc1_w4a16(q1, s1, x, experts, offsets, pairs, alpha, act, i, k)  # noqa: E731
        run()
        torch.cuda.synchronize()
        # reference: fp32 on dequantized weights, bf16 activations
        ref = torch.empty(m * k, i, device=dev)
        for p in range(m * k):
            t, ex = p // k, int(ids.view(-1)[p])
            hmid = x[t:t + 1].float() @ w["w1d"][ex].T
            ref[p] = (F.silu(hmid[:, i:]) * hmid[:, :i])[0]
        diff = (act.float() - ref)
        rel = float((diff.norm() / ref.norm()))
        maxrel = float((diff.abs() / ref.abs().clamp(min=1e-2)).max())
        first = act.clone()
        same = True
        for _ in range(50):
            run()
            same = same and bool(torch.equal(act, first))
        t_fc1 = floor.graph_time(run)
        ptrs = torch.tensor([q1.data_ptr() + int(x_) * 2 * i * h // 2 for x_ in experts.tolist()],
                            dtype=torch.int64, device=dev)
        # read-only floor for the codes the kernel reads (scales live in a separate tensor; codes are 8/9 of the bytes)
        t_read = floor.graph_time(lambda: fl.stream_read(ptrs, 2 * i * h // 2, 0, 2 * i * h // 2, sms * 4, 256, sink))
        print(f"{m:6d} | {experts.numel():7d} | {maxrel:18.4f} | {rel:8.5f} | {str(same):27s} | {t_fc1:13.1f} | {t_read:8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
