"""Stage 2, FC2 on tensor cores: out[t, n] = sum over t's routed experts of weight x alpha x W2[e][n, :] . act[pair, :].

The same machinery as scripts/fc1_mma.py (exact bf16 decode of FP4 x E4M3, k permuted identically for weights and
activations so a quad reads 64 contiguous bytes per load, every weight load issued first), arranged for FC2:

- One block per tile of 16 output (hidden) columns, the MMA's M; the N side is one expert's (token, expert) pairs
  (8 per tile, two tiles for 9 to 16); k runs over the intermediate dimension.
- The block's eight warps take the touched experts round-robin. After each expert a warp adds weight x alpha x its
  16 x 8 result into per-token accumulators in shared memory, then synchronises: a token can sit in a different lane
  for the next expert, and the synchronisation fixes the order of the two adds. The warps' partials are then added in
  warp order and each output written once. Every output has one summation order: deterministic, no atomics.
- Each activation row is read once per 16 output columns rather than once per column (scripts/moe_breakdown.py found
  the per-column re-reads cost FC2 up to 27 us at 16 tokens).

Layout as scripts/fc2_w4a16.py: codes [E, H, I/2], scales [E*H, I/16] E4M3, per-expert alpha; activations [pairs, I] bf16
from FC1; out [M, H] bf16. I must be a multiple of 128 and at most 1024.

    PYTHONPATH=. python scripts/fc2_mma.py --out reports/fc2-mma-<device>-<date>.json
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
for _name in ("bench_moe_baseline", "micro_floor", "fc1_w4a16", "fc2_w4a16", "fc1_mma"):
    _spec = importlib.util.spec_from_file_location(_name, _here / f"{_name}.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)
bench = sys.modules["bench_moe_baseline"]
floor = sys.modules["micro_floor"]
fc1 = sys.modules["fc1_w4a16"]
fc2 = sys.modules["fc2_w4a16"]
fc1m = sys.modules["fc1_mma"]

CPP = r"""
#include <torch/extension.h>
void fc2_mma(torch::Tensor q2, torch::Tensor s2, torch::Tensor act, torch::Tensor experts, torch::Tensor offsets,
             torch::Tensor pairs, torch::Tensor weights, torch::Tensor alpha, torch::Tensor out, int64_t top_k);
"""

CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <ATen/cuda/CUDAContext.h>

constexpr int WARPS = 8;
constexpr int COLS = 16;       // output columns per block (the MMA's M)
constexpr int MAXM = 16;       // tokens in the batch
constexpr int MAXCH = 8;       // 128-wide k chunks: I <= 1024

__device__ __forceinline__ float e4m3(unsigned b) {
  __nv_fp8_e4m3 v;
  v.__x = (unsigned char)b;
  return float(v);
}

__device__ __forceinline__ float2 fp4x2(unsigned byte) {
  unsigned o;
  unsigned short in = (unsigned short)byte;
  asm("{ .reg .b8 lo, hi;\n mov.b16 {lo, hi}, %1;\n cvt.rn.f16x2.e2m1x2 %0, lo; }\n" : "=r"(o) : "h"(in));
  return __half22float2(*reinterpret_cast<__half2*>(&o));
}

__device__ __forceinline__ unsigned pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<unsigned*>(&v);
}

__device__ __forceinline__ void decode_pairs(uint4 q, unsigned s01, unsigned* dst) {
  const float s0 = e4m3(s01 & 0xff), s1 = e4m3((s01 >> 8) & 0xff);
  const unsigned w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const float2 f = fp4x2((w[j >> 2] >> (8 * (j & 3))) & 0xff);
    const float sc = j < 8 ? s0 : s1;
    dst[j] = pack_bf16(f.x * sc, f.y * sc);
  }
}

__device__ __forceinline__ void mma(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int NT>   // 8-pair tiles per expert: 1 when the batch has at most 8 tokens, 2 up to 16
__global__ void __launch_bounds__(WARPS * 32)
k_fc2_mma(const unsigned char* __restrict__ q2, const unsigned char* __restrict__ s2, const __nv_bfloat16* __restrict__ act,
          const int* __restrict__ experts, const int* __restrict__ offsets, const int* __restrict__ pairs,
          const float* __restrict__ weights, const float* __restrict__ alpha, __nv_bfloat16* __restrict__ out,
          int U, int M, int H, int I, int top_k) {
  __shared__ float part[WARPS][COLS][MAXM];
  cudaGridDependencySynchronize();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
  const int n0 = blockIdx.x * COLS;
  const int chunks = I / 128;
  for (int z = threadIdx.x; z < WARPS * COLS * MAXM; z += blockDim.x) (&part[0][0][0])[z] = 0.f;
  __syncthreads();

  for (int u = warp; u < U; u += WARPS) {
    const int e = experts[u];
    if (e < 0) continue;                             // padding slot from the GPU router
    const int p0 = offsets[u];
    const int cnt = min(offsets[u + 1] - p0, 8 * NT);
    const long long r0 = (long long)e * H + n0 + gid, r1 = r0 + 8;
    uint4 wq[MAXCH][2];
    unsigned ws[MAXCH][2];
#pragma unroll
    for (int ch = 0; ch < MAXCH; ++ch) {
      if (ch < chunks) {
        const int k = ch * 128 + tig * 32;
        wq[ch][0] = __ldcs(reinterpret_cast<const uint4*>(q2 + r0 * (I / 2) + k / 2));
        wq[ch][1] = __ldcs(reinterpret_cast<const uint4*>(q2 + r1 * (I / 2) + k / 2));
        ws[ch][0] = *reinterpret_cast<const unsigned short*>(s2 + r0 * (I / 16) + k / 16);
        ws[ch][1] = *reinterpret_cast<const unsigned short*>(s2 + r1 * (I / 16) + k / 16);
      }
    }
    int pr[NT];
    bool valid[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      const int slot = t * 8 + gid;
      valid[t] = slot < cnt;
      pr[t] = valid[t] ? pairs[p0 + slot] : 0;
    }
    float c[NT][4];
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int i = 0; i < 4; ++i) c[t][i] = 0.f;
#pragma unroll
    for (int ch = 0; ch < MAXCH; ++ch) {
      if (ch < chunks) {
        const int k = ch * 128 + tig * 32;
        unsigned xb[NT][16];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
          const uint4* av = reinterpret_cast<const uint4*>(act + (long long)pr[t] * I + k);
#pragma unroll
          for (int v = 0; v < 4; ++v) {
            const uint4 q = valid[t] ? __ldg(av + v) : make_uint4(0, 0, 0, 0);
            xb[t][4 * v] = q.x; xb[t][4 * v + 1] = q.y; xb[t][4 * v + 2] = q.z; xb[t][4 * v + 3] = q.w;
          }
        }
        unsigned a[2][16];
        decode_pairs(wq[ch][0], ws[ch][0], a[0]);
        decode_pairs(wq[ch][1], ws[ch][1], a[1]);
#pragma unroll
        for (int s = 0; s < 8; ++s)
#pragma unroll
          for (int t = 0; t < NT; ++t)
            mma(c[t], a[0][2 * s], a[1][2 * s], a[0][2 * s + 1], a[1][2 * s + 1], xb[t][2 * s], xb[t][2 * s + 1]);
      }
    }
    // C fragment: c0, c1 -> column n0+gid, pair slots 2tig, 2tig+1; c2, c3 -> column n0+gid+8
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
    __syncwarp();                                    // the next expert may put a token in another lane
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < COLS * M; idx += blockDim.x) {
    const int r = idx / M, t = idx % M;
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) v += part[w][r][t];   // fixed warp order
    out[(long long)t * H + n0 + r] = __float2bfloat16(v);
  }
}

void fc2_mma(torch::Tensor q2, torch::Tensor s2, torch::Tensor act, torch::Tensor experts, torch::Tensor offsets,
             torch::Tensor pairs, torch::Tensor weights, torch::Tensor alpha, torch::Tensor out, int64_t top_k) {
  const int M = (int)out.size(0), H = (int)out.size(1), I = (int)act.size(1), U = (int)experts.numel();
  TORCH_CHECK(M <= MAXM && I % 128 == 0 && I <= 128 * MAXCH && H % COLS == 0, "shape");
  auto st = at::cuda::getCurrentCUDAStream();
#define FC2M_ARGS q2.data_ptr<uint8_t>(), s2.data_ptr<uint8_t>(), reinterpret_cast<const __nv_bfloat16*>(act.data_ptr()), \
    experts.data_ptr<int>(), offsets.data_ptr<int>(), pairs.data_ptr<int>(), weights.data_ptr<float>(),                   \
    alpha.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), U, M, H, I, (int)top_k
  if (M <= 8) k_fc2_mma<1><<<H / COLS, WARPS * 32, 0, st>>>(FC2M_ARGS);
  else k_fc2_mma<2><<<H / COLS, WARPS * 32, 0, st>>>(FC2M_ARGS);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "launch");
}
"""


def build():
    return load_inline(name="sm120fp4_fc2_mma", cpp_sources=CPP, cuda_sources=CUDA, functions=["fc2_mma"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a"], verbose=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="JSON report, written into the repository")
    a = ap.parse_args(argv)
    m2m, m2, m1, fl = build(), fc2.build(), fc1.build(), floor.build()
    dev = torch.device("cuda")
    e, k, h, i = 128, 8, 2048, 768
    w = bench.build(e, h, i, dev)
    q1, s1, q2, s2 = (w[n].contiguous() for n in ("q1", "s1", "q2", "s2"))
    alpha = torch.ones(e, device=dev)
    sink = torch.zeros(4, dtype=torch.int32, device=dev)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    rows = []
    print("routing | tokens | normwise vs fp32 MoE | bit-identical x50 | FC2 mma us | FC2 CUDA-core us | read of FC2 codes us")

    def case(label, m, ids, wts, x):
        experts, offsets, pairs = fc1.route(ids)
        act = torch.empty(ids.numel(), i, device=dev, dtype=torch.bfloat16)
        m1.fc1_w4a16(q1, s1, x, experts, offsets, pairs, alpha, act, i, k)
        out = torch.empty(m, h, device=dev, dtype=torch.bfloat16)
        wf = wts.reshape(-1).contiguous()
        go = lambda: m2m.fc2_mma(q2, s2, act, experts, offsets, pairs, wf, alpha, out, k)  # noqa: E731
        go()
        torch.cuda.synchronize()
        ref = bench.reference(x, w, ids, wts, i, act_quant=False)
        rel = float((out.float() - ref).norm() / ref.norm())
        first = out.clone()
        stable = True
        for _ in range(50):
            go()
            stable = stable and bool(torch.equal(out, first))
        out2 = torch.empty_like(out)
        t_m = floor.graph_time(go)
        t_c = floor.graph_time(lambda: m2.fc2_w4a16(q2, s2, act, experts, offsets, pairs, wf, alpha, out2, k))
        ptrs = torch.tensor([q2.data_ptr() + int(x_) * h * i // 2 for x_ in experts.tolist()], dtype=torch.int64, device=dev)
        t_r = floor.graph_time(lambda: fl.stream_read(ptrs, h * i // 2, 0, h * i // 2, sms * 4, 256, sink))
        rows.append({"routing": label, "tokens": m, "rel_err_vs_fp32_moe": rel, "bit_identical_50": stable,
                     "fc2_mma_us": t_m, "fc2_cuda_core_us": t_c, "read_fc2_codes_us": t_r})
        print(f"{label:7s} | {m:6d} | {rel:20.5f} | {str(stable):17s} | {t_m:10.1f} | {t_c:16.1f} | {t_r:8.1f}")

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
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"device": torch.cuda.get_device_name(0), "rows": rows}, indent=1) + "\n", encoding="utf-8")
    print(f"written {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
