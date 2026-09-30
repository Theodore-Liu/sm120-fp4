"""Stage 2, FC1 on tensor cores: FP4 decoded in registers to bf16 (exactly), mma.sync.m16n8k16 bf16 -> fp32, tokens as N.

scripts/moe_breakdown.py showed FC1 on CUDA cores is instruction-bound once an expert receives several tokens (15.1 us
at one token per expert, 105.2 at sixteen, bytes constant). This version moves the arithmetic to the tensor cores:

- One block per (touched expert, tile of 16 intermediate channels); its four warps split the hidden dimension into
  quarters and each computes a 16 x 8 tile of up and of gate outputs (channels x tokens) with mma.m16n8k16, tokens as the
  8-wide N side (a second N tile for 9 to 16 tokens). The four partial tiles are added in shared memory in warp order,
  so every output has one summation order and the result is deterministic.
- FP4 codes are decoded with cvt.rn.f16x2.e2m1x2, multiplied by the E4M3 block scale and rounded to bf16. The product of
  an E2M1 value (at most 2 significant bits) and an E4M3 scale (4) has at most 6 significant bits, so the bf16 weight
  equals the dequantized weight exactly.
- The reduction over the hidden dimension does not depend on the order of k, so k is permuted identically for weights and
  activations: in chunk ch of each warp's quarter, lane tig of a quad owns 32 consecutive k at ch*128 + tig*32, and in MMA
  step s its fragment columns {2tig, 2tig+1, 2tig+8, 2tig+9} are those 32 k's positions 4s + {0, 1, 2, 3}. Each thread
  reads 16-byte vectors of codes and of activations instead of the fragment layout's scattered bytes, and the four lanes
  of a quad read 64 contiguous bytes of a row per load.

Same layout and output as scripts/fc1_w4a16.py (codes [E, 2I, H/2], rows [up ; gate]; scales [E*2I, H/16] E4M3; act
[pairs, I] bf16 with pair = token * top_k + j). H must be a multiple of 512 (four warps x four quad lanes x 32).

    PYTHONPATH=. python scripts/fc1_mma.py
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
void fc1_mma(torch::Tensor q1, torch::Tensor s1, torch::Tensor x, torch::Tensor experts, torch::Tensor offsets,
             torch::Tensor pairs, torch::Tensor alpha, torch::Tensor act, int64_t inter, int64_t top_k);
"""

CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <ATen/cuda/CUDAContext.h>

constexpr int WARPS = 4;
constexpr int ROWS = 16;              // intermediate channels per block (the MMA's M)
constexpr int CHUNKS = 4;             // 16-byte code vectors per row per thread: H = WARPS * 4 * 32 * CHUNKS = 2048

__device__ __forceinline__ float e4m3(unsigned b) {
  __nv_fp8_e4m3 v;
  v.__x = (unsigned char)b;
  return float(v);
}

__device__ __forceinline__ float2 fp4x2(unsigned byte) {
  unsigned out;
  unsigned short in = (unsigned short)byte;
  asm("{ .reg .b8 lo, hi;\n mov.b16 {lo, hi}, %1;\n cvt.rn.f16x2.e2m1x2 %0, lo; }\n" : "=r"(out) : "h"(in));
  return __half22float2(*reinterpret_cast<__half2*>(&out));
}

__device__ __forceinline__ unsigned pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<unsigned*>(&v);
}

// 16 bytes of codes (32 values, element 2j in the low nibble of byte j) and their two block scales -> 16 bf16 pairs:
// pair j holds values 2j and 2j+1.
__device__ __forceinline__ void decode_pairs(uint4 q, unsigned s01, unsigned* out) {
  const float s0 = e4m3(s01 & 0xff), s1 = e4m3((s01 >> 8) & 0xff);
  const unsigned w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const float2 f = fp4x2((w[j >> 2] >> (8 * (j & 3))) & 0xff);
    const float sc = j < 8 ? s0 : s1;
    out[j] = pack_bf16(f.x * sc, f.y * sc);
  }
}

__device__ __forceinline__ void mma(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int NT>   // 8-token tiles: 1 for up to 8 tokens per expert, 2 for up to 16
__global__ void __launch_bounds__(WARPS * 32)
k_fc1_mma(const unsigned char* __restrict__ q1, const unsigned char* __restrict__ s1, const __nv_bfloat16* __restrict__ x,
          const int* __restrict__ experts, const int* __restrict__ offsets, const int* __restrict__ pairs,
          const float* __restrict__ alpha, __nv_bfloat16* __restrict__ act, int H, int I, int top_k) {
  __shared__ float red[2][WARPS][ROWS][8 * NT];
  cudaGridDependencySynchronize();
  cudaTriggerProgrammaticLaunchCompletion();
  const int tiles = I / ROWS;
  const int u = blockIdx.x / tiles, c0 = (blockIdx.x % tiles) * ROWS;
  const int e = experts[u];
  if (e < 0) return;
  const int p0 = offsets[u];
  const int n = min(offsets[u + 1] - p0, 8 * NT);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
  // chunk ch of lane tig covers physical k in [warp*H/4 + ch*128 + tig*32, +32): in each load instruction the quad's four
  // lanes read 64 contiguous bytes of a row (whole 32-byte sectors), and weights and activations share the permutation
  const int k0 = warp * (H / WARPS) + tig * 32;

  // rows: [up gid, up gid+8, gate gid, gate gid+8]
  const long long base = (long long)e * 2 * I + c0;
  const long long rows[4] = {base + gid, base + gid + 8, base + I + gid, base + I + gid + 8};
  int tok[NT];
  bool valid[NT];
#pragma unroll
  for (int t = 0; t < NT; ++t) {
    const int ti = t * 8 + gid;
    valid[t] = ti < n;
    tok[t] = valid[t] ? pairs[p0 + ti] / top_k : 0;
  }
  float cu[NT][4], cg[NT][4];
#pragma unroll
  for (int t = 0; t < NT; ++t)
#pragma unroll
    for (int i = 0; i < 4; ++i) { cu[t][i] = 0.f; cg[t][i] = 0.f; }

  // every weight load of the thread is issued before any is used: 4 rows x CHUNKS 16-byte vectors in flight
  uint4 wqa[CHUNKS][4];
  unsigned wsa[CHUNKS][4];
#pragma unroll
  for (int ch = 0; ch < CHUNKS; ++ch)
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int k = k0 + ch * 128;
      wqa[ch][r] = __ldcs(reinterpret_cast<const uint4*>(q1 + rows[r] * (H / 2) + k / 2));
      wsa[ch][r] = *reinterpret_cast<const unsigned short*>(s1 + rows[r] * (H / 16) + k / 16);
    }
#pragma unroll
  for (int ch = 0; ch < CHUNKS; ++ch) {
    const int k = k0 + ch * 128;
    const uint4* wq = wqa[ch];
    const unsigned* ws = wsa[ch];
    unsigned xb[NT][16];                            // 32 bf16 activations of token gid: pair j = values 2j, 2j+1
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      const uint4* xv = reinterpret_cast<const uint4*>(x + (long long)tok[t] * H + k);
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const uint4 q = valid[t] ? __ldg(xv + v) : make_uint4(0, 0, 0, 0);
        xb[t][4 * v] = q.x; xb[t][4 * v + 1] = q.y; xb[t][4 * v + 2] = q.z; xb[t][4 * v + 3] = q.w;
      }
    }
    unsigned a[2][16];
    // up rows, then gate rows: 8 MMA steps each; step s uses pairs 2s (fragment cols 2tig, 2tig+1) and 2s+1 (2tig+8, 2tig+9)
    decode_pairs(wq[0], ws[0], a[0]);
    decode_pairs(wq[1], ws[1], a[1]);
#pragma unroll
    for (int s = 0; s < 8; ++s)
#pragma unroll
      for (int t = 0; t < NT; ++t)
        mma(cu[t], a[0][2 * s], a[1][2 * s], a[0][2 * s + 1], a[1][2 * s + 1], xb[t][2 * s], xb[t][2 * s + 1]);
    decode_pairs(wq[2], ws[2], a[0]);
    decode_pairs(wq[3], ws[3], a[1]);
#pragma unroll
    for (int s = 0; s < 8; ++s)
#pragma unroll
      for (int t = 0; t < NT; ++t)
        mma(cg[t], a[0][2 * s], a[1][2 * s], a[0][2 * s + 1], a[1][2 * s + 1], xb[t][2 * s], xb[t][2 * s + 1]);
  }
  // C fragment: c0, c1 -> channel gid, tokens 2tig, 2tig+1; c2, c3 -> channel gid+8
#pragma unroll
  for (int t = 0; t < NT; ++t) {
    red[0][warp][gid][t * 8 + 2 * tig] = cu[t][0];
    red[0][warp][gid][t * 8 + 2 * tig + 1] = cu[t][1];
    red[0][warp][gid + 8][t * 8 + 2 * tig] = cu[t][2];
    red[0][warp][gid + 8][t * 8 + 2 * tig + 1] = cu[t][3];
    red[1][warp][gid][t * 8 + 2 * tig] = cg[t][0];
    red[1][warp][gid][t * 8 + 2 * tig + 1] = cg[t][1];
    red[1][warp][gid + 8][t * 8 + 2 * tig] = cg[t][2];
    red[1][warp][gid + 8][t * 8 + 2 * tig + 1] = cg[t][3];
  }
  __syncthreads();
  const float al = alpha[e];
  for (int idx = threadIdx.x; idx < ROWS * 8 * NT; idx += blockDim.x) {
    const int r = idx / (8 * NT), ti = idx % (8 * NT);
    if (ti >= n) continue;
    float up = 0.f, gate = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) { up += red[0][w][r][ti]; gate += red[1][w][r][ti]; }   // fixed warp order
    up *= al;
    gate *= al;
    const float silu = gate / (1.f + __expf(-gate));
    act[(long long)pairs[p0 + ti] * I + c0 + r] = __float2bfloat16(silu * up);
  }
}

void fc1_mma(torch::Tensor q1, torch::Tensor s1, torch::Tensor x, torch::Tensor experts, torch::Tensor offsets,
             torch::Tensor pairs, torch::Tensor alpha, torch::Tensor act, int64_t inter, int64_t top_k) {
  const int H = (int)x.size(1), I = (int)inter, M = (int)x.size(0), U = (int)experts.numel();
  TORCH_CHECK(H == WARPS * 4 * 32 * CHUNKS && I % ROWS == 0 && M <= 16, "shape");
  auto st = at::cuda::getCurrentCUDAStream();
  const dim3 grid(U * (I / ROWS)), block(WARPS * 32);
#define FC1M_ARGS q1.data_ptr<uint8_t>(), s1.data_ptr<uint8_t>(), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), \
    experts.data_ptr<int>(), offsets.data_ptr<int>(), pairs.data_ptr<int>(), alpha.data_ptr<float>(),                   \
    reinterpret_cast<__nv_bfloat16*>(act.data_ptr()), H, I, (int)top_k
  if (M <= 8) k_fc1_mma<1><<<grid, block, 0, st>>>(FC1M_ARGS);
  else k_fc1_mma<2><<<grid, block, 0, st>>>(FC1M_ARGS);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "launch");
}
"""


def build():
    return load_inline(name="sm120fp4_fc1_mma", cpp_sources=CPP, cuda_sources=CUDA, functions=["fc1_mma"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a", "-Xptxas=-v"], verbose=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None, help="write every row to this JSON report")
    a = ap.parse_args(argv)
    rows = []
    mm, mc, fl = build(), fc1.build(), floor.build()
    dev = torch.device("cuda")
    e, k, h, i = 128, 8, 2048, 768
    w = bench.build(e, h, i, dev)
    q1, s1 = w["q1"].contiguous(), w["s1"].contiguous()
    alpha = torch.ones(e, device=dev)
    sink = torch.zeros(4, dtype=torch.int32, device=dev)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    print("routing: random (as the baselines) | then 8 fixed experts with every token routed to all 8 (bytes constant)")
    print("case | tokens | normwise vs fp32 | bit-identical x50 | FC1 mma us | FC1 CUDA-core us | read of FC1 codes us")

    def run_case(label, m, ids, x):
        experts, offsets, pairs = fc1.route(ids)
        act = torch.empty(ids.numel(), i, device=dev, dtype=torch.bfloat16)
        go = lambda: mm.fc1_mma(q1, s1, x, experts, offsets, pairs, alpha, act, i, k)  # noqa: E731
        go()
        torch.cuda.synchronize()
        ref = torch.empty(ids.numel(), i, device=dev)
        for p in range(ids.numel()):
            t, ex = p // k, int(ids.view(-1)[p])
            hmid = x[t:t + 1].float() @ w["w1d"][ex].T
            ref[p] = (F.silu(hmid[:, i:]) * hmid[:, :i])[0]
        rel = float((act.float() - ref).norm() / ref.norm())
        first = act.clone()
        stable = True
        for _ in range(50):
            go()
            stable = stable and bool(torch.equal(act, first))
        act2 = torch.empty_like(act)
        t_mma = floor.graph_time(go)
        t_cc = floor.graph_time(lambda: mc.fc1_w4a16(q1, s1, x, experts, offsets, pairs, alpha, act2, i, k))
        ptrs = torch.tensor([q1.data_ptr() + int(x_) * 2 * i * h // 2 for x_ in experts.tolist()], dtype=torch.int64, device=dev)
        t_read = floor.graph_time(lambda: fl.stream_read(ptrs, 2 * i * h // 2, 0, 2 * i * h // 2, sms * 4, 256, sink))
        rows.append({"routing": label, "tokens": m, "rel_err": rel, "bit_identical_50": stable, "fc1_mma_us": t_mma,
                     "fc1_cuda_core_us": t_cc, "read_fc1_codes_us": t_read})
        print(f"{label:6s} | {m:6d} | {rel:16.5f} | {str(stable):17s} | {t_mma:10.1f} | {t_cc:16.1f} | {t_read:8.1f}")

    for m in (1, 2, 4, 8, 16):
        g = torch.Generator().manual_seed(1000 + m)
        x = torch.randn(m, h, generator=g).to(device=dev, dtype=torch.bfloat16)
        _, ids = torch.topk(F.softmax(torch.randn(m, e, generator=g), dim=-1), k, dim=-1)
        run_case("random", m, ids.to(torch.int32).to(dev), x)
    fixed = torch.arange(8, dtype=torch.int32, device=dev) * 16
    for m in (1, 4, 8, 16):
        x = torch.randn(m, h, generator=torch.Generator().manual_seed(7)).to(device=dev, dtype=torch.bfloat16)
        run_case("fixed8", m, fixed.repeat(m, 1).contiguous(), x)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"device": torch.cuda.get_device_name(0), "rows": rows}, indent=1) + "\n", encoding="utf-8")
        print(f"written {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
