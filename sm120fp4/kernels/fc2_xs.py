"""Stage 2, FC2 on tensor cores with one expert's activations shared by the block ("xs": expert-shared).

In fc2_mma_pf.py every warp walks its own experts and, for each expert and each 128-wide chunk, loads its pair rows' activation
columns from global memory; the block's eight warps together read each expert's activations eight times (once per 16-column tile),
and fc2_cols32.py showed halving that re-read is worth 16.4 us alone at 16 random tokens. The activations cannot be staged whole:
FC1 writes one row per (token, expert) pair, 192 KB at 16 tokens with top_k 8, twice SM120's shared memory (BACKLOG 7). What fits
is one expert at a time: here the block's eight warps take the SAME expert and eight different 16-column tiles (128 output columns
per block), the expert's at most 16 pair rows are copied once into shared memory (24 KB at I = 768, rows padded by 8 elements so a
warp's fragment reads hit distinct banks) and every warp reads its B fragments from there. Expert group g walks experts g, g + G,
... with the next expert's weights prefetched into registers as in the prefetch kernel; with H = 2048 the grid is 16 x G blocks, so
G is 8 or 16 here where the prefetch kernel used 1 or 2. Each group writes an fp32 partial and the last group to finish a column
block adds the G partials in group order, so the result is deterministic. It is not bit-identical to the prefetch kernel: a column's
experts are summed in a different grouping.

    PYTHONPATH=. python scripts/fc2_xs.py --check-only
    PYTHONPATH=. python scripts/fc2_xs.py --out reports/fc2-xs-<device>-<date>.json
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
for _name in ("bench_moe_baseline", "micro_floor", "fc1_w4a16", "fc2_w4a16", "fc1_mma", "fc2_mma", "fc2_mma_pf"):
    _spec = importlib.util.spec_from_file_location(_name, _here / f"{_name}.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)
bench = sys.modules["bench_moe_baseline"]
floor = sys.modules["micro_floor"]
fc1 = sys.modules["fc1_w4a16"]
fc2m = sys.modules["fc2_mma"]
fc2p = sys.modules["fc2_mma_pf"]

CPP = r"""
#include <torch/extension.h>
void fc2_xs(torch::Tensor q2, torch::Tensor s2, torch::Tensor act, torch::Tensor experts, torch::Tensor offsets,
            torch::Tensor pairs, torch::Tensor weights, torch::Tensor alpha, torch::Tensor out, torch::Tensor scratch,
            torch::Tensor counters, int64_t top_k, int64_t groups);
void fc2_xs_set_pdl(bool on);
"""

# The helpers (e4m3, fp4x2, pack_bf16, decode_pairs, mma) are v1's, reused verbatim.
_V1 = fc2m.CUDA
_HELPERS = _V1[: _V1.index("template <int NT>")]

CUDA = _HELPERS + r"""
constexpr int BCOLS = WARPS * COLS;   // output columns per block: eight warps, one 16-column tile each

template <int CH>
__device__ __forceinline__ void load_expert(const unsigned char* __restrict__ q2, const unsigned char* __restrict__ s2,
                                            long long r0, int I, int tig, uint4 (&wq)[CH][2], unsigned (&ws)[CH][2]) {
  const long long r1 = r0 + 8;
#pragma unroll
  for (int ch = 0; ch < CH; ++ch) {
    const int k = ch * 128 + tig * 32;
    wq[ch][0] = __ldcs(reinterpret_cast<const uint4*>(q2 + r0 * (I / 2) + k / 2));
    wq[ch][1] = __ldcs(reinterpret_cast<const uint4*>(q2 + r1 * (I / 2) + k / 2));
    ws[ch][0] = *reinterpret_cast<const unsigned short*>(s2 + r0 * (I / 16) + k / 16);
    ws[ch][1] = *reinterpret_cast<const unsigned short*>(s2 + r1 * (I / 16) + k / 16);
  }
}

template <int NT, int CH>
__global__ void __launch_bounds__(WARPS * 32)
k_fc2_xs(const unsigned char* __restrict__ q2, const unsigned char* __restrict__ s2, const __nv_bfloat16* __restrict__ act,
         const int* __restrict__ experts, const int* __restrict__ offsets, const int* __restrict__ pairs,
         const float* __restrict__ weights, const float* __restrict__ alpha, __nv_bfloat16* __restrict__ out,
         float* __restrict__ scratch, int* __restrict__ counters, int U, int M, int H, int I, int top_k) {
  constexpr int XW = CH * 128 + 8;                 // a staged row: I elements and an 8-element pad (16 bytes) against bank conflicts
  __shared__ float part[WARPS][COLS][MAXM];
  __shared__ __align__(16) unsigned short xs[8 * NT][XW];   // the current expert's pair rows
  __shared__ int last;
  cudaGridDependencySynchronize();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
  const int nb = blockIdx.x * BCOLS, n0 = nb + warp * COLS, g = blockIdx.y, G = gridDim.y;
  for (int z = threadIdx.x; z < WARPS * COLS * MAXM; z += blockDim.x) (&part[0][0][0])[z] = 0.f;
  __syncthreads();

  int u = g;
  int e = u < U ? experts[u] : -1;
  uint4 wq[CH][2];
  unsigned ws[CH][2];
  if (e >= 0) load_expert<CH>(q2, s2, (long long)e * H + n0 + gid, I, tig, wq, ws);
  for (; u < U; u += G) {
    // the next expert's weights, issued before this expert's staging and arithmetic
    const int un = u + G;
    const int en = un < U ? experts[un] : -1;
    uint4 nq[CH][2];
    unsigned ns[CH][2];
    if (en >= 0) {
      load_expert<CH>(q2, s2, (long long)en * H + n0 + gid, I, tig, nq, ns);
    } else {
#pragma unroll
      for (int ch = 0; ch < CH; ++ch) { nq[ch][0] = nq[ch][1] = make_uint4(0, 0, 0, 0); ns[ch][0] = ns[ch][1] = 0u; }
    }
    const int p0 = e >= 0 ? offsets[u] : 0;
    const int cnt = e >= 0 ? min(offsets[u + 1] - p0, 8 * NT) : 0;   // block-uniform
    __syncthreads();                                 // every warp is done reading the previous expert's rows
    if (cnt > 0) {
      const int nv = I / 8;                          // uint4 per activation row
      for (int z = threadIdx.x; z < cnt * nv; z += WARPS * 32) {
        const int slot = z / nv, c = z - slot * nv;
        const int p = pairs[p0 + slot];
        *reinterpret_cast<uint4*>(&xs[slot][c * 8]) = __ldg(reinterpret_cast<const uint4*>(act + (long long)p * I) + c);
      }
    }
    __syncthreads();
    if (cnt > 0) {
      bool valid[NT];
#pragma unroll
      for (int t = 0; t < NT; ++t) valid[t] = t * 8 + gid < cnt;
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
#pragma unroll
          for (int v = 0; v < 4; ++v) {
            const uint4 q = valid[t] ? *reinterpret_cast<const uint4*>(&xs[t * 8 + gid][k + 8 * v]) : make_uint4(0, 0, 0, 0);
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
    }
    __syncwarp();
#pragma unroll
    for (int ch = 0; ch < CH; ++ch) { wq[ch][0] = nq[ch][0]; wq[ch][1] = nq[ch][1]; ws[ch][0] = ns[ch][0]; ws[ch][1] = ns[ch][1]; }
    e = en;
  }
  __syncthreads();
  // each warp owns its 16 columns, so there is no cross-warp sum; the groups are summed in group order by the last to finish
  if (G == 1) {
    for (int idx = threadIdx.x; idx < BCOLS * M; idx += blockDim.x) {
      const int r = idx / M, t = idx % M;
      out[(long long)t * H + nb + r] = __float2bfloat16(part[r / COLS][r % COLS][t]);
    }
    return;
  }
  for (int idx = threadIdx.x; idx < BCOLS * M; idx += blockDim.x) {
    const int r = idx / M, t = idx % M;
    scratch[((long long)g * MAXM + t) * H + nb + r] = part[r / COLS][r % COLS][t];
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = (atomicAdd(&counters[blockIdx.x], 1) == G - 1);
  __syncthreads();
  if (!last) return;
  __threadfence();
  for (int idx = threadIdx.x; idx < BCOLS * M; idx += blockDim.x) {
    const int r = idx / M, t = idx % M;
    float v = 0.f;
    for (int gg = 0; gg < G; ++gg) v += __ldcg(&scratch[((long long)gg * MAXM + t) * H + nb + r]);   // fixed group order
    out[(long long)t * H + nb + r] = __float2bfloat16(v);
  }
  if (threadIdx.x == 0) counters[blockIdx.x] = 0;   // ready for the next launch
}

static bool g_pdl = false;
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
void fc2_xs_set_pdl(bool on) { g_pdl = on; }

template <int NT, int CH>
static void launch(dim3 grid, cudaStream_t st, const unsigned char* q2, const unsigned char* s2, const __nv_bfloat16* act,
                   const int* experts, const int* offsets, const int* pairs, const float* weights, const float* alpha,
                   __nv_bfloat16* out, float* scratch, int* counters, int U, int M, int H, int I, int top_k) {
  pdl_launch(k_fc2_xs<NT, CH>, grid, dim3(WARPS * 32), st, q2, s2, act, experts, offsets, pairs, weights, alpha, out,
             scratch, counters, U, M, H, I, top_k);
}

void fc2_xs(torch::Tensor q2, torch::Tensor s2, torch::Tensor act, torch::Tensor experts, torch::Tensor offsets,
            torch::Tensor pairs, torch::Tensor weights, torch::Tensor alpha, torch::Tensor out, torch::Tensor scratch,
            torch::Tensor counters, int64_t top_k, int64_t groups) {
  const int M = (int)out.size(0), H = (int)out.size(1), I = (int)act.size(1), U = (int)experts.numel();
  TORCH_CHECK(M <= MAXM && H % BCOLS == 0 && (I == 768 || I == 1024), "shape");
  TORCH_CHECK(groups >= 1 && groups <= 16 && scratch.numel() >= groups * MAXM * H && counters.numel() >= H / BCOLS, "scratch");
  auto st = at::cuda::getCurrentCUDAStream();
  dim3 grid(H / BCOLS, (unsigned)groups);
#define XS_ARGS grid, st, q2.data_ptr<uint8_t>(), s2.data_ptr<uint8_t>(), reinterpret_cast<const __nv_bfloat16*>(act.data_ptr()), \
    experts.data_ptr<int>(), offsets.data_ptr<int>(), pairs.data_ptr<int>(), weights.data_ptr<float>(), alpha.data_ptr<float>(), \
    reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), scratch.data_ptr<float>(), counters.data_ptr<int>(), U, M, H, I, (int)top_k
  if (I == 768) { if (M <= 8) launch<1, 6>(XS_ARGS); else launch<2, 6>(XS_ARGS); }
  else          { if (M <= 8) launch<1, 8>(XS_ARGS); else launch<2, 8>(XS_ARGS); }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "launch");
}
"""


def build(verbose: bool = False):
    return load_inline(name="sm120fp4_fc2_xs", cpp_sources=CPP, cuda_sources=CUDA, functions=["fc2_xs", "fc2_xs_set_pdl"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a"] + (["-Xptxas=-v"] if verbose else []),
                       verbose=verbose)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, help="JSON report, written into the repository")
    ap.add_argument("--check-only", action="store_true", help="correctness and determinism only, no timing")
    ap.add_argument("--ptxas", action="store_true", help="print register use and spills")
    a = ap.parse_args(argv)
    if not a.check_only and a.out is None:
        ap.error("--out is required unless --check-only")
    xs, pf, v1, m1, fl = build(a.ptxas), fc2p.build(), fc2m.build(), fc1.build(), floor.build()
    dev = torch.device("cuda")
    e, k, h, i = 128, 8, 2048, 768
    w = bench.build(e, h, i, dev)
    q1, s1, q2, s2 = (w[n].contiguous() for n in ("q1", "s1", "q2", "s2"))
    alpha = torch.ones(e, device=dev)
    sink = torch.zeros(4, dtype=torch.int32, device=dev)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    scratch = torch.zeros(16 * 16 * h, device=dev)
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
        for G in (4, 8, 16):
            variants[f"xs_g{G}"] = (lambda GG: lambda o: xs.fc2_xs(q2, s2, act, experts, offsets, pairs, wf, alpha, o, scratch, counters, k, GG))(G)
        variants["v1"](o1)
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
