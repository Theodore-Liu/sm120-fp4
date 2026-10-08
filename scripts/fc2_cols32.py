"""FC2 with 32 output columns per block: one activation read serves two 16-row weight tiles.

At 16 randomly routed tokens FC2 (scripts/fc2_mma_pf.py) takes about 87 us against a 50 us read of its codes, and the
experiments in docs/stage2-design.md left one measured cost standing: every column tile re-reads each expert's
activations, about 12 us at 16 tokens (fc2_split's no_act variant). Here each warp computes two 16-row tiles of the same
expert, so the activations are read once for both and the grid has half the column tiles.

Two tiles of an expert's weights are as many loads in flight per warp as the prefetch kernel's current and next expert,
so this kernel does not also prefetch the next expert (the registers would not hold three tiles). Expert assignment
(u = g * WARPS + warp, stride WARPS * G), the warp order and the group order are the prefetch kernel's, and each
column's products are the same MMAs, so the output must be bit-identical to it at the same group count.

    PYTHONPATH=. python scripts/fc2_cols32.py --check-only
    PYTHONPATH=. python scripts/fc2_cols32.py --out reports/fc2-cols32-<device>-<date>.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
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
pfm = sys.modules["fc2_mma_pf"]

CPP = r"""
#include <torch/extension.h>
void fc2_c32(torch::Tensor q2, torch::Tensor s2, torch::Tensor act, torch::Tensor experts, torch::Tensor offsets,
             torch::Tensor pairs, torch::Tensor weights, torch::Tensor alpha, torch::Tensor out, torch::Tensor scratch,
             torch::Tensor counters, int64_t top_k, int64_t groups);
void fc2_c32_set_pdl(bool on);
"""

# the prefetch kernel's helpers and load_expert, verbatim; its kernel and launcher are not used
_PF = pfm.CUDA
_HELPERS = _PF[: _PF.index("// PF_CHAIN: everything an expert's routing needs")]
# the prefetch kernel's routing-prefetch pieces (Meta, load_meta), verbatim; used only under C32_CHAIN
_CHAIN = _PF[_PF.index("// PF_CHAIN: everything an expert's routing needs"): _PF.index("template <int NT, int CH>\n__global__")]

CUDA = _HELPERS + _CHAIN + r"""
constexpr int TILES = 2;                // 16-row tiles per warp
constexpr int COLS2 = 16 * TILES;       // output columns per block

template <int NT, int CH>
__global__ void __launch_bounds__(WARPS * 32)
k_fc2_c32(const unsigned char* __restrict__ q2, const unsigned char* __restrict__ s2, const __nv_bfloat16* __restrict__ act,
          const int* __restrict__ experts, const int* __restrict__ offsets, const int* __restrict__ pairs,
          const float* __restrict__ weights, const float* __restrict__ alpha, __nv_bfloat16* __restrict__ out,
          float* __restrict__ scratch, int* __restrict__ counters, int U, int M, int H, int I, int top_k) {
  __shared__ float part[WARPS][COLS2][MAXM];
#ifdef C32_ONE_BLOCK
  // timing control: 40 KB of shared memory no thread reads, so two blocks (2 x 56 KB) no longer fit on an SM
  __shared__ volatile float occupancy_pad[10240];
  if (threadIdx.x == 0 && blockIdx.x == 0xFFFFFFF) occupancy_pad[0] = 0.f;
#endif
  __shared__ int last;
  cudaGridDependencySynchronize();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
  const int n0 = blockIdx.x * COLS2, g = blockIdx.y, G = gridDim.y;
  for (int z = threadIdx.x; z < WARPS * COLS2 * MAXM; z += blockDim.x) (&part[0][0][0])[z] = 0.f;
  __syncthreads();

  const int stride = WARPS * G;
  int u = g * WARPS + warp;
#ifdef C32_PREFETCH
  // the first expert's two tiles now; each next expert's two tiles one round ahead (the prefetch kernel's register double buffer, two tiles wide)
  uint4 pq[TILES][CH][2];
  unsigned ps[TILES][CH][2];
  if (u < U && experts[u] >= 0) {
#pragma unroll
    for (int j = 0; j < TILES; ++j) load_expert<CH>(q2, s2, (long long)experts[u] * H + n0 + 16 * j + gid, I, tig, pq[j], ps[j]);
  }
#endif
#ifdef C32_CHAIN
  // the routing of the first expert now, of each next expert one round ahead (the prefetch kernel's chain option)
  Meta<NT> cur, nxt;
  if (u < U && experts[u] >= 0) load_meta<NT>(offsets, pairs, weights, alpha, u, experts[u], gid, tig, top_k, cur);
#endif
  for (; u < U; u += stride) {
    const int e = experts[u];
#ifdef C32_CHAIN
    const int un = u + stride;
    const int en = un < U ? experts[un] : -1;
    if (en >= 0) load_meta<NT>(offsets, pairs, weights, alpha, un, en, gid, tig, top_k, nxt);
#endif
#ifdef C32_PREFETCH
    const int un_ = u + stride;
    const int en_ = un_ < U ? experts[un_] : -1;
    uint4 nq[TILES][CH][2];
    unsigned ns[TILES][CH][2];
    if (en_ >= 0) {
#pragma unroll
      for (int j = 0; j < TILES; ++j) load_expert<CH>(q2, s2, (long long)en_ * H + n0 + 16 * j + gid, I, tig, nq[j], ns[j]);
    } else {
#pragma unroll
      for (int j = 0; j < TILES; ++j)
#pragma unroll
        for (int ch = 0; ch < CH; ++ch) { nq[j][ch][0] = nq[j][ch][1] = make_uint4(0, 0, 0, 0); ns[j][ch][0] = ns[j][ch][1] = 0u; }
    }
#endif
    if (e >= 0) {                                   // a negative id is a padding slot from the GPU router
#ifdef C32_PREFETCH
      uint4 (&wq)[TILES][CH][2] = pq;
      unsigned (&ws)[TILES][CH][2] = ps;
#else
      uint4 wq[TILES][CH][2];
      unsigned ws[TILES][CH][2];
#pragma unroll
      for (int j = 0; j < TILES; ++j) load_expert<CH>(q2, s2, (long long)e * H + n0 + 16 * j + gid, I, tig, wq[j], ws[j]);
#endif
#ifdef C32_CHAIN
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
      float c[TILES][NT][4];
#pragma unroll
      for (int j = 0; j < TILES; ++j)
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
          for (int i = 0; i < 4; ++i) c[j][t][i] = 0.f;
#pragma unroll
      for (int ch = 0; ch < CH; ++ch) {
        const int k = ch * 128 + tig * 32;
        unsigned xb[NT][16];                        // read once, used by both tiles
#pragma unroll
        for (int t = 0; t < NT; ++t) {
          const uint4* av = reinterpret_cast<const uint4*>(act + (long long)pr[t] * I + k);
#pragma unroll
          for (int v = 0; v < 4; ++v) {
            const uint4 q = valid[t] ? __ldg(av + v) : make_uint4(0, 0, 0, 0);
            xb[t][4 * v] = q.x; xb[t][4 * v + 1] = q.y; xb[t][4 * v + 2] = q.z; xb[t][4 * v + 3] = q.w;
          }
        }
#pragma unroll
        for (int j = 0; j < TILES; ++j) {
          unsigned a[2][16];
          decode_pairs(wq[j][ch][0], ws[j][ch][0], a[0]);
          decode_pairs(wq[j][ch][1], ws[j][ch][1], a[1]);
#pragma unroll
          for (int s = 0; s < 8; ++s)
#pragma unroll
            for (int t = 0; t < NT; ++t)
              mma(c[j][t], a[0][2 * s], a[1][2 * s], a[0][2 * s + 1], a[1][2 * s + 1], xb[t][2 * s], xb[t][2 * s + 1]);
        }
      }
#ifdef C32_CHAIN
#pragma unroll
      for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          if (cur.ok[t][h]) {
            const float wt = cur.wt[t][h];
            const int tok = cur.tok[t][h];
#pragma unroll
            for (int j = 0; j < TILES; ++j) {
              part[warp][16 * j + gid][tok] += wt * c[j][t][h];
              part[warp][16 * j + gid + 8][tok] += wt * c[j][t][2 + h];
            }
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
#pragma unroll
            for (int j = 0; j < TILES; ++j) {
              part[warp][16 * j + gid][tok] += wt * c[j][t][h];
              part[warp][16 * j + gid + 8][tok] += wt * c[j][t][2 + h];
            }
          }
        }
#endif
    }
    __syncwarp();                                   // the next expert may put a token in another lane
#ifdef C32_PREFETCH
#pragma unroll
    for (int j = 0; j < TILES; ++j)
#pragma unroll
      for (int ch = 0; ch < CH; ++ch) { pq[j][ch][0] = nq[j][ch][0]; pq[j][ch][1] = nq[j][ch][1]; ps[j][ch][0] = ns[j][ch][0]; ps[j][ch][1] = ns[j][ch][1]; }
#endif
#ifdef C32_CHAIN
    cur = nxt;
#endif
  }
  __syncthreads();
  if (G == 1) {
    for (int idx = threadIdx.x; idx < COLS2 * M; idx += blockDim.x) {
      const int r = idx / M, t = idx % M;
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < WARPS; ++w) v += part[w][r][t];   // fixed warp order
      out[(long long)t * H + n0 + r] = __float2bfloat16(v);
    }
    return;
  }
  for (int idx = threadIdx.x; idx < COLS2 * M; idx += blockDim.x) {
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
  for (int idx = threadIdx.x; idx < COLS2 * M; idx += blockDim.x) {
    const int r = idx / M, t = idx % M;
    float v = 0.f;
    for (int gg = 0; gg < G; ++gg) v += __ldcg(&scratch[((long long)gg * MAXM + t) * H + n0 + r]);   // fixed group order
    out[(long long)t * H + n0 + r] = __float2bfloat16(v);
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
void fc2_c32_set_pdl(bool on) { g_pdl = on; }

template <int NT, int CH>
static void launch(dim3 grid, cudaStream_t st, const unsigned char* q2, const unsigned char* s2, const __nv_bfloat16* act,
                   const int* experts, const int* offsets, const int* pairs, const float* weights, const float* alpha,
                   __nv_bfloat16* out, float* scratch, int* counters, int U, int M, int H, int I, int top_k) {
  pdl_launch(k_fc2_c32<NT, CH>, grid, dim3(WARPS * 32), st, q2, s2, act, experts, offsets, pairs, weights, alpha, out,
             scratch, counters, U, M, H, I, top_k);
}

void fc2_c32(torch::Tensor q2, torch::Tensor s2, torch::Tensor act, torch::Tensor experts, torch::Tensor offsets,
             torch::Tensor pairs, torch::Tensor weights, torch::Tensor alpha, torch::Tensor out, torch::Tensor scratch,
             torch::Tensor counters, int64_t top_k, int64_t groups) {
  const int M = (int)out.size(0), H = (int)out.size(1), I = (int)act.size(1), U = (int)experts.numel();
  TORCH_CHECK(M <= MAXM && H % COLS2 == 0 && (I == 768 || I == 1024), "shape");
  TORCH_CHECK(groups >= 1 && groups <= 4 && scratch.numel() >= groups * MAXM * H && counters.numel() >= H / COLS2, "scratch");
  auto st = at::cuda::getCurrentCUDAStream();
  dim3 grid(H / COLS2, (unsigned)groups);
#define C32_ARGS grid, st, q2.data_ptr<uint8_t>(), s2.data_ptr<uint8_t>(), reinterpret_cast<const __nv_bfloat16*>(act.data_ptr()), \
    experts.data_ptr<int>(), offsets.data_ptr<int>(), pairs.data_ptr<int>(), weights.data_ptr<float>(), alpha.data_ptr<float>(), \
    reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), scratch.data_ptr<float>(), counters.data_ptr<int>(), U, M, H, I, (int)top_k
  if (I == 768) { if (M <= 8) launch<1, 6>(C32_ARGS); else launch<2, 6>(C32_ARGS); }
  else          { if (M <= 8) launch<1, 8>(C32_ARGS); else launch<2, 8>(C32_ARGS); }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "launch");
}
"""

NAME = "sm120fp4_fc2_c32"


def build(verbose: bool = False, one_block: bool = False, chain: bool = False, prefetch: bool = False):
    """one_block: timing control - pad the kernel's shared memory so one block fits per SM, as for the prefetch kernel.
    chain: load each expert's routing (offsets, pair indices, token and weight x alpha per slot) one expert ahead, as the prefetch kernel's
    chain option does; it changes only when loads are issued, so the output must equal the plain 32-column kernel's bit for bit."""
    return load_inline(name=NAME + ("_1blk" if one_block else "") + ("_chain" if chain else "") + ("_pf" if prefetch else ""), cpp_sources=CPP,
                       cuda_sources=("#define C32_ONE_BLOCK\n" if one_block else "") + ("#define C32_CHAIN\n" if chain else "") + ("#define C32_PREFETCH\n" if prefetch else "") + CUDA,
                       functions=["fc2_c32", "fc2_c32_set_pdl"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a"], verbose=verbose)


def resource_usage(name: str) -> dict:
    """Registers, spill stores/loads and shared memory of each kernel in a built extension, from cuobjdump."""
    from torch.utils.cpp_extension import _get_build_directory
    so = next(Path(_get_build_directory(name, False)).glob(f"{name}*.so"))
    txt = subprocess.run(["cuobjdump", "--dump-resource-usage", str(so)], capture_output=True, text=True).stdout
    out, fn = {}, None
    for line in txt.splitlines():
        m = re.search(r"Function (\S+):", line)
        if m:
            fn = m.group(1)
            continue
        m = re.search(r"REG:(\d+).*?SHARED:(\d+).*?LOCAL:(\d+)", line)
        if fn and m:
            out[fn] = {"registers": int(m.group(1)), "shared_bytes": int(m.group(2)), "local_bytes": int(m.group(3))}
            fn = None
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, help="JSON report, written into the repository")
    ap.add_argument("--check-only", action="store_true", help="correctness and determinism only, no timing")
    a = ap.parse_args(argv)
    if not a.check_only and a.out is None:
        ap.error("--out is required unless --check-only")
    c32, c32c, c32p, pf, m1, fl = build(), build(chain=True), build(prefetch=True), pfm.build(), fc1.build(), floor.build()
    usage = {"cols32": resource_usage(NAME), "cols32_prefetch": resource_usage(NAME + "_pf"), "cols16_prefetch": resource_usage("sm120fp4_fc2_pf")}
    for k_, v_ in usage.items():
        for fn, r in v_.items():
            if "k_fc2" in fn:
                print(f"{k_}: {fn[:60]} registers {r['registers']} local {r['local_bytes']} B")
    dev = torch.device("cuda")
    e, k, h, i = 128, 8, 2048, 768
    w = bench.build(e, h, i, dev)
    q1, s1, q2, s2 = (w[n].contiguous() for n in ("q1", "s1", "q2", "s2"))
    alpha = torch.ones(e, device=dev)
    sink = torch.zeros(4, dtype=torch.int32, device=dev)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    scratch = torch.zeros(4 * 16 * h, device=dev)
    counters = torch.zeros(h // 16, dtype=torch.int32, device=dev)
    rows = []
    print("routing | tokens | variant | normwise vs fp32 MoE | bit-identical x50 | equals twin same G | us")

    def case(label, m, ids, wts, x):
        experts, offsets, pairs = fc1.route(ids)
        act = torch.empty(ids.numel(), i, device=dev, dtype=torch.bfloat16)
        m1.fc1_w4a16(q1, s1, x, experts, offsets, pairs, alpha, act, i, k)
        wf = wts.reshape(-1).contiguous()
        ref = bench.reference(x, w, ids, wts, i, act_quant=False)
        row = {"routing": label, "tokens": m}
        variants = {}
        for G in (1, 2):
            variants[f"cols16_prefetch_g{G}"] = (lambda o, G=G: pf.fc2_pf(q2, s2, act, experts, offsets, pairs, wf, alpha, o,
                                                                         scratch, counters, k, G))
        for G in (1, 2, 4):
            variants[f"cols32_g{G}"] = (lambda o, G=G: c32.fc2_c32(q2, s2, act, experts, offsets, pairs, wf, alpha, o,
                                                                  scratch, counters, k, G))
        for G in (1, 2, 4):
            variants[f"cols32_chain_g{G}"] = (lambda o, G=G: c32c.fc2_c32(q2, s2, act, experts, offsets, pairs, wf, alpha, o,
                                                                         scratch, counters, k, G))
        for G in (1, 2, 4):
            variants[f"cols32_prefetch_g{G}"] = (lambda o, G=G: c32p.fc2_c32(q2, s2, act, experts, offsets, pairs, wf, alpha, o,
                                                                            scratch, counters, k, G))
        outs = {}
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
            outs[name] = first
            twin = (name.replace("cols32_chain", "cols32") if name.startswith("cols32_chain") else
                    name.replace("cols32_prefetch", "cols32") if name.startswith("cols32_prefetch") else name.replace("cols32", "cols16_prefetch"))
            same = bool(torch.equal(first, outs[twin])) if name.startswith("cols32") and twin in outs else None
            t = None if a.check_only else floor.graph_time(go)
            row[name] = {"rel_err_vs_fp32_moe": rel, "bit_identical_50": stable, "equals_cols16_same_groups": same, "us": t}
            print(f"{label:7s} | {m:6d} | {name:19s} | {rel:20.5f} | {str(stable):17s} | {str(same):20s} | "
                  + ("-" if t is None else f"{t:.1f}"))
        if not a.check_only:
            ptrs = torch.tensor([q2.data_ptr() + int(x_) * h * i // 2 for x_ in experts.tolist()], dtype=torch.int64, device=dev)
            row["read_fc2_codes_us"] = floor.graph_time(lambda: fl.stream_read(ptrs, h * i // 2, 0, h * i // 2, sms * 4, 256, sink))
            print(f"{'':7s} | {'':6s} | {'read floor':19s} | {'':20s} | {'':17s} | {'':20s} | {row['read_fc2_codes_us']:.1f}")
        rows.append(row)

    for m in (1, 2, 4, 8, 16):
        g = torch.Generator().manual_seed(1000 + m)
        x = torch.randn(m, h, generator=g).to(device=dev, dtype=torch.bfloat16)
        wts, ids = torch.topk(F.softmax(torch.randn(m, e, generator=g), dim=-1), k, dim=-1)
        wts = (wts / wts.sum(-1, keepdim=True)).float().to(dev).contiguous()
        case("random", m, ids.to(torch.int32).to(dev).contiguous(), wts, x)
    fixed = torch.arange(8, dtype=torch.int32, device=dev) * 16
    for m in (1, 16):
        x = torch.randn(m, h, generator=torch.Generator().manual_seed(7)).to(device=dev, dtype=torch.bfloat16)
        wts = torch.full((m, k), 1.0 / k, device=dev)
        case("fixed8", m, fixed.repeat(m, 1).contiguous(), wts, x)
    bad = [(r["routing"], r["tokens"], n) for r in rows for n, v in r.items() if isinstance(v, dict)
           and (not v["bit_identical_50"] or v["equals_cols16_same_groups"] is False)]
    if a.out is not None:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"device": torch.cuda.get_device_name(0), "check_only": a.check_only,
                                     "resource_usage": usage, "rows": rows, "failures": bad}, indent=1) + "\n",
                         encoding="utf-8")
        print(f"written {a.out}")
    if bad:
        print("FAILURES:", bad)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
