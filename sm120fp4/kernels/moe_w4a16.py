"""Stage 2: the whole decode MoE layer on the GPU with no host synchronisation - routing, FC1, FC2 - timed in one graph.

Routing runs as one block: per-expert counts in shared memory, an exclusive scan over experts, and each (token, expert)
pair placed at its expert's base plus its rank among the earlier pairs of that expert. Experts therefore come out in
ascending order and pairs in index order within an expert, which is what scripts/fc1_w4a16.py's torch router produces,
with no atomics deciding any position. Output buffers have a fixed size (min(E, pairs) expert slots); unused slots carry
expert id -1, which FC1 and FC2 skip, so nothing has to come back to the host to size a grid and the three kernels fit in
one CUDA graph. That makes the timing comparable with the baselines, whose calls include their own routing.

    PYTHONPATH=. python scripts/moe_w4a16.py --out reports/moe-w4a16-<device>-<date>.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_here = Path(__file__).resolve().parent
for _name in ("bench_moe_baseline", "micro_floor", "fc1_w4a16", "fc2_w4a16"):
    _spec = importlib.util.spec_from_file_location(_name, _here / f"{_name}.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)
bench = sys.modules["bench_moe_baseline"]
floor = sys.modules["micro_floor"]
fc1 = sys.modules["fc1_w4a16"]
fc2 = sys.modules["fc2_w4a16"]

CPP = r"""
#include <torch/extension.h>
void route(torch::Tensor ids, int64_t num_experts, torch::Tensor experts, torch::Tensor offsets, torch::Tensor pairs);
"""

CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

constexpr int MAXE = 1024;

__global__ void __launch_bounds__(1024)
k_route(const int* __restrict__ ids, int P, int E, int Umax, int* __restrict__ experts, int* __restrict__ offsets,
        int* __restrict__ pairs) {
  __shared__ int cnt[MAXE], base[MAXE], uidx[MAXE];
  const int t = threadIdx.x;
  cnt[t] = 0;
  __syncthreads();
  for (int p = t; p < P; p += blockDim.x) atomicAdd(&cnt[ids[p]], 1);   // counts only: order-independent
  __syncthreads();
  // inclusive scans of the counts and of the touched flags (Hillis-Steele over MAXE entries)
  int c = t < E ? cnt[t] : 0, f = c > 0 ? 1 : 0;
  base[t] = c;
  uidx[t] = f;
  __syncthreads();
  for (int o = 1; o < MAXE; o <<= 1) {
    const int b = t >= o ? base[t - o] : 0, u = t >= o ? uidx[t - o] : 0;
    __syncthreads();
    base[t] += b;
    uidx[t] += u;
    __syncthreads();
  }
  const int U = uidx[MAXE - 1];
  // pair p goes to its expert's base plus its rank among the earlier pairs of the same expert
  for (int p = t; p < P; p += blockDim.x) {
    const int e = ids[p];
    int r = 0;
    for (int q = 0; q < p; ++q) r += ids[q] == e;
    pairs[base[e] - cnt[e] + r] = p;
  }
  if (t < E && c > 0) {
    experts[uidx[t] - 1] = t;
    offsets[uidx[t] - 1] = base[t] - c;
  }
  for (int u = U + t; u < Umax; u += blockDim.x) { experts[u] = -1; offsets[u + 1] = P; }
  if (t == 0) offsets[U] = P;
}

void route(torch::Tensor ids, int64_t num_experts, torch::Tensor experts, torch::Tensor offsets, torch::Tensor pairs) {
  const int P = (int)ids.numel(), E = (int)num_experts, Umax = (int)experts.numel();
  TORCH_CHECK(E <= MAXE && offsets.numel() == Umax + 1 && pairs.numel() == P, "shape");
  k_route<<<1, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(ids.data_ptr<int>(), P, E, Umax, experts.data_ptr<int>(),
                                                          offsets.data_ptr<int>(), pairs.data_ptr<int>());
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "launch");
}
"""


def build():
    return load_inline(name="sm120fp4_moe_route", cpp_sources=CPP, cuda_sources=CUDA, functions=["route"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a"], verbose=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports") / f"moe-w4a16-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    ap.add_argument("--baseline", type=Path, default=Path("reports/moe-baseline-rtx5090-2026-09-29.json"))
    a = ap.parse_args(argv)
    mr, m1, m2 = build(), fc1.build(), fc2.build()
    dev = torch.device("cuda")
    e, k, h, i = 128, 8, 2048, 768
    w = bench.build(e, h, i, dev)
    q1, s1, q2, s2 = w["q1"].contiguous(), w["s1"].contiguous(), w["q2"].contiguous(), w["s2"].contiguous()
    alpha = torch.ones(e, device=dev)
    base = json.loads(a.baseline.read_text(encoding="utf-8"))
    best = {}
    for r in base["rows"]:
        if "graph_cold_median_us" in r:
            cur = best.get(r["tokens"])
            if cur is None or r["graph_cold_median_us"] < cur[1]:
                best[r["tokens"]] = (r["backend"], r["graph_cold_median_us"])
    report = {"device": torch.cuda.get_device_name(0), "baseline": str(a.baseline), "rows": []}
    print("tokens | router = torch router | normwise vs fp32 | bit-identical x50 | route+FC1+FC2 us (one graph, cold L2) | best existing (same method) | speedup")
    for m in (1, 2, 4, 8, 16):
        g = torch.Generator().manual_seed(1000 + m)
        x = torch.randn(m, h, generator=g).to(device=dev, dtype=torch.bfloat16)
        wts, ids = torch.topk(F.softmax(torch.randn(m, e, generator=g), dim=-1), k, dim=-1)
        wts = (wts / wts.sum(-1, keepdim=True)).float().to(dev).contiguous()
        ids = ids.to(torch.int32).to(dev).contiguous()
        P = m * k
        umax = min(e, P)
        experts = torch.empty(umax, dtype=torch.int32, device=dev)
        offsets = torch.empty(umax + 1, dtype=torch.int32, device=dev)
        pairs = torch.empty(P, dtype=torch.int32, device=dev)
        act = torch.empty(P, i, device=dev, dtype=torch.bfloat16)
        out = torch.empty(m, h, device=dev, dtype=torch.bfloat16)
        wflat = wts.view(-1)

        def layer():
            mr.route(ids, e, experts, offsets, pairs)
            m1.fc1_w4a16(q1, s1, x, experts, offsets, pairs, alpha, act, i, k)
            m2.fc2_w4a16(q2, s2, act, experts, offsets, pairs, wflat, alpha, out, k)

        layer()
        torch.cuda.synchronize()
        te, to, tp = fc1.route(ids)
        u = te.numel()
        same_route = (bool(torch.equal(experts[:u], te)) and bool(torch.equal(offsets[:u + 1], to))
                      and bool(torch.equal(pairs, tp)) and bool((experts[u:] == -1).all()))
        ref = bench.reference(x, w, ids, wts, i, act_quant=False)
        rel = float((out.float() - ref).norm() / ref.norm())
        first = out.clone()
        stable = True
        for _ in range(50):
            layer()
            stable = stable and bool(torch.equal(out, first))
        t = floor.graph_time(layer)
        m1.fc1_set_pdl(True)
        m2.fc2_set_pdl(True)
        layer()
        torch.cuda.synchronize()
        pdl_same = bool(torch.equal(out, first))
        t_pdl = floor.graph_time(layer)
        m1.fc1_set_pdl(False)
        m2.fc2_set_pdl(False)
        bname, bus = best[m]
        row = {"tokens": m, "experts_touched": u, "router_matches_torch": same_route, "rel_err": rel, "bit_stable_50": stable,
               "layer_us": t, "layer_pdl_us": t_pdl, "pdl_output_identical": pdl_same, "best_existing": bname, "best_existing_us": bus, "speedup": bus / t}
        report["rows"].append(row)
        print(f"{m:6d} | {str(same_route):21s} | {rel:16.5f} | {str(stable):17s} | {t:37.1f} | {bname} {bus:.1f} | {bus / t:.2f}x"
              f" || with PDL {t_pdl:.1f} us ({bus / t_pdl:.2f}x), output identical {pdl_same}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"written {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
