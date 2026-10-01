"""Drive routing, FC1 and each FC2 in sequence, for Nsight Compute to profile the FC2 kernels in the cache state the layer
leaves them (run under `ncu --cache-control none`, so the profiler does not flush L2 between kernels).

Synthetic weights of the Qwen3-30B-A3B shape (hidden 2048, expert 768, 128 experts, top 8); the bytes each kernel moves
and its L2 hit rate depend on the shapes and the routing, not on the values. One routing per run.

    ncu --cache-control none --clock-control none -k regex:k_fc2 --metrics <...> \
        python scripts/ncu_fc2_in_layer.py --tokens 16 --routing random
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_here = Path(__file__).resolve().parent
mods = {}
for _name in ("bench_moe_baseline", "fc1_w4a16", "fc1_mma", "fc2_mma_pf", "moe_w4a16", "moe_layer", "fc2_cols32"):
    _spec = importlib.util.spec_from_file_location(_name, _here / f"{_name}.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)
    mods[_name] = _mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--routing", choices=("random", "fixed8"), default="random")
    a = ap.parse_args(argv)
    bench, fc1, fc1m, pf, moe, layer_mod, c32m = (mods[n] for n in ("bench_moe_baseline", "fc1_w4a16", "fc1_mma",
                                                                    "fc2_mma_pf", "moe_w4a16", "moe_layer", "fc2_cols32"))
    dev = torch.device("cuda")
    e_n, k, h, i, m = 128, 8, 2048, 768, a.tokens
    w = bench.build(e_n, h, i, dev)
    q1, s1, q2, s2 = (w[n].contiguous() for n in ("q1", "s1", "q2", "s2"))
    alpha = torch.ones(e_n, device=dev)
    mr, m1, m1m, m2p, c32 = moe.build(), fc1.build(), fc1m.build(), pf.build(), c32m.build()
    c32one = c32m.build(one_block=True)   # the same kernel padded to one block per SM (a control)
    g = torch.Generator().manual_seed(1000 + m)
    x = torch.randn(m, h, generator=g).to(device=dev, dtype=torch.bfloat16)
    if a.routing == "random":
        wts, ids = torch.topk(F.softmax(torch.randn(m, e_n, generator=g), dim=-1), k, dim=-1)
        wts = (wts / wts.sum(-1, keepdim=True)).float().to(dev).contiguous()
        ids = ids.to(torch.int32).to(dev).contiguous()
    else:
        ids = (torch.arange(k, dtype=torch.int32, device=dev) * (e_n // k)).repeat(m, 1).contiguous()
        wts = torch.full((m, k), 1.0 / k, device=dev)
    P, umax = m * k, min(e_n, m * k)
    experts = torch.empty(umax, dtype=torch.int32, device=dev)
    offsets = torch.empty(umax + 1, dtype=torch.int32, device=dev)
    pairs = torch.empty(P, dtype=torch.int32, device=dev)
    act = torch.empty(P, i, device=dev, dtype=torch.bfloat16)
    out = torch.empty(m, h, device=dev, dtype=torch.bfloat16)
    scratch = torch.zeros(4 * 16 * h, device=dev)
    counters = torch.zeros(h // 16, dtype=torch.int32, device=dev)
    wflat = wts.reshape(-1).contiguous()
    f1, f2 = layer_mod.choice(m)
    big = torch.empty(256 * 1024 * 1024 // 4, device=dev)

    def prefix():
        big.zero_()                                   # evict L2 before the sequence, as the timed runs do
        mr.route(ids, e_n, experts, offsets, pairs)
        if f1 == "cuda_core":
            m1.fc1_w4a16(q1, s1, x, experts, offsets, pairs, alpha, act, i, k)
        else:
            m1m.fc1_mma(q1, s1, x, experts, offsets, pairs, alpha, act, i, k)

    groups = 1 if f2 == "prefetch" else 2
    prefix()
    m2p.fc2_pf(q2, s2, act, experts, offsets, pairs, wflat, alpha, out, scratch, counters, k, groups)   # k_fc2_pf
    prefix()
    c32.fc2_c32(q2, s2, act, experts, offsets, pairs, wflat, alpha, out, scratch, counters, k, 4)        # k_fc2_c32
    prefix()
    c32one.fc2_c32(q2, s2, act, experts, offsets, pairs, wflat, alpha, out, scratch, counters, k, 4)     # k_fc2_c32, one block per SM
    torch.cuda.synchronize()
    print(f"ran tokens={m} routing={a.routing} fc2 prefetch groups={groups}, cols32 groups=4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
