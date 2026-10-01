"""Drive the layer (routing, FC1, FC2 as `moe_layer.choice` picks them) and then vLLM's Marlin W4A16 MoE on the same
weights and routing, each after one L2 eviction, for Nsight Compute to time every kernel of both in the cache state its
predecessor leaves (run under `ncu --cache-control none`).

Synthetic weights of the Qwen3-30B-A3B shape (hidden 2048, expert 768, 128 experts, top 8), as scripts/ncu_fc2_in_layer.py:
the time each kernel takes depends on the shapes and the routing, not on the values. One routing and token count per run.

    ncu --cache-control none --clock-control none --metrics gpu__time_duration.sum \
        python scripts/ncu_layer_vs_marlin.py --tokens 16 --routing fixed8
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
for _name in ("bench_moe_baseline", "fc1_w4a16", "fc2_w4a16", "fc1_mma", "fc2_mma_pf", "moe_w4a16", "moe_layer"):
    _spec = importlib.util.spec_from_file_location(_name, _here / f"{_name}.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)
    mods[_name] = _mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--routing", choices=("random", "fixed8"), default="fixed8")
    ap.add_argument("--repeats", type=int, default=3, help="layer-then-Marlin sequences, each after an L2 eviction")
    a = ap.parse_args(argv)
    bench, fc1, fc2, fc1m, pf, moe, layer_mod = (mods[n] for n in ("bench_moe_baseline", "fc1_w4a16", "fc2_w4a16",
                                                                   "fc1_mma", "fc2_mma_pf", "moe_w4a16", "moe_layer"))
    dev = torch.device("cuda")
    e_n, k, h, i, m = 128, 8, 2048, 768, a.tokens
    w = bench.build(e_n, h, i, dev)
    q1, s1, q2, s2 = (w[n].contiguous() for n in ("q1", "s1", "q2", "s2"))
    alpha = torch.ones(e_n, device=dev)
    mr, m1, m2, m1m, m2p = moe.build(), fc1.build(), fc2.build(), fc1m.build(), pf.build()
    for fn in (m1.fc1_set_pdl, m2.fc2_set_pdl, m1m.fc1_mma_set_pdl, m2p.fc2_pf_set_pdl):
        fn(True)
    marlin_call, why = bench.marlin_setup(w, e_n, h, i, dev)
    if marlin_call is None:
        print(f"marlin unavailable: {why}")
        return 2
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
    outm = torch.empty(m, h, device=dev, dtype=torch.bfloat16)
    scratch = torch.zeros(4 * 16 * h, device=dev)
    counters = torch.zeros(h // 16, dtype=torch.int32, device=dev)
    wflat = wts.reshape(-1).contiguous()
    f1, f2 = layer_mod.choice(m)
    big = torch.empty(256 * 1024 * 1024 // 4, device=dev)

    def layer():
        mr.route(ids, e_n, experts, offsets, pairs)
        if f1 == "cuda_core":
            m1.fc1_w4a16(q1, s1, x, experts, offsets, pairs, alpha, act, i, k)
        else:
            m1m.fc1_mma(q1, s1, x, experts, offsets, pairs, alpha, act, i, k)
        if f2 == "cuda_core":
            m2.fc2_w4a16(q2, s2, act, experts, offsets, pairs, wflat, alpha, out, k)
        else:
            m2p.fc2_pf(q2, s2, act, experts, offsets, pairs, wflat, alpha, out, scratch, counters, k, 1)

    for _ in range(a.repeats):
        big.zero_()                   # evict L2 before each sequence, as the timed runs do (the zero kernel is the marker)
        layer()
        big.zero_()
        marlin_call(x, ids, wts, outm)
    torch.cuda.synchronize()
    print(f"ran tokens={m} routing={a.routing} fc1={f1} fc2={f2} repeats={a.repeats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
