"""Stage 2: where the layer's time goes at 8 and 16 tokens, every piece and Marlin timed by one method in one session.

scripts/moe_layer.py compared the layer against Marlin from an earlier session's baseline report, timed there by
bench_moe_baseline.graph_timed. This times, with micro_floor.graph_time (graph replay, L2 flushed) and in the same
session: the GPU router alone, the chosen FC1 alone, the chosen FC2 alone, the whole layer (PDL), a stream read of the
FC1 and FC2 codes the batch touches, and vLLM's Marlin W4A16 MoE on the same weights and routing.

    PYTHONPATH=. python scripts/moe_layer_breakdown.py --out reports/moe-layer-breakdown-<device>-<date>.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_here = Path(__file__).resolve().parent
for _name in ("bench_moe_baseline", "micro_floor", "fc1_w4a16", "fc2_w4a16", "fc1_mma", "fc2_mma", "fc2_mma_pf",
              "moe_w4a16", "moe_layer"):
    _spec = importlib.util.spec_from_file_location(_name, _here / f"{_name}.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)
bench = sys.modules["bench_moe_baseline"]
floor = sys.modules["micro_floor"]
fc1 = sys.modules["fc1_w4a16"]
fc2 = sys.modules["fc2_w4a16"]
fc1m = sys.modules["fc1_mma"]
fc2p = sys.modules["fc2_mma_pf"]
moe = sys.modules["moe_w4a16"]
layer_mod = sys.modules["moe_layer"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hidden", type=int, default=2048, help="hidden size; 2816 for the Gemma-4-26B-A4B shape")
    ap.add_argument("--inter", type=int, default=768, help="expert intermediate size; 704 for the Gemma-4-26B-A4B shape")
    a = ap.parse_args(argv)
    warps2, decode2 = layer_mod.fc2_warps(a.hidden, a.inter), layer_mod.fc2_decode(a.hidden, a.inter)   # the layer's own FC2 build at this shape
    mr, m1, m2, m1m, m2p, fl = moe.build(), fc1.build(), fc2.build(), fc1m.build(), fc2p.build(warps=warps2, decode=decode2), floor.build()
    dev = torch.device("cuda")
    e, k, h, i = 128, 8, a.hidden, a.inter
    print(f"shape: {e} experts, top-{k}, hidden {h}, intermediate {i}")
    w = bench.build(e, h, i, dev)
    q1, s1, q2, s2 = (w[n].contiguous() for n in ("q1", "s1", "q2", "s2"))
    alpha = torch.ones(e, device=dev)
    scratch = torch.zeros(4 * 16 * h, device=dev)
    counters = torch.zeros(h // 16, dtype=torch.int32, device=dev)
    sink = torch.zeros(4, dtype=torch.int32, device=dev)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    marlin_call, why = bench.marlin_setup(w, e, h, i, dev)
    if marlin_call is None:
        print(f"marlin unavailable: {why}")
    for fn in (m1.fc1_set_pdl, m2.fc2_set_pdl, m1m.fc1_mma_set_pdl, m2p.fc2_pf_set_pdl):
        fn(True)
    rows = []

    def case(label, m, ids, wts, x):
        P = m * k
        umax = min(e, P)
        experts = torch.empty(umax, dtype=torch.int32, device=dev)
        offsets = torch.empty(umax + 1, dtype=torch.int32, device=dev)
        pairs = torch.empty(P, dtype=torch.int32, device=dev)
        act = torch.empty(P, i, device=dev, dtype=torch.bfloat16)
        out = torch.empty(m, h, device=dev, dtype=torch.bfloat16)
        outm = torch.empty(m, h, device=dev, dtype=torch.bfloat16)
        wflat = wts.reshape(-1).contiguous()
        f1, f2 = layer_mod.choice(m, h, i)

        def route():
            mr.route(ids, e, experts, offsets, pairs)

        def run_fc1():
            if f1 == "cuda_core":
                m1.fc1_w4a16(q1, s1, x, experts, offsets, pairs, alpha, act, i, k)
            else:
                m1m.fc1_mma(q1, s1, x, experts, offsets, pairs, alpha, act, i, k)

        def run_fc2():
            if f2 == "cuda_core":
                m2.fc2_w4a16(q2, s2, act, experts, offsets, pairs, wflat, alpha, out, k)
            else:
                m2p.fc2_pf(q2, s2, act, experts, offsets, pairs, wflat, alpha, out, scratch, counters, k,
                           1 if f2 == "prefetch" else 2)

        def layer():
            route()
            run_fc1()
            run_fc2()

        layer()
        torch.cuda.synchronize()
        ref = bench.reference(x, w, ids, wts, i, act_quant=False)
        rel = float((out.float() - ref).norm() / ref.norm())
        touched = [int(v) for v in experts.tolist() if v >= 0]
        p1 = torch.tensor([q1.data_ptr() + t * 2 * i * h // 2 for t in touched], dtype=torch.int64, device=dev)
        p2 = torch.tensor([q2.data_ptr() + t * h * i // 2 for t in touched], dtype=torch.int64, device=dev)
        row = {"routing": label, "tokens": m, "fc1": f1, "fc2": f2, "experts_touched": len(touched), "rel_err": rel,
               "route_us": floor.graph_time(route), "fc1_us": floor.graph_time(run_fc1), "fc2_us": floor.graph_time(run_fc2),
               "layer_us": floor.graph_time(layer),
               "read_fc1_codes_us": floor.graph_time(lambda: fl.stream_read(p1, 2 * i * h // 2, 0, 2 * i * h // 2, sms * 4, 256, sink)),
               "read_fc2_codes_us": floor.graph_time(lambda: fl.stream_read(p2, h * i // 2, 0, h * i // 2, sms * 4, 256, sink))}
        if marlin_call is not None:
            marlin_call(x, ids, wts, outm)
            torch.cuda.synchronize()
            row["marlin_rel_err"] = float((outm.float() - ref).norm() / ref.norm())
            row["marlin_us"] = floor.graph_time(lambda: marlin_call(x, ids, wts, outm))
        rows.append(row)
        print(json.dumps({kk: (round(v, 2) if isinstance(v, float) else v) for kk, v in row.items()}), flush=True)

    for m in (8, 16):
        g = torch.Generator().manual_seed(1000 + m)
        x = torch.randn(m, h, generator=g).to(device=dev, dtype=torch.bfloat16)
        wts, ids = torch.topk(F.softmax(torch.randn(m, e, generator=g), dim=-1), k, dim=-1)
        wts = (wts / wts.sum(-1, keepdim=True)).float().to(dev).contiguous()
        case("random", m, ids.to(torch.int32).to(dev).contiguous(), wts, x)
    fixed = torch.arange(8, dtype=torch.int32, device=dev) * 16
    x = torch.randn(16, h, generator=torch.Generator().manual_seed(7)).to(device=dev, dtype=torch.bfloat16)
    case("fixed8", 16, fixed.repeat(16, 1).contiguous(), torch.full((16, k), 1.0 / k, device=dev), x)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"device": torch.cuda.get_device_name(0), "shape": {"hidden": h, "inter": i}, "fc2_warps": warps2, "fc2_decode": decode2, "rows": rows}, indent=1) + "\n", encoding="utf-8")
    print(f"written {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
