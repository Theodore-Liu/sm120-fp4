"""Where the decode MoE layer's time goes: each kernel against a read of exactly its own bytes, and what FC1 costs as
the number of tokens per expert grows with the bytes held fixed.

1. Per batch size (random routing, as everywhere else): router alone; FC1 alone; FC2 alone; and a streaming read of
   exactly the FP4 codes and scales each of FC1 and FC2 consumes for the touched experts. All CUDA-graph replays with L2
   flushed first (scripts/micro_floor.py), so every number carries the same 2.8 us replay overhead.
2. FC1 with the expert set held at 8 experts and every token routed to all 8 (so tokens per expert = batch size): the
   bytes read do not change with the batch, only the arithmetic does. If FC1's time grows with tokens per expert here,
   FC1 is bound by instructions, not memory, at that token count.

    PYTHONPATH=. python scripts/moe_breakdown.py --out reports/moe-breakdown-<device>-<date>.json
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

_here = Path(__file__).resolve().parent
for _name in ("bench_moe_baseline", "micro_floor", "fc1_w4a16", "fc2_w4a16", "moe_w4a16"):
    _spec = importlib.util.spec_from_file_location(_name, _here / f"{_name}.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)
bench = sys.modules["bench_moe_baseline"]
floor = sys.modules["micro_floor"]
fc1 = sys.modules["fc1_w4a16"]
fc2 = sys.modules["fc2_w4a16"]
moe = sys.modules["moe_w4a16"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports") / f"moe-breakdown-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    a = ap.parse_args(argv)
    mr, m1, m2, fl = moe.build(), fc1.build(), fc2.build(), floor.build()
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    e, k, h, i = 128, 8, 2048, 768
    w = bench.build(e, h, i, dev)
    q1, s1, q2, s2 = w["q1"].contiguous(), w["s1"].contiguous(), w["q2"].contiguous(), w["s2"].contiguous()
    alpha = torch.ones(e, device=dev)
    sink = torch.zeros(4, dtype=torch.int32, device=dev)
    fc1_codes, fc1_scales = 2 * i * h // 2, 2 * i * h // 16      # per expert
    fc2_codes, fc2_scales = h * i // 2, h * i // 16

    def read(base_tensor, per_expert, experts_list):
        ptrs = torch.tensor([base_tensor.data_ptr() + x * per_expert for x in experts_list], dtype=torch.int64, device=dev)
        return lambda: fl.stream_read(ptrs, per_expert, 0, per_expert, sms * 4, 256, sink)

    report = {"device": torch.cuda.get_device_name(0), "per_batch": [], "fixed_experts": []}
    print("tokens | experts | router | FC1 | FC1 read (codes+scales) | FC2 | FC2 read (codes+scales) | FC1 eff | FC2 eff  (us, graph replay, cold L2)")
    for m in (1, 2, 4, 8, 16):
        g = torch.Generator().manual_seed(1000 + m)
        x = torch.randn(m, h, generator=g).to(device=dev, dtype=torch.bfloat16)
        wts, ids = torch.topk(F.softmax(torch.randn(m, e, generator=g), dim=-1), k, dim=-1)
        wts = (wts / wts.sum(-1, keepdim=True)).float().to(dev).contiguous()
        ids = ids.to(torch.int32).to(dev).contiguous()
        P, umax = m * k, min(e, m * k)
        experts = torch.empty(umax, dtype=torch.int32, device=dev)
        offsets = torch.empty(umax + 1, dtype=torch.int32, device=dev)
        pairs = torch.empty(P, dtype=torch.int32, device=dev)
        act = torch.empty(P, i, device=dev, dtype=torch.bfloat16)
        out = torch.empty(m, h, device=dev, dtype=torch.bfloat16)
        mr.route(ids, e, experts, offsets, pairs)
        torch.cuda.synchronize()
        touched = sorted(set(ids.view(-1).tolist()))
        t_route = floor.graph_time(lambda: mr.route(ids, e, experts, offsets, pairs))
        t_fc1 = floor.graph_time(lambda: m1.fc1_w4a16(q1, s1, x, experts, offsets, pairs, alpha, act, i, k))
        t_fc2 = floor.graph_time(lambda: m2.fc2_w4a16(q2, s2, act, experts, offsets, pairs, wts.view(-1), alpha, out, k))
        m2.fc2_set_skip_act(True)       # same kernel, activation loads replaced by a constant: the traffic's share
        t_fc2_noact = floor.graph_time(lambda: m2.fc2_w4a16(q2, s2, act, experts, offsets, pairs, wts.view(-1), alpha, out, k))
        m2.fc2_set_skip_act(False)
        r1c, r1s = read(q1, fc1_codes, touched), read(s1, fc1_scales, touched)
        r2c, r2s = read(q2, fc2_codes, touched), read(s2, fc2_scales, touched)
        t_r1 = floor.graph_time(lambda: (r1c(), r1s()))
        t_r2 = floor.graph_time(lambda: (r2c(), r2s()))
        row = {"tokens": m, "experts": len(touched), "router_us": t_route, "fc1_us": t_fc1, "fc1_read_us": t_r1,
               "fc2_us": t_fc2, "fc2_no_act_loads_us": t_fc2_noact, "fc2_read_us": t_r2}
        report["per_batch"].append(row)
        print(f"{m:6d} | {len(touched):7d} | {t_route:6.1f} | {t_fc1:5.1f} | {t_r1:23.1f} | {t_fc2:5.1f} (no act loads {t_fc2_noact:5.1f}) | {t_r2:23.1f} | "
              f"{t_r1 / t_fc1:7.2f} | {t_r2 / t_fc2:7.2f}")

    print("\nFC1 with 8 experts fixed, every token routed to all 8 (bytes constant, tokens per expert = batch):")
    fixed = torch.arange(8, dtype=torch.int32, device=dev) * 16             # experts 0, 16, ..., 112
    for m in (1, 2, 4, 8, 16):
        x = torch.randn(m, h, generator=torch.Generator().manual_seed(7)).to(device=dev, dtype=torch.bfloat16)
        ids = fixed.repeat(m, 1).contiguous()
        P, umax = m * 8, min(e, m * 8)
        experts = torch.empty(umax, dtype=torch.int32, device=dev)
        offsets = torch.empty(umax + 1, dtype=torch.int32, device=dev)
        pairs = torch.empty(P, dtype=torch.int32, device=dev)
        act = torch.empty(P, i, device=dev, dtype=torch.bfloat16)
        mr.route(ids, e, experts, offsets, pairs)
        torch.cuda.synchronize()
        t = floor.graph_time(lambda: m1.fc1_w4a16(q1, s1, x, experts, offsets, pairs, alpha, act, i, 8))
        report["fixed_experts"].append({"tokens_per_expert": m, "fc1_us": t})
        print(f"  tokens per expert {m:2d}: FC1 {t:6.1f} us")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"written {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
