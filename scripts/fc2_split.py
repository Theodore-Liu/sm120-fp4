"""Stage 2, FC2 at 16 tokens: is it the loads or the arithmetic? Two timing variants, same session.

No performance counters on this machine, so the two sides are separated by construction (scripts/fc2_mma_pf.py modes):
"loads" keeps every weight, scale and activation load and replaces the decode and MMAs with a fold of the loaded
values; "math" keeps the decode and MMAs and makes the codes from the row index, reading no weights. Both give wrong
answers by design and are timing instruments only. "loads_contig" reads the same bytes as "loads" at lane stride
(512 contiguous bytes of codes per warp instruction), to see whether the load pattern or the load count costs. Each is timed beside the full kernel (G = 2, eight warps) and a
stream read of the FC2 codes; the full kernel is also checked against the fp32 reference.

    PYTHONPATH=. python scripts/fc2_split.py --out reports/fc2-split-<device>-<date>.json
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
for _name in ("bench_moe_baseline", "micro_floor", "fc1_w4a16", "fc2_mma", "fc2_mma_pf"):
    _spec = importlib.util.spec_from_file_location(_name, _here / f"{_name}.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)
bench = sys.modules["bench_moe_baseline"]
floor = sys.modules["micro_floor"]
fc1 = sys.modules["fc1_w4a16"]
fc2p = sys.modules["fc2_mma_pf"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    builds = {mode: fc2p.build(mode=mode) for mode in ("full", "loads", "math", "loads_contig")}
    m1, fl = fc1.build(), floor.build()
    dev = torch.device("cuda")
    e, k, h, i = 128, 8, 2048, 768
    w = bench.build(e, h, i, dev)
    q1, s1, q2, s2 = (w[n].contiguous() for n in ("q1", "s1", "q2", "s2"))
    alpha = torch.ones(e, device=dev)
    scratch = torch.zeros(4 * 16 * h, device=dev)
    counters = torch.zeros(h // 16, dtype=torch.int32, device=dev)
    sink = torch.zeros(4, dtype=torch.int32, device=dev)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    rows = []

    def case(label, m, ids, wts, x):
        experts, offsets, pairs = fc1.route(ids)
        act = torch.empty(ids.numel(), i, device=dev, dtype=torch.bfloat16)
        m1.fc1_w4a16(q1, s1, x, experts, offsets, pairs, alpha, act, i, k)
        wf = wts.reshape(-1).contiguous()
        row = {"routing": label, "tokens": m}
        for mode, mod in builds.items():
            out = torch.empty(m, h, device=dev, dtype=torch.bfloat16)
            fn = (lambda mm, o: (lambda: mm.fc2_pf(q2, s2, act, experts, offsets, pairs, wf, alpha, o, scratch, counters,
                                                    k, 2)))(mod, out)
            fn()
            torch.cuda.synchronize()
            row[mode] = {"us": floor.graph_time(fn)}
            if mode == "full":
                ref = bench.reference(x, w, ids, wts, i, act_quant=False)
                row[mode]["rel_err"] = float((out.float() - ref).norm() / ref.norm())
        ptrs = torch.tensor([q2.data_ptr() + int(t) * h * i // 2 for t in experts.tolist()], dtype=torch.int64, device=dev)
        row["read_fc2_codes_us"] = floor.graph_time(lambda: fl.stream_read(ptrs, h * i // 2, 0, h * i // 2, sms * 4, 256, sink))
        rows.append(row)
        print(label, m, {mode: round(row[mode]["us"], 1) for mode in builds}, "read", round(row["read_fc2_codes_us"], 1),
              "full err", round(row["full"]["rel_err"], 5), flush=True)

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
    a.out.write_text(json.dumps({"device": torch.cuda.get_device_name(0), "rows": rows}, indent=1) + "\n", encoding="utf-8")
    print(f"written {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
