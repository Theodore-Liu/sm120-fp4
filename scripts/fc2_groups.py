"""Stage 2, FC2 at 16 tokens: how many groups per column tile (scripts/fc2_mma_pf.py, G = 1 to 4), same session.

At 16 random tokens FC2 took 87.9 us against a 49.9 us read of its codes (reports/moe-layer-breakdown-*.json) with two
groups per tile. The kernel already takes G up to 4 (its scratch is sized for four): each group's warps walk a quarter
of the experts, the grid grows from 128 to 512 blocks on 170 SMs, and the last group to finish a tile adds the partials
in group order. This times G = 1, 2, 3, 4 and v1 against the read of the codes, with correctness and determinism, and
G = 1, 2 built with PF_SKIP_EMPTY (the tensor-core work of a pair tile the expert does not fill is skipped), whose
output must equal the plain kernel's bit for bit.

    PYTHONPATH=. python scripts/fc2_groups.py --out reports/fc2-groups-<device>-<date>.json
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
fc2m = sys.modules["fc2_mma"]
fc2p = sys.modules["fc2_mma_pf"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    pf, pfs, v1, m1, fl = fc2p.build(), fc2p.build(skip_empty=True), fc2m.build(), fc1.build(), floor.build()
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
        ref = bench.reference(x, w, ids, wts, i, act_quant=False)
        row = {"routing": label, "tokens": m, "experts_touched": int(experts.numel())}
        variants = {"v1": lambda o: v1.fc2_mma(q2, s2, act, experts, offsets, pairs, wf, alpha, o, k)}
        for g in (1, 2, 3, 4):
            variants[f"G{g}"] = (lambda gg: lambda o: pf.fc2_pf(q2, s2, act, experts, offsets, pairs, wf, alpha, o,
                                                                scratch, counters, k, gg))(g)
        for g in (1, 2):
            variants[f"G{g}_skip"] = (lambda gg: lambda o: pfs.fc2_pf(q2, s2, act, experts, offsets, pairs, wf, alpha, o,
                                                                      scratch, counters, k, gg))(g)
        outs = {}
        for name, fn in variants.items():
            out = torch.empty(m, h, device=dev, dtype=torch.bfloat16)
            fn(out)
            torch.cuda.synchronize()
            rel = float((out.float() - ref).norm() / ref.norm())
            first = out.clone()
            outs[name] = first
            stable = all(bool(torch.equal((fn(out), out)[1], first)) for _ in range(50))
            row[name] = {"rel_err": rel, "bit_identical_50": stable, "us": floor.graph_time(lambda: fn(out))}
        ptrs = torch.tensor([q2.data_ptr() + int(t) * h * i // 2 for t in experts.tolist()], dtype=torch.int64, device=dev)
        row["read_fc2_codes_us"] = floor.graph_time(lambda: fl.stream_read(ptrs, h * i // 2, 0, h * i // 2, sms * 4, 256, sink))
        row["skip_equals_plain"] = {g: bool(torch.equal(outs[g], outs[g + "_skip"])) for g in ("G1", "G2")}
        rows.append(row)
        print(label, m, {n: round(row[n]["us"], 1) for n in variants}, "read", round(row["read_fc2_codes_us"], 1),
              "stable", all(row[n]["bit_identical_50"] for n in variants), "skip==plain", row["skip_equals_plain"], flush=True)

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
