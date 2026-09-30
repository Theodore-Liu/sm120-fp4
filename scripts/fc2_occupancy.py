"""Stage 2, FC2 at 16 tokens: the occupancy hypothesis, same session.

Groups per tile and skipping empty pair tiles did not move FC2 at 16 random tokens (reports/fc2-groups-*.json). The
kernel's register count limits how many of its blocks an SM holds at once. This builds scripts/fc2_mma_pf.py four ways
and times them against the read of the FC2 codes, with correctness, determinism, and each build's registers per thread
and spill bytes read from the compiled extension (cuobjdump --dump-resource-usage):

- W8: eight warps per block (the kernel as it is), G = 2
- W4: four warps per block, G = 2 and G = 4 (G = 4 keeps sixteen warps per column tile, as W8 G = 2 has)
- W8 min 2: eight warps, __launch_bounds__(256, 2) (the compiler must fit two blocks per SM), G = 2

    PYTHONPATH=. python scripts/fc2_occupancy.py --out reports/fc2-occupancy-<device>-<date>.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import subprocess
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


def resources(module) -> dict[str, dict[str, int]]:
    """Registers and spill bytes per k_fc2_pf instantiation, from the extension's own cubin."""
    tool = shutil.which("cuobjdump") or "/usr/local/cuda-13.2/bin/cuobjdump"
    text = subprocess.run([tool, "--dump-resource-usage", module.__file__], capture_output=True, text=True).stdout
    out, fn = {}, None
    for line in text.splitlines():
        m = re.search(r"Function (\S*k_fc2_pf\S*):", line)
        if m:
            fn = m.group(1)
            continue
        if fn and "REG:" in line:
            reg = int(re.search(r"REG:(\d+)", line).group(1))
            st = re.search(r"STACK:(\d+)", line)
            out[fn] = {"registers": reg, "stack": int(st.group(1)) if st else None}
            fn = None
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    builds = {"W8": fc2p.build(), "W4": fc2p.build(warps=4), "W8min2": fc2p.build(min_blocks=2)}
    m1, fl = fc1.build(), floor.build()
    res = {name: resources(mod) for name, mod in builds.items()}
    dev = torch.device("cuda")
    e, k, h, i = 128, 8, 2048, 768
    w = bench.build(e, h, i, dev)
    q1, s1, q2, s2 = (w[n].contiguous() for n in ("q1", "s1", "q2", "s2"))
    alpha = torch.ones(e, device=dev)
    scratch = torch.zeros(4 * 16 * h, device=dev)
    counters = torch.zeros(h // 16, dtype=torch.int32, device=dev)
    sink = torch.zeros(4, dtype=torch.int32, device=dev)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    variants = [("W8_G2", "W8", 2), ("W4_G2", "W4", 2), ("W4_G4", "W4", 4), ("W8min2_G2", "W8min2", 2)]
    rows = []

    def case(label, m, ids, wts, x):
        experts, offsets, pairs = fc1.route(ids)
        act = torch.empty(ids.numel(), i, device=dev, dtype=torch.bfloat16)
        m1.fc1_w4a16(q1, s1, x, experts, offsets, pairs, alpha, act, i, k)
        wf = wts.reshape(-1).contiguous()
        ref = bench.reference(x, w, ids, wts, i, act_quant=False)
        row = {"routing": label, "tokens": m}
        for vname, bname, g in variants:
            mod = builds[bname]
            out = torch.empty(m, h, device=dev, dtype=torch.bfloat16)
            fn = (lambda mm, gg, o: (lambda: mm.fc2_pf(q2, s2, act, experts, offsets, pairs, wf, alpha, o, scratch,
                                                        counters, k, gg)))(mod, g, out)
            fn()
            torch.cuda.synchronize()
            rel = float((out.float() - ref).norm() / ref.norm())
            first = out.clone()
            stable = True
            for _ in range(50):
                fn()
                stable = stable and bool(torch.equal(out, first))
            row[vname] = {"rel_err": rel, "bit_identical_50": stable, "us": floor.graph_time(fn)}
        ptrs = torch.tensor([q2.data_ptr() + int(t) * h * i // 2 for t in experts.tolist()], dtype=torch.int64, device=dev)
        row["read_fc2_codes_us"] = floor.graph_time(lambda: fl.stream_read(ptrs, h * i // 2, 0, h * i // 2, sms * 4, 256, sink))
        rows.append(row)
        print(label, m, {v: round(row[v]["us"], 1) for v, _, _ in variants}, "read", round(row["read_fc2_codes_us"], 1),
              "stable", all(row[v]["bit_identical_50"] for v, _, _ in variants), flush=True)

    for m in (8, 16):
        g = torch.Generator().manual_seed(1000 + m)
        x = torch.randn(m, h, generator=g).to(device=dev, dtype=torch.bfloat16)
        wts, ids = torch.topk(F.softmax(torch.randn(m, e, generator=g), dim=-1), k, dim=-1)
        wts = (wts / wts.sum(-1, keepdim=True)).float().to(dev).contiguous()
        case("random", m, ids.to(torch.int32).to(dev).contiguous(), wts, x)
    fixed = torch.arange(8, dtype=torch.int32, device=dev) * 16
    x = torch.randn(16, h, generator=torch.Generator().manual_seed(7)).to(device=dev, dtype=torch.bfloat16)
    case("fixed8", 16, fixed.repeat(16, 1).contiguous(), torch.full((16, k), 1.0 / k, device=dev), x)
    print(json.dumps(res, indent=1))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"device": torch.cuda.get_device_name(0), "resources": res, "rows": rows}, indent=1) + "\n",
                     encoding="utf-8")
    print(f"written {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
