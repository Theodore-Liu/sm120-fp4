"""FC1 tensor-core kernel: is the bf16 decode bit-identical to the shipped decode? Both builds run on the same routing and weights at a shape; every output is compared element for element.
No timing, so it can run while the GPU is shared.

    PYTHONPATH=. python scripts/fc1_decode_identity.py --hidden 2816 --inter 704
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_here = Path(__file__).resolve().parent.parent / "sm120fp4" / "kernels"
for _name in ("bench_moe_baseline", "fc1_w4a16", "fc1_mma"):
    _spec = importlib.util.spec_from_file_location(_name, _here / f"{_name}.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)
bench = sys.modules["bench_moe_baseline"]
fc1 = sys.modules["fc1_w4a16"]
fc1m = sys.modules["fc1_mma"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--inter", type=int, default=768)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    mods = {d: fc1m.build(decode=d) for d in ("f32", "bf16", "auto")}
    dev = torch.device("cuda")
    e, k, h, i = 128, 8, a.hidden, a.inter
    w = bench.build(e, h, i, dev)
    q1, s1 = w["q1"].contiguous(), w["s1"].contiguous()
    alpha = torch.ones(e, device=dev)
    rows = []
    cases = []
    for m in (1, 2, 4, 8, 16):
        g = torch.Generator().manual_seed(1000 + m)
        x = torch.randn(m, h, generator=g).to(device=dev, dtype=torch.bfloat16)
        _, ids = torch.topk(F.softmax(torch.randn(m, e, generator=g), dim=-1), k, dim=-1)
        cases.append(("random", m, ids.to(torch.int32).to(dev), x))
    fixed = torch.arange(8, dtype=torch.int32, device=dev) * 16
    for m in (1, 8, 16):
        x = torch.randn(m, h, generator=torch.Generator().manual_seed(7)).to(device=dev, dtype=torch.bfloat16)
        cases.append(("fixed8", m, fixed.repeat(m, 1).contiguous(), x))
    ok = True
    for label, m, ids, x in cases:
        experts, offsets, pairs = fc1.route(ids)
        outs = {}
        for d, mod in mods.items():
            act = torch.empty(ids.numel(), i, device=dev, dtype=torch.bfloat16)
            mod.fc1_mma(q1, s1, x, experts, offsets, pairs, alpha, act, i, k)
            torch.cuda.synchronize()
            outs[d] = act
        same_bf16 = bool(torch.equal(outs["f32"], outs["bf16"]))
        same_auto = bool(torch.equal(outs["f32"], outs["auto"]))
        n_diff = int((outs["f32"] != outs["bf16"]).sum())
        rows.append({"routing": label, "tokens": m, "bf16_equals_f32": same_bf16, "auto_equals_f32": same_auto, "elements_differing_bf16": n_diff, "elements": outs["f32"].numel()})
        ok &= same_bf16 and same_auto
        print(f"{label:6s} {m:3d} tokens: bf16 == f32 {same_bf16} ({n_diff} of {outs['f32'].numel()} differ), auto == f32 {same_auto}")
    print("bit-identical on every case:", ok)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"device": torch.cuda.get_device_name(0), "shape": {"hidden": h, "inter": i}, "rows": rows, "all_identical": ok}, indent=1) + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
