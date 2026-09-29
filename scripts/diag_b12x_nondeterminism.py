"""Is FlashInfer b12x_fused_moe(quant_mode="nvfp4") deterministic, and how far is each run from the reference?

Found while building the stage 2 baseline: on the RTX 5090 with FlashInfer 0.6.16.post3, 20 identical calls of the W4A4
path return 20 different outputs, while the W4A16 path and cutlass_fused_moe return one. This script repeats identical
calls, measures each run's normwise error against two references (the NVFP4 reference arithmetic, and the same with the
SwiGLU output rounded to bf16 before it is quantized, which is how b12x stores it), and against the first run, so the
spread can be told apart from a fixed rounding rule.

    PYTHONPATH=. python scripts/diag_b12x_nondeterminism.py --out reports/b12x-nondeterminism-<device>-<date>.json
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
for _name in ("bench_moe_baseline", "diag_b12x_actquant"):
    _spec = importlib.util.spec_from_file_location(_name, _here / f"{_name}.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)
bench = sys.modules["bench_moe_baseline"]
diag = sys.modules["diag_b12x_actquant"]

from sm120fp4 import to_128x4  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports") / f"b12x-nondeterminism-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    ap.add_argument("--tokens", default="1,4,16")
    ap.add_argument("--runs", type=int, default=20)
    a = ap.parse_args(argv)
    import flashinfer
    from flashinfer import fused_moe as fm
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    dev = torch.device("cuda")
    e, k, h, i = 128, 8, 2048, 768
    w = bench.build(e, h, i, dev)
    s1 = convert_sf_to_mma_layout(to_128x4(w["s1"]), m=2 * i, k=h, num_groups=e, sf_vec_size=16)
    s2 = convert_sf_to_mma_layout(to_128x4(w["s2"]), m=h, k=i, num_groups=e, sf_vec_size=16)
    ones = torch.ones(e, device=dev)
    one = torch.ones(1, device=dev)
    report = {"flashinfer": flashinfer.__version__, "device": torch.cuda.get_device_name(0), "runs": a.runs, "cases": []}
    for m in [int(t) for t in a.tokens.split(",")]:
        g = torch.Generator().manual_seed(1000 + m)
        x = torch.randn(m, h, generator=g).to(device=dev, dtype=torch.bfloat16)
        wts, ids = torch.topk(F.softmax(torch.randn(m, e, generator=g), dim=-1), k, dim=-1)
        wts = (wts / wts.sum(-1, keepdim=True)).float().to(dev)
        ids = ids.to(torch.int32).to(dev)
        ref = diag.moe(x, w, ids, wts, i, diag.q_ref, diag.q_ref)
        ref_bf16 = diag.moe(x, w, ids, wts, i, diag.q_ref, diag.q_bf16_scaled)
        outs = []
        for _ in range(a.runs):
            o = fm.b12x_fused_moe(x, w["q1"], s1, w["q2"], s2, ids, wts, e, k, w1_alpha=ones, w2_alpha=ones,
                                  fc2_input_scale=one, quant_mode="nvfp4")
            outs.append(o.float().clone())
        torch.cuda.synchronize()
        nrm = lambda d, r: float(d.norm() / r.norm())  # noqa: E731
        case = {"tokens": m,
                "distinct_outputs": len({o.to(torch.bfloat16).view(torch.int16).cpu().numpy().tobytes() for o in outs}),
                "err_vs_ref": [nrm(o - ref, ref) for o in outs],
                "err_vs_ref_bf16_swiglu": [nrm(o - ref_bf16, ref_bf16) for o in outs],
                "diff_vs_first_run": [nrm(o - outs[0], outs[0]) for o in outs[1:]],
                # which output rows move between runs: per token, the largest normwise change against run 0
                "per_token_max_change": [max(float((o[t] - outs[0][t]).norm() / outs[0][t].norm()) for o in outs[1:]) for t in range(m)]}
        report["cases"].append(case)
        ev, eb = case["err_vs_ref_bf16_swiglu"], case["diff_vs_first_run"]
        print(f"M={m:2d}: {case['distinct_outputs']}/{a.runs} distinct; err vs bf16-SwiGLU reference min {min(ev):.4f} "
              f"median {sorted(ev)[len(ev)//2]:.4f} max {max(ev):.4f}; run-to-run max {max(eb):.4f}; "
              f"per-token max change {[round(v, 3) for v in case['per_token_max_change']]}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"written {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
