#!/usr/bin/env python3
"""Localisation for the INT8 quantiser sweep (vllm_scaled_int8_quant_probe.py, 2026-10-15): the static asymmetric kernel's
codes read one off the division model at exact ties. Fits two models on whole arrays: D = round(fp32(x) / scale) and
R = round(fp32(x) * fp32(1 / scale)), for the static symmetric and static asymmetric kernels.

    PYTHONPATH=<sm120-fp4> python vllm_scaled_int8_quant_diag.py --out <json>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    from vllm import _custom_ops as ops
    gen = torch.Generator().manual_seed(7)
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": []}
    one = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for tokens, hidden in ((1000, 8200), (4097, 1000), (3, 17)):
            x = (torch.randn((tokens, hidden), generator=gen) * 3).to(dtype).cuda()
            for sc, azp_v in ((2.1, None), (0.05, None), (2.1, 54), (0.05, 0), (0.1, -255)):
                scale = torch.tensor([sc], dtype=torch.float32, device="cuda")
                azp = torch.tensor([azp_v], dtype=torch.int32, device="cuda") if azp_v is not None else None
                q, _, _ = ops.scaled_int8_quant(x, scale, azp, symmetric=azp is None)
                xf = x.float()
                d = torch.round(xf / scale); rr = torch.round(xf * (one / scale))
                off = float(azp_v) if azp_v is not None else 0.0
                md = (d.double() + off).clamp(-128, 127).to(torch.int8); mr = (rr.double() + off).clamp(-128, 127).to(torch.int8)
                row = {"dtype": str(dtype).split(".")[-1], "tokens": tokens, "hidden": hidden, "mode": "static-asym" if azp is not None else "static-sym",
                       "scale": sc, "azp": azp_v, "elements": int(q.numel()), "off_division_model": int((q != md).sum()),
                       "off_reciprocal_model": int((q != mr).sum()), "division_vs_reciprocal_differ": int((md != mr).sum())}
                print(json.dumps(row), flush=True)
                report["rows"].append(row)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    sym = [r for r in report["rows"] if r["mode"] == "static-sym"]; asym = [r for r in report["rows"] if r["mode"] == "static-asym"]
    print("static-sym: off division", sum(r["off_division_model"] for r in sym), "off reciprocal", sum(r["off_reciprocal_model"] for r in sym),
          "| static-asym: off division", sum(r["off_division_model"] for r in asym), "off reciprocal", sum(r["off_reciprocal_model"] for r in asym),
          "| elements where the two models differ:", sum(r["division_vs_reciprocal_differ"] for r in report["rows"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
