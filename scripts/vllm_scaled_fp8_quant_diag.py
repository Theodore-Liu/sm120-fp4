#!/usr/bin/env python3
"""Localisation for the FP8 quantiser sweep (vllm_scaled_fp8_quant_probe.py, 2026-10-15): the dynamic per-token kernel's
codes read one e4m3 step off the model `x * fp32(1 / s)` on scattered elements while the per-tensor and static kernels
fit it. Fits candidate forms of the per-token conversion on whole arrays:
  R  x * fp32(1 / s)                      (the probe's model)
  D  x / s                                (a true division)
  A  x * fp32(448 / absmax)               (the reciprocal formed from absmax directly)
  F  fp32(x * fp32(1 / s)) via fma?       (not separable here; omitted)
  H  x * fp32(1 / s) with s un-floored    (the floor only matters for all-zero rows)

    PYTHONPATH=<sm120-fp4> python vllm_scaled_fp8_quant_diag.py --out <json>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

FP8 = torch.float8_e4m3fn
MIN_SCALE = 1.0 / (448.0 * 512.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    from vllm import _custom_ops as ops
    gen = torch.Generator().manual_seed(11)
    dev = "cuda"
    f32 = lambda v: torch.tensor(v, dtype=torch.float32, device=dev)  # noqa: E731

    def conv(x):
        r = torch.maximum(torch.minimum(x, f32(448.0)), f32(-448.0))
        return r.to(FP8).view(torch.uint8)

    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": []}
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for tokens, hidden in ((1000, 1000), (1000, 8200), (4097, 40), (3, 17)):
            x = ((torch.randn((tokens, hidden), generator=gen) * 3).to(dtype)).cuda()
            for mode in ("dyn-token", "dyn-tensor"):
                q, s = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=(mode == "dyn-token"))
                qb = q.view(torch.uint8)
                xf = x.float()
                if mode == "dyn-token":
                    am = xf.abs().amax(dim=-1, keepdim=True)
                else:
                    am = xf.abs().amax().reshape(1, 1)
                s_model = torch.maximum(am / f32(448.0), f32(MIN_SCALE)) if mode == "dyn-token" else am / f32(448.0)
                cands = {
                    "R_x_times_recip": conv(xf * (f32(1.0) / s_model)),
                    "D_x_div_s": conv(xf / s_model),
                    "A_x_times_448_over_absmax": conv(xf * (f32(448.0) / am)),
                    "R_with_returned_scale": conv(xf * (f32(1.0) / s.reshape(-1, 1).float() if mode == "dyn-token" else f32(1.0) / s.reshape(1, 1).float())),
                }
                row = {"dtype": str(dtype).split(".")[-1], "tokens": tokens, "hidden": hidden, "mode": mode, "elements": int(qb.numel()),
                       "scale_off_model": int((s.reshape(-1) != s_model.reshape(-1)).sum())}
                for k, v in cands.items():
                    row["off_" + k] = int((qb != v).sum())
                print(json.dumps(row), flush=True)
                report["rows"].append(row)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for mode in ("dyn-token", "dyn-tensor"):
        rows = [r for r in report["rows"] if r["mode"] == mode]
        print(mode, {k: sum(r[k] for r in rows) for k in rows[0] if k.startswith("off_")}, "scale off", sum(r["scale_off_model"] for r in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
