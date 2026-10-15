#!/usr/bin/env python3
"""Localisation 2 for the fused RMSNorm + per-block quant sweep (2026-10-15): the per-block path reads codes off the step
model in every dtype (fp32 included), not explained by any rms offset. Reports, for one case, which rows and groups are off
and the kernel's group scales against the model's; then fits alternative models on whole arrays:
  M0  model as registered: y = f32(scalar_t(x*rms)*w); s_g = max(absmax_g(y)/448, floor); q = y * (1/s_g)
  M1  q = y / s_g
  M2  s_g without the floor
  M3  absmax over x*rms (no weight) for the scale, codes from y
  M4  y = scalar_t(x*rms*w) single rounding
  M5  as M0 but the per-group scale replaced by the per-token scale (one scale per row)

    PYTHONPATH=<sm120-fp4> python vllm_fused_rmsnorm_quant_diag2.py --out <json>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

FP8 = torch.float8_e4m3fn
EPS = 1e-6
FLOOR = 1.0 / (448.0 * 512.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    from vllm import _custom_ops as ops
    dev = "cuda"
    f32 = lambda v: torch.tensor(v, dtype=torch.float32, device=dev)  # noqa: E731
    gen = torch.Generator().manual_seed(9)
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": []}

    def conv(v):
        return torch.minimum(torch.maximum(v, f32(-448.0)), f32(448.0)).to(FP8).view(torch.uint8)

    for dtype, tokens, hidden, group in ((torch.float32, 1000, 8192, 64), (torch.bfloat16, 1000, 8192, 64), (torch.bfloat16, 1000, 16384, 128), (torch.float16, 1000, 8192, 128)):
        x = (torch.randn((tokens, hidden), generator=gen) * 2).to(dtype).cuda()
        w = (torch.rand((hidden,), generator=gen) * 1.5 + 0.25).to(dtype).cuda()
        q, s = ops.rms_norm_per_block_quant(x, w, EPS, FP8, [1, group], scale_ub=None, residual=None)
        qb = q.view(torch.uint8); s = s.reshape(tokens, hidden // group).float()
        xf = x.float()
        rms = torch.rsqrt((xf.double() ** 2).sum(-1) / hidden + EPS).float()
        if dtype == torch.float32:
            y = xf * rms[:, None] * w.float(); y1 = y
        else:
            y = ((xf * rms[:, None]).to(dtype).float() * w.float()).to(dtype).float()
            y1 = (xf * rms[:, None] * w.float()).to(dtype).float()
        g = y.reshape(tokens, -1, group); g1 = y1.reshape(tokens, -1, group)
        am = g.abs().amax(-1)
        s0 = torch.maximum(am / f32(448.0), f32(FLOOR))
        s2 = am / f32(448.0)
        am3 = (xf * rms[:, None]).reshape(tokens, -1, group).abs().amax(-1)
        s3 = torch.maximum(am3 / f32(448.0), f32(FLOOR))
        s5 = torch.maximum(y.abs().amax(-1, keepdim=True) / f32(448.0), f32(FLOOR)).expand_as(s0)
        models = {
            "M0_registered": conv((g * (f32(1.0) / s0)[:, :, None]).reshape(tokens, hidden)),
            "M1_division": conv((g / s0[:, :, None]).reshape(tokens, hidden)),
            "M2_no_floor": conv((g * (f32(1.0) / s2)[:, :, None]).reshape(tokens, hidden)),
            "M3_absmax_without_weight": conv((g * (f32(1.0) / s3)[:, :, None]).reshape(tokens, hidden)),
            "M4_single_rounding_y": conv((g1 * (f32(1.0) / torch.maximum(g1.abs().amax(-1) / f32(448.0), f32(FLOOR)))[:, :, None]).reshape(tokens, hidden)),
            "M5_per_token_scale": conv((g * (f32(1.0) / s5)[:, :, None]).reshape(tokens, hidden)),
            "M6_kernel_scale_recip": conv((g * (f32(1.0) / s)[:, :, None]).reshape(tokens, hidden)),
            "M7_kernel_scale_div": conv((g / s[:, :, None]).reshape(tokens, hidden)),
        }
        row = {"dtype": str(dtype).split(".")[-1], "tokens": tokens, "hidden": hidden, "group": group, "elements": int(qb.numel()),
               "scales_off_M0": int((s != s0).sum()), "scales_off_M3": int((s != s3).sum()), "scales_off_M2": int((s != s2).sum()),
               "scale_ulps_from_M0_max": int((s.view(torch.int32) - s0.view(torch.int32)).abs().max())}
        for k, v in models.items():
            row["off_" + k] = int((qb != v).sum())
        off_rows = ((qb != models["M0_registered"]).reshape(tokens, -1, group).any(-1)).sum(-1)
        row["rows_with_off_groups"] = int((off_rows > 0).sum()); row["off_groups_total"] = int(off_rows.sum())
        bad = (qb != models["M0_registered"]).nonzero()[:3].tolist()
        row["examples"] = [{"row": r_, "col": c_, "group": c_ // group, "got": int(qb[r_, c_]), "M0": int(models["M0_registered"][r_, c_]),
                            "kernel_scale": float(s[r_, c_ // group]), "model_scale": float(s0[r_, c_ // group]), "y": float(y[r_, c_])} for r_, c_ in bad]
        print(json.dumps(row), flush=True)
        report["rows"].append(row)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
