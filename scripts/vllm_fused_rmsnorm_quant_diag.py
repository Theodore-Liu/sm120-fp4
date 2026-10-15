#!/usr/bin/env python3
"""Localisation for the fused RMSNorm + quant sweep (2026-10-15): 94 cases read a few codes off the step model, mostly at
hidden 8192 and 16384 with 1000 tokens. The model took rms as any fp32 within two ulps of the exact rsqrt; cub's fp32 sum
of 16384 squares can land many ulps from the float64 sum. This diag scans, per row, the rms offset in fp32 ulps from the
exact value over [-256, 256] and reports whether one offset per row makes every code of the row match (the rms rounding
explains the reading) and how large the needed offsets are; it also emulates the per-thread sequential fp32 partial sums
(thread i sums elements i, i + blockDim, ...) to see how far that alone moves rms.

    PYTHONPATH=<sm120-fp4> python vllm_fused_rmsnorm_quant_diag.py --out <json>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

FP8 = torch.float8_e4m3fn
EPS = 1e-6
FP8_MIN_SCALE = 1.0 / (448.0 * 512.0)
INT8_MIN_SCALE = 1.1920928955078125e-07


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    from vllm import _custom_ops as ops
    dev = "cuda"
    f32 = lambda v: torch.tensor(v, dtype=torch.float32, device=dev)  # noqa: E731
    gen = torch.Generator().manual_seed(3)
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": []}

    def ulps_apart(a_, b_):
        return (a_.view(torch.int32).long() - b_.view(torch.int32).long())

    def codes_for(x, res, w, dtype, quant, rms, group):
        xf = x.float() + (res.float() if res is not None else 0.0)
        if dtype == torch.float32:
            y = xf * rms[:, None] * w.float()
        else:
            y = ((xf * rms[:, None]).to(dtype).float() * w.float()).to(dtype).float()
        t, h = y.shape
        if group is None:
            am = y.abs().amax(dim=-1, keepdim=True)
            if quant == "fp8":
                s = torch.maximum(am / f32(448.0), f32(FP8_MIN_SCALE))
                return torch.minimum(torch.maximum(y / s, f32(-448.0)), f32(448.0)).to(FP8).view(torch.uint8)
            s = torch.maximum(am / f32(127.0), f32(INT8_MIN_SCALE))
            return torch.round(y * (f32(1.0) / s)).clamp(-128, 127).to(torch.int8).view(torch.uint8)
        g = y.reshape(t, h // group, group)
        s = torch.maximum(g.abs().amax(-1) / f32(448.0), f32(FP8_MIN_SCALE))
        return torch.minimum(torch.maximum(g * (f32(1.0) / s)[:, :, None], f32(-448.0)), f32(448.0)).reshape(t, h).to(FP8).view(torch.uint8)

    for dtype, tokens, hidden, quant, residual, group in ((torch.float32, 200, 16384, "fp8", False, None),
                                                          (torch.float32, 200, 16384, "int8", False, None),
                                                          (torch.bfloat16, 200, 8192, "fp8", True, 64),
                                                          (torch.float16, 200, 16384, "fp8", True, 128),
                                                          (torch.bfloat16, 200, 1023, "fp8", False, None)):
        x = (torch.randn((tokens, hidden), generator=gen) * 2).to(dtype).cuda()
        res = torch.randn((tokens, hidden), generator=gen).to(dtype).cuda() if residual else None
        w = (torch.rand((hidden,), generator=gen) * 1.5 + 0.25).to(dtype).cuda()
        r_copy = res.clone() if res is not None else None
        if group is None:
            q, s = ops.rms_norm_dynamic_per_token_quant(x, w, EPS, FP8 if quant == "fp8" else torch.int8, scale_ub=None, residual=r_copy)
        else:
            q, s = ops.rms_norm_per_block_quant(x, w, EPS, FP8, [1, group], scale_ub=None, residual=r_copy)
        qb = q.view(torch.uint8)
        xd = x.double() + (res.double() if res is not None else 0.0)
        rms_exact = torch.rsqrt((xd * xd).sum(-1) / hidden + EPS).float()
        # per-thread sequential fp32 partial sums, blockDim = min(hidden, 1024)
        bd = min(hidden, 1024)
        xf = (x.float() + (res.float() if res is not None else 0.0))
        pad = (-hidden) % bd
        xp = torch.nn.functional.pad(xf, (0, pad)).reshape(tokens, -1, bd)
        part = torch.zeros((tokens, bd), dtype=torch.float32, device=dev)
        for k in range(xp.shape[1]):
            part = part + xp[:, k, :] * xp[:, k, :]
        ss_thread = part.double().sum(-1)                      # exact combination of the fp32 partials
        rms_thread = torch.rsqrt(ss_thread / hidden + EPS).float()
        # scan offsets
        base_off = (qb != codes_for(x, res, w, dtype, quant, rms_exact, group)).sum(-1)
        found = torch.full((tokens,), 10 ** 6, dtype=torch.long, device=dev)
        cur = rms_exact.clone()
        for sign in (1, -1):
            cur = rms_exact.clone()
            for k in range(1, 257):
                cur = torch.nextafter(cur, torch.full_like(cur, float("inf") if sign > 0 else float("-inf")))
                off = (qb != codes_for(x, res, w, dtype, quant, cur, group)).sum(-1)
                hit = (off == 0) & (found.abs() > k)
                found = torch.where(hit, torch.full_like(found, sign * k), found)
        found = torch.where(base_off == 0, torch.zeros_like(found), found)
        unresolved = int((found.abs() == 10 ** 6).sum())
        f_ok = found[found.abs() < 10 ** 6]
        row = {"dtype": str(dtype).split(".")[-1], "tokens": tokens, "hidden": hidden, "quant": quant, "residual": residual, "group": group,
               "rows_off_at_exact_rms": int((base_off > 0).sum()), "rows_unresolved_within_256_ulps": unresolved,
               "needed_offset_ulps_max_abs": int(f_ok.abs().max()) if f_ok.numel() else 0,
               "needed_offset_ulps_mean_abs": round(float(f_ok.abs().float().mean()), 2) if f_ok.numel() else 0.0,
               "thread_partial_rms_ulps_from_exact_mean_abs": round(float(ulps_apart(rms_thread, rms_exact).abs().float().mean()), 2),
               "rows_where_thread_partial_rms_matches_needed": int((ulps_apart(rms_thread, rms_exact) == found).sum())}
        print(json.dumps(row), flush=True)
        report["rows"].append(row)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
