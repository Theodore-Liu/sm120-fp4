#!/usr/bin/env python3
"""Localisation for the group FP8 quantiser probe (2026-10-15): the instrument check read 880 of 2^21 codes off the model
x * fp32(1 / y_s) with every scale matching. Fits candidate conversions on whole arrays for the register path (bf16, group
128) and the shared-memory path (group 64; fp32 input):
  R  x * fp32(1 / y_s)       D  x / y_s       A  x * fp32(448 / absmax)       M  x * fp32(1 / y_s) with y_s = absmax * fp32(1/448)

    PYTHONPATH=<sm120-fp4> python vllm_group_quant_fp8_diag.py --out <json>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

FP8 = torch.float8_e4m3fn


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
    gen = torch.Generator().manual_seed(5)
    dev = "cuda"
    f32 = lambda v: torch.tensor(v, dtype=torch.float32, device=dev)  # noqa: E731

    def conv(v):
        return torch.minimum(torch.maximum(v, f32(-448.0)), f32(448.0)).to(FP8).view(torch.uint8)

    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": []}
    for dtype, group in ((torch.bfloat16, 128), (torch.bfloat16, 64), (torch.float32, 128), (torch.float16, 128)):
        for tokens, hidden in ((2048, 1024), (1000, 8320)):
            x = ((torch.randn((tokens, hidden), generator=gen) * 3).to(dtype)).cuda()
            for ue8m0 in (False, True):
                q, s = per_token_group_quant_fp8(x, group, eps=1e-10, column_major_scales=False, use_ue8m0=ue8m0)
                qb = q.view(torch.uint8)
                xf = x.float(); g = xf.reshape(tokens, hidden // group, group)
                absmax = torch.maximum(g.abs().amax(-1), f32(1e-10))
                y_s = absmax / f32(448.0)
                if ue8m0:
                    y_s = torch.exp2(torch.ceil(torch.log2(torch.maximum(y_s.abs(), f32(1e-10)))))
                y_m = absmax * f32(1.0 / 448.0)
                cands = {
                    "R_x_times_recip": conv((g * (f32(1.0) / y_s)[:, :, None]).reshape(tokens, hidden)),
                    "D_x_div_ys": conv((g / y_s[:, :, None]).reshape(tokens, hidden)),
                    "A_x_times_448_over_absmax": conv((g * (f32(448.0) / absmax)[:, :, None]).reshape(tokens, hidden)),
                    "M_recip_of_absmax_times_inv448": conv((g * (f32(1.0) / y_m)[:, :, None]).reshape(tokens, hidden)),
                    "S_x_times_recip_returned_scale": conv((g * (f32(1.0) / s.float())[:, :, None]).reshape(tokens, hidden)),
                }
                row = {"dtype": str(dtype).split(".")[-1], "group": group, "tokens": tokens, "hidden": hidden, "ue8m0": ue8m0,
                       "elements": int(qb.numel()), "scale_off_model": int((s.float() != y_s).sum())}
                for k, v in cands.items():
                    row["off_" + k] = int((qb != v).sum())
                print(json.dumps(row), flush=True)
                report["rows"].append(row)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
