#!/usr/bin/env python3
"""Where the GPU quantisers' codes differ from the CPU reference: are the differences one e2m1 step at near-tie positions?

    PYTHONPATH=<sm120-fp4> python scripts/nvfp4_quantize_cross_library_diag.py --outdir reports/nvfp4-quantize-cross-library-rtx5090
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def unpack(codes: torch.Tensor) -> torch.Tensor:
    lo = codes & 0x0F
    hi = (codes >> 4) & 0x0F
    return torch.stack([lo, hi], dim=-1).reshape(codes.shape[0], -1).to(torch.int64)


def decode(c: torch.Tensor) -> torch.Tensor:
    mag = E2M1[c & 7]
    return torch.where(c >= 8, -mag, mag)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args(argv)
    xs = torch.load(a.outdir / "inputs.pt", weights_only=False)
    ref = torch.load(a.outdir / "quant-reference.pt", weights_only=False)["outs"]
    gpu = torch.load(a.outdir / "quant-vllm.pt", weights_only=False)["outs"]
    report = []
    for shp, (x, gs) in xs.items():
        if "codes" not in ref.get(shp, {}) or "codes" not in gpu.get(shp, {}):
            continue
        cr, cg = unpack(ref[shp]["codes"]), unpack(gpu[shp]["codes"])
        sf = ref[shp]["sf"].view(torch.float8_e4m3fn).float()   # scales are identical across all runs
        m, k = x.shape
        scale = sf.repeat_interleave(16, dim=1)[:, :k] * (1.0 / gs.float())  # per-element block scale times global scale
        xs_scaled = x.float() / torch.where(scale == 0, torch.ones_like(scale), scale)
        diff = cr != cg
        n = int(diff.sum())
        rec = {"shape": list(shp), "elements": int(diff.numel()), "codes_differing": n}
        if n:
            dr, dg = decode(cr[diff]), decode(cg[diff])
            # one-step neighbours on the e2m1 grid?
            idx_r = (cr[diff] & 7); idx_g = (cg[diff] & 7)
            same_sign = ((cr[diff] >= 8) == (cg[diff] >= 8))
            one_step = (same_sign & ((idx_r - idx_g).abs() == 1)) | ((idx_r == 0) & (idx_g == 0))
            v = xs_scaled[diff]
            # distance of the scaled value from the midpoint between the two decoded codes, in units of the step
            mid = (dr + dg) / 2; step = (dr - dg).abs()
            rel = ((v - mid).abs() / torch.where(step == 0, torch.ones_like(step), step))
            rec.update({"one_step_apart": int(one_step.sum()), "within_1e-3_step_of_midpoint": int((rel < 1e-3).sum()),
                        "within_1e-2_step_of_midpoint": int((rel < 1e-2).sum()), "max_rel_distance_from_midpoint": float(rel.max()),
                        "example": {"x_scaled": [round(float(t), 5) for t in v[:4]], "reference_code_value": [float(t) for t in dr[:4]], "gpu_code_value": [float(t) for t in dg[:4]]}})
        report.append(rec); print(json.dumps(rec))
    (a.outdir / "diag-20261016.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
