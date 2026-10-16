#!/usr/bin/env python3
"""Sanity for the cross-library NVFP4 comparison: the saved outputs must be non-trivial (non-zero, varying) and conform to the exact
oracle, otherwise 'bit-identical across libraries' would be an empty statement.

    PYTHONPATH=<sm120-fp4> python scripts/nvfp4_cross_library_sanity.py --outdir reports/nvfp4-cross-library-rtx5090
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--oracle-shape", default="256,4096,4096")
    a = ap.parse_args(argv)
    from sm120fp4 import oracle
    ops = torch.load(a.outdir / "operands.pt", weights_only=False)
    oshape = tuple(int(v) for v in a.oracle_shape.split(","))
    out = []
    for f in sorted(glob.glob(str(a.outdir / "*-*.pt"))):
        r = torch.load(f, weights_only=False)
        for shp, o in r["outs"].items():
            if "out" not in o:
                continue
            x = o["out"].float()
            rec = {"run": f"{r['lib']} {r['backend']}", "shape": list(shp), "absmax": round(float(x.abs().max()), 4),
                   "nonzero_fraction": round(float((x != 0).float().mean()), 5), "distinct_values": int(torch.unique(x).numel()), "dtype": str(o["out"].dtype)}
            if shp == oshape:
                (a_p, a_sf, a_gs), (b_p, b_sf, b_gs) = ops[shp]
                exact, bound = oracle.exact_gemm_nvfp4(a_p, a_sf, a_gs, b_p, b_sf, b_gs, out_dtype=torch.bfloat16)
                c = oracle.conformance(x, exact, bound)
                rec.update({"oracle_outside_bound": c["outside_bound"], "oracle_elements": c["elements"], "oracle_max_abs_err": round(float((x.double() - exact).abs().max()), 5)})
            out.append(rec); print(json.dumps(rec))
    (a.outdir / "sanity-20261016.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
