#!/usr/bin/env python3
"""Does FlashInfer `grouped_mm_fp4` (cuDNN) need each expert's token segment to start on a 128-row boundary of the swizzled scale
layout? The first probe (flashinfer_grouped_fp4_probe.py) saw segments of 16/0/24/24 and random lengths return wrong values
and NaNs without raising, and a single segment starting at row 0 return conformant output; FlashInfer's own tests use segment
lengths that are multiples of 128 only. This sweep varies the segment lengths and nothing else, with the activation scales
swizzled over all tokens at once (the layout the test suite uses), and judges each case with the exact grouped oracle.

    PYTHONPATH=/path/to/sm120-fp4 python scripts/flashinfer_grouped_fp4_alignment.py --out reports/...json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sm120fp4 import oracle
from sm120fp4.layouts import to_128x4
from sm120fp4.reference import global_scale_for, quantize_nvfp4

SEGMENTS = {
    "4x128 aligned": [128, 128, 128, 128],
    "256,128,384,256 aligned (their test)": [256, 128, 384, 256],
    "0,128,0,128 aligned with empties": [0, 128, 0, 128],
    "64,64 half tiles": [64, 64],
    "120,136 straddling": [120, 136],
    "16,48,64 small": [16, 48, 64],
    "32,96 two in one tile": [32, 96],
    "128,64,64 first aligned": [128, 64, 64],
    "192,64 second starts mid-tile": [192, 64],
}


def run(fi, segs, n, k, gen, dev):
    e_count = len(segs)
    m_total = sum(segs)
    indptr = [0]
    for s in segs:
        indptr.append(indptr[-1] + s)
    x = torch.randn(m_total, k, generator=gen).to(torch.bfloat16).float()
    w = torch.randn(e_count, n, k, generator=gen).to(torch.bfloat16).float()
    gs_a, gs_b = global_scale_for(x), global_scale_for(w)
    ap_, asf, _ = quantize_nvfp4(x, gs_a)
    bq = [quantize_nvfp4(w[e], gs_b) for e in range(e_count)]
    bp = torch.stack([b[0] for b in bq]); bsf = torch.stack([b[1] for b in bq])
    exact, bound = oracle.exact_grouped_gemm_nvfp4(ap_, asf, gs_a, bp, bsf, gs_b, torch.tensor(indptr))
    a_sw = to_128x4(asf).view(torch.float8_e4m3fn).to(dev)
    b_sw = torch.stack([to_128x4(bsf[e]) for e in range(e_count)]).view(torch.float8_e4m3fn).to(dev)
    alpha = (1.0 / (gs_a.double() * gs_b.double())).float().reshape(1).to(dev)
    out = fi.grouped_mm_fp4(ap_.to(dev), bp.to(dev), a_sw, b_sw, torch.tensor(indptr, dtype=torch.int32, device=dev),
                            alpha=alpha, out_dtype=torch.bfloat16, block_size=16, backend="cudnn")
    torch.cuda.synchronize()
    v = oracle.conformance(out.float(), exact, bound)
    # which segments carry the bad elements
    err = (out.double().cpu() - exact).abs() > bound
    per_seg = [int(err[indptr[e]:indptr[e + 1]].sum()) for e in range(e_count)]
    v["outside_bound_per_segment"] = per_seg
    v["nan_per_segment"] = [int(torch.isnan(out[indptr[e]:indptr[e + 1]].float()).sum()) for e in range(e_count)]
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports") / f"flashinfer-grouped-fp4-alignment-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--k", type=int, default=512)
    a = ap.parse_args(argv)
    import flashinfer
    dev = torch.device("cuda")
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "flashinfer": flashinfer.__version__, "cudnn": torch.backends.cudnn.version(), "n": a.n, "k": a.k, "cases": []}
    gen = torch.Generator().manual_seed(20261010)
    for name, segs in SEGMENTS.items():
        rec = {"name": name, "segments": segs}
        try:
            rec.update(run(flashinfer, segs, a.n, a.k, gen, dev))
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        report["cases"].append(rec)
        print(json.dumps(rec)[:400])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
