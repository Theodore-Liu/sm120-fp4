#!/usr/bin/env python3
"""Does the alignment behaviour of FlashInfer's cuDNN grouped GEMM depend on the FP4 block-scale layout, or is it the grouped
kernel's handling of m_indptr itself? The same segment sweep as flashinfer_grouped_fp4_alignment.py, run through
`grouped_mm_fp8` (e4m3 operands, one per-tensor alpha, no block scales), judged against the exact fp64 product with the
fp32-accumulation bound. If mid-tile segments are wrong here too, the scale swizzle is not the cause.

    PYTHONPATH=/path/to/sm120-fp4 python scripts/flashinfer_grouped_fp8_alignment.py --out reports/...json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sm120fp4 import oracle

SEGMENTS = {"4x128 aligned": [128, 128, 128, 128], "64,64 half tiles": [64, 64], "16,48,64 small": [16, 48, 64], "192,64 second mid-tile": [192, 64], "120,136 straddling": [120, 136]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports") / f"flashinfer-grouped-fp8-alignment-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--k", type=int, default=512)
    a = ap.parse_args(argv)
    import flashinfer
    dev = torch.device("cuda")
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "flashinfer": flashinfer.__version__, "cudnn": torch.backends.cudnn.version(), "cases": []}
    gen = torch.Generator().manual_seed(20261010)
    for name, segs in SEGMENTS.items():
        indptr = [0]
        for s in segs:
            indptr.append(indptr[-1] + s)
        m_total, e_count = indptr[-1], len(segs)
        rec = {"name": name, "segments": segs}
        try:
            x = (torch.randn(m_total, a.k, generator=gen) * 2).to(torch.float8_e4m3fn)
            w = torch.randn(e_count, a.n, a.k, generator=gen).to(torch.float8_e4m3fn)
            alpha = torch.tensor([0.5], dtype=torch.float32)
            xd, wd = x.double(), w.double()
            exact = torch.zeros(m_total, a.n, dtype=torch.float64); bound = torch.zeros_like(exact)
            for e in range(e_count):
                s, t = indptr[e], indptr[e + 1]
                if t > s:
                    ex = (xd[s:t] @ wd[e].t()) * 0.5
                    sabs = (xd[s:t].abs() @ wd[e].abs().t()) * 0.5
                    exact[s:t] = ex
                    bound[s:t] = a.k * oracle.U32 * sabs + 0.5 * oracle._ulp(ex, torch.bfloat16) + 2 * oracle.U64 * ex.abs()
            out = flashinfer.grouped_mm_fp8(x.to(dev), w.to(dev), torch.tensor(indptr, dtype=torch.int32, device=dev), alpha=alpha.to(dev), out_dtype=torch.bfloat16)
            torch.cuda.synchronize()
            v = oracle.conformance(out.float(), exact, bound)
            err = (out.double().cpu() - exact).abs() > bound
            v["outside_bound_per_segment"] = [int(err[indptr[e]:indptr[e + 1]].sum()) for e in range(e_count)]
            rec.update(v)
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
