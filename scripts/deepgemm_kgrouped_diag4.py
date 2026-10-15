#!/usr/bin/env python3
"""Fourth pass: the square-SF hypothesis for the k-grouped NT reading. Every wrong pattern so far has sum_k = 1536 at
m = 12, i.e. sum_k / gran_k == m, so the K-major SF tensor (m, sum_k / 128) is square. Prediction: wrong for every m with
sum_k = 128 * m, clean for sum_k = 128 * (m +- 1); and the same operands with the SF passed contiguous (not a transposed
view) tell whether the stride, not the shape, is what the library reads.

    PYTHONPATH=<sm120-fp4>:<sm120-fp4>/scripts:<DeepGEMM>/tests python deepgemm_kgrouped_diag4.py --out <json>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from deepgemm_edge_probe import bound
from deepgemm_kgrouped_probe import dequant_nt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    import deep_gemm
    from deep_gemm.testing import calc_diff
    from generators import MajorTypeAB, generate_k_grouped_contiguous

    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "deep_gemm": getattr(deep_gemm, "__version__", "?"), "rows": []}
    deep_gemm.set_mk_alignment_for_contiguous_layout(128)

    def case(m, n, ks, label, sf_contiguous=False):
        row = {"label": label, "m": m, "n": n, "ks": ks if len(ks) <= 8 else f"{len(ks)} groups", "sum_k": sum(ks),
               "sf_cols": sum(ks) // 128, "square": sum(ks) // 128 == m, "sf_contiguous": sf_contiguous}
        try:
            torch.manual_seed(7)
            total_k, qa, qb, c, d, ref_d, layout, host_ks = generate_k_grouped_contiguous(
                len(ks), m, n, MajorTypeAB.KMajor, MajorTypeAB.KMajor, list(ks), use_ue8m0=True, gran_k=128,
                k_alignment=128, use_psum_layout=False, accumulate=True, out_dtype=torch.float32)
            row["sfa_shape"] = list(qa[1].shape)
            row["sfa_strides"] = list(qa[1].stride())
            if sf_contiguous:
                qa = (qa[0], qa[1].contiguous())
                qb = (qb[0], qb[1].contiguous())
            ad, bd = dequant_nt(qa[0], qa[1], host_ks, m, 128), dequant_nt(qb[0], qb[1], host_ks, n, 128)
            c0 = c.detach().cpu().double()
            exact = torch.zeros((len(ks), m, n), dtype=torch.float64)
            bnd = torch.ones((len(ks), m, n), dtype=torch.float64)
            start = 0
            for g, (lk, hk) in enumerate(zip(ks, host_ks)):
                ag, bg = ad[start: start + hk], bd[start: start + hk]
                exact[g] = ag.T @ bg + c0[g]
                bnd[g] = bound(ag.T, bg.T, exact[g], lk + 1, torch.float32)
                start += hk
            dd = c.clone()
            deep_gemm.k_grouped_fp8_gemm_nt_contiguous(qa, qb, dd, host_ks, layout, dd, recipe=(1, 1, 128))
            torch.cuda.synchronize()
            out = dd.double().cpu()
            w = (out - exact).abs() > bnd
            row["outside"] = int(w.sum())
            row["elements"] = int(w.numel())
            row["calc_diff_vs_ref_d"] = float(calc_diff(out.float().cuda(), ref_d.float()))
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        print(json.dumps(row)[:400], flush=True)
        report["rows"].append(row)

    n = 136
    for m in (4, 8, 12, 16, 24, 32, 48, 64):
        case(m, n, [128] * m, f"square-m{m}")               # sum_k = 128 m
        case(m, n, [128] * (m - 1), f"below-m{m}")          # one column fewer
        case(m, n, [128] * (m + 1), f"above-m{m}")          # one column more
    case(12, n, [384] * 4, "square-m12-4x384")
    case(12, n, [384] * 4, "square-m12-4x384-sf-contiguous", sf_contiguous=True)
    case(16, n, [512] * 4, "square-m16-4x512")
    case(16, n, [512] * 4, "square-m16-4x512-sf-contiguous", sf_contiguous=True)
    case(128, n, [256] * 64, "square-m128-64x256")
    case(300, 1000, [128] * 300, "square-m300-n1000")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
