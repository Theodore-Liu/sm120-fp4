#!/usr/bin/env python3
"""Second localisation pass for the k-grouped NT reading at m = 12 (deepgemm_kgrouped_diag.py, 2026-10-15): which ks
patterns trigger it, whether repeated calls in one process agree, whether a wrong row equals some other row's correct
value, and what DeepGEMM's own calc_diff metric says about the same output.

    PYTHONPATH=<sm120-fp4>:<sm120-fp4>/scripts:<DeepGEMM>/tests python deepgemm_kgrouped_diag2.py --out <json>
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

    def case(m, n, ks, label, runs=3):
        row = {"label": label, "m": m, "n": n, "ks": ks, "groups": len(ks)}
        try:
            torch.manual_seed(7)
            total_k, qa, qb, c, d, ref_d, layout, host_ks = generate_k_grouped_contiguous(
                len(ks), m, n, MajorTypeAB.KMajor, MajorTypeAB.KMajor, list(ks), use_ue8m0=True, gran_k=128,
                k_alignment=128, use_psum_layout=False, accumulate=True, out_dtype=torch.float32)
            ad, bd = dequant_nt(qa[0], qa[1], host_ks, m, 128), dequant_nt(qb[0], qb[1], host_ks, n, 128)
            c0 = c.detach().cpu().double()
            exact = torch.zeros((len(ks), m, n), dtype=torch.float64)
            bnd = torch.ones((len(ks), m, n), dtype=torch.float64)
            start = 0
            for g, (lk, hk) in enumerate(zip(ks, host_ks)):
                if hk:
                    ag, bg = ad[start: start + hk], bd[start: start + hk]
                    exact[g] = ag.T @ bg + c0[g]
                    bnd[g] = bound(ag.T, bg.T, exact[g], lk + 1, torch.float32)
                else:
                    exact[g] = c0[g]
                start += hk
            outs = []
            for _ in range(runs):
                dd = c.clone()
                deep_gemm.k_grouped_fp8_gemm_nt_contiguous(qa, qb, dd, host_ks, layout, dd, recipe=(1, 1, 128))
                torch.cuda.synchronize()
                outs.append(dd.double().cpu())
            row["distinct_outputs_in_process"] = len({o.numpy().tobytes() for o in outs})
            row["outside_per_run"] = [int(((o - exact).abs() > bnd).sum()) for o in outs]
            # DeepGEMM's own metric on the first output (the library's tests judge k-grouped with calc_diff against ref_d)
            row["calc_diff_vs_ref_d"] = float(calc_diff(outs[0].float().cuda(), ref_d.float()))
            row["calc_diff_exact_vs_ref_d"] = float(calc_diff(exact.float().cuda(), ref_d.float()))
            # wrong rows of the first output: does the row equal another row's correct value (same or other group)?
            out = outs[0]
            wrong = ((out - exact).abs() > bnd).any(dim=2).nonzero().tolist()
            row["wrong_rows"] = wrong[:12]
            rowmatch = []
            allrows = exact.reshape(-1, n)
            for g, r in wrong[:6]:
                diffs = (allrows - out[g, r][None, :]).abs().max(dim=1).values
                j = int(diffs.argmin())
                rowmatch.append({"group": g, "row": r, "nearest_correct_row": [j // m, j % m], "max_abs_diff": round(float(diffs[j]), 4),
                                 "vs_c_row": round(float((out[g, r] - c0[g, r]).abs().max()), 4),
                                 "abs_max_of_out_row": round(float(out[g, r].abs().max()), 3)})
            row["row_matches"] = rowmatch
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        print(json.dumps(row)[:700], flush=True)
        report["rows"].append(row)

    n = 136
    case(12, n, [256] * 8, "8x256")
    case(12, n, [128] * 8, "8x128")
    case(12, n, [256], "1x256")
    case(12, n, [256, 128], "256,128")
    case(12, n, [128, 256], "128,256")
    case(12, n, [384] * 4, "4x384")
    case(12, n, [512] * 4, "4x512")
    case(12, n, [1024] * 4, "4x1024")
    case(12, n, [256] * 64, "64x256")
    case(12, 64, [256] * 8, "8x256-n64")
    case(12, 2048, [256] * 8, "8x256-n2048")
    for m in (12, 44, 76, 108, 140, 204, 268):
        case(m, n, [256] * 8, f"8x256-m{m}")
    for m in (10, 11, 13, 14, 15):
        case(m, n, [256] * 8, f"8x256-m{m}-mn-not-4")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
