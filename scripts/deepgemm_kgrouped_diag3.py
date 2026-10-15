#!/usr/bin/env python3
"""Third localisation pass for the k-grouped NT reading at m = 12: a map of ks patterns, the TN control on a wrong pattern,
c separate vs aliased, and a stale-memory test (the same call repeated after the allocator's free memory is overwritten
with a sentinel: if the wrong rows change, the kernel reads memory it did not write).

    PYTHONPATH=<sm120-fp4>:<sm120-fp4>/scripts:<DeepGEMM>/tests python deepgemm_kgrouped_diag3.py --out <json>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from deepgemm_edge_probe import bound
from deepgemm_kgrouped_probe import dequant_nt, dequant_tn


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

    def prepare(entry, m, n, ks):
        major = MajorTypeAB.KMajor if entry == "nt" else MajorTypeAB.MNMajor
        torch.manual_seed(7)
        total_k, qa, qb, c, d, ref_d, layout, host_ks = generate_k_grouped_contiguous(
            len(ks), m, n, major, major, list(ks), use_ue8m0=True, gran_k=128, k_alignment=128,
            use_psum_layout=False, accumulate=True, out_dtype=torch.float32)
        if entry == "nt":
            ad, bd = dequant_nt(qa[0], qa[1], host_ks, m, 128), dequant_nt(qb[0], qb[1], host_ks, n, 128)
        else:
            ad, bd = dequant_tn(qa[0], qa[1], 128), dequant_tn(qb[0], qb[1], 128)
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
        return qa, qb, c, d, ref_d, layout, host_ks, exact, bnd

    def call(entry, qa, qb, c, host_ks, layout, c_mode):
        fn = deep_gemm.k_grouped_fp8_gemm_nt_contiguous if entry == "nt" else deep_gemm.k_grouped_fp8_gemm_tn_contiguous
        dd = c.clone() if c_mode == "alias" else torch.empty_like(c)
        fn(qa, qb, dd, host_ks, layout, dd if c_mode == "alias" else c, recipe=(1, 1, 128))
        torch.cuda.synchronize()
        return dd.double().cpu()

    def scribble():
        free, _ = torch.cuda.mem_get_info()
        x = torch.full((max(1, (free // 2) // 4),), float("nan"), dtype=torch.float32, device="cuda")
        y = torch.full((2 << 20,), -7.0, dtype=torch.float32, device="cuda")
        del x, y
        torch.cuda.synchronize()

    def case(entry, m, n, ks, label, c_mode="alias"):
        row = {"label": label, "entry": entry, "m": m, "n": n, "ks": ks if len(ks) <= 12 else f"{len(ks)} groups", "c": c_mode}
        try:
            qa, qb, c, d, ref_d, layout, host_ks, exact, bnd = prepare(entry, m, n, ks)
            o1 = call(entry, qa, qb, c, host_ks, layout, c_mode)
            scribble()
            o2 = call(entry, qa, qb, c, host_ks, layout, c_mode)
            w1, w2 = ((o1 - exact).abs() > bnd), ((o2 - exact).abs() > bnd)
            row["outside_before_scribble"] = int(w1.sum())
            row["outside_after_scribble"] = int(w2.sum())
            row["same_output_before_after"] = bool(torch.equal(o1, o2))
            row["wrong_rows_before"] = w1.any(dim=2).nonzero().tolist()[:10]
            row["wrong_rows_after"] = w2.any(dim=2).nonzero().tolist()[:10]
            row["calc_diff_vs_ref_d"] = float(calc_diff(o1.float().cuda(), ref_d.float()))
            if int(w1.sum()):
                g, r = w1.any(dim=2).nonzero().tolist()[0]
                row["example_wrong_row"] = {"group": g, "row": r, "out_first8": [round(float(v), 3) for v in o1[g, r][:8]],
                                           "exact_first8": [round(float(v), 3) for v in exact[g, r][:8]],
                                           "c_first8": [round(float(v), 3) for v in c.detach().cpu().double()[g, r][:8]],
                                           "is_nan_any": bool(torch.isnan(o1[g, r]).any())}
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        print(json.dumps(row)[:600], flush=True)
        report["rows"].append(row)

    n = 136
    case("nt", 12, n, [384] * 4, "4x384")
    case("nt", 12, n, [384] * 4, "4x384-c-separate", c_mode="separate")
    case("tn", 12, n, [384] * 4, "4x384-tn-control")
    case("nt", 12, n, [384], "1x384")
    case("nt", 12, n, [384] * 2, "2x384")
    case("nt", 12, n, [384, 128], "384,128")
    case("nt", 12, n, [128, 384], "128,384")
    case("nt", 12, n, [640] * 2, "2x640")
    case("nt", 12, n, [768] * 2, "2x768")
    case("nt", 12, n, [896] * 2, "2x896")
    case("nt", 12, n, [128, 256] * 4, "128,256 x4")
    case("nt", 12, n, [256, 128] * 2, "256,128 x2")
    case("nt", 12, n, [128] * 3, "3x128")
    case("nt", 12, n, [128] * 12, "12x128")
    case("nt", 12, n, [256] * 3, "3x256")
    case("nt", 12, n, [512] * 3, "3x512")
    for m in (12, 16, 24, 36, 44, 60, 76, 92, 108, 124):
        case("nt", m, n, [384] * 4, f"4x384-m{m}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
