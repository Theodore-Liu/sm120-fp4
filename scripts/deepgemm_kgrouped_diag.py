#!/usr/bin/env python3
"""Localise the k-grouped NT reading outside the bound (deepgemm_kgrouped_probe.py, 2026-10-15: m = 12, 64 groups with 56
empty, every n). For one case: which groups and rows are outside, what the kernel's value matches (another group's exact
product, the product without the accumulator, zero), and how the reading changes with m, the number of groups and the
accumulator mode. Writes a JSON; prints each row.

    PYTHONPATH=<sm120-fp4>:<sm120-fp4>/scripts:<DeepGEMM>/tests python deepgemm_kgrouped_diag.py --out <json>
"""
from __future__ import annotations

import argparse
import json
import random
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
    from generators import MajorTypeAB, generate_k_grouped_contiguous

    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "deep_gemm": getattr(deep_gemm, "__version__", "?"), "rows": []}

    def pattern64(seed=20261015):
        rnd = random.Random(seed)
        next(iter([rnd.sample(range(64), 8)]))  # keep the probe's draw order: first draw unused here
        return None

    # the probe's 64-group pattern, reproduced from its seed (same Random sequence: sample then choices)
    rnd = random.Random(20261015)
    lens = [0] * 64
    for i in rnd.sample(range(64), 8):
        lens[i] = rnd.choice([128, 256])
    PROBE64 = lens

    def case(entry, m, n, ks, c_mode, gran_k=128, k_alignment=128, label=""):
        major = MajorTypeAB.KMajor if entry == "nt" else MajorTypeAB.MNMajor
        fn = deep_gemm.k_grouped_fp8_gemm_nt_contiguous if entry == "nt" else deep_gemm.k_grouped_fp8_gemm_tn_contiguous
        row = {"label": label, "entry": entry, "m": m, "n": n, "groups": len(ks), "nonempty": sum(1 for k in ks if k), "c": c_mode}
        try:
            deep_gemm.set_mk_alignment_for_contiguous_layout(k_alignment)
            total_k, qa, qb, c, d, ref_d, layout, host_ks = generate_k_grouped_contiguous(
                len(ks), m, n, major, major, list(ks), use_ue8m0=True, gran_k=gran_k, k_alignment=k_alignment,
                use_psum_layout=False, accumulate=True, out_dtype=torch.float32)
            if entry == "nt":
                ad, bd = dequant_nt(qa[0], qa[1], host_ks, m, gran_k), dequant_nt(qb[0], qb[1], host_ks, n, gran_k)
            else:
                ad, bd = dequant_tn(qa[0], qa[1], gran_k), dequant_tn(qb[0], qb[1], gran_k)
            c0 = c.detach().cpu().double()
            prod = torch.zeros((len(ks), m, n), dtype=torch.float64)
            bnd = torch.ones((len(ks), m, n), dtype=torch.float64)
            start = 0
            for g, (lk, hk) in enumerate(zip(ks, host_ks)):
                if hk:
                    ag, bg = ad[start: start + hk], bd[start: start + hk]
                    prod[g] = ag.T @ bg
                    bnd[g] = bound(ag.T, bg.T, prod[g] + c0[g], max(lk, 1) + 1, torch.float32)
                else:
                    bnd[g] = bound(torch.zeros(m, 1, dtype=torch.float64), torch.zeros(n, 1, dtype=torch.float64), c0[g], 1, torch.float32)
                start += hk
            exact = prod + c0
            dd = c.clone() if c_mode == "alias" else torch.empty_like(d)
            cc = dd if c_mode == "alias" else c
            fn(qa, qb, dd, host_ks, layout, cc, recipe=(1, 1, gran_k))
            torch.cuda.synchronize()
            out = dd.double().cpu()
            err = (out - exact).abs()
            outside = err > bnd
            row["outside"] = int(outside.sum())
            row["elements"] = int(outside.numel())
            per_group = [int(outside[g].sum()) for g in range(len(ks))]
            row["outside_groups"] = {str(g): {"count": per_group[g], "k": ks[g], "rows_outside": sorted(set(torch.nonzero(outside[g])[:, 0].tolist()))[:16]}
                                     for g in range(len(ks)) if per_group[g]}
            # what the wrong groups' outputs match
            matches = {}
            for g in row["outside_groups"]:
                g = int(g)
                cand = {"c_only": c0[g], "product_without_c": prod[g], "zero": torch.zeros(m, n, dtype=torch.float64)}
                for g2 in range(len(ks)):
                    if g2 != g and host_ks[g2]:
                        cand[f"exact_of_group_{g2}"] = exact[g2]
                        cand[f"c_g_plus_product_of_group_{g2}"] = c0[g] + prod[g2]
                best = sorted(((float((out[g] - v).abs().max()), k) for k, v in cand.items()))[:3]
                matches[str(g)] = [{"candidate": k, "max_abs_diff": round(v, 6)} for v, k in best]
                matches[str(g)].append({"candidate": "exact_of_this_group", "max_abs_diff": round(float(err[g].max()), 6)})
            row["matches"] = matches
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        print(json.dumps(row)[:900], flush=True)
        report["rows"].append(row)
        return row

    # 1. the flagged case, both accumulator modes, and the TN control
    case("nt", 12, 136, PROBE64, "alias", label="flagged")
    case("nt", 12, 136, PROBE64, "separate", label="flagged-c-separate")
    case("tn", 12, 136, PROBE64, "alias", label="tn-control")
    # 2. m dependence at the same groups
    for m in (4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 68):
        case("nt", m, 136, PROBE64, "alias", label="m-sweep")
    # 3. group-count dependence at m = 12: the 8 non-empty groups alone, then with empties in different places
    nonempty = [k for k in PROBE64 if k]
    case("nt", 12, 136, nonempty, "alias", label="8-nonempty-only")
    case("nt", 12, 136, [0] + nonempty, "alias", label="one-leading-empty")
    case("nt", 12, 136, nonempty + [0], "alias", label="one-trailing-empty")
    case("nt", 12, 136, [nonempty[0], 0, 0, 0] + nonempty[1:], "alias", label="three-empties-after-first")
    case("nt", 12, 136, PROBE64[:33], "alias", label="first-33-groups")
    case("nt", 12, 136, [128] * 64, "alias", label="64-groups-all-128")
    case("nt", 12, 136, [256, 0, 128] + [0] * 61, "alias", label="3-nonempty-of-64")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
