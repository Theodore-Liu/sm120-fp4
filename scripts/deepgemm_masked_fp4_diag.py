#!/usr/bin/env python3
"""Instrument check for deepgemm_masked_fp4_edge_probe.py (every case read outside the bound on 2026-10-15): the same
case with (a) the generator's own masked_m, (b) masked_m overridden below the generator's, (c) overridden above it (rows
whose SF the generator zeroed), (d) FP8 x FP8 under the same override path (known conformant at edge shapes), and which
rows are outside relative to the generator's masked_m.

    PYTHONPATH=<sm120-fp4>:<sm120-fp4>/scripts:<DeepGEMM>/tests python deepgemm_masked_fp4_diag.py --out <json>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from deepgemm_grouped_probe import bound_for, dequant, judge


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    import deep_gemm
    import generators as G

    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": []}
    use_ue8m0 = G.get_ue8m0_usage(G.KernelType.Kernel1D1D)
    qcs = {"fp8xfp8": G.QuantConfig(), **{f"{'fp4' if q.is_fp4_a else 'fp8'}x{'fp4' if q.is_fp4_b else 'fp8'}": q
                                           for q in G.QuantConfig.get_list_from_dtype(torch.float8_e4m3fn) if q.is_fp4_a or q.is_fp4_b}}

    def case(name, qc, num_groups, max_m, expected_m, n, k, override, label):
        row = {"label": label, "pairing": name, "groups": num_groups, "max_m": max_m, "n": n, "k": k, "override": override}
        try:
            torch.manual_seed(3)
            recipe, recipe_a, recipe_b = qc.get_recipes()
            deep_gemm.set_mk_alignment_for_contiguous_layout(deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(expected_m))
            aa, bb, masked_m, d, ref_d = G.generate_m_grouped_masked(num_groups, max_m, expected_m, n, k, use_ue8m0=use_ue8m0,
                                                                     quant_config=qc)[:5]
            gen_masked = masked_m.cpu().tolist()
            row["generator_masked_m"] = gen_masked
            if override is not None:
                masked = list(override)
                sfa = aa[1]
                for j in range(num_groups):
                    sfa[j, masked[j]:] = 0
                aa = (aa[0], sfa)
                masked_m = torch.tensor(masked, dtype=torch.int32, device="cuda")
            else:
                masked = gen_masked
            row["masked_m"] = masked
            ad = dequant(aa[0], aa[1], qc.is_fp4_a, 1, qc.gran_k_a)
            bd = dequant(bb[0], bb[1], qc.is_fp4_b, 1 if qc.is_fp4_b else qc.gran_k_b, qc.gran_k_b)
            d.zero_()
            deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(aa, bb, d, masked_m, int(expected_m * 1.2), disable_ue8m0_cast=not use_ue8m0,
                                                      recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)
            torch.cuda.synchronize()
            out = d.double().cpu()
            res = {"elements": 0, "outside": 0, "rows_outside_below_gen": 0, "rows_outside_at_or_above_gen": 0}
            examples = []
            for g, r in enumerate(masked):
                if not r:
                    continue
                ex = ad[g, :r] @ bd[g].T
                bn = bound_for(ad[g, :r], bd[g], ex, k)
                w = (out[g, :r] - ex).abs() > bn
                res["elements"] += int(w.numel()); res["outside"] += int(w.sum())
                rows = w.any(dim=1).nonzero().flatten().tolist()
                res["rows_outside_below_gen"] += sum(1 for x in rows if x < gen_masked[g])
                res["rows_outside_at_or_above_gen"] += sum(1 for x in rows if x >= gen_masked[g])
                if rows and len(examples) < 3:
                    x = rows[0]
                    examples.append({"group": g, "row": x, "gen_masked": gen_masked[g], "out_first4": [round(float(v), 4) for v in out[g, x][:4]],
                                     "exact_first4": [round(float(v), 4) for v in ex[x][:4]], "sfa_row_absmax": float(aa[1][g, x].abs().max())})
            row.update(res)
            row["examples"] = examples
            row["calc_diff_vs_ref_d_valid"] = float(G.calc_diff(out[0:1, :masked[0]].float().cuda(), ref_d[0:1, :masked[0]].float())) if masked[0] else None
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        print(json.dumps(row)[:700], flush=True)
        report["rows"].append(row)

    for name, qc in qcs.items():
        case(name, qc, 3, 128, 64, 136, 384, None, "generator-masked")
        case(name, qc, 3, 128, 64, 136, 384, [32, 0, 16], "override-below")
        case(name, qc, 3, 128, 64, 136, 384, [128, 0, 128], "override-above")
        case(name, qc, 3, 68, 34, 136, 128, [68, 0, 68], "edge-B-like")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
