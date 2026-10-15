#!/usr/bin/env python3
"""DeepGEMM's SM120 masked m-grouped GEMM (`m_grouped_fp8_fp4_gemm_nt_masked`), the pairings with an FP4 operand, at shapes
the library's own tests do not reach, against the exact oracle (vllm-project/DeepGEMM dev).

Registered before any case ran: mlsys-5090-research docs/d1-masked-fp4-pairings-registration-2026-10-15.md. Operands come
from DeepGEMM's own `generate_m_grouped_masked` with the pairing's `QuantConfig`; `masked_m` is then set to the case's
pattern and the SF rows past it are zeroed, as the generator does. The decoder, bound and judge are
`deepgemm_grouped_probe.py`'s (its FP4 decode is checked against DeepGEMM's `cast_back_from_fp4` at start-up).

    cd <outside the DeepGEMM source tree>
    PYTHONPATH=<sm120-fp4>:<sm120-fp4>/scripts:<DeepGEMM>/tests python deepgemm_masked_fp4_edge_probe.py --log <marker> --out <json>
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from deepgemm_grouped_probe import bound_for, dequant, judge, selftest_decode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-cases", type=int, default=0)
    a = ap.parse_args(argv)

    def mark(s: str) -> None:
        with open(a.log, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} masked-fp4 {s}\n")

    import deep_gemm
    import generators as G
    from deep_gemm.utils.math import cast_back_from_fp4, per_token_cast_to_fp4

    selftest_decode()
    xs = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
    for gran in (32, 128):
        pk, sf = per_token_cast_to_fp4(xs, use_ue8m0=True, gran_k=gran)
        assert torch.equal(cast_back_from_fp4(pk, sf, gran_k=gran).double().cpu(), dequant(pk, sf, True, 1, gran)), gran

    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "deep_gemm": getattr(deep_gemm, "__version__", "?"),
              "deep_gemm_path": deep_gemm.__file__, "cases": []}
    rnd = random.Random(20261015)
    torch.manual_seed(20261015)

    pairings = [qc for qc in G.QuantConfig.get_list_from_dtype(torch.float8_e4m3fn) if qc.is_fp4_a or qc.is_fp4_b]
    report["pairings"] = [(qc.gran_k_a, qc.gran_k_b, qc.is_fp4_a, qc.is_fp4_b) for qc in pairings]
    use_ue8m0 = G.get_ue8m0_usage(G.KernelType.Kernel1D1D)

    def run_case(qc, label, num_groups, max_m, masked, n, k, expected_m, runs=2):
        rec = {"label": label, "fp4_a": bool(qc.is_fp4_a), "fp4_b": bool(qc.is_fp4_b), "gran_k_a": qc.gran_k_a,
               "gran_k_b": qc.gran_k_b, "groups": num_groups, "max_m": max_m, "masked_m": masked, "n": n, "k": k,
               "expected_m": expected_m, "empty_groups": sum(1 for x in masked if x == 0)}
        try:
            recipe, recipe_a, recipe_b = qc.get_recipes()
            deep_gemm.set_mk_alignment_for_contiguous_layout(deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(max(1, expected_m)))
            # Operands as `generate_m_grouped_masked` builds them (tests/generators.py), but with this case's masked_m from
            # the start: the generator draws its own masked_m and zeroes the SF rows past it, and a zero fp32 scale is not a
            # zero UE8M0 scale (it packs as 2^-127), so overriding masked_m upward after generation judged rows the kernel
            # computes at ~1e-37 against an exact 0 (the first full run of 2026-10-15, void).
            x = torch.randn((num_groups, max_m, k), device="cuda", dtype=torch.bfloat16)
            w = torch.randn((num_groups, n, k), device="cuda", dtype=torch.bfloat16)
            aa = G.grouped_cast_fp8_fp4_with_major(x, G.MajorTypeAB.KMajor, qc.gran_k_a, qc.is_fp4_a, use_ue8m0)
            bb = G.grouped_cast_fp8_fp4_with_major(w, G.MajorTypeAB.KMajor, qc.gran_k_b, qc.is_fp4_b, use_ue8m0,
                                                  use_block_cast_for_fp8=True)
            sfa = aa[1].clone()
            for j in range(num_groups):           # as the generator: SF rows past masked_m are zero
                sfa[j, masked[j]:] = 0
            aa = (aa[0], sfa)
            masked_m = torch.tensor(masked, dtype=torch.int32, device="cuda")
            d = torch.empty((num_groups, max_m, n), device="cuda", dtype=torch.bfloat16)
            ad = dequant(aa[0], aa[1], qc.is_fp4_a, 1, qc.gran_k_a)
            bd = dequant(bb[0], bb[1], qc.is_fp4_b, 1 if qc.is_fp4_b else qc.gran_k_b, qc.gran_k_b)
            outs = []
            for _ in range(runs):
                d.zero_()
                mark(f"{label} fp4_a={qc.is_fp4_a} fp4_b={qc.is_fp4_b} groups={num_groups} max_m={max_m} n={n} k={k} exp={expected_m}")
                deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(aa, bb, d, masked_m, expected_m, disable_ue8m0_cast=not use_ue8m0,
                                                          recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)
                torch.cuda.synchronize()
                outs.append(d.double().cpu())
            res = {"elements": 0, "outside": 0, "nan": 0, "max_err_over_bound": 0.0}
            sel = []
            for g, r in enumerate(masked):
                if r:
                    ex = ad[g, :r] @ bd[g].T
                    j = judge(outs[0][g, :r], ex, bound_for(ad[g, :r], bd[g], ex, k))
                    for kk in ("elements", "outside", "nan"):
                        res[kk] += j[kk]
                    res["max_err_over_bound"] = max(res["max_err_over_bound"], j["max_err_over_bound"])
                    sel.append((g, r))
            rec.update(res)
            rec["distinct_outputs"] = len({b"".join(o[g, :r].numpy().tobytes() for g, r in sel) for o in outs})
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        return rec

    plan = []
    for qc in pairings:
        # edge A: 64 groups, most empty, the rest at 1, 63, 64, 65, 127 rows
        masked = [0] * 64
        for i, r in zip(rnd.sample(range(64), 8), (1, 63, 64, 65, 127, 1, 65, 127)):
            masked[i] = r
        for n in (8, 24, 136, 1000):
            for k in (128, 384, 1152):
                plan.append((qc, "edge-A", 64, 128, list(masked), n, k, 64))
        # edge B: capacity shapes, every row valid, one group empty
        for max_m in (68, 127, 129, 257):
            for n in (24, 136):
                plan.append((qc, "edge-B", 3, max_m, [max_m, 0, max_m], n, 128, max_m))
        # the hint far from the rows
        plan.append((qc, "hint-1", 64, 128, list(masked), 136, 384, 1))
        plan.append((qc, "hint-max", 64, 128, list(masked), 136, 384, 128))

    for i, (qc, label, groups, max_m, masked, n, k, exp) in enumerate(plan):
        if a.max_cases and i >= a.max_cases:
            break
        rec = run_case(qc, label, groups, max_m, masked, n, k, exp)
        report["cases"].append(rec)
        flag = rec.get("outside") or rec.get("nan") or "error" in rec or rec.get("distinct_outputs", 1) != 1
        if flag or (i + 1) % 10 == 0:
            print(json.dumps({kk: v for kk, v in rec.items() if kk != "masked_m"})[:400], flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    judged = [c for c in report["cases"] if "outside" in c]
    print(f"cases {len(report['cases'])}, judged {len(judged)}, outside in {sum(1 for c in judged if c['outside'])}, nan in "
          f"{sum(1 for c in judged if c['nan'])}, nondeterministic {sum(1 for c in judged if c.get('distinct_outputs', 1) != 1)}, "
          f"refused/errors {sum(1 for c in report['cases'] if 'error' in c)}, elements {sum(c['elements'] for c in judged)}; "
          f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
