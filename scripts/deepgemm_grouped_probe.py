#!/usr/bin/env python3
"""DeepGEMM's SM120 m-grouped FP8 GEMMs (`m_grouped_fp8_fp4_gemm_nt_contiguous` with the per-row index layout and the PSUM
layout, `m_grouped_fp8_fp4_gemm_nt_masked`) against the exact oracle, on vllm-project/DeepGEMM dev.

Inside the documented contract only: README, "each expert segment must be aligned to the GEMM M block size
(`get_mk_alignment_for_contiguous_layout()`)". Operands, layouts and call arguments come from DeepGEMM's own test machinery
(`tests/generators.py`: `enumerate_m_grouped_contiguous`, `generate_m_grouped_contiguous`, `enumerate_m_grouped_masked`,
`generate_m_grouped_masked`), called exactly as `tests/test_fp8_fp4.py` calls them, so the operand contract is the library's.
That test judges with `calc_diff` (a normalised squared difference against a bf16 product of the unquantised operands); here
each valid output element is judged against the exact product of the quantised operands (scales expanded element by element,
float64 sum) under the census's bound. Rows outside every group (padding, PSUM gaps, masked tails) are not judged. Each case runs
three times and the distinct-output count over the valid rows is recorded. FP4 operands are skipped in this version (their
packing is not decoded here); they are listed as skipped, not judged.

    PYTHONPATH=<sm120-fp4>:<DeepGEMM>/tests python scripts/deepgemm_grouped_probe.py --out reports/<file>.json [--max-cases N]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def sf_to_float(sf: torch.Tensor) -> torch.Tensor:
    """Scale factors as float64: fp32 as is; packed UE8M0 (4 exponent bytes per int32, low byte first) expanded."""
    sf = sf.detach().cpu()
    if sf.dtype in (torch.float32, torch.float64, torch.bfloat16):
        return sf.double()
    if sf.dtype == torch.int32:
        b = sf.contiguous().view(torch.uint8).reshape(*sf.shape[:-1], sf.shape[-1] * 4)
        return torch.pow(2.0, b.double() - 127.0)
    if sf.dtype == torch.uint8:
        return torch.pow(2.0, sf.double() - 127.0)
    raise TypeError(sf.dtype)


def expand(sf: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Expand a [.., R, C] scale grid to [.., rows, cols] by the integer granularities rows/R (rounded up) and cols/C."""
    r, c = sf.shape[-2], sf.shape[-1]
    gr = -(-rows // r)
    gc = -(-cols // c)
    return sf.repeat_interleave(gr, dim=-2)[..., :rows, :].repeat_interleave(gc, dim=-1)[..., :cols]


def bound_for(ad: torch.Tensor, bd: torch.Tensor, exact: torch.Tensor, k: int):
    from sm120fp4 import oracle
    sabs = ad.abs() @ bd.abs().transpose(-1, -2)
    pre = k * oracle.U32 * sabs + 2 * oracle.U64 * sabs
    return pre + 0.5 * oracle._ulp(exact.abs() + pre, torch.bfloat16)


def judge(d: torch.Tensor, exact: torch.Tensor, bound: torch.Tensor) -> dict:
    err = (d - exact).abs()
    return {"elements": int(err.numel()), "outside": int((err > bound).sum()),
            "max_err_over_bound": float((err / bound.clamp(min=1e-300)).max()) if err.numel() else 0.0,
            "nan": int(torch.isnan(d).sum())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-cases", type=int, default=24)
    a = ap.parse_args(argv)
    import deep_gemm
    import generators as G
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "deep_gemm": getattr(deep_gemm, "__version__", "?"), "cases": []}

    # ---------------------------------------------------------------- contiguous (index layout and PSUM layout)
    n_done = 0
    for cfg in G.enumerate_m_grouped_contiguous(dtype=torch.float8_e4m3fn):
        kernel_type, qc, num_groups, expected_m, n, k, major_a, major_b, use_psum, ensure_zero = cfg
        rec = {"layout": "contiguous-psum" if use_psum else "contiguous-index", "groups": num_groups, "expected_m": expected_m,
               "n": n, "k": k, "major_a": str(major_a), "major_b": str(major_b), "ensure_zero_padding": bool(ensure_zero),
               "fp4_a": bool(qc.is_fp4_a), "fp4_b": bool(qc.is_fp4_b)}
        if qc.is_fp4_a or qc.is_fp4_b:
            rec["skipped"] = "FP4 operand (packing not decoded in this version)"
            report["cases"].append(rec)
            continue
        if not (major_a.is_k_major() and major_b.is_k_major()):
            rec["skipped"] = "non-K-major operand (kept to the NT form here)"
            report["cases"].append(rec)
            continue
        try:
            use_ue8m0 = G.get_ue8m0_usage(kernel_type)
            recipe, recipe_a, recipe_b = qc.get_recipes()
            deep_gemm.set_mk_alignment_for_contiguous_layout(deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout())
            m, aa, bb, layout, d, _, valid = G.generate_m_grouped_contiguous(num_groups, expected_m, n, k, major_a, major_b,
                                                                             use_ue8m0=use_ue8m0, use_psum_layout=use_psum,
                                                                             quant_config=qc)
            outs = []
            for _ in range(3):
                d.zero_()
                deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(aa, bb, d, layout, disable_ue8m0_cast=not use_ue8m0,
                                                              use_psum_layout=use_psum, ensure_zero_padding=ensure_zero,
                                                              recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)
                torch.cuda.synchronize()
                outs.append(d.double().cpu())
            ad = aa[0].cpu().double() * expand(sf_to_float(aa[1]), m, k)
            bd = bb[0].cpu().double() * expand(sf_to_float(bb[1]), n, k)
            lay = layout.cpu().long()
            vm = valid.cpu()
            exact = torch.zeros(m, n, dtype=torch.float64)
            bnd = torch.zeros(m, n, dtype=torch.float64)
            if use_psum:
                start = 0
                align = deep_gemm.get_mk_alignment_for_contiguous_layout()
                for g in range(num_groups):
                    end = int(lay[g])
                    rows = torch.arange(start, end)
                    exact[rows] = ad[rows] @ bd[g].T
                    bnd[rows] = bound_for(ad[rows], bd[g], exact[rows], k)
                    start = -(-end // align) * align
            else:
                for g in range(num_groups):
                    rows = (lay == g).nonzero(as_tuple=True)[0]
                    if rows.numel():
                        exact[rows] = ad[rows] @ bd[g].T
                        bnd[rows] = bound_for(ad[rows], bd[g], exact[rows], k)
            rec["m"] = m
            rec["valid_rows"] = int(vm.sum())
            rec.update(judge(outs[0][vm], exact[vm], bnd[vm]))
            rec["distinct_outputs"] = len({o[vm].to(torch.bfloat16).view(torch.int16).numpy().tobytes() for o in outs})
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
        report["cases"].append(rec)
        print(json.dumps(rec)[:400], flush=True)
        n_done += 1
        if n_done >= a.max_cases:
            break

    # ---------------------------------------------------------------- masked
    n_done = 0
    for cfg in G.enumerate_m_grouped_masked(dtype=torch.float8_e4m3fn):
        kernel_type, qc, num_groups, max_m, expected_m, n, k, use_psum = cfg[:8] if len(cfg) >= 8 else (*cfg, False)
        rec = {"layout": "masked" + ("-psum" if use_psum else ""), "groups": num_groups, "max_m": max_m, "expected_m": expected_m,
               "n": n, "k": k, "fp4_a": bool(qc.is_fp4_a), "fp4_b": bool(qc.is_fp4_b)}
        if qc.is_fp4_a or qc.is_fp4_b or use_psum:
            rec["skipped"] = "FP4 operand" if (qc.is_fp4_a or qc.is_fp4_b) else "masked PSUM variant (not modelled here)"
            report["cases"].append(rec)
            continue
        try:
            use_ue8m0 = G.get_ue8m0_usage(kernel_type)
            recipe, recipe_a, recipe_b = qc.get_recipes()
            exp_m = int(expected_m * 1.2)                   # as tests/test_fp8_fp4.py
            deep_gemm.set_mk_alignment_for_contiguous_layout(deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(exp_m))
            aa, bb, masked_m, d, _ = G.generate_m_grouped_masked(num_groups, max_m, expected_m, n, k, use_ue8m0=use_ue8m0,
                                                                 quant_config=qc)[:5]
            outs = []
            for _ in range(3):
                d.zero_()
                deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(aa, bb, d, masked_m, exp_m, disable_ue8m0_cast=not use_ue8m0,
                                                          recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)
                torch.cuda.synchronize()
                outs.append(d.double().cpu())
            ad = aa[0].cpu().double() * expand(sf_to_float(aa[1]), max_m, k)
            bd = bb[0].cpu().double() * expand(sf_to_float(bb[1]), n, k)
            mm = masked_m.cpu().long()
            res = {"elements": 0, "outside": 0, "max_err_over_bound": 0.0, "nan": 0}
            sel = []
            for g in range(num_groups):
                r = int(mm[g])
                if r == 0:
                    continue
                ex = ad[g, :r] @ bd[g].T
                bn = bound_for(ad[g, :r], bd[g], ex, k)
                j = judge(outs[0][g, :r], ex, bn)
                res["elements"] += j["elements"]; res["outside"] += j["outside"]; res["nan"] += j["nan"]
                res["max_err_over_bound"] = max(res["max_err_over_bound"], j["max_err_over_bound"])
                sel.append((g, r))
            rec.update(res)
            rec["masked_m"] = mm.tolist()
            rec["distinct_outputs"] = len({b"".join(o[g, :r].to(torch.bfloat16).view(torch.int16).numpy().tobytes() for g, r in sel)
                                           for o in outs})
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
        report["cases"].append(rec)
        print(json.dumps(rec)[:400], flush=True)
        n_done += 1
        if n_done >= a.max_cases:
            break

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    judged = [c for c in report["cases"] if "outside" in c]
    print(f"judged {len(judged)}, outside in {sum(1 for c in judged if c['outside'])}, errors "
          f"{sum(1 for c in report['cases'] if 'error' in c)}, skipped {sum(1 for c in report['cases'] if 'skipped' in c)}; wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
