#!/usr/bin/env python3
"""DeepGEMM's SM120 m-grouped GEMMs at the two enumerated sets deepgemm_grouped_probe.py (2026-10-12) left unjudged: the
contiguous GEMM with an MN-major B operand (48 cases) and the masked GEMM in its PSUM layout, which the library's test runs
through the contiguous entry point with `use_psum_layout=True` (64 cases). Inside the library's own enumeration; the rows
are inside rows for the inside-against-outside count.

Registered before any case ran: mlsys-5090-research docs/d1-grouped-inside-registration-2026-10-15.md. Instrument check
first: the same bf16 `b` cast K-major and MN-major must dequantise identically under the two decodes (FP8 and FP4).

    cd <outside the DeepGEMM source tree>
    PYTHONPATH=<sm120-fp4>:<sm120-fp4>/scripts:<DeepGEMM>/tests python deepgemm_grouped_inside_probe.py --out <json>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from deepgemm_grouped_probe import bound_for, decode_fp4, dequant, judge, selftest_decode, sf_to_float


def dequant_mn_major(x: torch.Tensor, sf: torch.Tensor, is_fp4: bool, row_gran: int, k_gran: int) -> torch.Tensor:
    """An MN-major operand as the generator hands it over: FP8 as a transposed view (logical [.., rows, k], read through
    the view), FP4 as the `.mT` view of codes packed along the rows (unpack along rows, transpose back)."""
    if is_fp4:
        packed_kn = x.mT.contiguous()                          # [.., k, rows / 2], pairs of consecutive rows in one byte
        v = decode_fp4(packed_kn).transpose(-1, -2)            # [.., rows, k]
        rows, k = v.shape[-2], v.shape[-1]
        s = sf_to_float(sf)[..., : -(-rows // row_gran), : -(-k // k_gran)]
        s = s.repeat_interleave(row_gran, dim=-2)[..., :rows, :].repeat_interleave(k_gran, dim=-1)[..., :k]
        return v * s
    return dequant(x, sf, False, row_gran, k_gran)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-cases", type=int, default=0)
    a = ap.parse_args(argv)
    import deep_gemm
    import generators as G
    selftest_decode()

    # ---- instrument check: K-major and MN-major casts of the same b dequantise identically
    b = torch.randn((2, 256, 512), device="cuda", dtype=torch.bfloat16)
    for is_fp4, gran in ((False, 128), (True, 32)):
        km = G.grouped_cast_fp8_fp4_with_major(b, G.MajorTypeAB.KMajor, gran, is_fp4, True, use_block_cast_for_fp8=True)
        mn = G.grouped_cast_fp8_fp4_with_major(b, G.MajorTypeAB.MNMajor, gran, is_fp4, True, use_block_cast_for_fp8=True)
        dk = dequant(km[0], km[1], is_fp4, 1 if is_fp4 else gran, gran)
        dm = dequant_mn_major(mn[0], mn[1], is_fp4, 1 if is_fp4 else gran, gran)
        assert torch.equal(dk, dm), f"MN-major decode disagrees with the K-major decode (fp4={is_fp4})"
    print("decode check: K-major and MN-major casts dequantise identically (FP8, FP4)", flush=True)

    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "deep_gemm": getattr(deep_gemm, "__version__", "?"),
              "deep_gemm_path": deep_gemm.__file__, "cases": []}
    n_done = 0

    # ---------------------------------------------------------------- contiguous, MN-major B
    for cfg in G.enumerate_m_grouped_contiguous(dtype=torch.float8_e4m3fn):
        kernel_type, qc, num_groups, expected_m, n, k, major_a, major_b, use_psum, ensure_zero = cfg
        if major_b.is_k_major():
            continue
        rec = {"layout": "contiguous-psum" if use_psum else "contiguous-index", "set": "mn-major-b", "groups": num_groups,
               "expected_m": expected_m, "n": n, "k": k, "major_a": str(major_a), "major_b": str(major_b),
               "ensure_zero_padding": bool(ensure_zero), "fp4_a": bool(qc.is_fp4_a), "fp4_b": bool(qc.is_fp4_b),
               "gran_k_a": qc.gran_k_a, "gran_k_b": qc.gran_k_b}
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
            ad = dequant(aa[0], aa[1], qc.is_fp4_a, 1, qc.gran_k_a)
            bd = dequant_mn_major(bb[0], bb[1], qc.is_fp4_b, 1 if qc.is_fp4_b else qc.gran_k_b, qc.gran_k_b)
            lay = layout.cpu().long(); vm = valid.cpu()
            exact = torch.zeros(m, n, dtype=torch.float64); bnd = torch.zeros(m, n, dtype=torch.float64)
            if use_psum:
                start = 0; align = deep_gemm.get_mk_alignment_for_contiguous_layout()
                for g in range(num_groups):
                    end = int(lay[g]); rows = torch.arange(start, end)
                    if rows.numel():
                        exact[rows] = ad[rows] @ bd[g].T; bnd[rows] = bound_for(ad[rows], bd[g], exact[rows], k)
                    start = -(-end // align) * align
            else:
                for g in range(num_groups):
                    rows = (lay == g).nonzero(as_tuple=True)[0]
                    if rows.numel():
                        exact[rows] = ad[rows] @ bd[g].T; bnd[rows] = bound_for(ad[rows], bd[g], exact[rows], k)
            rec["m"] = m; rec["valid_rows"] = int(vm.sum())
            rec.update(judge(outs[0][vm], exact[vm], bnd[vm]))
            rec["distinct_outputs"] = len({o[vm].to(torch.bfloat16).view(torch.int16).numpy().tobytes() for o in outs})
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
        report["cases"].append(rec)
        if rec.get("outside") or "error" in rec or rec.get("distinct_outputs", 1) != 1 or n_done % 8 == 0:
            print(json.dumps(rec)[:400], flush=True)
        n_done += 1
        if a.max_cases and n_done >= a.max_cases:
            break

    # ---------------------------------------------------------------- masked, PSUM layout (through the contiguous entry point)
    n_done2 = 0
    for cfg in G.enumerate_m_grouped_masked(dtype=torch.float8_e4m3fn):
        kernel_type, qc, num_groups, max_m, expected_m, n, k, use_psum = cfg[:8]
        if not use_psum:
            continue
        rec = {"layout": "masked-psum", "set": "masked-psum", "groups": num_groups, "max_m": max_m, "expected_m": expected_m,
               "n": n, "k": k, "fp4_a": bool(qc.is_fp4_a), "fp4_b": bool(qc.is_fp4_b), "gran_k_a": qc.gran_k_a, "gran_k_b": qc.gran_k_b}
        try:
            use_ue8m0 = G.get_ue8m0_usage(kernel_type)
            recipe, recipe_a, recipe_b = qc.get_recipes()
            exp_m = int(expected_m * 1.2)
            deep_gemm.set_mk_alignment_for_contiguous_layout(deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(exp_m))
            aa, bb, psum_m, d, _, valid = G.generate_m_grouped_masked(num_groups, max_m, expected_m, n, k, use_ue8m0=use_ue8m0,
                                                                      use_psum_layout=True, quant_config=qc)
            outs = []
            for _ in range(3):
                d.zero_()
                deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(aa, bb, d, psum_m, use_psum_layout=True, expected_m_for_psum_layout=exp_m,
                                                              disable_ue8m0_cast=not use_ue8m0, recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)
                torch.cuda.synchronize()
                outs.append(d.double().cpu())
            ad = dequant(aa[0], aa[1], qc.is_fp4_a, 1, qc.gran_k_a)
            bd = dequant(bb[0], bb[1], qc.is_fp4_b, 1 if qc.is_fp4_b else qc.gran_k_b, qc.gran_k_b)
            ends = psum_m.cpu().long(); vm = valid.cpu(); total_m = d.shape[0]
            exact = torch.zeros(total_m, n, dtype=torch.float64); bnd = torch.ones(total_m, n, dtype=torch.float64)
            start = 0; align = deep_gemm.get_mk_alignment_for_contiguous_layout()
            for g in range(num_groups):
                end = int(ends[g]); rows = torch.arange(start, end)
                if rows.numel():
                    exact[rows] = ad[rows] @ bd[g].T; bnd[rows] = bound_for(ad[rows], bd[g], exact[rows], k)
                start = -(-end // align) * align
            rec["total_m"] = total_m; rec["valid_rows"] = int(vm.sum())
            rec.update(judge(outs[0][vm], exact[vm], bnd[vm]))
            rec["distinct_outputs"] = len({o[vm].to(torch.bfloat16).view(torch.int16).numpy().tobytes() for o in outs})
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
        report["cases"].append(rec)
        if rec.get("outside") or "error" in rec or rec.get("distinct_outputs", 1) != 1 or n_done2 % 8 == 0:
            print(json.dumps(rec)[:400], flush=True)
        n_done2 += 1
        if a.max_cases and n_done2 >= a.max_cases:
            break

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    judged = [c for c in report["cases"] if "outside" in c]
    print(f"cases {len(report['cases'])}, judged {len(judged)}, outside in {sum(1 for c in judged if c['outside'])}, nan in "
          f"{sum(1 for c in judged if c['nan'])}, nondeterministic {sum(1 for c in judged if c.get('distinct_outputs', 1) != 1)}, "
          f"errors {sum(1 for c in report['cases'] if 'error' in c)}, elements {sum(c['elements'] for c in judged)}; wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
