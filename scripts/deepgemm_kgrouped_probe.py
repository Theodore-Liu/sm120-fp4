#!/usr/bin/env python3
"""DeepGEMM's SM120 k-grouped FP8 GEMMs (`k_grouped_fp8_gemm_tn_contiguous`, MN-major operands, and
`k_grouped_fp8_gemm_nt_contiguous`, K-major operands) at shapes the contract allows but the library's own test enumerations
do not reach, against the exact oracle (vllm-project/DeepGEMM dev).

The smoke run of 2026-10-15 found a host-side rule the README does not state: the k-grouped SF packing asserts
`num_groups <= 128 and mn % 4 == 0` (csrc/jit_kernels/impls/smxx_layout.hpp:231), so m = 1, 7, 65, 129 are refused with a
message. They are kept as one refusal probe each; the judged m set is 4, 12, 68, 132, 300.

Registered before any case ran: mlsys-5090-research docs/d1-next-sweep-outside-enumeration-2026-10-14.md. The contract, read
from the host checks (csrc/apis/gemm.hpp `k_grouped_fp8_gemm_*_contiguous`, `check_k_grouped_args`): recipe (1, 1, gran_k)
with gran_k 32 or 128 (the NT entry point pins (1, 1, 128) and requires an accumulator c); `k_alignment % 128 == 0`; every
group's k a multiple of k_alignment; a, b, d, c contiguous; d `(groups, m, n)` bf16 or fp32 with c of d's dtype; m, n > 0
with no alignment stated. Operands come from the library's own `tests/generators.py::generate_k_grouped_contiguous`, called
as `tests/test_fp8_fp4.py` calls it, so the operand contract (per-group compact SF rows padded to k_alignment, zero padding
codes) is DeepGEMM's. Before any verdict the probe's decode of that layout is checked on one case against the generator's
own bf16 reference of the unquantised operands.

Outside the enumeration (the SM120 file tests m in {48, 128, 256, 768}, n in {128, 256, 2048}, a handful of ks lists, and
constant operands with unit scales; the generic test m in {768, 4096, 7168}, n in {2048, 7168}): m in {4, 12, 68, 132, 300},
n in {8, 24, 136, 1000}, three group patterns (3 groups [8192, 128, 256]; 64 groups with 56 empty; 8 groups near 1024),
TN at (gran_k, k_alignment) in {(128, 128), (32, 128), (32, 256), (32, 384)}, NT at (128, 128); bf16 output without c,
fp32 with c aliased to d, fp32 with c separate. Each valid output element is judged against the exact product of the
quantised operands (float64) under the census bound; each case runs twice and the distinct-output count is recorded; a
marker line is written before every call. Refusals are recorded with their message.

    cd <anywhere outside the DeepGEMM source tree>
    PYTHONPATH=<sm120-fp4>:<sm120-fp4>/scripts:<DeepGEMM>/tests python scripts/deepgemm_kgrouped_probe.py --log <marker> --out <json>
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from deepgemm_edge_probe import bound, judge

M_LIST = (4, 12, 68, 132, 300)  # off the 64-row tile; the SF packing kernel asserts mn % 4 == 0 (smxx_layout.hpp:231)
M_REFUSAL_PROBES = (1, 7)  # recorded once each as refusals
N_LIST = (8, 24, 136, 1000)
TN_LAYOUTS = ((128, 128), (32, 128), (32, 256), (32, 384))  # (gran_k, k_alignment)


def group_patterns(rnd: random.Random):
    yield "3-uneven", [8192, 128, 256]
    lens = [0] * 64
    for i in rnd.sample(range(64), 8):
        lens[i] = rnd.choice([128, 256])
    yield "64-mostly-empty", lens
    yield "8-near-1024", [max(1, int(1024 * rnd.uniform(0.7, 1.3))) for _ in range(8)]


def dequant_tn(x, sf, gran_k):
    """MN-major operand `(total_k, mn)` with SF `(total_k / gran_k, mn)`: float64 `(total_k, mn)`."""
    xd = x.detach().cpu().double()
    sd = sf.detach().cpu().double().repeat_interleave(gran_k, dim=0)[: xd.shape[0]]
    return xd * sd


def dequant_nt(data, sf, host_ks, mn, gran_k):
    """K-major operand: flat per-group `(mn, host_k)` row-major blocks, SF `(mn, total_k / gran_k)`: float64 `(total_k, mn)`."""
    out = torch.zeros((sum(host_ks), mn), dtype=torch.float64)
    sd = sf.detach().cpu().double().repeat_interleave(gran_k, dim=1)
    start = 0
    for hk in host_ks:
        if hk:
            blk = data[start * mn: (start + hk) * mn].detach().cpu().double().view(mn, hk)
            out[start: start + hk] = (blk * sd[:, start: start + hk]).T
        start += hk
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-cases", type=int, default=0)
    a = ap.parse_args(argv)

    def mark(s: str) -> None:
        with open(a.log, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} kgrouped {s}\n")

    import deep_gemm
    from deep_gemm.testing import calc_diff
    from generators import MajorTypeAB, generate_k_grouped_contiguous

    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "deep_gemm": getattr(deep_gemm, "__version__", "?"),
              "deep_gemm_path": deep_gemm.__file__, "cases": []}
    rnd = random.Random(20261015)
    torch.manual_seed(20261015)

    def run_case(entry, m, n, pattern_name, ks, gran_k, k_alignment, out_dtype, accumulate, c_mode, runs=2):
        """One case; returns the record. c_mode: 'none' | 'alias' | 'separate'."""
        rec = {"entry": entry, "m": m, "n": n, "pattern": pattern_name, "groups": len(ks), "logical_ks": ks, "gran_k": gran_k,
               "k_alignment": k_alignment, "out_dtype": str(out_dtype).replace("torch.", ""), "c": c_mode}
        major = MajorTypeAB.KMajor if entry == "nt" else MajorTypeAB.MNMajor
        fn = deep_gemm.k_grouped_fp8_gemm_nt_contiguous if entry == "nt" else deep_gemm.k_grouped_fp8_gemm_tn_contiguous
        try:
            deep_gemm.set_mk_alignment_for_contiguous_layout(k_alignment)
            total_k, qa, qb, c, d, ref_d, layout, host_ks = generate_k_grouped_contiguous(
                len(ks), m, n, major, major, list(ks), use_ue8m0=True, gran_k=gran_k, k_alignment=k_alignment,
                use_psum_layout=False, accumulate=accumulate, out_dtype=out_dtype)
            rec["host_ks"] = host_ks
            if entry == "nt":
                ad = dequant_nt(qa[0], qa[1], host_ks, m, gran_k)
                bd = dequant_nt(qb[0], qb[1], host_ks, n, gran_k)
            else:
                ad = dequant_tn(qa[0], qa[1], gran_k)
                bd = dequant_tn(qb[0], qb[1], gran_k)
            c0 = c.detach().cpu().double() if accumulate else None
            exact = torch.zeros((len(ks), m, n), dtype=torch.float64)
            bnd = torch.ones((len(ks), m, n), dtype=torch.float64)
            start = 0
            for g, (lk, hk) in enumerate(zip(ks, host_ks)):
                if hk:
                    ag, bg = ad[start: start + hk], bd[start: start + hk]
                    exact[g] = ag.T @ bg
                    if accumulate:
                        exact[g] += c0[g]
                    bnd[g] = bound(ag.T, bg.T, exact[g], max(lk, 1) + (1 if accumulate else 0), out_dtype)
                elif accumulate:
                    exact[g] = c0[g]
                    bnd[g] = bound(torch.zeros(m, 1, dtype=torch.float64), torch.zeros(n, 1, dtype=torch.float64), exact[g], 1, out_dtype)
                start += hk
            rec["calc_diff_vs_generator_ref"] = float(calc_diff(exact.float().cuda(), ref_d.float()))
            outs = []
            for _ in range(runs):
                if c_mode == "alias":
                    dd = c.clone()
                    cc = dd
                elif c_mode == "separate":
                    dd = torch.empty_like(d)
                    cc = c
                else:
                    dd = torch.empty_like(d)
                    cc = None
                mark(f"{entry} m={m} n={n} {pattern_name} gran_k={gran_k} align={k_alignment} out={rec['out_dtype']} c={c_mode}")
                fn(qa, qb, dd, host_ks, layout, cc, recipe=(1, 1, gran_k))
                torch.cuda.synchronize()
                outs.append(dd.double().cpu())
            rec.update(judge(outs[0], exact, bnd))
            rec["distinct_outputs"] = len({o.numpy().tobytes() for o in outs})
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        return rec

    # ---- instrument check: the probe's decode of the generator's layout against the generator's own bf16 reference
    for entry in ("tn", "nt"):
        chk = run_case(entry, 128, 256, "check", [128, 256], 128, 128, torch.float32, True, "alias", runs=1)
        report[f"decode_check_{entry}"] = chk
        if "error" in chk:
            print(json.dumps(chk)[:600])
            raise SystemExit(f"decode check failed on {entry}: {chk['error']}")
        if chk["calc_diff_vs_generator_ref"] > 1e-2:
            print(json.dumps(chk)[:600])
            raise SystemExit(f"decode check: the probe's exact product is far from the generator's reference on {entry} "
                             f"(calc_diff {chk['calc_diff_vs_generator_ref']:.3g}); the layout decode is wrong, no verdict")
        print(f"decode check {entry}: calc_diff vs generator ref {chk['calc_diff_vs_generator_ref']:.2e}, "
              f"outside {chk['outside']} of {chk['elements']}", flush=True)

    # ---- refusal probes (m not a multiple of 4), one each
    for m in M_REFUSAL_PROBES:
        rec = run_case("tn", m, 24, "3-uneven", [8192, 128, 256], 128, 128, torch.bfloat16, False, "none")
        rec["refusal_probe"] = True
        report["cases"].append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k not in ("logical_ks", "host_ks")})[:400], flush=True)

    # ---- the registered cases
    n_done = 0
    for pattern_name, ks in group_patterns(rnd):
        for m in M_LIST:
            for n in N_LIST:
                plan = []
                plan.append(("tn", 128, 128, torch.bfloat16, False, "none"))
                plan.append(("tn", 128, 128, torch.float32, True, "alias"))
                plan.append(("tn", 128, 128, torch.float32, True, "separate"))
                for gran_k, k_alignment in TN_LAYOUTS[1:]:
                    # the SM120 k-grouped kernel asserts c.has_value() (sm120_fp8_fp4_gemm_1d1d.hpp:337): bf16 without c
                    # is a refusal on every shape (240 of them in the first full run), so these layouts run with fp32 + c
                    plan.append(("tn", gran_k, k_alignment, torch.float32, True, "alias"))
                plan.append(("nt", 128, 128, torch.float32, True, "alias"))
                for entry, gran_k, k_alignment, out_dtype, accumulate, c_mode in plan:
                    if a.max_cases and n_done >= a.max_cases:
                        break
                    rec = run_case(entry, m, n, pattern_name, ks, gran_k, k_alignment, out_dtype, accumulate, c_mode)
                    report["cases"].append(rec)
                    n_done += 1
                    flag = rec.get("outside") or rec.get("nan") or "error" in rec or rec.get("distinct_outputs", 1) != 1
                    if flag or n_done % 20 == 0:
                        print(json.dumps({k: v for k, v in rec.items() if k not in ("logical_ks", "host_ks")})[:400], flush=True)

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
