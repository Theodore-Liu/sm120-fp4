#!/usr/bin/env python3
"""DeepGEMM's SM120 FP8 GEMMs at shapes the contract allows but the library's test enumerations do not reach, against the exact
oracle (vllm-project/DeepGEMM dev).

The contract, read from the host checks (csrc/apis/gemm.hpp, csrc/apis/sm120_dispatch.hpp, csrc/utils/layout.hpp) and the
README: m, n, k positive; d bf16 (or fp32 with an optional fp32 accumulator c, `d = a @ b.T + c`); only mixed FP8 x FP4 needs
k % 128 == 0; grouped contiguous segments aligned to `get_mk_alignment_for_contiguous_layout()`; masked_m per group in [0, max_m].
FP8 rows are read by TMA, so k is kept a multiple of 16 (16-byte rows); the output's row stride must also be a 16-byte
multiple, so outputs are allocated with padded rows and passed as column slices (`strided`), as DeepGEMM's own SM120 odd-n test
does; odd n is then in contract. The library's casts (`per_token_cast_to_fp8`, `per_block_cast_to_fp8`, UE8M0 scales)
pad unaligned shapes themselves, so unpadded m, n, k are supported inputs.

Cases:
  dense   fp8_gemm_nt, legacy recipe (A per token x 128, B per 128 x 128 block): m in {1, 2, 7, 65, 129, 300}, n in {8, 9, 24,
          136, 1000, 4100}, k in {16, 48, 144, 208, 1040, 4112}; bf16 d; plus fp32 d with c at a subset
  grouped m_grouped contiguous (index and PSUM layouts) and masked with many groups, most empty, and segment lengths of 0, 1
          and alignment +- 1, at n and k off 128
Each valid output element is judged against the exact product of the quantised operands (float64) under the census bound; each
case runs twice and the distinct-output count is recorded. A marker line is written before every call.

    PYTHONPATH=<sm120-fp4> python scripts/deepgemm_edge_probe.py --log <marker file> --out reports/<file>.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

M_LIST = (1, 2, 7, 65, 129, 300)
N_LIST = (8, 9, 24, 136, 1000, 4100)
K_LIST = (16, 48, 144, 208, 1040, 4112)


def expand(sf: torch.Tensor, row_gran: int, k_gran: int, rows: int, k: int) -> torch.Tensor:
    s = sf.detach().cpu().double()
    s = s[..., : -(-rows // row_gran), : -(-k // k_gran)]
    return s.repeat_interleave(row_gran, dim=-2)[..., :rows, :].repeat_interleave(k_gran, dim=-1)[..., :k]


def strided(shape, dtype, fill=0.0):
    """An output (or accumulator) whose rows start on 16-byte boundaries: the TMA descriptor requires the row stride in bytes
    to be a multiple of 16 (`runtime_utils.hpp: num_gmem_outer_stride_bytes % 16u == 0u`), so an n whose rows are not 16-byte
    multiples is written into a wider buffer and passed as a column slice, as DeepGEMM's own SM120 odd-n test does."""
    *lead, n = shape
    per = 16 // torch.empty((), dtype=dtype).element_size()
    width = -(-n // per) * per
    return torch.full((*lead, width), fill, dtype=dtype, device="cuda")[..., :n]


def bound(ad, bd, exact, k, out_dtype):
    from sm120fp4 import oracle
    sabs = ad.abs() @ bd.abs().transpose(-1, -2)
    pre = k * oracle.U32 * sabs + 2 * oracle.U64 * sabs
    return pre + 0.5 * oracle._ulp(exact.abs() + pre, out_dtype)


def judge(d, exact, bnd):
    err = (d - exact).abs()
    return {"elements": int(err.numel()), "outside": int((err > bnd).sum()), "nan": int(torch.isnan(d).sum()),
            "max_err_over_bound": float((err / bnd.clamp(min=1e-300)).max()) if err.numel() else 0.0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    def mark(s: str) -> None:
        with open(a.log, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} edge {s}\n")

    import deep_gemm
    from deep_gemm.utils.math import per_block_cast_to_fp8, per_token_cast_to_fp8
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "deep_gemm": getattr(deep_gemm, "__version__", "?"), "cases": []}
    gen = torch.Generator().manual_seed(20261012)

    def cast_a(x):   # per token x 128
        q, s = per_token_cast_to_fp8(x.cuda(), use_ue8m0=True)
        return q, s

    def cast_b(w):   # per 128 x 128 block
        q, s = per_block_cast_to_fp8(w.cuda(), use_ue8m0=True)
        return q, s

    # ---------------------------------------------------------------- dense
    for m in M_LIST:
        for n in N_LIST:
            for k in K_LIST:
                for acc in ((False, True) if (m in (7, 129) and n in (9, 136) and k in (144, 1040)) else (False,)):
                    rec = {"kind": "dense", "m": m, "n": n, "k": k, "fp32_with_c": acc}
                    try:
                        x = torch.randn(m, k, generator=gen).to(torch.bfloat16)
                        w = torch.randn(n, k, generator=gen).to(torch.bfloat16)
                        xq, xs = cast_a(x)
                        wq, ws = cast_b(w)
                        ad = xq.cpu().double() * expand(xs, 1, 128, m, k)
                        bd = wq.cpu().double() * expand(ws, 128, 128, n, k)
                        exact = ad @ bd.T
                        if acc:
                            c = torch.randn(m, n, generator=gen).float()
                            exact = exact + c.double()
                            bnd = bound(ad, bd, exact, k + 1, torch.float32)
                        else:
                            bnd = bound(ad, bd, exact, k, torch.bfloat16)
                        outs = []
                        for _ in range(2):
                            d = strided((m, n), torch.float32 if acc else torch.bfloat16)
                            mark(f"dense m={m} n={n} k={k} acc={acc}")
                            if acc:
                                cc = strided((m, n), torch.float32); cc.copy_(c.cuda())
                                deep_gemm.fp8_gemm_nt((xq, xs), (wq, ws), d, c=cc)
                            else:
                                deep_gemm.fp8_gemm_nt((xq, xs), (wq, ws), d)
                            torch.cuda.synchronize()
                            outs.append(d.double().cpu())
                        rec.update(judge(outs[0], exact, bnd))
                        rec["distinct_outputs"] = len({o.numpy().tobytes() for o in outs})
                    except Exception as exc:  # noqa: BLE001
                        rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                    report["cases"].append(rec)
                    if rec.get("outside") or rec.get("nan") or "error" in rec or rec.get("distinct_outputs", 1) != 1:
                        print(json.dumps(rec)[:400], flush=True)

    # ---------------------------------------------------------------- grouped
    deep_gemm.set_mk_alignment_for_contiguous_layout(deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout())
    align = deep_gemm.get_mk_alignment_for_contiguous_layout()
    report["mk_alignment"] = align
    rnd = random.Random(7)
    for num_groups, n, k in ((64, 136, 208), (64, 1000, 1040), (17, 4100, 144), (8, 9, 4112)):
        # segment lengths: most groups empty, the rest 1, align - 1, align, align + 1 or random
        lens = []
        for g in range(num_groups):
            r = rnd.random()
            lens.append(0 if r < 0.6 else rnd.choice([1, align - 1, align, align + 1, rnd.randint(1, 3 * align)]))
        if sum(lens) == 0:
            lens[-1] = 1
        bs = [cast_b(torch.randn(n, k, generator=gen).to(torch.bfloat16)) for _ in range(num_groups)]
        bq = torch.stack([b[0] for b in bs])
        bsf = torch.stack([b[1] for b in bs])
        bd = torch.stack([b[0].cpu().double() * expand(b[1], 128, 128, n, k) for b in bs])
        for layout in ("index", "psum"):
            rec = {"kind": f"contiguous-{layout}", "groups": num_groups, "n": n, "k": k, "segment_lengths": lens,
                   "empty_groups": sum(1 for x in lens if x == 0)}
            try:
                starts, m = [], 0
                for ln in lens:
                    starts.append(m)
                    m += -(-ln // align) * align
                m = max(m, align)
                x = torch.zeros(m, k, dtype=torch.bfloat16)
                gl = torch.full((m,), -1, dtype=torch.int32) if layout == "index" else torch.zeros(num_groups, dtype=torch.int32)
                valid = torch.zeros(m, dtype=torch.bool)
                for g, (s0, ln) in enumerate(zip(starts, lens)):
                    x[s0:s0 + ln] = torch.randn(ln, k, generator=gen).to(torch.bfloat16)
                    valid[s0:s0 + ln] = True
                    if layout == "index":
                        gl[s0:s0 + ln] = g
                    else:
                        gl[g] = s0 + ln
                xq, xs = cast_a(x)
                ad = xq.cpu().double() * expand(xs, 1, 128, m, k)
                exact = torch.zeros(m, n, dtype=torch.float64)
                bnd = torch.ones(m, n, dtype=torch.float64)
                for g, (s0, ln) in enumerate(zip(starts, lens)):
                    if ln:
                        exact[s0:s0 + ln] = ad[s0:s0 + ln] @ bd[g].T
                        bnd[s0:s0 + ln] = bound(ad[s0:s0 + ln], bd[g], exact[s0:s0 + ln], k, torch.bfloat16)
                outs = []
                for _ in range(2):
                    d = strided((m, n), torch.bfloat16)
                    mark(f"{rec['kind']} groups={num_groups} n={n} k={k}")
                    deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous((xq, xs), (bq, bsf), d, gl.cuda(), disable_ue8m0_cast=False,
                                                                  use_psum_layout=(layout == "psum"))
                    torch.cuda.synchronize()
                    outs.append(d.double().cpu())
                rec["m"] = m
                rec.update(judge(outs[0][valid], exact[valid], bnd[valid]))
                rec["distinct_outputs"] = len({o[valid].numpy().tobytes() for o in outs})
            except Exception as exc:  # noqa: BLE001
                rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            report["cases"].append(rec)
            print(json.dumps({kk: v for kk, v in rec.items() if kk != "segment_lengths"})[:400], flush=True)
        # masked
        max_m = max(-(-max(lens) // align) * align, align)
        rec = {"kind": "masked", "groups": num_groups, "n": n, "k": k, "masked_m": lens, "max_m": max_m,
               "empty_groups": sum(1 for x in lens if x == 0)}
        try:
            x = torch.zeros(num_groups, max_m, k, dtype=torch.bfloat16)
            for g, ln in enumerate(lens):
                x[g, :ln] = torch.randn(ln, k, generator=gen).to(torch.bfloat16)
            casts = [cast_a(x[g]) for g in range(num_groups)]
            xq = torch.stack([c[0] for c in casts])
            xs = torch.stack([c[1] for c in casts])
            for g, ln in enumerate(lens):
                xs[g, ln:] = 0                     # as tests/generators.py: SFA rows past masked_m zeroed
            ad = torch.stack([casts[g][0].cpu().double() * expand(casts[g][1], 1, 128, max_m, k) for g in range(num_groups)])
            mm = torch.tensor(lens, dtype=torch.int32)
            exp_m = max(1, int(sum(lens) / max(1, num_groups - rec["empty_groups"])))
            outs = []
            for _ in range(2):
                d = strided((num_groups, max_m, n), torch.bfloat16)
                mark(f"masked groups={num_groups} n={n} k={k}")
                deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked((xq, xs), (bq, bsf), d, mm.cuda(), exp_m, disable_ue8m0_cast=False)
                torch.cuda.synchronize()
                outs.append(d.double().cpu())
            res = {"elements": 0, "outside": 0, "nan": 0, "max_err_over_bound": 0.0}
            for g, ln in enumerate(lens):
                if ln:
                    ex = ad[g, :ln] @ bd[g].T
                    j = judge(outs[0][g, :ln], ex, bound(ad[g, :ln], bd[g], ex, k, torch.bfloat16))
                    for kk in ("elements", "outside", "nan"):
                        res[kk] += j[kk]
                    res["max_err_over_bound"] = max(res["max_err_over_bound"], j["max_err_over_bound"])
            rec.update(res)
            rec["distinct_outputs"] = len({b"".join(o[g, :ln].numpy().tobytes() for g, ln in enumerate(lens)) for o in outs})
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        report["cases"].append(rec)
        print(json.dumps({kk: v for kk, v in rec.items() if kk != "masked_m"})[:400], flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    judged = [c for c in report["cases"] if "outside" in c]
    print(f"cases {len(report['cases'])}, judged {len(judged)}, outside in {sum(1 for c in judged if c['outside'])}, nan in "
          f"{sum(1 for c in judged if c['nan'])}, nondeterministic {sum(1 for c in judged if c.get('distinct_outputs', 1) != 1)}, "
          f"errors {sum(1 for c in report['cases'] if 'error' in c)}; wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
