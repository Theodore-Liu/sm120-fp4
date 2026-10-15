#!/usr/bin/env python3
"""DeepGEMM's SM120 MQA logits kernel (`fp8_fp4_mqa_logits`, FP8 q and kv, fp32 per-token kv scales) at spans, head counts
and row counts its test does not enumerate, against a float64 oracle of the README's formula.

Registered before any case ran: mlsys-5090-research docs/d1-mqa-logits-registration-2026-10-15.md. Output row i, column c
(c < end_i - start_i, j = start_i + c): sum_h weights[i, h] * relu(q[i, h, :] . kv[j, :] * kv_sf[j]). The oracle is first
run on the test's own enumerated case (tests/test_sm120_mqa.py: integer operands, unit scales) and must agree with the
kernel within the test's 1e-5 and with the census bound, else the probe stops with no verdict.

    cd <outside the DeepGEMM source tree>
    PYTHONPATH=<sm120-fp4>:<sm120-fp4>/scripts python deepgemm_mqa_logits_probe.py --log <marker> --out <json>
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

U32 = 2.0 ** -23


def ulp32(x: torch.Tensor) -> torch.Tensor:
    return torch.ldexp(torch.ones_like(x), torch.floor(torch.log2(x.clamp(min=2.0 ** -126))) - 23)


def oracle(qd, kvd, sfd, wd, starts, ends, max_seqlen_k):
    """float64 logits [seq_len, max_seqlen_k] of the README formula, the per-element bound, and the valid mask."""
    seq_len, heads, hd = qd.shape
    out = torch.zeros((seq_len, max_seqlen_k), dtype=torch.float64)
    bnd = torch.ones((seq_len, max_seqlen_k), dtype=torch.float64)
    valid = torch.zeros((seq_len, max_seqlen_k), dtype=torch.bool)
    for i in range(seq_len):
        s, e = int(starts[i]), int(ends[i])
        span = min(e - s, max_seqlen_k)
        if span <= 0:
            continue
        k = kvd[s: s + span]                                   # [span, hd]
        dots = torch.einsum("hd,nd->hn", qd[i], k) * sfd[s: s + span][None, :]   # [heads, span]
        absdots = torch.einsum("hd,nd->hn", qd[i].abs(), k.abs()) * sfd[s: s + span][None, :]
        v = (wd[i][:, None] * dots.clamp(min=0)).sum(0)        # [span]
        slack = (hd + heads + 2) * U32 * (wd[i].abs()[:, None] * absdots).sum(0)
        out[i, :span] = v
        bnd[i, :span] = slack + 0.5 * ulp32((v.abs() + slack).float()).double()
        valid[i, :span] = True
    return out, bnd, valid


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-cases", type=int, default=0)
    a = ap.parse_args(argv)

    def mark(s: str) -> None:
        with open(a.log, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} mqa {s}\n")

    import deep_gemm
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "deep_gemm": getattr(deep_gemm, "__version__", "?"),
              "deep_gemm_path": deep_gemm.__file__, "cases": []}
    gen = torch.Generator().manual_seed(20261015)
    rnd = random.Random(20261015)

    def run_case(label, head_dim, heads, starts, ends, tokens, max_seqlen_k, integer=False, runs=2):
        seq_len = len(starts)
        rec = {"label": label, "head_dim": head_dim, "heads": heads, "seq_len": seq_len, "tokens": tokens,
               "max_seqlen_k": max_seqlen_k, "spans": [int(e - s) for s, e in zip(starts, ends)][:12],
               "max_start": int(max(starts)), "integer_operands": integer}
        try:
            if integer:
                q_cpu = torch.randint(-2, 3, (seq_len, heads, head_dim), generator=gen).float()
                kv_cpu = torch.randint(-2, 3, (tokens, head_dim), generator=gen).float()
                w_cpu = torch.randint(-2, 3, (seq_len, heads), generator=gen).float()
                sf_cpu = torch.ones(tokens)
            else:
                q_cpu = torch.randn((seq_len, heads, head_dim), generator=gen)
                kv_cpu = torch.randn((tokens, head_dim), generator=gen)
                w_cpu = torch.randn((seq_len, heads), generator=gen)
                sf_cpu = torch.ldexp(torch.ones(tokens), torch.randint(-3, 4, (tokens,), generator=gen).float())
            q = q_cpu.to(torch.float8_e4m3fn).cuda()
            kv = kv_cpu.to(torch.float8_e4m3fn).cuda()
            qd, kvd = q.cpu().double(), kv.cpu().double()        # the quantised operands, as the kernel sees them
            weights = w_cpu.cuda().float()
            scales = sf_cpu.cuda().float()
            st = torch.tensor(starts, dtype=torch.int32, device="cuda")
            en = torch.tensor(ends, dtype=torch.int32, device="cuda")
            exact, bnd, valid = oracle(qd, kvd, sf_cpu.double(), w_cpu.double(), starts, ends, max_seqlen_k)
            outs = []
            for _ in range(runs):
                mark(f"{label} hd={head_dim} heads={heads} rows={seq_len} tokens={tokens} max={max_seqlen_k}")
                o = deep_gemm.fp8_fp4_mqa_logits((q, None), (kv, scales), weights, st, en, max_seqlen_k=max_seqlen_k)
                torch.cuda.synchronize()
                outs.append(o.double().cpu())
            err = (outs[0] - exact).abs()
            w = (err > bnd) & valid
            rec["elements"] = int(valid.sum()); rec["outside"] = int(w.sum()); rec["nan"] = int(torch.isnan(outs[0][valid]).sum())
            rec["max_err_over_bound"] = float((err[valid] / bnd[valid]).max()) if valid.any() else 0.0
            rec["max_abs_err"] = float(err[valid].max()) if valid.any() else 0.0
            rec["distinct_outputs"] = len({o[valid].numpy().tobytes() for o in outs})
            if rec["outside"]:
                rows = w.any(dim=1).nonzero().flatten().tolist()
                rec["rows_outside"] = rows[:10]
                r0 = rows[0]; cols = w[r0].nonzero().flatten().tolist()
                rec["example"] = {"row": r0, "span": int(ends[r0] - starts[r0]), "start": int(starts[r0]), "cols_outside_first": cols[:8],
                                  "cols_outside_count": len(cols), "got": [round(float(x), 5) for x in outs[0][r0, cols[:4]]],
                                  "exact": [round(float(x), 5) for x in exact[r0, cols[:4]]]}
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        return rec

    # ---- instrument check: the test's own case
    chk = run_case("test-case", 128, 16, [0, 0, 0, 0], [0, 1, 65, 193], 256, 256, integer=True)
    report["oracle_check"] = chk
    print(json.dumps(chk)[:500], flush=True)
    if "error" in chk or chk["outside"] or chk["max_abs_err"] > 1e-5:
        raise SystemExit("oracle check failed on the test's own case; no verdict")

    plan = []
    for hd in (32, 64, 128):
        for heads in (1, 2, 4, 8, 16, 32, 64, 128):
            plan.append(("A-starts0", hd, heads, [0] * 5, [0, 1, 127, 128, 129], 256, 129))
            plan.append(("B-starts", hd, heads, [5, 100, 300, 1000], [6, 227, 429, 1255], 1300, 255))
            plan.append(("C-long", hd, heads, [0, 50, 103], [4095, 4146, 4200], 4200, 4097))
            st = [rnd.randint(0, 200) for _ in range(129)]
            en = [s + rnd.randint(0, 300) for s in st]
            plan.append(("D-129rows", hd, heads, st, en, 600, max(e - s for s, e in zip(st, en))))
    plan.append(("E-max-twice", 128, 16, [5, 100, 300, 1000], [6, 227, 429, 1255], 1300, 510))
    plan.append(("E-max-short", 128, 16, [5, 100, 300, 1000], [6, 227, 429, 1255], 1300, 100))

    for i, (label, hd, heads, st, en, tokens, mx) in enumerate(plan):
        if a.max_cases and i >= a.max_cases:
            break
        rec = run_case(label, hd, heads, st, en, tokens, mx)
        report["cases"].append(rec)
        flag = rec.get("outside") or rec.get("nan") or "error" in rec or rec.get("distinct_outputs", 1) != 1
        if flag or (i + 1) % 12 == 0:
            print(json.dumps({k: v for k, v in rec.items() if k != "spans"})[:420], flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    judged = [c for c in report["cases"] if "outside" in c]
    print(f"cases {len(report['cases'])}, judged {len(judged)}, outside in {sum(1 for c in judged if c['outside'])}, nan in "
          f"{sum(1 for c in judged if c['nan'])}, nondeterministic {sum(1 for c in judged if c.get('distinct_outputs', 1) != 1)}, "
          f"refused/errors {sum(1 for c in report['cases'] if 'error' in c)}, elements {sum(c['elements'] for c in judged)}; wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
