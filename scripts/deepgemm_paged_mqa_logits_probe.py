#!/usr/bin/env python3
"""DeepGEMM's SM120 paged MQA logits kernel (`fp8_fp4_paged_mqa_logits`, FP8 q and keys, fp32 per-token scales) outside its
test enumeration, against a float64 oracle of the README formula with the page gather.

Registered before any case ran: mlsys-5090-research docs/d1-paged-mqa-logits-registration-2026-10-15.md. The cache is built
as tests/test_sm120_mqa.py builds it (FP8 keys then fp32 scales per page row, a 128-byte gap between pages). The oracle is
first run on the test's own geometry with its dyadic operands and must agree with the kernel within the test's 1e-5 and
with the census bound, else the probe stops with no verdict.

    cd <outside the DeepGEMM source tree>
    PYTHONPATH=<sm120-fp4>:<sm120-fp4>/scripts python deepgemm_paged_mqa_logits_probe.py --log <marker> --out <json>
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-cases", type=int, default=0)
    a = ap.parse_args(argv)

    def mark(s: str) -> None:
        with open(a.log, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} paged-mqa {s}\n")

    import deep_gemm
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "deep_gemm": getattr(deep_gemm, "__version__", "?"),
              "deep_gemm_path": deep_gemm.__file__, "cases": []}
    gen = torch.Generator().manual_seed(20261015)
    rnd = random.Random(20261015)

    def run_case(label, head_dim, heads, page_kv, next_n, lengths, table_cpu, num_pages, max_context_len, dyadic=False,
                 runs=2):
        """lengths: per-request context (the last token's); table_cpu: int64 [batch, pages_per_request]."""
        batch = len(lengths)
        rows = batch * next_n
        rec = {"label": label, "head_dim": head_dim, "heads": heads, "page_kv": page_kv, "next_n": next_n, "batch": batch,
               "num_pages": num_pages, "table_cols": int(table_cpu.shape[1]), "max_context_len": max_context_len,
               "lengths": [int(x) for x in lengths][:12], "max_length": int(max(lengths)),
               "pages_reused": int(table_cpu.numel() - len(set(table_cpu.flatten().tolist()))), "dyadic": dyadic}
        try:
            if dyadic:
                q_cpu = torch.randint(-4, 5, (batch, next_n, heads, head_dim), generator=gen).float() / 2
                k_cpu = torch.randint(-4, 5, (num_pages, page_kv, head_dim), generator=gen).float() / 2
                scale_cpu = 2.0 ** torch.randint(-3, 3, (num_pages, page_kv), generator=gen).float()
                w_cpu = torch.randint(-4, 5, (rows, heads), generator=gen).float() / 8
            else:
                q_cpu = torch.randn((batch, next_n, heads, head_dim), generator=gen)
                k_cpu = torch.randn((num_pages, page_kv, head_dim), generator=gen)
                scale_cpu = torch.ldexp(torch.ones(num_pages, page_kv), torch.randint(-3, 4, (num_pages, page_kv), generator=gen).float())
                w_cpu = torch.randn((rows, heads), generator=gen)
            q = q_cpu.to(torch.float8_e4m3fn).cuda()
            k8 = k_cpu.to(torch.float8_e4m3fn)
            qd, kd = q.cpu().double(), k8.double()                 # the quantised operands, as the kernel sees them
            weights = w_cpu.cuda().float()
            row_bytes = page_kv * (head_dim + 4)
            fused = torch.zeros((num_pages, row_bytes + 128), dtype=torch.uint8, device="cuda")
            fused[:, : page_kv * head_dim] = k8.cuda().reshape(num_pages, -1).view(torch.uint8)
            fused[:, page_kv * head_dim: row_bytes] = scale_cpu.cuda().float().view(torch.uint8)
            cache = fused[:, :row_bytes].view(num_pages, page_kv, 1, head_dim + 4)
            context_cpu = torch.tensor(lengths, dtype=torch.int32)[:, None]
            context_cpu = (context_cpu - next_n + 1 + torch.arange(next_n, dtype=torch.int32)).clamp_min(0)
            context = context_cpu.cuda()
            table = table_cpu.to(device="cuda", dtype=torch.int32)
            # the oracle: keys gathered page by page through the table
            exact = torch.zeros((rows, max_context_len), dtype=torch.float64)
            bnd = torch.ones((rows, max_context_len), dtype=torch.float64)
            valid = torch.zeros((rows, max_context_len), dtype=torch.bool)
            for b in range(batch):
                keys = kd[table_cpu[b]].reshape(-1, head_dim)
                scales = scale_cpu[table_cpu[b]].reshape(-1).double()
                for t in range(next_n):
                    c = min(int(context_cpu[b, t]), max_context_len, keys.shape[0])
                    if c <= 0:
                        continue
                    r = b * next_n + t
                    dots = torch.einsum("hd,nd->hn", qd[b, t], keys[:c])               # [heads, c]
                    absdots = torch.einsum("hd,nd->hn", qd[b, t].abs(), keys[:c].abs())
                    wd = w_cpu[r].double()
                    v = (wd[:, None] * dots.clamp(min=0)).sum(0) * scales[:c]
                    slack = (head_dim + heads + 3) * U32 * (wd.abs()[:, None] * absdots).sum(0) * scales[:c]
                    exact[r, :c] = v
                    bnd[r, :c] = slack + 0.5 * ulp32((v.abs() + slack).float()).double()
                    valid[r, :c] = True
            outs = []
            for _ in range(runs):
                mark(f"{label} hd={head_dim} heads={heads} page={page_kv} next_n={next_n} batch={batch} pages={num_pages} max={max_context_len}")
                metadata = deep_gemm.get_paged_mqa_logits_metadata(context, page_kv, deep_gemm.get_num_sms())
                o = deep_gemm.fp8_fp4_paged_mqa_logits((q, None), cache, weights, context, table, metadata, max_context_len)
                torch.cuda.synchronize()
                outs.append(o.double().cpu())
            err = (outs[0] - exact).abs()
            w = (err > bnd) & valid
            rec["elements"] = int(valid.sum()); rec["outside"] = int(w.sum()); rec["nan"] = int(torch.isnan(outs[0][valid]).sum())
            rec["max_err_over_bound"] = float((err[valid] / bnd[valid]).max()) if valid.any() else 0.0
            rec["max_abs_err"] = float(err[valid].max()) if valid.any() else 0.0
            rec["distinct_outputs"] = len({o[valid].numpy().tobytes() for o in outs})
            if rec["outside"]:
                rws = w.any(dim=1).nonzero().flatten().tolist()
                rec["rows_outside"] = rws[:10]
                r0 = rws[0]; cols = w[r0].nonzero().flatten().tolist()
                rec["example"] = {"row": r0, "request": r0 // next_n, "cols_outside_first": cols[:8], "cols_outside_count": len(cols),
                                  "got": [round(float(x), 5) for x in outs[0][r0, cols[:4]]],
                                  "exact": [round(float(x), 5) for x in exact[r0, cols[:4]]]}
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        return rec

    def perm_table(batch, pages_per_request, num_pages):
        return torch.randperm(num_pages, generator=gen)[: batch * pages_per_request].view(batch, pages_per_request)

    # ---- instrument check: the test's own geometry and dyadic operands
    lengths = [0, 1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257]
    chk = run_case("test-geometry", 128, 16, 64, 2, lengths, perm_table(12, 6, 72), 72, 384, dyadic=True)
    report["oracle_check"] = chk
    print(json.dumps({k: v for k, v in chk.items() if k != "lengths"})[:500], flush=True)
    if "error" in chk or chk["outside"] or chk["max_abs_err"] > 1e-5:
        raise SystemExit("oracle check failed on the test's own geometry; no verdict")

    plan = []
    for hd in (128, 64):
        for heads in (16, 64):
            for page_kv in (64, 32):
                ppr = 384 // page_kv
                plan.append(("A-test-geometry-random", hd, heads, page_kv, 2, lengths, perm_table(12, ppr, 12 * ppr), 12 * ppr, 384))
                long_lengths = [4095, 4096, 4097, 8191]
                ppr_l = -(-8191 // page_kv)
                plan.append(("B-long", hd, heads, page_kv, 1, long_lengths, perm_table(4, ppr_l, 4 * ppr_l), 4 * ppr_l, 8191))
                # C: page reuse across requests and inside one request, and descending page order
                t = perm_table(4, ppr, 4 * ppr).clone()
                t[1] = t[0]                                   # request 1 reads request 0's pages
                t[2, 1] = t[2, 0]                             # one page twice inside request 2
                t[3] = t[3].flip(0)                           # descending physical order
                plan.append(("C-page-reuse", hd, heads, page_kv, 1, [300, 300, 300, 300], t, 4 * ppr, 384))
                # D: 129 requests, random contexts up to 700
                ppr_d = -(-700 // page_kv)
                lens_d = [rnd.randint(0, 700) for _ in range(129)]
                plan.append(("D-129-requests", hd, heads, page_kv, 1, lens_d, perm_table(129, ppr_d, 129 * ppr_d), 129 * ppr_d, 700))
            # E: next_n 4 and 8 (page 64)
            for nn in (4, 8):
                plan.append((f"E-next_n-{nn}", hd, heads, 64, nn, [8, 65, 129, 257, 383, 384], perm_table(6, 6, 36), 36, 384))
    # F: max_context_len twice the longest, and a table with spare columns
    plan.append(("F-max-twice", 128, 16, 64, 1, lengths, perm_table(12, 6, 72), 72, 768))
    plan.append(("F-spare-table-cols", 128, 16, 64, 1, lengths, perm_table(12, 12, 144), 144, 384))

    for i, (label, hd, heads, page_kv, nn, lens, tbl, npages, mx) in enumerate(plan):
        if a.max_cases and i >= a.max_cases:
            break
        rec = run_case(label, hd, heads, page_kv, nn, lens, tbl, npages, mx)
        report["cases"].append(rec)
        flag = rec.get("outside") or rec.get("nan") or "error" in rec or rec.get("distinct_outputs", 1) != 1
        if flag or (i + 1) % 8 == 0:
            print(json.dumps({k: v for k, v in rec.items() if k != "lengths"})[:420], flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    judged = [c for c in report["cases"] if "outside" in c]
    print(f"cases {len(report['cases'])}, judged {len(judged)}, outside in {sum(1 for c in judged if c['outside'])}, nan in "
          f"{sum(1 for c in judged if c['nan'])}, nondeterministic {sum(1 for c in judged if c.get('distinct_outputs', 1) != 1)}, "
          f"refused/errors {sum(1 for c in report['cases'] if 'error' in c)}, elements {sum(c['elements'] for c in judged)}; wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
