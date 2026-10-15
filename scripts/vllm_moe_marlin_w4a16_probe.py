#!/usr/bin/env python3
"""vLLM's routed Marlin W4A16 GEMM (`ops.moe_wna16_marlin_gemm`, the kernel behind `fused_marlin_moe`) on an SM120 device
under routings, shapes and block sizes its test does not enumerate, judged against a float64 oracle per output row.

Registered before any case ran: mlsys-5090-research docs/d1-vllm-moe-marlin-registration-2026-10-15.md. Weights from
vLLM's own quantisers per expert, stacked as tests/kernels/moe/test_moe.py stacks them; routing from vLLM's
`moe_align_block_size`; both call forms of `fused_marlin_moe` (gate-up: input by token, no weighting; down: input by row,
times the top-k weight). The oracle is first checked on the test's own routing (133 tokens, 5 experts, top-k 2).

    PYTHONPATH=<sm120-fp4> python vllm_moe_marlin_w4a16_probe.py --log <marker> --out <json>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

U32 = 2.0 ** -23


def ulp(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    mant = 7 if dtype == torch.bfloat16 else 10
    return torch.ldexp(torch.ones_like(x), torch.floor(torch.log2(x.clamp(min=2.0 ** -120))) - mant)


def choose_block(m: int, topk: int, e: int) -> int:
    for b in (8, 16, 32, 48, 64):                      # vLLM's occupancy rule in fused_marlin_moe
        if m * topk / e / b < 0.9:
            break
    return b


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-cases", type=int, default=0)
    a = ap.parse_args(argv)

    def mark(s: str) -> None:
        with open(a.log, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} moe-marlin {s}\n")

    import vllm
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import awq_marlin_quantize, marlin_quantize
    from vllm.scalar_type import scalar_types

    dtype = torch.bfloat16
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "vllm": vllm.__version__, "cases": []}
    gen = torch.Generator().manual_seed(20261015)
    workspace = marlin_make_workspace_new(torch.device("cuda"), 4)

    def make_weights(variant, e, n, k, group):
        """Per-expert Marlin weights stacked as MarlinMoEWeightData.make does. Returns w_ref [E, K, N] (double) and the stacks."""
        w = (torch.randn((e, n, k), generator=gen) / 10).to(dtype).cuda()        # [E, N, K] as vLLM stores w1/w2
        refs, qws, ss, gs, sis, zps = [], [], [], [], [], []
        for i in range(e):
            if variant == "awq":
                w_ref, qw, s, zp = awq_marlin_quantize(w[i].T.contiguous(), scalar_types.uint4, group)
                refs.append(w_ref); qws.append(qw); ss.append(s); zps.append(zp)
            else:
                act = variant == "gptq-act_order"
                perm = torch.randperm(k, generator=gen)
                w_ref, qw, s, g_idx, si, _ = marlin_quantize(w[i].T.contiguous(), scalar_types.uint4b8, group, act, perm)
                refs.append(w_ref); qws.append(qw); ss.append(s); gs.append(g_idx); sis.append(si)
        stack = lambda l: torch.stack(l).contiguous() if l else None
        return (torch.stack(refs).double().cpu(), stack(qws), stack(ss), stack(gs), stack(sis), stack(zps),
                scalar_types.uint4 if variant == "awq" else scalar_types.uint4b8)

    def routing(kind, m, e, topk):
        if kind == "R1-fused_topk":          # the test's fused_topk semantics: top-k of softmax(scores), no renormalisation
            score = torch.softmax(torch.randn((m, e), generator=gen).float(), dim=-1)
            tw, ti = torch.topk(score, topk, dim=-1)
            return tw.float().cuda(), ti.int().cuda()
        if kind == "R2-uneven-64":
            used = torch.randperm(e, generator=gen)[:8]
            ti = used[torch.randint(0, 8, (m, topk), generator=gen)]
            for t in range(m):                               # no duplicate expert inside a token
                row = ti[t].tolist()
                for s in range(1, topk):
                    while row[s] in row[:s]:
                        row[s] = int(used[torch.randint(0, 8, (1,), generator=gen)])
                ti[t] = torch.tensor(row)
            tw = torch.softmax(torch.randn((m, topk), generator=gen), dim=-1)
            return tw.float().cuda(), ti.int().cuda()
        if kind == "R3-one-expert":
            ti = torch.zeros((m, topk), dtype=torch.int32)
            ti[:, 0] = 3
            for s in range(1, topk):
                ti[:, s] = (3 + s) % e
            tw = torch.softmax(torch.randn((m, topk), generator=gen), dim=-1)
            return tw.float().cuda(), ti.cuda()
        raise ValueError(kind)

    def run_case(label, variant, group, m, n, k, e, topk, kind, form, block=None, runs=2):
        rec = {"label": label, "variant": variant, "group_size": group, "m": m, "n": n, "k": k, "experts": e, "top_k": topk,
               "routing": kind, "form": form}
        try:
            w_ref, qw, s, g_idx, si, zp, b_type = make_weights(variant, e, n, k, group)
            tw, ti = routing(kind, m, e, topk)
            block = block or choose_block(m, topk, e)
            rec["block_size_m"] = block
            sorted_ids, expert_ids, num_post = moe_align_block_size(ti, block, e)
            rows = m * topk
            if form == "gate-up":
                a_in = (torch.randn((m, k), generator=gen) / 2).to(dtype).cuda()
                row_input = lambda r: a_in[r // topk]
                mul = False
            else:
                a_in = (torch.randn((rows, k), generator=gen) / 2).to(dtype).cuda()
                row_input = lambda r: a_in[r]
                mul = True
            ad = a_in.double().cpu(); twc = tw.double().cpu(); tic = ti.long().cpu()
            exact = torch.zeros((rows, n), dtype=torch.float64); bnd = torch.ones((rows, n), dtype=torch.float64)
            for r in range(rows):
                t, sl = r // topk, r % topk
                x = ad[r // topk] if form == "gate-up" else ad[r]
                wr = w_ref[int(tic[t, sl])]
                v = x @ wr
                sabs = x.abs() @ wr.abs()
                scale = float(twc[t, sl]) if mul else 1.0
                v = v * scale; sabs = sabs * abs(scale)
                slack = (k + 1) * U32 * sabs
                exact[r] = v; bnd[r] = slack + 0.5 * ulp((v.abs() + slack).float(), dtype).double()
            outs = []
            for _ in range(runs):
                out = torch.empty((rows, n), dtype=dtype, device="cuda")
                mark(f"{label} {variant} g={group} m={m} n={n} k={k} e={e} topk={topk} {kind} {form} block={block}")
                # as fused_marlin_moe: the gate-up call takes M tokens and top_k; the down call takes the M * top_k rows
                # as its own tokens (top_k = 1, size_m = M * top_k) with the same sorted ids and expert ids
                call_topk, call_m = (topk, m) if form == "gate-up" else (1, rows)
                out = ops.moe_wna16_marlin_gemm(a_in, out, qw, None, s, None, None, zp, g_idx, si, workspace, sorted_ids, expert_ids,
                                                num_post, tw, moe_block_size=block, top_k=call_topk, mul_topk_weights=mul, b_q_type=b_type,
                                                size_m=call_m, size_n=n, size_k=k, is_k_full=True, use_atomic_add=False,
                                                use_fp32_reduce=True, is_zp_float=False)
                torch.cuda.synchronize()
                outs.append(out.double().cpu())
            err = (outs[0] - exact).abs()
            rec["elements"] = int(err.numel()); rec["outside"] = int((err > bnd).sum()); rec["nan"] = int(torch.isnan(outs[0]).sum())
            rec["max_err_over_bound"] = float((err / bnd).max()); rec["max_abs_err"] = float(err.max())
            rec["distinct_outputs"] = len({o.numpy().tobytes() for o in outs})
            if rec["outside"]:
                bad_rows = (err > bnd).any(dim=1).nonzero().flatten().tolist()
                rec["rows_outside"] = [bad_rows[0], bad_rows[-1], len(bad_rows)]
                r0 = bad_rows[0]; cols = (err[r0] > bnd[r0]).nonzero().flatten().tolist()
                rec["example"] = {"row": r0, "token": r0 // topk, "slot": r0 % topk, "expert": int(tic[r0 // topk, r0 % topk]),
                                  "cols_count": len(cols), "got": [round(float(x), 4) for x in outs[0][r0, cols[:4]]],
                                  "exact": [round(float(x), 4) for x in exact[r0, cols[:4]]]}
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        return rec

    # ---- instrument check: the test's own routing and shape
    chk = run_case("test-routing", "gptq", 128, 133, 256, 256, 5, 2, "R1-fused_topk", "gate-up")
    report["instrument_check"] = chk
    print(json.dumps(chk)[:500], flush=True)
    if "error" in chk or chk["outside"]:
        raise SystemExit("instrument check failed on the test's own routing; no verdict")

    plan = []
    for variant in ("gptq", "gptq-act_order", "awq"):
        for group in (-1, 32, 128):
            if variant == "gptq-act_order" and group == -1:
                continue
            for m in (1, 7, 129, 300):
                for n, k in ((64, 128), (320, 1152), (4160, 4224)):
                    if variant == "gptq-act_order" and group == k:
                        continue
                    for kind, e, topk in (("R1-fused_topk", 8, 2), ("R2-uneven-64", 64, 2), ("R3-one-expert", 8, 2),
                                          ("R1-fused_topk", 8, 1), ("R1-fused_topk", 16, 8)):
                        if kind == "R2-uneven-64" and m == 1:
                            continue
                        for form in ("gate-up", "down"):
                            plan.append(("edge", variant, group, m, n, k, e, topk, kind, form, None))
    for block in (8, 16, 32, 64):
        plan.append((f"block-{block}", "gptq", 128, 129, 320, 1152, 8, 2, "R1-fused_topk", "gate-up", block))
        plan.append((f"block-{block}", "gptq", 128, 7, 320, 1152, 64, 2, "R2-uneven-64", "down", block))

    for i, (label, variant, group, m, n, k, e, topk, kind, form, block) in enumerate(plan):
        if a.max_cases and i >= a.max_cases:
            break
        rec = run_case(label, variant, group, m, n, k, e, topk, kind, form, block)
        report["cases"].append(rec)
        flag = rec.get("outside") or rec.get("nan") or "error" in rec or rec.get("distinct_outputs", 1) != 1
        if flag or (i + 1) % 50 == 0:
            print(json.dumps(rec)[:420], flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    judged = [c for c in report["cases"] if "outside" in c]
    print(f"cases {len(report['cases'])}, judged {len(judged)}, outside in {sum(1 for c in judged if c['outside'])}, nan in "
          f"{sum(1 for c in judged if c['nan'])}, nondeterministic {sum(1 for c in judged if c.get('distinct_outputs', 1) != 1)}, "
          f"refused/errors {sum(1 for c in report['cases'] if 'error' in c)}, elements {sum(c['elements'] for c in judged)}; wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
