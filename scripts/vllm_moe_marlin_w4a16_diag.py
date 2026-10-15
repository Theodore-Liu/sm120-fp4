#!/usr/bin/env python3
"""Rounding-model diagnostic for the routed Marlin GEMM's "down" form (mul_topk_weights=True), where grouped cases read
1 to 2 output ulps outside the one-rounding oracle (vllm_moe_marlin_w4a16_probe.py, 2026-10-15). Models:
  M0  bf16( (a @ w_ref) * topk_w )           one rounding, the probe's oracle
  M1  bf16( bf16(a @ w_ref) * topk_w )       the GEMM rounded to the output dtype, then weighted, rounded again
  M2  bf16( (a @ w_ref) * bf16(topk_w) )     the fp32 weight rounded to the output dtype first
Also the gate-up form (no weighting) as the control.

    PYTHONPATH=<sm120-fp4> python vllm_moe_marlin_w4a16_diag.py --out <json>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import awq_marlin_quantize, marlin_quantize
    from vllm.scalar_type import scalar_types

    dtype = torch.bfloat16
    gen = torch.Generator().manual_seed(11)
    workspace = marlin_make_workspace_new(torch.device("cuda"), 4)
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": []}

    def case(variant, group, m, n, k, e, topk, form):
        row = {"variant": variant, "group_size": group, "m": m, "n": n, "k": k, "experts": e, "top_k": topk, "form": form}
        w = (torch.randn((e, n, k), generator=gen) / 10).to(dtype).cuda()
        refs, qws, ss, gs, sis, zps = [], [], [], [], [], []
        for i in range(e):
            if variant == "awq":
                w_ref, qw, s, zp = awq_marlin_quantize(w[i].T.contiguous(), scalar_types.uint4, group)
                refs.append(w_ref); qws.append(qw); ss.append(s); zps.append(zp)
            else:
                w_ref, qw, s, g_idx, si, _ = marlin_quantize(w[i].T.contiguous(), scalar_types.uint4b8, group, False, torch.randperm(k, generator=gen))
                refs.append(w_ref); qws.append(qw); ss.append(s); gs.append(g_idx); sis.append(si)
        stack = lambda l: torch.stack(l).contiguous() if l else None
        w_ref = torch.stack(refs).double().cpu(); qw, s, g_idx, si, zp = stack(qws), stack(ss), stack(gs), stack(sis), stack(zps)
        b_type = scalar_types.uint4 if variant == "awq" else scalar_types.uint4b8
        score = torch.softmax(torch.randn((m, e), generator=gen).float(), dim=-1)
        tw, ti = torch.topk(score, topk, dim=-1); tw = tw.float().cuda(); ti = ti.int().cuda()
        block = 64
        sorted_ids, expert_ids, num_post = moe_align_block_size(ti, block, e)
        rows = m * topk
        a_in = (torch.randn((m if form == "gate-up" else rows, k), generator=gen) / 2).to(dtype).cuda()
        out = torch.empty((rows, n), dtype=dtype, device="cuda")
        call_topk, call_m, mul = (topk, m, False) if form == "gate-up" else (1, rows, True)
        out = ops.moe_wna16_marlin_gemm(a_in, out, qw, None, s, None, None, zp, g_idx, si, workspace, sorted_ids, expert_ids, num_post, tw,
                                        moe_block_size=block, top_k=call_topk, mul_topk_weights=mul, b_q_type=b_type, size_m=call_m,
                                        size_n=n, size_k=k, is_k_full=True, use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)
        torch.cuda.synchronize()
        o = out.double().cpu(); ad = a_in.double().cpu(); twc = tw.double().cpu(); tic = ti.long().cpu()
        acc = torch.zeros((rows, n), dtype=torch.float64); wts = torch.ones((rows,), dtype=torch.float64)
        for r in range(rows):
            t, sl = r // topk, r % topk
            x = ad[t] if form == "gate-up" else ad[r]
            acc[r] = x @ w_ref[int(tic[t, sl])]
            if mul:
                wts[r] = float(twc[t, sl])
        models = {"M0_one_rounding": (acc * wts[:, None]).to(dtype).double(),
                  "M1_round_then_weight": (acc.to(dtype).double() * wts[:, None]).to(dtype).double(),
                  "M2_weight_rounded_first": (acc * wts.to(dtype).double()[:, None]).to(dtype).double(),
                  "M3_both_rounded": (acc.to(dtype).double() * wts.to(dtype).double()[:, None]).to(dtype).double()}
        if mul:   # the weight applied to the activation row before the GEMM, in the activation dtype
            for name, wv in (("M7_a_times_w_rounded", wts), ("M8_a_times_bf16w_rounded", wts.to(dtype).double())):
                acc_w = torch.zeros((rows, n), dtype=torch.float64)
                for r in range(rows):
                    t, sl = r // topk, r % topk
                    xw = (ad[r] * wv[r]).to(dtype).double()
                    acc_w[r] = xw @ w_ref[int(tic[t, sl])]
                models[name] = acc_w.to(dtype).double()
            # the weight applied in fp32 to the fp32 accumulator, with the product rounded to fp32 before the bf16 store
            models["M9_fp32_product"] = ((acc.float() * wts.float()[:, None])).to(dtype).double()
        for name, mdl in models.items():
            d = (o - mdl).abs()
            row[name] = {"bit_exact_fraction": round(float((d == 0).double().mean()), 4), "max_abs_diff": float(d.max())}
        print(json.dumps(row), flush=True)
        report["rows"].append(row)

    case("gptq", 128, 129, 320, 1152, 8, 2, "down")
    case("awq", 32, 129, 320, 1152, 8, 2, "down")
    case("gptq", 128, 7, 320, 1152, 64, 2, "down")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
