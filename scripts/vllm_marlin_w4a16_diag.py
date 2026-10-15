#!/usr/bin/env python3
"""Rounding-model diagnostic for the Marlin W4A16 readings outside the bound (vllm_marlin_w4a16_probe.py, 2026-10-15):
which of these reproduces the kernel bit for bit on the channelwise (group_size = -1) and grouped cases?
  M0  bf16(fp32 sum of a * w_ref)                       -- the probe's oracle (one output rounding)
  M1  bf16( bf16(a @ (q - 8)) * s )                      -- integer accumulate, round, then scale (two roundings)
  M2  bf16( a @ bf16((q - 8) * s) ) with fp32 accumulate  -- the same as M0 but the dequant rounded in bf16 (w_ref already is)
  M3  sum over k-slices of 64 (or 128) rounded to bf16 between slices
The weights come from vLLM's `gptq_quantize_weights` (the same function marlin_quantize calls) so q and s are known.

    PYTHONPATH=<sm120-fp4> python vllm_marlin_w4a16_diag.py --out <json>
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
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new, marlin_permute_scales
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import get_weight_perm, marlin_weights
    from vllm.model_executor.layers.quantization.utils.quant_utils import gptq_quantize_weights
    from vllm.scalar_type import scalar_types

    gen = torch.Generator().manual_seed(7)
    workspace = marlin_make_workspace_new(torch.device("cuda"))
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": []}

    def case(dtype, group_size, m, n, k):
        row = {"dtype": str(dtype).replace("torch.", ""), "group_size": group_size, "m": m, "n": n, "k": k}
        a_in = (torch.randn((m, k), generator=gen) / 2).to(dtype).cuda()
        w = (torch.randn((k, n), generator=gen) / 10).to(dtype).cuda()
        gs = k if group_size == -1 else group_size
        w_ref, q_w, s, g_idx, _ = gptq_quantize_weights(w, scalar_types.uint4b8, gs, False)   # q_w uint4 codes, s [k/gs, n]
        perm = get_weight_perm(4)
        mq = marlin_weights(q_w, k, n, 4, perm)
        ms = marlin_permute_scales(s, k, n, gs)
        out = torch.empty((m, n), dtype=dtype, device="cuda")
        out = ops.marlin_gemm(a_in, out, mq, None, ms, None, None, None, None, None, workspace, scalar_types.uint4b8, m, n, k,
                              is_k_full=True, use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)
        torch.cuda.synchronize()
        o = out.double().cpu()
        ad = a_in.double().cpu()
        qd = (q_w.double().cpu() - 8.0)                                  # [k, n] integers in [-8, 7]
        sd = s.double().cpu().repeat_interleave(gs, dim=0)[:k]           # [k, n]
        wref = w_ref.double().cpu()
        models = {}
        models["M0_one_rounding"] = (ad @ wref).to(dtype).double()
        acc_int = ad @ qd                                                 # exact integer-weighted sums (fp64)
        if group_size == -1:
            models["M1_round_then_scale"] = ((acc_int.to(dtype).double()) * s.double().cpu()[0][None, :]).to(dtype).double()
            models["M1b_fp32acc_round_then_scale"] = ((acc_int.float().to(dtype).double()) * s.double().cpu()[0][None, :]).to(dtype).double()
        models["M2_dequant_bf16_then_fp32"] = (ad.float() @ wref.float()).double().to(dtype).double()
        for slab in (64, 128):
            acc = torch.zeros(m, n, dtype=torch.float64)
            for k0 in range(0, k, slab):
                acc = (acc + (ad[:, k0:k0 + slab] @ wref[k0:k0 + slab])).to(dtype).double()
            models[f"M3_slab{slab}_bf16_between"] = acc
        for name, mdl in models.items():
            d = (o - mdl).abs()
            row[name] = {"bit_exact_fraction": float((d == 0).double().mean()), "max_abs_diff": float(d.max())}
        print(json.dumps(row)[:600], flush=True)
        report["rows"].append(row)

    for dtype in (torch.bfloat16, torch.float16):
        for g in (-1, 128):
            for m, n, k in ((1, 64, 128), (1, 320, 1152), (7, 320, 1152), (129, 320, 1152), (129, 4160, 4224)):
                case(dtype, g, m, n, k)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
