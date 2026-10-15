#!/usr/bin/env python3
"""Localise the one grouped Marlin case read outside the bound and non-deterministic (vllm_marlin_w4a16_probe.py,
2026-10-15: GPTQ, bf16, group 128, m = 129, n = 320, k = 1152, 7,111 of 41,280 elements outside, distinct outputs > 1):
the same shape over several seeds and 10 runs each, the reduce and atomic options, and neighbouring m, n, k.

    PYTHONPATH=<sm120-fp4> python vllm_marlin_w4a16_diag2.py --out <json>
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
    from vllm.scalar_type import scalar_types

    workspace = marlin_make_workspace_new(torch.device("cuda"))
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": []}

    def case(label, dtype, g, m, n, k, seed, fp32_reduce=True, atomic=False, runs=10, fresh_workspace=False):
        gen = torch.Generator().manual_seed(seed)
        row = {"label": label, "dtype": str(dtype).replace("torch.", ""), "group_size": g, "m": m, "n": n, "k": k, "seed": seed,
               "fp32_reduce": fp32_reduce, "atomic": atomic, "fresh_workspace": fresh_workspace}
        try:
            a_in = (torch.randn((m, k), generator=gen) / 2).to(dtype).cuda()
            w = (torch.randn((k, n), generator=gen) / 10).to(dtype).cuda()
            torch.manual_seed(seed)
            w_ref, q_w, s, g_idx, sort_idx, _ = marlin_quantize(w, scalar_types.uint4b8, g, False)
            exact = a_in.double().cpu() @ w_ref.double().cpu()
            slack = k * U32 * (a_in.double().cpu().abs() @ w_ref.double().cpu().abs())
            bnd = slack + 0.5 * ulp((exact.abs() + slack).float(), dtype).double()
            outs = []
            for _ in range(runs):
                ws = marlin_make_workspace_new(torch.device("cuda")) if fresh_workspace else workspace
                out = torch.empty((m, n), dtype=dtype, device="cuda")
                out = ops.marlin_gemm(a_in, out, q_w, None, s, None, None, None, g_idx, sort_idx, ws, scalar_types.uint4b8, m, n, k,
                                      is_k_full=True, use_atomic_add=atomic, use_fp32_reduce=fp32_reduce, is_zp_float=False)
                torch.cuda.synchronize()
                outs.append(out.double().cpu())
            per_run = [int(((o - exact).abs() > bnd).sum()) for o in outs]
            row["outside_per_run"] = per_run
            row["distinct_outputs"] = len({o.numpy().tobytes() for o in outs})
            row["max_err_over_bound"] = float(max(((o - exact).abs() / bnd).max() for o in outs))
            row["max_abs_err"] = float(max((o - exact).abs().max() for o in outs))
            if max(per_run):
                o = outs[per_run.index(max(per_run))]
                w_ = (o - exact).abs() > bnd
                rows_ = w_.any(dim=1).nonzero().flatten().tolist()
                row["rows_outside"] = [rows_[0], rows_[-1], len(rows_)]
                cols_ = w_.any(dim=0).nonzero().flatten().tolist()
                row["cols_outside"] = [cols_[0], cols_[-1], len(cols_)]
                r0 = rows_[0]; c0 = w_[r0].nonzero().flatten().tolist()[:4]
                row["example"] = {"row": r0, "cols": c0, "got": [round(float(x), 4) for x in o[r0, c0]], "exact": [round(float(x), 4) for x in exact[r0, c0]]}
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        print(json.dumps(row)[:520], flush=True)
        report["rows"].append(row)

    bf = torch.bfloat16
    for seed in range(6):
        case("same-shape-seeds", bf, 128, 129, 320, 1152, seed)
    case("fresh-workspace", bf, 128, 129, 320, 1152, 0, fresh_workspace=True)
    case("fp16", torch.float16, 128, 129, 320, 1152, 0)
    case("no-fp32-reduce", bf, 128, 129, 320, 1152, 0, fp32_reduce=False)
    case("atomic-add", bf, 128, 129, 320, 1152, 0, atomic=True)
    for m in (64, 128, 130, 256, 257, 1024):
        case("m-neighbours", bf, 128, m, 320, 1152, 0)
    for n in (64, 256, 4160):
        case("n-neighbours", bf, 128, 129, n, 1152, 0)
    for k in (128, 2304, 4224):
        case("k-neighbours", bf, 128, 129, 320, k, 0)
    for g in (32, 64):
        case("group-neighbours", bf, g, 129, 320, 1152, 0)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
