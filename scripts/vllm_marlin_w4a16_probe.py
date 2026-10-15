#!/usr/bin/env python3
"""vLLM's Marlin W4A16 GEMM (`ops.marlin_gemm`, GPTQ uint4b8 with and without act_order, AWQ uint4 with zero points) on an
SM120 device at shapes and options outside vLLM's own test enumeration, judged against a float64 oracle.

Registered before any case ran: mlsys-5090-research docs/d1-vllm-marlin-w4a16-registration-2026-10-15.md. Operands come
from vLLM's own quantisers (`marlin_quantize`, `awq_marlin_quantize`), whose `w_ref` is the dequantised weight the kernel
is contracted to multiply; exact = a.double() @ w_ref.double(); bound = k fp32 roundings over |a| |w_ref| plus the output
ulp. The instrument is checked first on one of the test's own shapes against the test's own reference and tolerance.

    PYTHONPATH=<sm120-fp4> python vllm_marlin_w4a16_probe.py --log <marker> --out <json>
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
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-cases", type=int, default=0)
    a = ap.parse_args(argv)

    def mark(s: str) -> None:
        with open(a.log, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} marlin {s}\n")

    import vllm
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import awq_marlin_quantize, marlin_quantize
    from vllm.scalar_type import scalar_types

    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "vllm": vllm.__version__, "cases": []}
    gen = torch.Generator().manual_seed(20261015)
    workspace = marlin_make_workspace_new(torch.device("cuda"))

    def run_case(label, variant, dtype, group_size, m, n, k, atomic=False, runs=2, test_ref=False):
        rec = {"label": label, "variant": variant, "dtype": str(dtype).replace("torch.", ""), "group_size": group_size,
               "m": m, "n": n, "k": k, "use_atomic_add": atomic}
        try:
            a_in = (torch.randn((m, k), generator=gen) / 2).to(dtype).cuda()
            w = (torch.randn((k, n), generator=gen) / 10).to(dtype).cuda()
            if variant == "awq":
                w_ref, q_w, s, zp = awq_marlin_quantize(w, scalar_types.uint4, group_size)
                g_idx = sort_idx = None; b_type = scalar_types.uint4
            else:
                act = variant == "gptq-act_order"
                torch.manual_seed(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=gen)))   # the quantiser's own permutation draw
                w_ref, q_w, s, g_idx, sort_idx, _ = marlin_quantize(w, scalar_types.uint4b8, group_size, act)
                zp = None; b_type = scalar_types.uint4b8
            exact = a_in.double().cpu() @ w_ref.double().cpu()
            slack = k * U32 * (a_in.double().cpu().abs() @ w_ref.double().cpu().abs())
            bnd = slack + 0.5 * ulp((exact.abs() + slack).float(), dtype).double()
            outs = []
            for _ in range(runs):
                out = torch.empty((m, n), dtype=dtype, device="cuda")
                mark(f"{label} {variant} {rec['dtype']} g={group_size} {m}x{n}x{k}")
                out = ops.marlin_gemm(a_in, out, q_w, None, s, None, None, zp, g_idx, sort_idx, workspace, b_type, m, n, k,
                                      is_k_full=True, use_atomic_add=atomic, use_fp32_reduce=True, is_zp_float=False)
                torch.cuda.synchronize()
                outs.append(out.double().cpu())
            err = (outs[0] - exact).abs()
            rec["elements"] = int(err.numel()); rec["outside"] = int((err > bnd).sum()); rec["nan"] = int(torch.isnan(outs[0]).sum())
            rec["max_err_over_bound"] = float((err / bnd).max()); rec["max_abs_err"] = float(err.max())
            rec["distinct_outputs"] = len({o.numpy().tobytes() for o in outs})
            if test_ref:   # the test's own judge: compute_max_diff(output, matmul(a, w_ref)) < 0.04
                ref = torch.matmul(a_in, w_ref).double().cpu()
                rec["test_max_diff_kernel_vs_test_ref"] = float(((outs[0] - ref).abs().max()) / ref.abs().max())
                rec["test_max_diff_oracle_vs_test_ref"] = float(((exact - ref).abs().max()) / ref.abs().max())
            if rec["outside"]:
                rows = (err > bnd).any(dim=1).nonzero().flatten().tolist()
                rec["rows_outside"] = rows[:8]
                r0 = rows[0]; cols = (err[r0] > bnd[r0]).nonzero().flatten().tolist()
                rec["example"] = {"row": r0, "cols_first": cols[:6], "cols_count": len(cols),
                                  "got": [round(float(x), 5) for x in outs[0][r0, cols[:4]]], "exact": [round(float(x), 5) for x in exact[r0, cols[:4]]]}
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        return rec

    # ---- instrument check: one of the test's own shapes, judged two ways
    chk = run_case("test-shape", "gptq", torch.bfloat16, 128, 26, 37 * 64, 13 * 128, test_ref=True)
    report["instrument_check"] = chk
    print(json.dumps(chk)[:600], flush=True)
    if "error" in chk or chk["outside"] or chk["test_max_diff_oracle_vs_test_ref"] > 0.04 or chk["test_max_diff_kernel_vs_test_ref"] > 0.04:
        raise SystemExit("instrument check failed on the test's own shape; no verdict")

    plan = []
    for dtype in (torch.bfloat16, torch.float16):
        for variant in ("gptq", "gptq-act_order", "awq"):
            for g in (-1, 32, 128):
                if variant == "gptq-act_order" and g == -1:
                    continue
                for m in (1, 7, 129):
                    for n in (64, 320, 4160):
                        for k in (128, 1152, 4224):
                            plan.append(("edge", variant, dtype, g, m, n, k, False))
    plan.append(("m4097", "gptq", torch.bfloat16, 128, 4097, 320, 1152, False))
    plan.append(("m4097", "gptq", torch.bfloat16, 128, 4097, 4160, 4224, False))
    plan.append(("atomic-add", "gptq", torch.bfloat16, 128, 129, 320, 1152, True))
    plan.append(("refusal-n32", "gptq", torch.bfloat16, 128, 7, 32, 128, False))
    plan.append(("refusal-k64", "gptq", torch.bfloat16, -1, 7, 64, 64, False))

    for i, (label, variant, dtype, g, m, n, k, atomic) in enumerate(plan):
        if a.max_cases and i >= a.max_cases:
            break
        rec = run_case(label, variant, dtype, g, m, n, k, atomic)
        report["cases"].append(rec)
        flag = rec.get("outside") or rec.get("nan") or "error" in rec or rec.get("distinct_outputs", 1) != 1
        if flag or (i + 1) % 40 == 0:
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
