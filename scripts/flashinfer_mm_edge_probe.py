#!/usr/bin/env python3
"""FlashInfer's dense block-scaled GEMMs on SM120 at shapes the contract allows but earlier census rows did not reach, against
the exact oracle: `mm_fp4` (NVFP4, block 16) and `mm_mxfp8` (block 32, UE8M0), on every backend that serves SM120.

The contract, read from `flashinfer/gemm/gemm_base.py` (`_check_mm_fp4_problem_size`, the per-backend `_*_gemm_fp4_requirement`
functions and the `mm_fp4` docstring): 2D operands, matching K, bf16 output, block 16 for NVFP4 and 32 for MXFP4, 128x4 scale
layout; no M, N or K alignment is stated for the cudnn, cutlass or b12x backends on SM120 ("Logical M may be ragged" for
cute-dsl, which additionally requires N and K divisible by 64). Hardware rules the call cannot escape are kept: packed FP4 rows
are read as 16-byte rows, so K is a multiple of 32 for FP4 and of 16 for FP8; the bf16 output row is a 16-byte multiple, so N is
a multiple of 8. Within that: M in {1, 3, 7, 65, 129, 300}, N in {8, 24, 136, 1000, 4104}, K off 128 ({32, 96, 160, 1056, 4128}
for NVFP4; {32, 160, 1056, 4128} for MXFP8, whose block is 32). A refusal is recorded as a refusal.

Operands as the census's existing probes build them: NVFP4 by the census's reference quantiser (`sm120fp4.reference`) with
scales swizzled 128x4 and alpha = 1 / (gs_a * gs_b), judged by `sm120fp4.oracle.exact_gemm_nvfp4`; MXFP8 by FlashInfer's own
`mxfp8_quantize` and `mxfp8_dequantize_host`, judged against the float64 product. Two runs per case for determinism; a marker
line before every call.

    PYTHONPATH=<sm120-fp4> python scripts/flashinfer_mm_edge_probe.py --log <marker file> --out reports/<file>.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

M_LIST = (1, 3, 7, 65, 129, 300)
N_LIST = (8, 24, 136, 1000, 4104)
K_FP4 = (32, 96, 160, 1056, 4128)
K_MX8 = (32, 160, 1056, 4128)
FP4_BACKENDS = ("cudnn", "cutlass", "b12x", "auto")
MX8_BACKENDS = ("cudnn", "cutlass", "b12x", "auto")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    def mark(s: str) -> None:
        with open(a.log, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} mm-edge {s}\n")

    import flashinfer
    from sm120fp4 import oracle
    from sm120fp4.layouts import to_128x4
    from sm120fp4.reference import quantize_nvfp4
    dev = torch.device("cuda")
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "flashinfer": flashinfer.__version__, "cases": []}

    def record(rec):
        report["cases"].append(rec)
        if rec.get("outside") or rec.get("nan") or rec.get("distinct_outputs", 1) != 1 or "error" in rec:
            print(json.dumps(rec)[:400], flush=True)

    # ---------------------------------------------------------------- mm_fp4 (NVFP4)
    for m in M_LIST:
        for n in N_LIST:
            for k in K_FP4:
                g = torch.Generator().manual_seed(20261012 + 7 * m + 3 * n + k)
                x = torch.randn(m, k, generator=g).to(torch.bfloat16).float()
                w = torch.randn(n, k, generator=g).to(torch.bfloat16).float()
                aq, asf, ags = quantize_nvfp4(x)
                bq, bsf, bgs = quantize_nvfp4(w)
                exact, bound = oracle.exact_gemm_nvfp4(aq, asf, ags, bq, bsf, bgs, out_dtype=torch.bfloat16)
                alpha = (1.0 / (ags * bgs)).reshape(1).float().to(dev)
                a_d, b_d = aq.to(dev), bq.to(dev)
                asw = to_128x4(asf).view(torch.float8_e4m3fn).to(dev)
                bsw = to_128x4(bsf).view(torch.float8_e4m3fn).to(dev)
                for be in FP4_BACKENDS:
                    rec = {"op": "mm_fp4", "backend": be, "m": m, "n": n, "k": k}
                    try:
                        outs = []
                        for _ in range(2):
                            mark(f"mm_fp4 {be} m={m} n={n} k={k}")
                            o = flashinfer.mm_fp4(a_d, b_d.t(), asw, bsw, alpha, torch.bfloat16, None, 16, False, be)
                            torch.cuda.synchronize()
                            outs.append(o.double().cpu())
                        v = oracle.conformance(outs[0].float(), exact, bound)
                        rec.update({"elements": v["elements"], "outside": v["outside_bound"], "nan": int(torch.isnan(outs[0]).sum()),
                                    "max_err_over_bound": v["max_err_over_bound"],
                                    "distinct_outputs": len({t.numpy().tobytes() for t in outs})})
                    except Exception as exc:  # noqa: BLE001
                        rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                    record(rec)

    # ---------------------------------------------------------------- mm_mxfp8
    for m in M_LIST:
        for n in N_LIST:
            for k in K_MX8:
                g = torch.Generator().manual_seed(20261013 + 7 * m + 3 * n + k)
                rec0 = {"op": "mm_mxfp8", "m": m, "n": n, "k": k}
                try:
                    x = (torch.randn(m, k, generator=g) * 2).to(torch.bfloat16).to(dev)
                    w = torch.randn(n, k, generator=g).to(torch.bfloat16).to(dev)
                    xq, xsf = flashinfer.mxfp8_quantize(x, True, 32)
                    wq, wsf = flashinfer.mxfp8_quantize(w, True, 32)
                    torch.cuda.synchronize()
                    xd = flashinfer.mxfp8_dequantize_host(xq.view(torch.uint8).cpu(), xsf.view(torch.uint8).cpu(), True).double()[:m, :k]
                    wd = flashinfer.mxfp8_dequantize_host(wq.view(torch.uint8).cpu(), wsf.view(torch.uint8).cpu(), True).double()[:n, :k]
                    exact = xd @ wd.t()
                    sabs = xd.abs() @ wd.abs().t()
                    pre = k * oracle.U32 * sabs + 2 * oracle.U64 * sabs
                    bound = pre + 0.5 * oracle._ulp(exact.abs() + pre, torch.bfloat16)
                except Exception as exc:  # noqa: BLE001
                    rec0["error"] = f"quantise: {type(exc).__name__}: {str(exc)[:300]}"
                    record(rec0)
                    continue
                for be in MX8_BACKENDS:
                    rec = dict(rec0, backend=be)
                    try:
                        outs = []
                        for _ in range(2):
                            mark(f"mm_mxfp8 {be} m={m} n={n} k={k}")
                            o = flashinfer.mm_mxfp8(xq, wq.t(), xsf.view(torch.uint8).reshape(-1), wsf.view(torch.uint8).reshape(-1),
                                                    out_dtype=torch.bfloat16, backend=be)
                            torch.cuda.synchronize()
                            outs.append(o.double().cpu())
                        err = (outs[0] - exact).abs()
                        rec.update({"elements": int(err.numel()), "outside": int((err > bound).sum()), "nan": int(torch.isnan(outs[0]).sum()),
                                    "max_err_over_bound": float((err / bound.clamp(min=1e-300)).max()),
                                    "distinct_outputs": len({t.numpy().tobytes() for t in outs})})
                    except Exception as exc:  # noqa: BLE001
                        rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                    record(rec)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    judged = [c for c in report["cases"] if "outside" in c]
    by = {}
    for c in report["cases"]:
        key = (c["op"], c.get("backend", "-"))
        s = by.setdefault(key, {"judged": 0, "outside_cases": 0, "errors": 0})
        if "outside" in c:
            s["judged"] += 1
            s["outside_cases"] += int(c["outside"] > 0 or c["nan"] > 0)
        else:
            s["errors"] += 1
    for key, s in sorted(by.items()):
        print(key, s)
    print(f"cases {len(report['cases'])}, judged {len(judged)}; wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
