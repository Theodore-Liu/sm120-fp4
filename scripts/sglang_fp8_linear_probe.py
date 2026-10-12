#!/usr/bin/env python3
"""SGLang's SM12x small-M static per-tensor FP8 linear (`python/sglang/kernels/kda_kernels/sm120_fp8.py`,
`try_sm120_fp8_linear`, sgl-project/sglang#38082) against the exact oracle. The dispatcher quantises the bf16 input with
`static_quant_fp8` and serves the call with one of two new kernels: the native FP8 GEMV for m = 1
(`ops/gemm/sm120_fp8_gemv.py`, gate: k % 512 == 0, 512 <= k <= 32768, n % 16 == 0, 256 <= n <= 32768) or the KDA skinny CUTLASS
GEMM at its production shapes (`_KDA_M_BY_PROJECTION`, capability (12, 0)); any other call returns None.

Contract (`_supports_common`): input bf16 (m, k) contiguous; weight e4m3 of shape (k, n) with strides (1, k), i.e. the transposed
view of a row-major (n, k); input_scale and output_scale one fp32 each; no bias. The output is
`(static_quant_fp8(input, input_scale) @ weight) * output_scale` in bf16. The oracle takes the quantised input from the same
`static_quant_fp8` call the dispatcher makes (it is deterministic), checks those bytes against the round-to-nearest e4m3 of
input / input_scale (a separate quantiser record), and judges the GEMM against the float64 product of the e4m3 values times
output_scale under the census bound. Cases: every KDA production shape, and the GEMV at its production m = 1 shapes plus
in-gate edges (k in {512, 1536, 5120, 32768} x n in {256, 272, 1040, 4112}, and n = 32768 at k <= 5120). Three runs per case;
a None return is recorded as a fallback; the path that served is read from the dispatcher's own gate functions.

    PYTHONPATH=<sm120-fp4> python scripts/sglang_fp8_linear_probe.py --out reports/<file>.json   (SGLang installed)
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
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args(argv)
    from sglang.kernels.kda_kernels import sm120_fp8 as disp
    from sglang.kernels.ops.gemm.sm120_fp8_gemv import use_sm120_fp8_gemv
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8
    from sm120fp4 import oracle
    dev = torch.device("cuda")
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "capability": list(torch.cuda.get_device_capability()), "torch": torch.__version__, "cases": []}
    shapes = sorted(disp._KDA_PRODUCTION_SHAPES)
    gemv = [(1, k, n) for k in (512, 1536, 5120, 32768) for n in (256, 272, 1040, 4112)] + [(1, k, 32768) for k in (512, 1536, 5120)]
    gemv += [(1, k, n) for (k, n) in disp._KDA_M_BY_PROJECTION if use_sm120_fp8_gemv(1, n, k)]
    for s in gemv:
        if s not in shapes:
            shapes.append(s)
    for m, k, n in shapes:
        path = "gemv" if use_sm120_fp8_gemv(m, n, k) else ("kda-skinny" if (m, k, n) in disp._KDA_PRODUCTION_SHAPES else "none")
        rec = {"m": m, "k": k, "n": n, "expected_path": path}
        try:
            g = torch.Generator().manual_seed(20261012 + 37 * m + k + 3 * n)
            x = (torch.randn(m, k, generator=g) * 2).to(torch.bfloat16)
            w8 = (torch.randn(n, k, generator=g) * 0.5).to(torch.float8_e4m3fn)
            in_s = (torch.rand(1, generator=g) * 0.02 + 0.01).float()
            out_s = (torch.rand(1, generator=g) * 0.02 + 0.005).float()
            xd, wd = x.to(dev).contiguous(), w8.to(dev).contiguous().t()          # weight (k, n), strides (1, k)
            isd, osd = in_s.to(dev), out_s.to(dev)
            q, _ = static_quant_fp8(xd, isd, repeat_scale=False)
            torch.cuda.synchronize()
            qd = q.cpu().double()
            ref_q = (x.float() / in_s).clamp(-448, 448).to(torch.float8_e4m3fn).double()
            ref_q2 = (x.float() * (1.0 / in_s)).clamp(-448, 448).to(torch.float8_e4m3fn).double()
            rec["quantiser_mismatch_div"] = int((qd != ref_q).sum())
            rec["quantiser_mismatch_mul_reciprocal"] = int((qd != ref_q2).sum())
            p = qd @ w8.double().T
            ex = p * out_s.double()
            sabs = (qd.abs() @ w8.double().abs().T) * out_s.double()
            pre = k * oracle.U32 * sabs + 2 * oracle.U32 * ex.abs() + 2 * oracle.U64 * sabs
            bd = pre + 0.5 * oracle._ulp(ex.abs() + pre, torch.bfloat16)
            outs = []
            for _ in range(a.runs):
                y = disp.try_sm120_fp8_linear(xd, wd, isd, osd, None)
                torch.cuda.synchronize()
                if y is None:
                    break
                outs.append(y.double().cpu())
            rec["served"] = bool(outs)
            if outs:
                err = (outs[0] - ex).abs()
                rec.update({"elements": int(err.numel()), "outside": int((err > bd).sum()), "nan": int(torch.isnan(outs[0]).sum()),
                            "max_err_over_bound": float((err / bd.clamp(min=1e-300)).max()),
                            "distinct_outputs": len({t.numpy().tobytes() for t in outs})})
                if rec["outside"]:
                    bad = (err > bd).nonzero()
                    rec["outside_rows"] = sorted({int(i) for i in bad[:, 0].tolist()})[:20]
                    rec["outside_cols_first"] = sorted({int(j) for j in bad[:, 1].tolist()})[:20]
                    rec["rel_err_normwise"] = float(err.norm() / ex.norm())
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
        report["cases"].append(rec)
        print(json.dumps(rec)[:500], flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    served = [c for c in report["cases"] if c.get("served")]
    for p in ("gemv", "kda-skinny"):
        sp = [c for c in served if c["expected_path"] == p]
        print(p, "served", len(sp), "outside in", sum(1 for c in sp if c["outside"]), "nondeterministic", sum(1 for c in sp if c["distinct_outputs"] != 1))
    print(f"cases {len(report['cases'])}, served {len(served)}, fallbacks {sum(1 for c in report['cases'] if c.get('served') is False)}, "
          f"errors {sum(1 for c in report['cases'] if 'error' in c)}; wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
