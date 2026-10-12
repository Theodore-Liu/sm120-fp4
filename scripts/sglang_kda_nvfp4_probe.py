#!/usr/bin/env python3
"""SGLang's KDA (Kernel Design Agents) NVFP4 GEMM for SM120 (`python/sglang/kernels/kda_kernels/qwen3x_nvfp4_gemm_sm120.py`,
sgl-project/sglang#36865, a CuTe DSL kernel produced by an automated kernel-optimisation workflow and specialised for Qwen3.x
decode shapes) against the exact oracle. SGLang's own test of it is end to end only (GSM8K accuracy above 0.64,
`test/registered/e2e/quantization/test_nvfp4_gemm_sm120.py`); no element-level check exists.

The contract is the public dispatcher `try_qwen3x_nvfp4_gemm` (`qwen3x_nvfp4_gemm.py`): exactly 17 (m, k, n) shapes (m in
{1, 2, 4, 8} x (k, n) in {(2560, 18432), (9216, 2560), (4096, 24576), (12288, 4096)}, plus (9, 17408, 5120)); input packed uint8
(m, k/2) row-major; weight packed uint8 of shape (k/2, n) with strides (1, k/2), i.e. the transposed view of a row-major (n, k/2);
input_sf (round_up(m, 128), k/16) row-major, uint8 or e4m3; weight_sf e4m3 of shape (k/16, n) with strides (1, k/16); alpha one
fp32; bf16 output; device capability (12, 0). Any other call returns None and SGLang falls back. The kernel reads both scale
buffers in the 128x4 layout (`(32, 4, sf_rows, 4, sf_k)`), so the scales are the census's `to_128x4` swizzle of the reference
quantiser's row-major scales, and weight_sf is the transposed view of the swizzled (n, k/16) tensor.

Each shape runs three times through the dispatcher; a None return is recorded as a fallback (the call was not served). The
oracle is `sm120fp4.oracle.exact_gemm_nvfp4` with the half output ulp taken at the top of the interval.

    PYTHONPATH=<sm120-fp4> python scripts/sglang_kda_nvfp4_probe.py --sglang <sglang checkout> --out reports/<file>.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
import types
from pathlib import Path

import torch

SHAPES = [(m, k, n) for m in (1, 2, 4, 8) for k, n in ((2560, 18432), (9216, 2560), (4096, 24576), (12288, 4096))] + [(9, 17408, 5120)]


def load_kda(sglang_root: Path):
    """Import the two KDA modules from a checkout without importing the rest of SGLang (whose import pulls the server)."""
    base = sglang_root / "python" / "sglang" / "kernels" / "kda_kernels"
    for name in ("sglang", "sglang.kernels", "sglang.kernels.kda_kernels"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []
            sys.modules[name] = mod
    loaded = {}
    for short in ("qwen3x_nvfp4_gemm_sm120", "qwen3x_nvfp4_gemm"):
        full = f"sglang.kernels.kda_kernels.{short}"
        spec = importlib.util.spec_from_file_location(full, base / f"{short}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
        loaded[short] = mod
    return loaded["qwen3x_nvfp4_gemm"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sglang", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args(argv)
    from sm120fp4 import oracle
    from sm120fp4.layouts import to_128x4
    from sm120fp4.reference import quantize_nvfp4
    gate = load_kda(a.sglang)
    try:
        commit = subprocess.run(["git", "-C", str(a.sglang), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"
    import cutlass
    dev = torch.device("cuda")
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "capability": list(torch.cuda.get_device_capability()), "torch": torch.__version__,
              "cutlass_dsl": getattr(cutlass, "__version__", "?"), "sglang_commit": commit, "cases": []}
    for m, k, n in SHAPES:
        rec = {"m": m, "k": k, "n": n}
        try:
            g = torch.Generator().manual_seed(20261012 + 31 * m + k + n)
            aq, asf, ags = quantize_nvfp4(torch.randn(m, k, generator=g).to(torch.bfloat16).float())
            bq, bsf, bgs = quantize_nvfp4((torch.randn(n, k, generator=g) * 0.05).to(torch.bfloat16).float())
            ex, bd = oracle.exact_gemm_nvfp4(aq, asf, ags, bq, bsf, bgs, out_dtype=torch.bfloat16)
            pre = bd - 0.5 * oracle._ulp(ex, torch.bfloat16)
            bd = pre + 0.5 * oracle._ulp(ex.abs() + pre, torch.bfloat16)
            x = aq.to(dev).contiguous()                                    # (m, k/2) row-major
            w = bq.to(dev).contiguous().t()                                # (k/2, n), strides (1, k/2)
            xsf = to_128x4(asf).view(torch.float8_e4m3fn).to(dev).contiguous()     # (round_up(m,128), k/16)
            wsf = to_128x4(bsf).view(torch.float8_e4m3fn).to(dev).contiguous().t()  # (k/16, n), strides (1, k/16)
            alpha = (1.0 / (ags.double() * bgs.double())).float().reshape(1).to(dev)
            rec["input_sf_shape"], rec["weight_sf_shape"] = list(xsf.shape), list(wsf.shape)
            outs = []
            for _ in range(a.runs):
                y = gate.try_qwen3x_nvfp4_gemm(x, w, xsf, wsf, alpha, torch.bfloat16, n)
                torch.cuda.synchronize()
                if y is None:
                    break
                outs.append(y.double().cpu())
            if not outs:
                rec["served"] = False
            else:
                rec["served"] = True
                err = (outs[0] - ex).abs()
                rec.update({"elements": int(err.numel()), "outside": int((err > bd).sum()), "nan": int(torch.isnan(outs[0]).sum()),
                            "max_err_over_bound": float((err / bd.clamp(min=1e-300)).max()),
                            "distinct_outputs": len({t.numpy().tobytes() for t in outs})})
                if rec["outside"]:
                    bad = (err > bd)
                    rec["outside_rows"] = sorted({int(i) for i in bad.nonzero()[:, 0].tolist()})[:20]
                    rec["outside_cols_first"] = sorted({int(j) for j in bad.nonzero()[:, 1].tolist()})[:20]
                    rec["rel_err_normwise"] = float(err.norm() / ex.norm())
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
        report["cases"].append(rec)
        print(json.dumps(rec)[:500], flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    served = [c for c in report["cases"] if c.get("served")]
    print(f"shapes {len(SHAPES)}, served {len(served)}, outside in {sum(1 for c in served if c['outside'])}, "
          f"nondeterministic {sum(1 for c in served if c['distinct_outputs'] != 1)}, errors {sum(1 for c in report['cases'] if 'error' in c)}; wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
