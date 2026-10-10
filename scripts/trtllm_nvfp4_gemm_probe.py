#!/usr/bin/env python3
"""TensorRT-LLM's `torch.ops.trtllm.nvfp4_gemm` on this SM120 device, judged by the exact oracle (sm120fp4.oracle).

Why: TensorRT-LLM issue 14154 reports `nvfp4_gemm` numerically wrong on an RTX 5060 Ti (sm_120) and a GB10 (sm_121), with a
correlation of 0.113 against a bf16 reference and identical failure on the CUTLASS, cuBLASLt and CuTe-DSL backends, while an
RTX PRO 6000 (sm_120) reproduction found the op correct (correlation 0.9915; lossless operands bit-exact). This probe runs the
same two readings on the RTX 5090 (sm_120) and adds the oracle's verdict, which needs no tolerance: for lossless operands
(every element on the E2M1 grid, every block scale 1.0, per-tensor scales 1.0) the exact product is an integer and a correct
kernel returns it exactly after the output rounding; for random bf16 operands quantised by the reference quantiser the exact
value and the fp32-accumulation bound are computed per element.

Operands are built with sm120fp4.reference (row-major scales) and swizzled with sm120fp4.layouts.to_128x4, the layout
TensorRT-LLM and FlashInfer take; alpha = 1 / (gs_a * gs_b) as a float32 tensor. Backends: nvfp4_gemm_cutlass,
nvfp4_gemm_cublaslt, nvfp4_gemm (its default backend list), cute_dsl_nvfp4_gemm_blackwell (expected to refuse off SM100).

    PYTHONPATH=/path/to/sm120-fp4 python scripts/trtllm_nvfp4_gemm_probe.py --out reports/trtllm-nvfp4-gemm-rtx5090-<date>.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

from sm120fp4 import oracle
from sm120fp4.layouts import to_128x4
from sm120fp4.reference import E2M1_GRID, pack_e2m1, quantize_nvfp4

SHAPES = ((16, 4096, 4096), (128, 4096, 4096), (1, 4096, 4096), (64, 512, 1024), (3, 256, 2048))


def swizzle(sf_bits: torch.Tensor) -> torch.Tensor:
    """row-major uint8 [rows, K/16] -> 128x4 swizzled uint8, flattened as the kernels take it."""
    return to_128x4(sf_bits.view(torch.uint8)).contiguous()


def backends():
    ops = torch.ops.trtllm
    out = {"cutlass": lambda a, b, asf, bsf, alpha: ops.nvfp4_gemm_cutlass(a, b, asf, bsf, alpha, torch.bfloat16, False),
           "cublaslt": lambda a, b, asf, bsf, alpha: ops.nvfp4_gemm_cublaslt(a, b, asf, bsf, alpha, torch.bfloat16, False),
           "default": lambda a, b, asf, bsf, alpha: ops.nvfp4_gemm(a, b, asf, bsf, alpha, torch.bfloat16, False, "cutlass,cublaslt,cuda_core")}
    if hasattr(ops, "cute_dsl_nvfp4_gemm_blackwell"):
        out["cute_dsl"] = lambda a, b, asf, bsf, alpha: ops.cute_dsl_nvfp4_gemm_blackwell(a, b, asf, bsf, alpha, torch.bfloat16, False, True)
    return out


def lossless_operands(m: int, n: int, k: int, gen: torch.Generator):
    """Codes drawn from the E2M1 grid with sign, block scales exactly 1.0 (E4M3 bits 0x38), per-tensor scales 1.0: the
    exact product is an integer or half-integer sum and a correct kernel returns it to the bit after bf16 rounding."""
    def one(rows):
        idx = torch.randint(0, 8, (rows, k), generator=gen)
        sign = torch.randint(0, 2, (rows, k), generator=gen)
        codes = (sign << 3) | idx
        packed = pack_e2m1(codes.to(torch.uint8))
        sf = torch.full((rows, k // 16), 0x38, dtype=torch.uint8)    # E4M3 1.0
        return packed, sf, torch.ones(1)
    return one(m), one(n)


def random_operands(m: int, n: int, k: int, gen: torch.Generator):
    a = torch.randn(m, k, generator=gen) * 2.0
    b = torch.randn(n, k, generator=gen)
    return quantize_nvfp4(a), quantize_nvfp4(b), a, b


def run_case(name, fn, a_packed, a_sf, a_gs, b_packed, b_sf, b_gs, device) -> dict:
    exact, bound = oracle.exact_gemm_nvfp4(a_packed, a_sf, a_gs, b_packed, b_sf, b_gs, out_dtype=torch.bfloat16)
    alpha = (1.0 / (a_gs.double() * b_gs.double())).float().to(device)
    rec = {"backend": name}
    try:
        out = fn(a_packed.to(device), b_packed.to(device), swizzle(a_sf).to(device), swizzle(b_sf).to(device), alpha)
        torch.cuda.synchronize()
        rec.update(oracle.conformance(out.float(), exact, bound))
        o = out.double().cpu().flatten()
        e = exact.flatten()
        if e.std() > 0 and o.std() > 0:
            rec["correlation_vs_exact"] = float(torch.corrcoef(torch.stack([o, e]))[0, 1])
        rec["out_absmax"] = float(o.abs().max())
        rec["exact_absmax"] = float(e.abs().max())
    except Exception as exc:  # a refusal is a result
        rec.update({"error": f"{type(exc).__name__}: {str(exc)[:300]}"})
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports") / f"trtllm-nvfp4-gemm-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    ap.add_argument("--shapes", default=",".join("x".join(map(str, s)) for s in SHAPES))
    a = ap.parse_args(argv)
    import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm)
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "device": props.name, "capability": list(torch.cuda.get_device_capability()), "sms": props.multi_processor_count,
              "driver": None, "torch": torch.__version__, "cuda": torch.version.cuda,
              "tensorrt_llm": tensorrt_llm.__version__, "python": platform.python_version(), "cases": []}
    try:
        import subprocess
        report["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    gen = torch.Generator().manual_seed(20261010)
    fns = backends()
    for shp in a.shapes.split(","):
        m, n, k = map(int, shp.split("x"))
        (ap_, asf, ags), (bp_, bsf, bgs) = lossless_operands(m, n, k, gen)
        for name, fn in fns.items():
            rec = run_case(name, fn, ap_, asf, ags, bp_, bsf, bgs, device)
            rec.update({"shape": [m, n, k], "operands": "lossless"})
            report["cases"].append(rec)
            print(json.dumps(rec))
        (ap_, asf, ags), (bp_, bsf, bgs), a_bf, b_bf = random_operands(m, n, k, gen)
        for name, fn in fns.items():
            rec = run_case(name, fn, ap_, asf, ags, bp_, bsf, bgs, device)
            rec.update({"shape": [m, n, k], "operands": "random-bf16"})
            # the issue's own reading: correlation against the bf16 product of the unquantised operands
            if "error" not in rec:
                ref = (a_bf.double() @ b_bf.double().t()).flatten()
                exact, _ = oracle.exact_gemm_nvfp4(ap_, asf, ags, bp_, bsf, bgs)
                rec["correlation_exact_vs_bf16_product"] = float(torch.corrcoef(torch.stack([exact.flatten(), ref]))[0, 1])
            report["cases"].append(rec)
            print(json.dumps(rec))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    ok = [c for c in report["cases"] if c.get("conformant")]
    bad = [c for c in report["cases"] if "error" not in c and not c.get("conformant")]
    err = [c for c in report["cases"] if "error" in c]
    print(f"wrote {a.out}: {len(ok)} conformant, {len(bad)} non-conformant, {len(err)} refused")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
