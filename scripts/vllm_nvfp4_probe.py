#!/usr/bin/env python3
"""vLLM's NVFP4 quantiser and GEMM on this SM120 device, judged by the exact oracle (sm120fp4.oracle).

Two shipped ops of vllm._custom_ops:
  * `scaled_fp4_quant(input, input_global_scale, is_sf_swizzled_layout=True)` -> (codes uint8 [M, K/2], scales e4m3 swizzled 128x4):
    checked element by element against the reference quantiser under the same global scale (`quantizer_conformance`; midpoint
    ties reported apart), after un-swizzling the scales with sm120fp4.layouts.from_128x4;
  * `cutlass_scaled_fp4_mm(a, b, block_scale_a, block_scale_b, alpha, out_dtype)`: every output element against the exact
    product and the fp32-accumulation bound (`exact_gemm_nvfp4`, `conformance`), on operands from vLLM's own quantiser and
    on lossless operands (codes on the grid, scales 1.0) built by the reference.

    PYTHONPATH=/path/to/sm120-fp4 python scripts/vllm_nvfp4_probe.py --out reports/vllm-nvfp4-<device>-<date>.json
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import torch

from sm120fp4 import oracle
from sm120fp4.layouts import from_128x4, to_128x4
from sm120fp4.reference import global_scale_for, pack_e2m1, quantize_nvfp4

SHAPES = ((16, 4096, 4096), (128, 4096, 4096), (1, 4096, 4096), (64, 512, 1024), (3, 256, 2048))


def lossless(rows: int, k: int, gen: torch.Generator):
    idx = torch.randint(0, 8, (rows, k), generator=gen)
    sign = torch.randint(0, 2, (rows, k), generator=gen)
    return pack_e2m1(((sign << 3) | idx).to(torch.uint8)), torch.full((rows, k // 16), 0x38, dtype=torch.uint8), torch.ones(1)


def unswizzle(sf_sw: torch.Tensor, rows: int, k: int) -> torch.Tensor:
    return from_128x4(sf_sw.reshape(-1).view(torch.uint8).cpu(), rows, k // 16).contiguous()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports") / f"vllm-nvfp4-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    ap.add_argument("--shapes", default=",".join("x".join(map(str, s)) for s in SHAPES))
    a = ap.parse_args(argv)
    import vllm
    import vllm._custom_ops as ops
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": props.name,
              "capability": list(torch.cuda.get_device_capability()), "torch": torch.__version__, "cuda": torch.version.cuda,
              "vllm": vllm.__version__, "python": platform.python_version(),
              "driver": subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
              "supports_fp4": bool(ops.cutlass_scaled_mm_supports_fp4(torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1])) if hasattr(ops, "cutlass_scaled_mm_supports_fp4") else None,
              "quantizer": [], "gemm": []}
    gen = torch.Generator().manual_seed(20261010)
    for shp in a.shapes.split(","):
        m, n, k = map(int, shp.split("x"))
        # 1. vLLM's quantiser against the format
        x_a = (torch.randn(m, k, generator=gen) * 2.0).to(torch.bfloat16)
        x_b = torch.randn(n, k, generator=gen).to(torch.bfloat16)
        rec_q = {"shape": [m, n, k]}
        try:
            gs_a = global_scale_for(x_a.float()).to(device)
            gs_b = global_scale_for(x_b.float()).to(device)
            qa, sfa_sw = ops.scaled_fp4_quant(x_a.to(device), gs_a)
            qb, sfb_sw = ops.scaled_fp4_quant(x_b.to(device), gs_b)
            torch.cuda.synchronize()
            sfa = unswizzle(sfa_sw, m, k)
            sfb = unswizzle(sfb_sw, n, k)
            rec_q["a"] = oracle.quantizer_conformance(x_a.float(), qa.cpu(), sfa, gs_a.cpu())
            rec_q["b"] = oracle.quantizer_conformance(x_b.float(), qb.cpu(), sfb, gs_b.cpu())
            # 2. the GEMM on vLLM's own operands
            exact, bound = oracle.exact_gemm_nvfp4(qa.cpu(), sfa, gs_a.cpu(), qb.cpu(), sfb, gs_b.cpu(), out_dtype=torch.bfloat16)
            alpha = (1.0 / (gs_a.double() * gs_b.double())).float().reshape(1)
            out = ops.cutlass_scaled_fp4_mm(qa, qb, sfa_sw, sfb_sw, alpha, torch.bfloat16)
            torch.cuda.synchronize()
            rec_g = {"shape": [m, n, k], "operands": "vllm-quantised", **oracle.conformance(out.float(), exact, bound)}
        except Exception as exc:  # noqa: BLE001
            rec_q["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            rec_g = {"shape": [m, n, k], "operands": "vllm-quantised", "error": rec_q["error"]}
        report["quantizer"].append(rec_q)
        report["gemm"].append(rec_g)
        print(json.dumps(rec_q)[:400]); print(json.dumps(rec_g)[:400])
        # 3. the GEMM on lossless operands built by the reference
        (la, lsa, lga), (lb, lsb, lgb) = lossless(m, k, gen), lossless(n, k, gen)
        rec_l = {"shape": [m, n, k], "operands": "lossless"}
        try:
            exact, bound = oracle.exact_gemm_nvfp4(la, lsa, lga, lb, lsb, lgb, out_dtype=torch.bfloat16)
            out = ops.cutlass_scaled_fp4_mm(la.to(device), lb.to(device), to_128x4(lsa).view(torch.float8_e4m3fn).to(device),
                                            to_128x4(lsb).view(torch.float8_e4m3fn).to(device), torch.ones(1, device=device), torch.bfloat16)
            torch.cuda.synchronize()
            rec_l.update(oracle.conformance(out.float(), exact, bound))
        except Exception as exc:  # noqa: BLE001
            rec_l["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        report["gemm"].append(rec_l)
        print(json.dumps(rec_l)[:400])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
