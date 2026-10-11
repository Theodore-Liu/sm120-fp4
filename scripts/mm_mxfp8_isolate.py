#!/usr/bin/env python3
"""FlashInfer's dense MXFP8 GEMM (`mm_mxfp8`, block size 32, UE8M0 scales) on one backend against the exact oracle, one
backend per process, with a marker file written on the Windows side before every kernel call.

Why one backend per process: `bmm_fp8`'s cutlass backend took the WSL2 virtual machine down on 2026-10-11 and left no report
and no error behind; a marker that says which call was in flight is the only record such a crash leaves. The Windows-side loop
(`scripts/run_mm_mxfp8_census.sh` or the command in `docs/`) runs this script once per backend and checks the VM between runs.

The oracle is exact: `mxfp8_dequantize_host` returns the quantised values (e4m3 times a power of two, exact in float64); the
product is summed in float64 and the fp32-accumulation bound of the census applies (`sm120fp4.oracle.U32`, the tensor core's
truncation unit). Operands follow the docstring: A (m, k) row-major e4m3 with its 1D swizzled 128x4 scales, B (k, n) column
major (the transposed view of a row-major (n, k)) with its 1D swizzled scales, both from `mxfp8_quantize(..., True, 32)`.

    PYTHONPATH=. python scripts/mm_mxfp8_isolate.py --backend cutlass --log <marker file> --out reports/<file>.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

SHAPES = ((1, 4096, 4096), (16, 2048, 7168), (128, 4096, 1024), (256, 1024, 2048), (512, 512, 512))  # (m, n, k)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--log", required=True, help="marker file (put it on the Windows side)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    def mark(s: str) -> None:
        with open(a.log, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {a.backend} {s}\n")
            f.flush()

    mark("start")
    import flashinfer
    from sm120fp4 import oracle
    dev = torch.device("cuda")
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "flashinfer": flashinfer.__version__, "backend": a.backend, "cases": []}
    for m, n, k in SHAPES:
        gen = torch.Generator().manual_seed(20261011 + 7 * m + n + k)
        rec = {"m": m, "n": n, "k": k}
        try:
            x = (torch.randn(m, k, generator=gen) * 2).to(torch.bfloat16).to(dev)
            w = torch.randn(n, k, generator=gen).to(torch.bfloat16).to(dev)
            xq, xsf = flashinfer.mxfp8_quantize(x, True, 32)
            wq, wsf = flashinfer.mxfp8_quantize(w, True, 32)
            torch.cuda.synchronize()
            xd = flashinfer.mxfp8_dequantize_host(xq.view(torch.uint8).cpu(), xsf.view(torch.uint8).cpu(), True).double()
            wd = flashinfer.mxfp8_dequantize_host(wq.view(torch.uint8).cpu(), wsf.view(torch.uint8).cpu(), True).double()
            exact = xd @ wd.t()
            bound = k * oracle.U32 * (xd.abs() @ wd.abs().t()) + 0.5 * oracle._ulp(exact, torch.bfloat16) + 2 * oracle.U64 * exact.abs()
            mark(f"calling m={m} n={n} k={k}")
            out = flashinfer.mm_mxfp8(xq, wq.t(), xsf.view(torch.uint8).reshape(-1), wsf.view(torch.uint8).reshape(-1),
                                      out_dtype=torch.bfloat16, backend=a.backend)
            torch.cuda.synchronize()
            rec.update(oracle.conformance(out.float(), exact, bound))
            mark(f"ok outside={rec['outside_bound']}")
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
            mark(f"error {type(exc).__name__}: {str(exc)[:120]}")
        report["cases"].append(rec)
        print(json.dumps(rec)[:300])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    mark("wrote report")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
