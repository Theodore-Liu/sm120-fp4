#!/usr/bin/env python3
"""FlashInfer `mm_bf16_fp4` (W4A16: bf16 activations, NVFP4 weights) on this SM120 device against the exact W4A16 oracle, on the
backends it offers (cudnn, cute-dsl), with weights prepared by `prepare_bf16_fp4_weights` as the API requires. The weights are
quantised by the reference quantiser (row-major scales, one global scale); the prepared layout is the backend's business.

    PYTHONPATH=/path/to/sm120-fp4 python scripts/flashinfer_mm_bf16_fp4_probe.py --out reports/...json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sm120fp4 import oracle
from sm120fp4.layouts import to_128x4
from sm120fp4.reference import quantize_nvfp4

SHAPES = ((16, 4096, 4096), (1, 4096, 4096), (128, 2048, 2048), (64, 512, 1024), (3, 256, 2048))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports") / f"flashinfer-mm-bf16-fp4-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    ap.add_argument("--backends", default="cudnn,cute-dsl,cute-dsl-native,blackwell-native,blackwell-tiled")
    a = ap.parse_args(argv)
    import flashinfer
    dev = torch.device("cuda")
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "flashinfer": flashinfer.__version__, "cudnn": torch.backends.cudnn.version(), "cases": []}
    gen = torch.Generator().manual_seed(20261010)
    for m, n, k in SHAPES:
        x = torch.randn(m, k, generator=gen).to(torch.bfloat16)
        w = torch.randn(n, k, generator=gen).to(torch.bfloat16).float()
        bp, bsf, bgs = quantize_nvfp4(w)
        exact, bound = oracle.exact_gemm_bf16_nvfp4(x, bp, bsf, bgs, out_dtype=torch.bfloat16)
        for backend in a.backends.split(","):
            rec = {"shape": [m, n, k], "backend": backend}
            try:
                alpha = (1.0 / bgs.double()).float().reshape(1).to(dev)
                # the contract: canonical (N, K/2) codes and the 128x4-swizzled E4M3 scales, as nvfp4_quantize(sfLayout=layout_128x4) emits them
                b_p, sf_p, alpha_p = flashinfer.prepare_bf16_fp4_weights(bp.to(dev), to_128x4(bsf).view(torch.float8_e4m3fn).to(dev), alpha, backend=backend)
                out = flashinfer.mm_bf16_fp4(x.to(dev), b_p, sf_p, alpha_p, backend=backend, out_dtype=torch.bfloat16)
                torch.cuda.synchronize()
                rec.update(oracle.conformance(out.float(), exact, bound))
            except Exception as exc:  # noqa: BLE001
                rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            report["cases"].append(rec)
            print(json.dumps(rec)[:400])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
