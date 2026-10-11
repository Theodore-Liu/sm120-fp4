#!/usr/bin/env python3
"""DeepGEMM's SM120 FP8 block-scaled GEMM (`deep_gemm.fp8_gemm_nt`, vllm-project/DeepGEMM dev, which vendors the
lucifer1004/DeepGEMM-sm120 device layer) against the exact oracle.

Operands follow DeepGEMM's own test (`tests/generators.py`, legacy recipe on SM120): A cast per token with one scale per 128 of K
(`per_token_cast_to_fp8(..., use_ue8m0=True)`), B cast per 128x128 block (`per_block_cast_to_fp8(..., use_ue8m0=True)`), scales
fp32 rounded up to powers of two (UE8M0), bf16 output, no accumulator. The oracle expands both scale grids element by element
(powers of two, so every scaled e4m3 value is exact in float64), sums in float64, and applies the census's bound (the tensor
core's truncation unit per addition, half a bf16 ulp at the top of the interval). A marker line is written before every call
(the census's crash rule).

    PYTHONPATH=. python scripts/deepgemm_fp8_probe.py --log <marker file> --out reports/<file>.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch

SHAPES = ((1, 4096, 4096), (16, 2048, 7168), (128, 4096, 1024), (256, 1024, 2048), (512, 512, 512))  # (m, n, k)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    def mark(s: str) -> None:
        with open(a.log, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} deepgemm {s}\n")

    mark("start")
    import deep_gemm
    from deep_gemm.utils.math import per_block_cast_to_fp8, per_token_cast_to_fp8
    from sm120fp4 import oracle
    dg_dir = Path(deep_gemm.__file__).parent
    try:
        commit = subprocess.run(["git", "-C", str(Path.home() / "oss/DeepGEMM-vllm"), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "deep_gemm": getattr(deep_gemm, "__version__", "?"), "deep_gemm_commit": commit,
              "deep_gemm_path": str(dg_dir), "cases": []}
    for m, n, k in SHAPES:
        gen = torch.Generator().manual_seed(20261011 + 7 * m + n + k)
        rec = {"m": m, "n": n, "k": k}
        try:
            x = torch.randn(m, k, generator=gen).to(torch.bfloat16).cuda()
            w = torch.randn(n, k, generator=gen).to(torch.bfloat16).cuda()
            xq, xsf = per_token_cast_to_fp8(x, use_ue8m0=True)          # (m, k), (m, k/128)
            wq, wsf = per_block_cast_to_fp8(w, use_ue8m0=True)          # (n, k), (n/128, k/128)
            torch.cuda.synchronize()
            xs = xsf.double().cpu().repeat_interleave(128, dim=1)[:, :k]
            ws = wsf.double().cpu().repeat_interleave(128, dim=0)[:n].repeat_interleave(128, dim=1)[:, :k]
            ad = xq.cpu().double() * xs
            wd = wq.cpu().double() * ws
            exact = ad @ wd.t()
            sabs = ad.abs() @ wd.abs().t()
            pre = k * oracle.U32 * sabs + 2 * oracle.U64 * sabs
            bound = pre + 0.5 * oracle._ulp(exact.abs() + pre, torch.bfloat16)
            d = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
            mark(f"calling m={m} n={n} k={k}")
            deep_gemm.fp8_gemm_nt((xq, xsf), (wq, wsf), d)
            torch.cuda.synchronize()
            rec.update(oracle.conformance(d.float(), exact, bound))
            mark(f"ok outside={rec['outside_bound']}")
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
            mark(f"error {type(exc).__name__}: {str(exc)[:150]}")
        report["cases"].append(rec)
        print(json.dumps(rec)[:300])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    mark("wrote report")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
