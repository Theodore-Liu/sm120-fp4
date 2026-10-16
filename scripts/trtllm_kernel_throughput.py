#!/usr/bin/env python3
"""The TensorRT-LLM half of R1's cost column: wall time of `torch.ops.trtllm.nvfp4_gemm*` on the 5090 at the shared dense shapes
(CUDA events, median of 50 after 10 warm-ups), reusing the census probe's operand construction and backend table
(`trtllm_nvfp4_gemm_probe.py`: cutlass, cublaslt, the default list "cutlass,cublaslt,cuda_core", cute_dsl where present).

    PYTHONPATH=<sm120-fp4>:<sm120-fp4>/scripts python scripts/trtllm_kernel_throughput.py --out <json> [--shapes small|paper]
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def timeit(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000.0)
    times.sort()
    q = statistics.quantiles(times, n=4)
    return {"median_us": round(statistics.median(times), 2), "q1_us": round(q[0], 2), "q3_us": round(q[2], 2), "min_us": round(times[0], 2), "iters": iters}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shapes", default="paper")
    a = ap.parse_args(argv)
    import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm)
    from trtllm_nvfp4_gemm_probe import backends, random_operands, swizzle

    dev = torch.device("cuda")
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "capability": list(torch.cuda.get_device_capability()), "torch": torch.__version__, "tensorrt_llm": tensorrt_llm.__version__,
              "clock_note": "WSL2 host; clocks not pinned; median and IQR of 50 timings after 10 warm-ups", "rows": []}
    shapes = [(1, 4096, 4096), (16, 4096, 4096), (256, 4096, 4096), (2048, 4096, 4096), (16, 14336, 4096), (2048, 14336, 4096), (256, 8192, 8192)]
    if a.shapes == "small":
        shapes = shapes[:3]
    gen = torch.Generator().manual_seed(1)
    fns = backends()
    for (M, N, K) in shapes:
        (a_p, a_sf, a_gs), (b_p, b_sf, b_gs), _, _ = random_operands(M, N, K, gen)
        alpha = (1.0 / (a_gs.double() * b_gs.double())).float().to(dev)
        ad, bd, asw, bsw = a_p.to(dev), b_p.to(dev), swizzle(a_sf).to(dev), swizzle(b_sf).to(dev)
        for name, fn in fns.items():
            try:
                t = timeit(lambda: fn(ad, bd, asw, bsw, alpha))
                r = {"kernel": f"trtllm nvfp4_gemm {name}", "shape": [M, N, K], "dtype": "nvfp4 x nvfp4 -> bf16",
                     "tflops": round(2 * M * N * K / (t["median_us"] * 1e-6) / 1e12, 2), **t}
            except Exception as exc:  # noqa: BLE001
                r = {"kernel": f"trtllm nvfp4_gemm {name}", "shape": [M, N, K], "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
            report["rows"].append(r); print(json.dumps(r), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("rows", len(report["rows"]), "errors", sum("error" in r for r in report["rows"]), "wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
