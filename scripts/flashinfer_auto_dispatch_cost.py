#!/usr/bin/env python3
"""Where the time goes when `mm_fp4` is called with backend="auto" on the 5090 (R1 cost column, follow-up to the 2026-10-16 table
where auto was 20 to 45 percent slower than either explicit backend at every shape).

Reading of FlashInfer v0.7.1 `flashinfer/utils.py` (`backend_requirement`, lines 1447 to 1486 and 1520 to 1600): for
backend="auto" the wrapper calls `suitable_auto_backends(cc, *args)` on every call; that runs every backend's requirement
checker on the arguments, then the heuristic, and only then dispatches. Nothing is cached. The hypothesis is therefore a
constant host-side cost per call, independent of the shape.

This script measures, at one shape, the wall time per call (host plus device, CUDA sync after each call) for auto and for
the backend auto resolves to, and profiles the auto path with cProfile so the host time is attributed to functions.

    PYTHONPATH=<sm120-fp4> python scripts/flashinfer_auto_dispatch_cost.py [--m 256] [--n 4096] [--k 4096] [--iters 200]
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import statistics
import time

import torch


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=200)
    a = ap.parse_args(argv)
    import flashinfer
    from sm120fp4.layouts import to_128x4
    from sm120fp4.reference import quantize_nvfp4

    dev = torch.device("cuda")
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(a.m, a.k, generator=gen).to(torch.bfloat16).float()
    w = torch.randn(a.n, a.k, generator=gen).to(torch.bfloat16).float()
    aq, asf, ags = quantize_nvfp4(x)
    bq, bsf, bgs = quantize_nvfp4(w)
    alpha = (1.0 / (ags * bgs)).reshape(1).float().to(dev)
    a_d, b_d = aq.to(dev), bq.to(dev)
    asw = to_128x4(asf).view(torch.float8_e4m3fn).to(dev)
    bsw = to_128x4(bsf).view(torch.float8_e4m3fn).to(dev)

    def call(be):
        return flashinfer.mm_fp4(a_d, b_d.t(), asw, bsw, alpha, torch.bfloat16, None, 16, False, be)

    # what auto resolves to
    call("auto"); torch.cuda.synchronize()
    resolved = list(getattr(flashinfer.mm_fp4, "suitable_auto_backends", []) or [])
    print(json.dumps({"flashinfer": flashinfer.__version__, "device": torch.cuda.get_device_name(0), "shape": [a.m, a.n, a.k],
                      "auto_resolves_to": resolved}))

    def wall_us(be, iters):
        for _ in range(20):
            call(be)
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter(); call(be); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e6)
        ts.sort()
        q = statistics.quantiles(ts, n=4)
        return {"median_us": round(statistics.median(ts), 1), "q1_us": round(q[0], 1), "q3_us": round(q[2], 1), "min_us": round(ts[0], 1)}

    def device_us(be, iters):
        for _ in range(20):
            call(be)
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record(); call(be); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) * 1000.0)
        return round(statistics.median(ts), 1)

    rows = {}
    for be in ["auto", *resolved[:1], "cutlass", "b12x"]:
        if be in rows:
            continue
        try:
            rows[be] = {"wall": wall_us(be, a.iters), "device_median_us": device_us(be, a.iters)}
        except Exception as exc:  # noqa: BLE001
            rows[be] = {"error": str(exc)[:160]}
    for be, r in rows.items():
        print(json.dumps({"backend": be, **r}))
    if "auto" in rows and resolved and resolved[0] in rows and "wall" in rows["auto"] and "wall" in rows[resolved[0]]:
        gap = rows["auto"]["wall"]["median_us"] - rows[resolved[0]]["wall"]["median_us"]
        print(json.dumps({"auto_minus_resolved_wall_us": round(gap, 1), "note": "constant across shapes if the cost is host-side dispatch"}))

    # attribute the auto path's host time
    pr = cProfile.Profile()
    for _ in range(20):
        call("auto")
    torch.cuda.synchronize()
    pr.enable()
    for _ in range(100):
        call("auto")
    torch.cuda.synchronize()
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(18)
    print("--- cProfile of 100 auto calls (cumulative):")
    for line in s.getvalue().splitlines():
        if "flashinfer" in line or "ncalls" in line or "function calls" in line:
            print(line[:170])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
