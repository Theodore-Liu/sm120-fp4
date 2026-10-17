#!/usr/bin/env python3
"""Why does the best NVFP4 GEMM on the RTX 5090 reach half the card's rated dense FP4 rate? Step 1: timing conditions.

For TensorRT-LLM's nvfp4_gemm backends (cutlass, cublaslt, default) at the census's two large shapes, this measures the
same CUDA-event median as the cost column under three conditions and reads the SM clock while timing:
    warm      operands resident, nothing flushed (the cost column's method)
    flushed   a 256 MB device buffer is written between iterations, evicting the 5090's 96 MB L2
    clock     nvidia-smi clocks.sm sampled during the timed loop (median of the samples)
From the measured TFLOPS and the sampled clock it derives achieved FLOP per clock per SM against the figure implied by the
specification: 1,676 TFLOPS dense FP4 over 170 SMs at the 2,407 MHz boost clock is 4,096 FLOP/clock/SM, i.e. one
m16n8k64 e2m1 mma.sync (16,384 FLOP) every four clocks per SM. If the achieved rate per clock is half of that at a clock near
boost, the gap is in the kernel's pipeline use, not the host; if the clock sits far below boost, the gap is the host.

    python scripts/nvfp4_gemm_rate_gap.py --out reports/nvfp4-rate-gap-rtx5090/trtllm.json     (trtllm venv)
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

SPEC_TFLOPS_DENSE_FP4 = 1676.0
SMS = 170
BOOST_MHZ = 2407.0
SHAPES = [(2048, 14336, 4096), (2048, 4096, 4096), (256, 8192, 8192)]


def sample_clocks(stop: threading.Event, out: list):
    while not stop.is_set():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5)
            sm, pw, tp = [x.strip() for x in r.stdout.strip().split(",")]
            out.append((float(sm), float(pw), float(tp)))
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.05)


def timeit(fn, flush_buf, iters=50, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    times = []
    stop = threading.Event(); samples = []
    th = threading.Thread(target=sample_clocks, args=(stop, samples), daemon=True); th.start()
    for _ in range(iters):
        if flush_buf is not None:
            flush_buf.fill_(1.0)          # 256 MB write: evicts L2
            torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000.0)
    stop.set(); th.join(timeout=2)
    q = statistics.quantiles(times, n=4)
    clk = statistics.median([x[0] for x in samples]) if samples else None
    pw = statistics.median([x[1] for x in samples]) if samples else None
    tp = statistics.median([x[2] for x in samples]) if samples else None
    return {"median_us": q[1], "iqr_us": [q[0], q[2]], "clock_mhz": clk, "power_w": pw, "temp_c": tp, "clock_samples": len(samples)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import tensorrt_llm  # noqa: F401
    from trtllm_nvfp4_gemm_probe import backends, lossless_operands
    from sm120fp4.layouts import to_128x4
    dev = torch.device("cuda")
    fns = {k: v for k, v in backends().items() if k != "cute_dsl"}
    gen = torch.Generator().manual_seed(7)
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.float32, device=dev)   # 256 MB
    rows = []
    for (M, N, K) in SHAPES:
        (ap_, asf, ags), (bp_, bsf, bgs) = lossless_operands(M, N, K, gen)
        a_d, b_d = ap_.to(dev), bp_.to(dev)
        asw = to_128x4(asf).contiguous().to(dev); bsw = to_128x4(bsf).contiguous().to(dev)
        alpha = torch.ones(1, device=dev)
        for be, fn in fns.items():
            call = lambda: fn(a_d, b_d, asw, bsw, alpha)  # noqa: E731
            for cond, fb in (("warm", None), ("flushed", flush)):
                try:
                    t = timeit(call, fb)
                    tflops = 2 * M * N * K / (t["median_us"] * 1e-6) / 1e12
                    per_clk_sm = (tflops * 1e12 / (t["clock_mhz"] * 1e6) / SMS) if t["clock_mhz"] else None
                    r = {"backend": be, "shape": [M, N, K], "condition": cond, **t, "tflops": round(tflops, 1),
                         "flop_per_clock_per_sm": round(per_clk_sm, 0) if per_clk_sm else None,
                         "fraction_of_spec_rate_per_clock": round(per_clk_sm / 4096.0, 3) if per_clk_sm else None,
                         "fraction_of_spec_tflops": round(tflops / SPEC_TFLOPS_DENSE_FP4, 3)}
                except Exception as e:  # noqa: BLE001
                    r = {"backend": be, "shape": [M, N, K], "condition": cond, "error": repr(e)[:200]}
                rows.append(r); print(json.dumps(r), flush=True)
    rep = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
           "torch": torch.__version__, "spec_tflops_dense_fp4": SPEC_TFLOPS_DENSE_FP4, "sms": SMS, "boost_mhz": BOOST_MHZ,
           "spec_flop_per_clock_per_sm": SPEC_TFLOPS_DENSE_FP4 * 1e12 / (BOOST_MHZ * 1e6) / SMS, "rows": rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
