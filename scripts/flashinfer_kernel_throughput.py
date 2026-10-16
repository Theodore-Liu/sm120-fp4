#!/usr/bin/env python3
"""The FlashInfer half of R1's cost column: wall time of the FlashInfer NVFP4 kernels the census read, at the same GEMM shapes as
`vllm_kernel_throughput.py`, measured with CUDA events after warm-up (median, inter-quartile range, minimum of repeated timings).

What is timed (one row per shape and backend):
  mm_fp4 (NVFP4 x NVFP4, 128x4-swizzled E4M3 scales)       backends cudnn / cutlass / b12x / auto, as each accepts the shape
  mm_bf16_fp4 (bf16 activations x NVFP4 weights, W4A16)     backends cudnn / cute-dsl, via prepare_bf16_fp4_weights
  grouped_mm_fp4 (cuDNN MoE grouped GEMM, 128-aligned segments so the measured code is correct)   the census's MoE shapes

    PYTHONPATH=<sm120-fp4> python scripts/flashinfer_kernel_throughput.py --out <json> [--shapes small|paper]
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
    ap.add_argument("--fp4-backends", default="cudnn,cutlass,b12x,auto")
    ap.add_argument("--w4a16-backends", default="cudnn,cute-dsl")
    a = ap.parse_args(argv)
    import flashinfer
    from flashinfer import SfLayout
    from flashinfer.fp4_quantization import nvfp4_quantize
    from sm120fp4.layouts import to_128x4
    from sm120fp4.reference import quantize_nvfp4

    dev = torch.device("cuda")
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "capability": list(torch.cuda.get_device_capability()), "torch": torch.__version__, "flashinfer": flashinfer.__version__,
              "cudnn": torch.backends.cudnn.version(), "clock_note": "WSL2 host; clocks not pinned; median and IQR of 50 timings after 10 warm-ups",
              "rows": []}
    gen = torch.Generator().manual_seed(1)
    gemm_shapes = [(1, 4096, 4096), (16, 4096, 4096), (256, 4096, 4096), (2048, 4096, 4096), (16, 14336, 4096), (2048, 14336, 4096), (256, 8192, 8192)]
    moe_shapes = [(8, 128, 4096, 4096), (8, 1024, 4096, 4096), (64, 128, 2048, 7168), (128, 64, 2048, 7168), (128, 512, 1536, 4096)]
    if a.shapes == "small":
        gemm_shapes = gemm_shapes[:3]; moe_shapes = moe_shapes[:2]

    def row(kernel, shape, extra, t):
        r = {"kernel": kernel, "shape": shape, **extra, **t}
        report["rows"].append(r); print(json.dumps(r), flush=True)

    def tflops(M, N, K, t):
        return round(2 * M * N * K / (t["median_us"] * 1e-6) / 1e12, 2)

    # ---- dense GEMMs
    for (M, N, K) in gemm_shapes:
        x = torch.randn(M, K, generator=gen).to(torch.bfloat16)
        w = torch.randn(N, K, generator=gen).to(torch.bfloat16)
        aq, asf, ags = quantize_nvfp4(x.float())
        bq, bsf, bgs = quantize_nvfp4(w.float())
        alpha = (1.0 / (ags * bgs)).reshape(1).float().to(dev)
        a_d, b_d = aq.to(dev), bq.to(dev)
        asw = to_128x4(asf).view(torch.float8_e4m3fn).to(dev)
        bsw = to_128x4(bsf).view(torch.float8_e4m3fn).to(dev)
        for be in a.fp4_backends.split(","):
            try:
                t = timeit(lambda: flashinfer.mm_fp4(a_d, b_d.t(), asw, bsw, alpha, torch.bfloat16, None, 16, False, be))
                row(f"mm_fp4 {be}", [M, N, K], {"dtype": "nvfp4 x nvfp4 -> bf16", "tflops": tflops(M, N, K, t)}, t)
            except Exception as exc:  # noqa: BLE001
                row(f"mm_fp4 {be}", [M, N, K], {"error": str(exc)[:160]}, {})
        for be in a.w4a16_backends.split(","):
            try:
                alpha_w = (1.0 / bgs.double()).float().reshape(1).to(dev)
                b_p, sf_p, alpha_p = flashinfer.prepare_bf16_fp4_weights(bq.to(dev), bsw, alpha_w, backend=be)
                xd = x.to(dev)
                t = timeit(lambda: flashinfer.mm_bf16_fp4(xd, b_p, sf_p, alpha_p, backend=be, out_dtype=torch.bfloat16))
                row(f"mm_bf16_fp4 {be} (W4A16)", [M, N, K], {"dtype": "bf16 x nvfp4 -> bf16", "tflops": tflops(M, N, K, t)}, t)
            except Exception as exc:  # noqa: BLE001
                row(f"mm_bf16_fp4 {be} (W4A16)", [M, N, K], {"error": str(exc)[:160]}, {})

    # ---- grouped MoE GEMM (cuDNN), segments 128-aligned
    for (e_count, tpe, n, k) in moe_shapes:
        try:
            g = torch.Generator(device=dev).manual_seed(5)
            cum_m = e_count * tpe
            a_bf = torch.randn(cum_m, k, generator=g, device=dev, dtype=torch.bfloat16)
            b_bf = torch.randn(e_count * n, k, generator=g, device=dev, dtype=torch.bfloat16)
            gsa = (448 * 6) / a_bf.float().abs().max(); gsb = (448 * 6) / b_bf.float().abs().max()
            a_fp4, a_sf = nvfp4_quantize(a_bf, gsa, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
            b_fp4, b_sf = nvfp4_quantize(b_bf, gsb, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
            a_sf = a_sf.view(torch.float8_e4m3fn).reshape(-1, k // 16)
            b_fp4 = b_fp4.reshape(e_count, n, k // 2); b_sf = b_sf.view(torch.float8_e4m3fn).reshape(e_count, -1, k // 16)
            m_indptr = (torch.arange(e_count + 1, device=dev) * tpe).to(torch.int32)
            alpha_g = torch.tensor([1.0 / float(gsa * gsb)], device=dev)
            t = timeit(lambda: flashinfer.grouped_mm_fp4(a_fp4, b_fp4, a_sf, b_sf, m_indptr, alpha=alpha_g, out_dtype=torch.bfloat16, block_size=16))
            row("grouped_mm_fp4 cudnn", [e_count, tpe, n, k], {"dtype": "nvfp4 x nvfp4 -> bf16", "tflops": tflops(cum_m, n, k, t), "note": "experts x tokens/expert x n x k; segments 128-aligned"}, t)
        except Exception as exc:  # noqa: BLE001
            row("grouped_mm_fp4 cudnn", [e_count, tpe, n, k], {"error": str(exc)[:160]}, {})

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("rows", len(report["rows"]), "errors", sum("error" in r for r in report["rows"]), "wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
