#!/usr/bin/env python3
"""The cost column for R1 (the kernel reference-semantics characterisation, ISPASS 2027): wall time of each vLLM 0.28.0
kernel the census read, at a fixed set of shapes on this device, measured with CUDA events after warm-up, median and
inter-quartile range of repeated timings. Numbers are for the card at hand (the 5090 under WSL2 by default); the same script
runs on a pod for a second card.

What is timed (one row each, per shape):
  cutlass_scaled_mm fp8 (per-token x per-channel)       the FP8 W8A8 GEMM
  triton_scaled_mm int8                                   the INT8 fallback vLLM never reaches on SM120 (its price)
  scaled_fp8_quant per-token / per-tensor                 activation quantisers
  scaled_int8_quant dynamic symmetric
  per_token_group_quant_fp8 (1,128)
  rms_norm_dynamic_per_token_quant fp8                    the fused norm + quantiser
  rms_norm + scaled_fp8_quant                             its unfused composition
  marlin_gemm W4A16 (GPTQ uint4b8, group 128, bf16)       the weight-only path

    PYTHONPATH=<sm120-fp4> python vllm_kernel_throughput.py --out <json> [--shapes small|paper]
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

FP8 = torch.float8_e4m3fn


def timeit(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000.0)   # microseconds
    times.sort()
    q = statistics.quantiles(times, n=4)
    return {"median_us": round(statistics.median(times), 2), "q1_us": round(q[0], 2), "q3_us": round(q[2], 2), "min_us": round(times[0], 2), "iters": iters}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shapes", default="paper")
    a = ap.parse_args(argv)
    import vllm
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import triton_scaled_mm
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
    from vllm.scalar_type import scalar_types

    dev = "cuda"
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "capability": list(torch.cuda.get_device_capability()), "torch": torch.__version__, "vllm": vllm.__version__,
              "clock_note": "WSL2 host; clocks not pinned; median and IQR of 50 timings after 10 warm-ups", "rows": []}
    gen = torch.Generator().manual_seed(1)

    # GEMM shapes: (M tokens, N out, K in): a decode step and a prefill chunk for a 4k and a 8k hidden size
    gemm_shapes = [(1, 4096, 4096), (16, 4096, 4096), (256, 4096, 4096), (2048, 4096, 4096), (16, 14336, 4096), (2048, 14336, 4096), (256, 8192, 8192)]
    act_shapes = [(1, 4096), (16, 4096), (256, 4096), (2048, 4096), (2048, 8192), (16384, 4096)]
    if a.shapes == "small":
        gemm_shapes = gemm_shapes[:3]; act_shapes = act_shapes[:3]

    def row(kernel, shape, extra, t):
        r = {"kernel": kernel, "shape": shape, **extra, **t}
        report["rows"].append(r); print(json.dumps(r), flush=True)

    # ---- GEMMs
    for (M, N, K) in gemm_shapes:
        a8 = torch.randn((M, K), generator=gen).to(FP8).cuda()
        w8 = torch.randn((N, K), generator=gen).to(FP8).cuda().t()
        sa = (torch.rand((M, 1), generator=gen) * 0.01 + 1e-3).cuda(); sb = (torch.rand((N, 1), generator=gen) * 0.01 + 1e-3).cuda()
        try:
            t = timeit(lambda: ops.cutlass_scaled_mm(a8, w8, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16))
            row("cutlass_scaled_mm fp8", [M, N, K], {"dtype": "fp8->bf16", "tflops": round(2 * M * N * K / (t["median_us"] * 1e-6) / 1e12, 2)}, t)
        except Exception as exc:  # noqa: BLE001
            row("cutlass_scaled_mm fp8", [M, N, K], {"error": str(exc)[:160]}, {})
        ai = torch.randint(-128, 128, (M, K), generator=gen).to(torch.int8).cuda()
        wi = torch.randint(-128, 128, (N, K), generator=gen).to(torch.int8).cuda().t()
        try:
            t = timeit(lambda: triton_scaled_mm(ai, wi, sa, sb, torch.bfloat16, None))
            row("triton_scaled_mm int8 (the SM120 fallback)", [M, N, K], {"dtype": "int8->bf16", "tflops": round(2 * M * N * K / (t["median_us"] * 1e-6) / 1e12, 2)}, t)
        except Exception as exc:  # noqa: BLE001
            row("triton_scaled_mm int8 (the SM120 fallback)", [M, N, K], {"error": str(exc)[:160]}, {})
        # Marlin W4A16 at the same shape (GPTQ uint4b8, group 128), K-major weight [K, N]
        try:
            w = (torch.randn((K, N), generator=gen) / 10).to(torch.bfloat16).cuda()
            w_ref, qw, s, g_idx, si, _ = marlin_quantize(w, scalar_types.uint4b8, 128, False, torch.randperm(K, generator=gen))
            ws = marlin_make_workspace_new(torch.device(dev), 4)
            x = torch.randn((M, K), generator=gen).to(torch.bfloat16).cuda()
            c = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
            t = timeit(lambda: ops.marlin_gemm(x, c, qw, None, s, None, None, None, g_idx, si, ws, scalar_types.uint4b8, M, N, K, True, False, True, False))
            row("marlin_gemm W4A16 gptq g128", [M, N, K], {"dtype": "bf16", "tflops": round(2 * M * N * K / (t["median_us"] * 1e-6) / 1e12, 2)}, t)
        except Exception as exc:  # noqa: BLE001
            row("marlin_gemm W4A16 gptq g128", [M, N, K], {"error": str(exc)[:160]}, {})

    # ---- activation quantisers and the fused norm
    for (T, H) in act_shapes:
        x = torch.randn((T, H), generator=gen).to(torch.bfloat16).cuda()
        w = torch.rand((H,), generator=gen).to(torch.bfloat16).cuda()
        gb = T * H * 2 / 1e9
        def bw(t): return round(gb / (t["median_us"] * 1e-6), 1) if t else None  # GB/s read of the input (write not counted)
        for name, fn in (("scaled_fp8_quant per-token", lambda: ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)),
                         ("scaled_fp8_quant per-tensor", lambda: ops.scaled_fp8_quant(x)),
                         ("scaled_int8_quant dynamic sym", lambda: ops.scaled_int8_quant(x, None, None, symmetric=True)),
                         ("per_token_group_quant_fp8 (1,128)", lambda: per_token_group_quant_fp8(x, 128, use_ue8m0=False)),
                         ("rms_norm_dynamic_per_token_quant fp8 (fused)", lambda: ops.rms_norm_dynamic_per_token_quant(x, w, 1e-6, FP8)),
                         ("rms_norm + scaled_fp8_quant per-token (unfused)", None)):
            try:
                if fn is None:
                    y = torch.empty_like(x)
                    def fn2():
                        ops.rms_norm(y, x, w, 1e-6); return ops.scaled_fp8_quant(y, use_per_token_if_dynamic=True)
                    t = timeit(fn2)
                else:
                    t = timeit(fn)
                row(name, [T, H], {"dtype": "bf16 in", "input_gbps": bw(t)}, t)
            except Exception as exc:  # noqa: BLE001
                row(name, [T, H], {"error": str(exc)[:160]}, {})

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("rows", len(report["rows"]), "wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
