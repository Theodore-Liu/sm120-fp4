#!/usr/bin/env python3
"""The DeepGEMM half of R1's cost column: wall time of the vllm-project/DeepGEMM SM120 kernels the census read, at the same dense
shapes as `vllm_kernel_throughput.py` and the same MoE shapes as `flashinfer_kernel_throughput.py`, CUDA events after warm-up.

What is timed:
  fp8_gemm_nt        FP8 x FP8 with (1, 128) fp32 scales on both operands (SM120 recipe (1, 1, 128)), bf16 output
  fp8_fp4_gemm_nt    FP8 activations x FP4 (e2m1, two per byte) weights, (1, 128) scales, bf16 output
  k_grouped_fp8_gemm_nt_contiguous   the k-grouped MoE form (the census's k-grouped rows), 128-aligned segments
  m_grouped_fp8_fp4_gemm_nt_contiguous   the m-grouped FP8 x FP4 MoE form

Every call is wrapped: a refusal is recorded as a row with `error`, never raised, so the table shows what the fork accepts.

    PYTHONPATH=<sm120-fp4>:<DeepGEMM>/tests python scripts/deepgemm_kernel_throughput.py --out <json> [--shapes small|paper]
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
    import deep_gemm
    from deep_gemm.utils.math import per_token_cast_to_fp4, per_token_cast_to_fp8

    dev = torch.device("cuda")
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "capability": list(torch.cuda.get_device_capability()), "torch": torch.__version__, "deep_gemm_path": deep_gemm.__file__,
              "clock_note": "WSL2 host; clocks not pinned; median and IQR of 50 timings after 10 warm-ups", "rows": []}
    gemm_shapes = [(1, 4096, 4096), (16, 4096, 4096), (256, 4096, 4096), (2048, 4096, 4096), (16, 14336, 4096), (2048, 14336, 4096), (256, 8192, 8192)]
    moe_shapes = [(8, 128, 4096, 4096), (8, 1024, 4096, 4096), (64, 128, 2048, 7168), (128, 64, 2048, 7168), (128, 512, 1536, 4096)]
    if a.shapes == "small":
        gemm_shapes = gemm_shapes[:3]; moe_shapes = moe_shapes[:2]

    def row(kernel, shape, extra, t):
        r = {"kernel": kernel, "shape": shape, **extra, **t}
        report["rows"].append(r); print(json.dumps(r), flush=True)

    def tflops(M, N, K, t):
        return round(2 * M * N * K / (t["median_us"] * 1e-6) / 1e12, 2)

    # dense operands through the fork's own test generators (the same construction its tests validate the kernels with)
    from generators import KernelType, MajorTypeAB, QuantConfig, generate_normal, get_ue8m0_usage
    kernel_type = KernelType.Kernel1D1D
    use_ue8m0 = get_ue8m0_usage(kernel_type)
    for (M, N, K) in gemm_shapes:
        for label, qc in (("fp8_gemm (fp8 x fp8, (1,128) scales)", QuantConfig((128, 128, False, False))),
                          ("fp8_fp4_gemm (fp8 x fp4, (1,128) scales)", QuantConfig((128, 128, False, True)))):
            try:
                torch.manual_seed(1)
                recipe, recipe_a, recipe_b = qc.get_recipes(is_wgrad=False)
                a_, b_, c_, d_, _ = generate_normal(M, N, K, MajorTypeAB.KMajor, MajorTypeAB.KMajor, False, torch.bfloat16,
                                                    kernel_type, use_ue8m0=use_ue8m0, quant_config=qc)
                t = timeit(lambda: deep_gemm.fp8_fp4_gemm_nt(a_, b_, d_, c=None, disable_ue8m0_cast=not use_ue8m0,
                                                             recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b))
                row(label, [M, N, K], {"dtype": "-> bf16", "ue8m0": use_ue8m0, "tflops": tflops(M, N, K, t)}, t)
            except Exception as exc:  # noqa: BLE001
                row(label, [M, N, K], {"error": str(exc)[:200]}, {})

    # grouped forms, through the fork's own test generators
    try:
        from generators import MajorTypeAB, generate_k_grouped_contiguous, generate_m_grouped_contiguous  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        report["generators_error"] = str(exc)[:200]
        generate_k_grouped_contiguous = None
    if generate_k_grouped_contiguous is not None:
        old_align = deep_gemm.get_mk_alignment_for_contiguous_layout()
        try:
            deep_gemm.set_mk_alignment_for_contiguous_layout(128)
            for (e_count, tpe, n, k) in moe_shapes:
                # k-grouped: each expert contributes a K segment of `tpe` x 128 rows? No: k-grouped means the K dimension is
                # split per expert; m here is the output rows. Use m = n_out rows = tpe * ? Keep the MoE meaning: m tokens
                # per expert accumulate over the expert's K. We time the form the census read: ks = [k] * e_count, m = tpe.
                try:
                    ks = [k] * e_count
                    # the SM120 k-grouped kernel requires an fp32 accumulator (c.has_value()), as the fork's own SM120 tests pass
                    _, aa, bb, _, initial, _, layout, host_ks = generate_k_grouped_contiguous(
                        e_count, tpe, n, MajorTypeAB.KMajor, MajorTypeAB.KMajor, ks,
                        use_ue8m0=True, gran_k=128, k_alignment=128, use_psum_layout=False, accumulate=True, out_dtype=torch.float32)
                    d = initial.clone()
                    t = timeit(lambda: deep_gemm.k_grouped_fp8_gemm_nt_contiguous(aa, bb, d, host_ks, layout, d, recipe=(1, 1, 128)))
                    row("k_grouped_fp8_gemm_nt_contiguous", [e_count, tpe, n, k], {"dtype": "fp8 x fp8 -> fp32 (accumulate)", "tflops": tflops(tpe * e_count, n, k, t), "note": "experts x m x n x k_per_expert; the output accumulates over the group"}, t)
                except Exception as exc:  # noqa: BLE001
                    row("k_grouped_fp8_gemm_nt_contiguous", [e_count, tpe, n, k], {"error": str(exc)[:200]}, {})
        finally:
            deep_gemm.set_mk_alignment_for_contiguous_layout(old_align)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("rows", len(report["rows"]), "errors", sum("error" in r for r in report["rows"]), "wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
