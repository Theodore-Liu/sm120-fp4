"""Enumerate every autotuner tactic FlashInfer's mm_fp4 runners offer on this device, run each one alone against the reference
GEMM, and write the list a caller can pin.

Why: on SM120 the autotuner has presented tactics that cannot run (flashinfer-ai/flashinfer#4841 counted 14 of 60 for the fused
MoE), and a tactic chosen under memory pressure at one boot differs from the next. A pinned, tested tactic per shape removes
the lottery. This probe uses the runners exactly as mm_fp4 constructs them (flashinfer.gemm.gemm_base), so what it lists is
what the autotuner sees.

    PYTHONPATH=. python scripts/probe_tactics.py --out reports/tactics-<device>-<date>.json
"""
from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path

import torch

from sm120fp4 import quantize_nvfp4, reference_gemm_nvfp4, to_128x4


def operands(m, n, k, device):
    torch.manual_seed(3)
    a = torch.randn(m, k, device=device).to(torch.bfloat16)
    b = torch.randn(n, k, device=device).to(torch.bfloat16)
    a_q, a_sf, a_gs = quantize_nvfp4(a.float())
    b_q, b_sf, b_gs = quantize_nvfp4(b.float())
    ref = reference_gemm_nvfp4(a_q, a_sf, a_gs, b_q, b_sf, b_gs)
    return a_q, to_128x4(a_sf).view(torch.float8_e4m3fn), b_q, to_128x4(b_sf).view(torch.float8_e4m3fn), (1.0 / (a_gs * b_gs)).reshape(1), ref


def _time(fn, warmup: int, iters: int) -> float:
    """Median wall time of fn on the current stream, in ms, from CUDA events."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports") / f"tactics-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    ap.add_argument("--shapes", default="128x256x512,16x4096x4096,1x1024x2048,512x512x1024,4096x4096x4096")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    a = ap.parse_args(argv)
    import flashinfer.gemm as g
    M = vars(inspect.getmodule(inspect.unwrap(g.mm_fp4)))
    device = torch.device("cuda")
    maj, mi = torch.cuda.get_device_capability()
    # mm_fp4 resolves enable_pdl from the device before it builds a runner; the probe does the same, otherwise the
    # CuTe-DSL runners fail in their cache-key formatting (int(None)) and that failure would be the probe's, not the kernel's.
    from flashinfer.utils import device_support_pdl
    pdl = bool(device_support_pdl(device))
    factories = {
        "cutlass": lambda: M["get_cutlass_fp4_gemm_module"](maj, mi).cutlass_fp4_gemm_runner(),
        "b12x": lambda: M["_b12x_gemm_fp4_runner"](maj, mi, pdl, torch.bfloat16, True),
        "cute-dsl": lambda: M["_cute_dsl_gemm_fp4_runner"](maj, mi, pdl, torch.bfloat16, True),
    }
    report = {"device": torch.cuda.get_device_name(0), "capability": [maj, mi], "flashinfer": __import__("flashinfer").__version__,
              "torch": torch.__version__, "enable_pdl": pdl, "timing": {"warmup": a.warmup, "iters": a.iters, "statistic": "median ms"},
              "backends": {}}
    for name, mk in factories.items():
        entry = {"runner": None, "shapes": {}}
        try:
            runner = mk()
            entry["runner"] = type(runner).__name__
        except Exception as e:  # noqa: BLE001
            entry["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            report["backends"][name] = entry
            continue
        for shp in a.shapes.split(","):
            m, n, k = (int(x) for x in shp.split("x"))
            a_q, a_sf, b_q, b_sf, alpha, ref = operands(m, n, k, device)
            out = torch.empty(m, n, device=device, dtype=torch.bfloat16)
            ws = M["_get_cache_buf"]("mm_fp4_workspace", M["DEFAULT_WORKSPACE_SIZE"], device)
            inputs = [a_q, b_q.t(), a_sf, b_sf, alpha, torch.bfloat16, out, 16, True, ws]
            try:
                tactics = runner.get_valid_tactics(inputs, None)
            except Exception as e:  # noqa: BLE001
                entry["shapes"][shp] = {"error": f"get_valid_tactics: {type(e).__name__}: {str(e)[:160]}"}
                continue
            rows = []
            scale = ref.abs().max().item() + 1e-6
            # The runner's own default (tactic -1 for cutlass, None for b12x/cute-dsl) is what a caller gets without the
            # autotuner. It is timed too, so the report can say whether any presented tactic beats it.
            default_tactic = -1 if name == "cutlass" else None
            for i, t in enumerate([default_tactic] + list(tactics)):
                row = {"tactic": "default" if i == 0 else str(t)}
                try:
                    out.zero_()
                    runner(inputs=inputs, tactic=t)
                    torch.cuda.synchronize()
                    err = (out.float() - ref).abs().max().item()
                    row.update({"ran": True, "rel_err": err / scale, "all_zero": bool((out == 0).all()), "correct": (err / scale < 2e-2) and not bool((out == 0).all())})
                    row["median_ms"] = _time(lambda: runner(inputs=inputs, tactic=t), a.warmup, a.iters)
                except Exception as e:  # noqa: BLE001
                    row.update({"ran": False, "error": f"{type(e).__name__}: {str(e)[:160]}"})
                rows.append(row)
            presented = rows[1:]
            timed = [r for r in presented if r.get("correct")]
            entry["shapes"][shp] = {"presented": len(tactics), "ran": sum(r["ran"] for r in presented), "correct": len(timed),
                                    "default_ms": rows[0].get("median_ms"),
                                    "fastest": min(timed, key=lambda r: r["median_ms"])["tactic"] if timed else None,
                                    "fastest_ms": min(r["median_ms"] for r in timed) if timed else None,
                                    "distinct_median_ms": len({round(r["median_ms"], 4) for r in timed}),
                                    "pinnable": [r["tactic"] for r in timed], "rows": rows}
        report["backends"][name] = entry
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for name, e in report["backends"].items():
        if "error" in e:
            print(f"{name}: {e['error']}")
            continue
        for shp, s in e["shapes"].items():
            if "error" in s:
                print(f"{name} {shp}: {s['error']}")
            else:
                print(f"{name} {shp}: presented {s['presented']}, ran {s['ran']}, correct {s['correct']}, "
                      f"default {s['default_ms']} ms, fastest {s['fastest']} at {s['fastest_ms']} ms, {s['distinct_median_ms']} distinct timings")
    print(f"written {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
