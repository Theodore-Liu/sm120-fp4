"""`sm120fp4 conformance`: run the conformance checks against the installed libraries and write one JSON report.

The report records the device, driver, CUDA and library versions, and for every check its outcome and the numbers
behind it, so that two machines' reports can be compared line by line. The pytest suite under tests/ is the same set
of checks with assertions; this command is the non-asserting form for people who want the report even when something
fails.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

from . import __version__
from .layouts import describe, to_128x4
from .reference import quantize_nvfp4, reference_gemm_nvfp4, unpack_e2m1


def _versions() -> dict:
    out = {"sm120fp4": __version__, "python": platform.python_version(), "torch": torch.__version__,
           "torch_cuda": torch.version.cuda, "platform": platform.platform()}
    try:
        import flashinfer  # type: ignore
        out["flashinfer"] = flashinfer.__version__
    except Exception as e:  # noqa: BLE001
        out["flashinfer"] = f"unavailable: {type(e).__name__}"
    try:
        out["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                           capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # noqa: BLE001
        out["nvidia_smi"] = f"unavailable: {type(e).__name__}"
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        out["device"] = {"name": p.name, "capability": list(torch.cuda.get_device_capability()), "sms": p.multi_processor_count,
                         "total_memory_gb": round(p.total_memory / 2**30, 1)}
    return out


def check_layout(fi, device) -> dict:
    res = {"name": "128x4 layout vs flashinfer.nvfp4_block_scale_interleave", "cases": []}
    for rows, cols in [(128, 4), (256, 32), (384, 8), (130, 6), (1000, 40)]:
        sf = torch.randint(1, 255, (rows, cols), dtype=torch.uint8, device=device)
        theirs = fi.nvfp4_block_scale_interleave(sf).reshape(-1)
        ours = to_128x4(sf).reshape(-1)
        res["cases"].append({"rows": rows, "sf_cols": cols, "equal": bool(theirs.numel() == ours.numel() and torch.equal(theirs, ours))})
    res["pass"] = all(c["equal"] for c in res["cases"])
    return res


def check_quantize(fi, device) -> dict:
    res = {"name": "fp4_quantize vs reference quantizer", "cases": []}
    torch.manual_seed(1)
    for m, k in [(128, 512), (256, 2048), (64, 4096)]:
        x = (torch.randn(m, k, device=device) * 2).to(torch.bfloat16)
        gs = (448.0 * 6.0 / x.abs().max().float()).reshape(1)
        q_fi, sf_fi = fi.fp4_quantize(x, gs, 16, False, False)
        sf_fi = sf_fi.view(-1)[: m * (k // 16)].view(m, k // 16)
        q_us, sf_us, _ = quantize_nvfp4(x.float(), gs)
        codes_differ = int((unpack_e2m1(q_fi) != unpack_e2m1(q_us)).sum())
        res["cases"].append({"m": m, "k": k, "scales_equal": bool(torch.equal(sf_fi, sf_us)), "codes_differ": codes_differ,
                             "codes_total": m * k, "codes_differ_fraction": codes_differ / (m * k)})
    res["pass"] = all(c["scales_equal"] and c["codes_differ_fraction"] < 0.005 for c in res["cases"])
    return res


def _operands(m, n, k, device):
    torch.manual_seed(3)
    a = torch.randn(m, k, device=device).to(torch.bfloat16)
    b = torch.randn(n, k, device=device).to(torch.bfloat16)
    a_q, a_sf, a_gs = quantize_nvfp4(a.float())
    b_q, b_sf, b_gs = quantize_nvfp4(b.float())
    ref = reference_gemm_nvfp4(a_q, a_sf, a_gs, b_q, b_sf, b_gs)
    return a_q, to_128x4(a_sf).view(torch.float8_e4m3fn), b_q, to_128x4(b_sf).view(torch.float8_e4m3fn), (1.0 / (a_gs * b_gs)).reshape(1), ref


def check_mm_fp4(fi, device, backends=("cutlass", "cudnn", "trtllm"), replays=200) -> list[dict]:
    out = []
    for backend in backends:
        r = {"name": f"mm_fp4[{backend}]", "backend": backend, "shapes": [], "graph_replay": None}
        for m, n, k in [(128, 256, 512), (16, 4096, 4096), (1, 1024, 2048), (512, 512, 1024)]:
            a_q, a_sf, b_q, b_sf, alpha, ref = _operands(m, n, k, device)
            try:
                t0 = time.time()
                o = fi.mm_fp4(a_q, b_q.t(), a_sf, b_sf, alpha, torch.bfloat16, None, 16, False, backend)
                torch.cuda.synchronize()
                o = o.float()
                err = (o - ref).abs().max().item()
                scale = ref.abs().max().item() + 1e-6
                r["shapes"].append({"m": m, "n": n, "k": k, "ran": True, "all_zero": bool((o == 0).all()), "max_abs_err": err,
                                    "ref_max": scale, "rel_err": err / scale, "first_call_s": round(time.time() - t0, 2)})
            except Exception as e:  # noqa: BLE001
                r["shapes"].append({"m": m, "n": n, "k": k, "ran": False, "error": f"{type(e).__name__}: {str(e)[:200]}"})
        ran = [s for s in r["shapes"] if s["ran"]]
        r["available"] = bool(ran)
        r["correct"] = bool(ran) and all((not s["all_zero"]) and s["rel_err"] < 2e-2 for s in ran)
        if ran:
            m, n, k = 64, 2048, 2048
            a_q, a_sf, b_q, b_sf, alpha, ref = _operands(m, n, k, device)
            try:
                run = lambda: fi.mm_fp4(a_q, b_q.t(), a_sf, b_sf, alpha, torch.bfloat16, None, 16, False, backend)  # noqa: E731
                eager = run().clone()
                s = torch.cuda.Stream()
                with torch.cuda.stream(s):
                    for _ in range(2):
                        run()
                torch.cuda.synchronize()
                outb = torch.empty_like(eager)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    outb.copy_(run())
                bad = 0
                for _ in range(replays):
                    g.replay()
                    torch.cuda.synchronize()
                    bad += int(not torch.equal(outb, eager))
                r["graph_replay"] = {"replays": replays, "differ_from_eager": bad, "pass": bad == 0}
            except Exception as e:  # noqa: BLE001
                r["graph_replay"] = {"error": f"{type(e).__name__}: {str(e)[:200]}", "pass": False}
        out.append(r)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sm120fp4")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("conformance", help="run the checks against the installed libraries and write a JSON report")
    c.add_argument("--out", type=Path, default=Path("reports") / f"conformance-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    c.add_argument("--backends", default="cutlass,cudnn,trtllm")
    c.add_argument("--replays", type=int, default=200)
    a = ap.parse_args(argv)
    if a.cmd == "conformance":
        report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "versions": _versions(), "layout": describe()}
        if not torch.cuda.is_available():
            report["error"] = "no CUDA device"
        else:
            major, _ = torch.cuda.get_device_capability()
            report["sm12x"] = major == 12
            try:
                import flashinfer as fi  # type: ignore
            except Exception as e:  # noqa: BLE001
                report["error"] = f"FlashInfer unavailable: {type(e).__name__}"
                fi = None
            if fi is not None:
                dev = torch.device("cuda")
                report["checks"] = {"layout": check_layout(fi, dev), "quantize": check_quantize(fi, dev),
                                    "mm_fp4": check_mm_fp4(fi, dev, tuple(a.backends.split(",")), a.replays)}
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=1))
        for name, chk in (report.get("checks") or {}).items():
            if isinstance(chk, list):
                for r in chk:
                    print(f"{r['name']}: available={r['available']} correct={r['correct']} graph_replay={r['graph_replay']}")
            else:
                print(f"{chk['name']}: pass={chk['pass']}")
        print(f"report written to {a.out}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
