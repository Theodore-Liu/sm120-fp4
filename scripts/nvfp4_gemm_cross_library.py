#!/usr/bin/env python3
"""Do the NVFP4 x NVFP4 GEMMs of different libraries return the same bits for the same operands? (R1 reference-semantics table, the
cross-library row; ISPASS 2027.)

Each library's kernel was judged against the exact oracle in the census (within the fp32-accumulation bound, not bit-exact),
so two libraries may both pass and still differ by their summation order. This measures that directly.

Two phases, because the libraries live in different Python environments:
  --phase dump   --lib flashinfer|trtllm   builds the operands from a fixed seed (first run) or loads them from --operands,
                                            runs every backend of that library, saves outputs to <outdir>/<lib>-<backend>.pt
  --phase compare                           loads every saved output and reports, per shape, which pairs are bit-identical and the
                                            max |difference| in bf16 ulps otherwise

    PYTHONPATH=<sm120-fp4>:<sm120-fp4>/scripts python scripts/nvfp4_gemm_cross_library.py --phase dump --lib flashinfer --outdir <dir>
    PYTHONPATH=... python scripts/nvfp4_gemm_cross_library.py --phase dump --lib trtllm --outdir <dir>
    python scripts/nvfp4_gemm_cross_library.py --phase compare --outdir <dir> --out <json>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

SHAPES = [(16, 4096, 4096), (256, 4096, 4096), (2048, 4096, 4096), (256, 8192, 8192), (1000, 4104, 4128)]


def operands(outdir: Path):
    f = outdir / "operands.pt"
    if f.exists():
        return torch.load(f, weights_only=False)   # our own file
    from sm120fp4.reference import quantize_nvfp4
    gen = torch.Generator().manual_seed(20261016)
    ops = {}
    for (M, N, K) in SHAPES:
        a = torch.randn(M, K, generator=gen) * 2.0
        b = torch.randn(N, K, generator=gen)
        ops[(M, N, K)] = (quantize_nvfp4(a), quantize_nvfp4(b))
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(ops, f)
    return ops


def dump(lib: str, outdir: Path) -> int:
    from sm120fp4.layouts import to_128x4
    dev = torch.device("cuda")
    ops = operands(outdir)
    if lib == "flashinfer":
        import flashinfer
        ver = flashinfer.__version__
        fns = {be: (lambda a, b, asw, bsw, alpha, be=be: flashinfer.mm_fp4(a, b.t(), asw, bsw, alpha, torch.bfloat16, None, 16, False, be))
               for be in ("cutlass", "b12x")}
    elif lib == "trtllm":
        import tensorrt_llm
        from trtllm_nvfp4_gemm_probe import backends
        ver = tensorrt_llm.__version__
        fns = {k: v for k, v in backends().items() if k != "cute_dsl"}
    else:
        raise SystemExit(lib)
    for be, fn in fns.items():
        outs = {}
        for shp, ((a_p, a_sf, a_gs), (b_p, b_sf, b_gs)) in ops.items():
            alpha = (1.0 / (a_gs.double() * b_gs.double())).float().to(dev)
            # the same swizzled bytes; FlashInfer takes them as float8_e4m3fn, TensorRT-LLM's ops as uint8 (as the census probe passes them)
            asw = to_128x4(a_sf.view(torch.uint8)).contiguous().to(dev)
            bsw = to_128x4(b_sf.view(torch.uint8)).contiguous().to(dev)
            if lib == "flashinfer":
                asw, bsw = asw.view(torch.float8_e4m3fn), bsw.view(torch.float8_e4m3fn)
            try:
                o1 = fn(a_p.to(dev), b_p.to(dev), asw, bsw, alpha); torch.cuda.synchronize()
                o2 = fn(a_p.to(dev), b_p.to(dev), asw, bsw, alpha); torch.cuda.synchronize()
                outs[shp] = {"out": o1.cpu(), "deterministic": bool(torch.equal(o1, o2))}
            except Exception as exc:  # noqa: BLE001
                outs[shp] = {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
            print(json.dumps({"lib": lib, "backend": be, "shape": list(shp), **({"error": outs[shp]["error"]} if "error" in outs[shp] else {"deterministic": outs[shp]["deterministic"]})}), flush=True)
        torch.save({"lib": lib, "version": ver, "torch": torch.__version__, "backend": be, "outs": outs}, outdir / f"{lib}-{be}.pt")
    return 0


def compare(outdir: Path, out: Path) -> int:
    files = sorted(outdir.glob("*-*.pt"))
    runs = [torch.load(f, weights_only=False) for f in files if f.name != "operands.pt"]   # our own files
    names = [f"{r['lib']} {r['backend']}" for r in runs]
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runs": [{"name": n, "version": r["version"], "torch": r["torch"]} for n, r in zip(names, runs)], "shapes": []}
    for shp in SHAPES:
        rec = {"shape": list(shp), "pairs": []}
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                a = runs[i]["outs"].get(shp, {}); b = runs[j]["outs"].get(shp, {})
                if "out" not in a or "out" not in b:
                    rec["pairs"].append({"a": names[i], "b": names[j], "skipped": a.get("error") or b.get("error")}); continue
                x, y = a["out"].float(), b["out"].float()
                diff = (x - y).abs()
                # bf16 ulp of the larger magnitude, elementwise
                ulp = torch.maximum(x.abs(), y.abs()).clamp(min=1e-30)
                ulp = torch.pow(2.0, torch.floor(torch.log2(ulp)) - 7)
                n_diff = int((diff > 0).sum()); total = diff.numel()
                rec["pairs"].append({"a": names[i], "b": names[j], "identical": n_diff == 0, "elements_differing": n_diff, "fraction": round(n_diff / total, 6),
                                     "max_abs_diff": float(diff.max()), "max_diff_ulps": float((diff / ulp).max())})
        rec["deterministic"] = {names[i]: runs[i]["outs"].get(shp, {}).get("deterministic") for i in range(len(runs))}
        report["shapes"].append(rec)
        for p in rec["pairs"]:
            print(json.dumps({"shape": list(shp), **p}))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote", out)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["dump", "compare"], required=True)
    ap.add_argument("--lib", choices=["flashinfer", "trtllm"])
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    if a.phase == "dump":
        return dump(a.lib, a.outdir)
    return compare(a.outdir, a.out or (a.outdir / "cross-library.json"))


if __name__ == "__main__":
    raise SystemExit(main())
