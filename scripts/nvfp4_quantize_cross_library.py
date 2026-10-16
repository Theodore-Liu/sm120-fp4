#!/usr/bin/env python3
"""Do the NVFP4 activation quantisers of different libraries emit the same codes and block scales for the same bf16 input? (R1
reference-semantics table, the cross-library quantiser row; ISPASS 2027.)

The GEMMs agree to the bit (nvfp4_gemm_cross_library.py); the quantiser is the stage where libraries can legitimately differ
(tie rounding of the E4M3 scale, of the E2M1 code, the global-scale convention). Each library is run on the same bf16 inputs
and the same per-tensor global scale, with linear (unswizzled) scale layout where the library offers it, and the packed codes and
scales are compared byte for byte, against each other and against the CPU reference quantiser (`sm120fp4.reference`).

  --phase dump --lib vllm|flashinfer|trtllm --outdir <dir>     one run per Python environment; inputs saved once from a seed
  --phase compare --outdir <dir> [--out <json>]

    PYTHONPATH=<sm120-fp4>:<sm120-fp4>/scripts python scripts/nvfp4_quantize_cross_library.py --phase dump --lib vllm --outdir <dir>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

SHAPES = [(16, 4096), (256, 4096), (2048, 4096), (1000, 4128), (4097, 8192)]


def inputs(outdir: Path):
    f = outdir / "inputs.pt"
    if f.exists():
        return torch.load(f, weights_only=False)
    from sm120fp4.reference import global_scale_for
    gen = torch.Generator().manual_seed(20261016)
    xs = {}
    for shp in SHAPES:
        x = (torch.randn(*shp, generator=gen) * 2.0).to(torch.bfloat16)
        xs[shp] = (x, global_scale_for(x.float()))
    # a block built to sit on E2M1 code ties after scaling: values at the midpoints of the e2m1 grid times a scale of 1
    tie = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5.0, 6.0, -6.0] * 8, dtype=torch.bfloat16).reshape(1, 128)
    xs[(1, 128)] = (tie, global_scale_for(tie.float()))
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(xs, f)
    return xs


def dump(lib: str, outdir: Path) -> int:
    dev = torch.device("cuda")
    xs = inputs(outdir)
    outs = {}
    if lib == "vllm":
        import vllm
        from vllm import _custom_ops as ops
        ver = vllm.__version__
        def q(x, gs):
            codes, sf = ops.scaled_fp4_quant(x.to(dev), gs.to(dev), is_sf_swizzled_layout=False)
            return codes, sf
    elif lib == "flashinfer":
        import flashinfer
        from flashinfer import SfLayout
        from flashinfer.fp4_quantization import nvfp4_quantize
        ver = flashinfer.__version__
        def q(x, gs):
            codes, sf = nvfp4_quantize(x.to(dev), gs.to(dev), sfLayout=SfLayout.layout_linear, do_shuffle=False)
            return codes, sf
    elif lib == "trtllm":
        import tensorrt_llm
        ver = tensorrt_llm.__version__
        def q(x, gs):
            codes, sf = torch.ops.trtllm.fp4_quantize(x.to(dev), gs.to(dev), 16, False, False)
            return codes, sf
    elif lib == "reference":
        from sm120fp4.reference import quantize_nvfp4
        ver = "sm120fp4.reference (CPU)"
        def q(x, gs):
            packed, sf, gs_used = quantize_nvfp4(x.float())
            return packed, sf
    else:
        raise SystemExit(lib)
    for shp, (x, gs) in xs.items():
        try:
            codes, sf = q(x, gs)
            m, k = x.shape
            codes = codes.reshape(m, -1).cpu().contiguous().view(torch.uint8)
            sf = sf.reshape(-1).cpu().contiguous().view(torch.uint8)[: m * (k // 16)].reshape(m, k // 16)
            outs[shp] = {"codes": codes, "sf": sf}
            print(json.dumps({"lib": lib, "shape": list(shp), "codes_bytes": list(codes.shape), "sf_bytes": list(sf.shape)}), flush=True)
        except Exception as exc:  # noqa: BLE001
            outs[shp] = {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
            print(json.dumps({"lib": lib, "shape": list(shp), "error": outs[shp]["error"]}), flush=True)
    torch.save({"lib": lib, "version": ver, "torch": torch.__version__, "outs": outs}, outdir / f"quant-{lib}.pt")
    return 0


def compare(outdir: Path, out: Path) -> int:
    runs = [torch.load(f, weights_only=False) for f in sorted(outdir.glob("quant-*.pt"))]
    names = [r["lib"] for r in runs]
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runs": [{"lib": r["lib"], "version": r["version"], "torch": r["torch"]} for r in runs], "shapes": []}
    xs = inputs(outdir)
    for shp in xs:
        rec = {"shape": list(shp), "pairs": []}
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                a = runs[i]["outs"].get(shp, {}); b = runs[j]["outs"].get(shp, {})
                if "codes" not in a or "codes" not in b:
                    rec["pairs"].append({"a": names[i], "b": names[j], "skipped": a.get("error") or b.get("error")}); continue
                cd = int((a["codes"] != b["codes"]).sum()); sd = int((a["sf"] != b["sf"]).sum())
                rec["pairs"].append({"a": names[i], "b": names[j], "codes_bytes_differing": cd, "codes_bytes": int(a["codes"].numel()),
                                     "scales_differing": sd, "scales": int(a["sf"].numel()), "identical": cd == 0 and sd == 0})
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
    ap.add_argument("--lib", choices=["vllm", "flashinfer", "trtllm", "reference"])
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    if a.phase == "dump":
        return dump(a.lib, a.outdir)
    return compare(a.outdir, a.out or (a.outdir / "quant-cross-library.json"))


if __name__ == "__main__":
    raise SystemExit(main())
