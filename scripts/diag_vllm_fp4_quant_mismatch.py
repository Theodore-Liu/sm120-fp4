#!/usr/bin/env python3
"""Replay vllm_nvfp4_probe.py's operand stream and dissect every block where vLLM's `scaled_fp4_quant` and the reference quantiser
disagree on the E4M3 block scale: the block's amax, the fp32 scale value both start from, both E4M3 results, the distance to the
nearest E4M3 rounding boundary, and the global scale. Also quantises the same block with FlashInfer's `fp4_quantize` when it is
installed, to see which of the two the third library agrees with.

    PYTHONPATH=/path/to/sm120-fp4 python scripts/diag_vllm_fp4_quant_mismatch.py [--out reports/...json]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sm120fp4.layouts import from_128x4
from sm120fp4.reference import E2M1_MAX, E4M3_MAX, global_scale_for, quantize_nvfp4
from scripts.vllm_nvfp4_probe import SHAPES, lossless  # the same generator consumption as the probe


def replay_operands():
    gen = torch.Generator().manual_seed(20261010)
    for shp in SHAPES:
        m, n, k = shp
        x_a = (torch.randn(m, k, generator=gen) * 2.0).to(torch.bfloat16)
        x_b = torch.randn(n, k, generator=gen).to(torch.bfloat16)
        yield (m, n, k), "a", x_a
        yield (m, n, k), "b", x_b
        lossless(m, k, gen); lossless(n, k, gen)


def e4m3_neighbours(v: float):
    """The E4M3 values just below and above v (v > 0), from the float8_e4m3fn grid."""
    grid = torch.arange(0, 256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    grid = grid[torch.isfinite(grid) & (grid > 0)].unique().sort().values
    lo = grid[grid <= v].max().item() if (grid <= v).any() else 0.0
    hi = grid[grid >= v].min().item() if (grid >= v).any() else float("inf")
    return lo, hi


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports") / f"vllm-fp4-quant-mismatch-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    a = ap.parse_args(argv)
    import vllm
    import vllm._custom_ops as ops
    try:
        import flashinfer
        fi = flashinfer
    except Exception:  # noqa: BLE001
        fi = None
    dev = torch.device("cuda")
    report = {"vllm": vllm.__version__, "flashinfer": getattr(fi, "__version__", None), "torch": torch.__version__, "device": torch.cuda.get_device_name(0), "operands": []}
    for shape, which, x in replay_operands():
        m, k = x.shape
        gs = global_scale_for(x.float())
        ref_q, ref_sf, _ = quantize_nvfp4(x.float(), gs)
        q_v, sf_v = ops.scaled_fp4_quant(x.to(dev), gs.to(dev), is_sf_swizzled_layout=False)
        torch.cuda.synchronize()
        sf_v = sf_v.view(torch.uint8).cpu().reshape(m, k // 16)
        diff = (sf_v != ref_sf)
        rec = {"shape": list(shape), "operand": which, "rows": m, "k": k, "global_scale": float(gs), "scales_differ": int(diff.sum()), "codes_differ": int((q_v.cpu() != ref_q).sum())}
        if fi is not None:
            try:
                q_f, sf_f = fi.fp4_quantize(x.to(dev), gs.to(dev), 16, False, False)
                torch.cuda.synchronize()
                sf_f = sf_f.view(torch.uint8).cpu().reshape(m, k // 16)
                rec["flashinfer_scales_differ_from_ref"] = int((sf_f != ref_sf).sum())
                rec["flashinfer_scales_differ_from_vllm"] = int((sf_f != sf_v).sum())
            except Exception as exc:  # noqa: BLE001
                rec["flashinfer_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        if diff.any():
            ex = []
            for r, c in diff.nonzero().tolist()[:12]:
                blk = x.float()[r, c * 16:(c + 1) * 16]
                amax = float(blk.abs().max())
                sf_f32 = float(torch.tensor(amax / E2M1_MAX) * gs.float())
                lo, hi = e4m3_neighbours(sf_f32)
                ex.append({"row": r, "block": c, "amax": amax, "sf_fp32": sf_f32, "e4m3_below": lo, "e4m3_above": hi,
                           "midpoint": (lo + hi) / 2, "ref_value": float(ref_sf[r, c:c + 1].view(torch.float8_e4m3fn).float()),
                           "vllm_value": float(sf_v[r, c:c + 1].view(torch.float8_e4m3fn).float()),
                           "flashinfer_value": float(sf_f[r, c:c + 1].view(torch.float8_e4m3fn).float()) if fi is not None and "flashinfer_error" not in rec else None})
            rec["examples"] = ex
        # codes that differ with the library's own scale, outside blocks whose scale differs: how far from an E2M1 midpoint?
        codes_v = __import__("sm120fp4.reference", fromlist=["unpack_e2m1"]).unpack_e2m1(q_v.cpu()).reshape(m, k)
        from sm120fp4.reference import E2M1_GRID, e2m1_encode
        sf_val = sf_v.view(torch.float8_e4m3fn).float()
        mult = torch.where(sf_val > 0, gs.float() / sf_val, torch.zeros_like(sf_val))
        y = (x.float().view(m, k // 16, 16) * mult[..., None]).reshape(m, k)
        ref_codes_own = e2m1_encode(y).reshape(m, k)
        cdiff = (codes_v != ref_codes_own)
        block_ok = (~diff).repeat_interleave(16, dim=1)
        cdiff &= block_ok
        rec["codes_differ_under_library_scale_outside_tie_blocks"] = int(cdiff.sum())
        if cdiff.any():
            mids = (E2M1_GRID[1:] + E2M1_GRID[:-1]) / 2
            ya = y.abs()[cdiff]
            dist = (ya[:, None] - mids[None, :]).abs().min(dim=1).values
            ulp = torch.finfo(torch.float32).eps * ya.clamp(min=1e-30)
            rec["code_mismatch_distance_to_midpoint_in_fp32_ulps"] = {"max": float((dist / ulp).max()), "median": float((dist / ulp).median()), "n": int(cdiff.sum())}
            ex = []
            for r, c in cdiff.nonzero().tolist()[:6]:
                ex.append({"row": r, "col": c, "x": float(x.float()[r, c]), "scale_mult": float(mult[r, c // 16]), "y": float(y[r, c]), "ref_code": int(ref_codes_own[r, c]), "vllm_code": int(codes_v[r, c])})
            rec["code_examples"] = ex
            if fi is not None and "flashinfer_error" not in rec:
                codes_f = __import__("sm120fp4.reference", fromlist=["unpack_e2m1"]).unpack_e2m1(q_f.cpu()).reshape(m, k)
                rec["flashinfer_codes_differ_from_vllm_at_those"] = int((codes_f[cdiff] != codes_v[cdiff]).sum())
        report["operands"].append(rec)
        print(json.dumps(rec)[:900])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
