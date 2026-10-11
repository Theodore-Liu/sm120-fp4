#!/usr/bin/env python3
"""Where does vLLM's `scaled_fp4_quant` disagree with the format's definition? (follow-up to vllm_nvfp4_probe.py)

The probe found one operand (M=64, K=1024) whose block scales from vLLM differ from the reference quantiser's under the same
global scale, with 119 codes differing off any midpoint tie; every other shape agreed to the byte. This script asks the
narrower questions, with the swizzle taken out of the comparison by asking vLLM for the row-major layout directly
(`is_sf_swizzled_layout=False`): how many block scales differ, by how many E4M3 steps and in which direction, whether the set
depends on the shape (M, K) or on the values, whether it is deterministic across calls, and whether the row-major and the
swizzled calls agree with each other.

    PYTHONPATH=/path/to/sm120-fp4 python scripts/diag_vllm_fp4_quant_scales.py [--out reports/...json]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sm120fp4.layouts import from_128x4
from sm120fp4.reference import E2M1_MAX, E4M3_MAX, global_scale_for, quantize_nvfp4

SHAPES = ((64, 1024), (16, 1024), (128, 1024), (64, 2048), (64, 512), (64, 4096), (3, 1024), (1, 1024))


def e4m3_steps(a_bits: torch.Tensor, b_bits: torch.Tensor) -> torch.Tensor:
    """Signed distance in E4M3 code steps (the byte difference is the step count for same-sign finite values)."""
    return a_bits.to(torch.int16) - b_bits.to(torch.int16)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports") / f"vllm-fp4-quant-scales-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args(argv)
    import vllm
    import vllm._custom_ops as ops
    dev = torch.device("cuda")
    report = {"vllm": vllm.__version__, "torch": torch.__version__, "device": torch.cuda.get_device_name(0), "cases": []}
    gen = torch.Generator().manual_seed(20261010)
    for m, k in SHAPES:
        x = (torch.randn(m, k, generator=gen) * 2.0).to(torch.bfloat16)
        gs = global_scale_for(x.float())
        ref_q, ref_sf, _ = quantize_nvfp4(x.float(), gs)
        rec = {"m": m, "k": k, "blocks": m * k // 16}
        # row-major from vLLM, no swizzle in the comparison
        q_rm, sf_rm = ops.scaled_fp4_quant(x.to(dev), gs.to(dev), is_sf_swizzled_layout=False)
        torch.cuda.synchronize()
        sf_rm = sf_rm.view(torch.uint8).cpu().reshape(m, k // 16)
        d = e4m3_steps(sf_rm, ref_sf)
        diff = d != 0
        rec["rowmajor_scales_differ"] = int(diff.sum())
        rec["rowmajor_steps_hist"] = {int(v): int((d == v).sum()) for v in torch.unique(d).tolist()}
        rec["rowmajor_codes_differ"] = int((q_rm.cpu() != ref_q).sum())
        # the swizzled call, un-swizzled by our converter: does it match the row-major call?
        q_sw, sf_sw = ops.scaled_fp4_quant(x.to(dev), gs.to(dev), is_sf_swizzled_layout=True)
        torch.cuda.synchronize()
        sf_un = from_128x4(sf_sw.reshape(-1).view(torch.uint8).cpu(), m, k // 16).reshape(m, k // 16)
        rec["swizzled_vs_rowmajor_scales_differ"] = int((sf_un != sf_rm).sum())
        rec["swizzled_vs_rowmajor_codes_differ"] = int((q_sw.cpu() != q_rm.cpu()).sum())
        # deterministic across calls?
        reps = []
        for _ in range(a.repeats):
            q2, sf2 = ops.scaled_fp4_quant(x.to(dev), gs.to(dev), is_sf_swizzled_layout=False)
            torch.cuda.synchronize()
            reps.append(bool(torch.equal(q2.cpu(), q_rm.cpu()) and torch.equal(sf2.view(torch.uint8).cpu().reshape(m, k // 16), sf_rm)))
        rec["deterministic"] = all(reps)
        # where the differing blocks are, and what the reference computed for them
        if diff.any():
            idx = diff.nonzero()[:8].tolist()
            rows = []
            for r, c in idx:
                blk = x.float()[r, c * 16:(c + 1) * 16]
                amax = float(blk.abs().max())
                sf_f = amax / E2M1_MAX * float(gs)
                rows.append({"row": r, "block": c, "block_amax": amax, "sf_fp32": sf_f, "ref_e4m3": int(ref_sf[r, c]), "vllm_e4m3": int(sf_rm[r, c]),
                             "ref_value": float(ref_sf[r, c:c + 1].view(torch.float8_e4m3fn).float()), "vllm_value": float(sf_rm[r, c:c + 1].view(torch.float8_e4m3fn).float())})
            rec["examples"] = rows
            rec["differing_rows"] = sorted(set(int(r) for r, _ in diff.nonzero().tolist()))[:16]
            rec["differing_cols"] = sorted(set(int(c) for _, c in diff.nonzero().tolist()))[:16]
        report["cases"].append(rec)
        print(json.dumps(rec)[:700])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
