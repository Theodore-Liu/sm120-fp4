#!/usr/bin/env python3
"""FlashInfer's `cutlass_fused_moe` in NVFP4 mode (NVFP4 activations quantised in the kernel, NVFP4 weights, SwiGLU) against
the W4A4 fused-MoE oracle (`sm120fp4/moe_oracle_w4a4.py`) on SM120. This is the backend vLLM uses for NVFP4 MoE on SM120
(`FLASHINFER_CUTLASS`); FlashInfer's own test (`tests/moe/test_trtllm_cutlass_fused_moe.py::test_moe_nvfp4`) judges it with
`assert_close(rtol=0.2, atol=0.2)` against a bf16 dequantised reference.

Operand contract as in that test: w1 [E, 2I, H/2] packed E2M1 with the gate in the SECOND half (the test's reference takes
`weight[m//2:]` as the silu input), block scales 128x4-swizzled per expert and passed as an int32 view, quant_scales =
[a1_gs, w1_sf, 1/(a1_gs*w1_gs), a2_gs, w2_sf, 1/(a2_gs*w2_gs)], unquantised bf16 input (the kernel quantises it with a1_gs).

First read: the kernel's rounding points between FC1, the activation, the mid quantiser, FC2 and the finalize are not yet
fitted, so each output is judged under two combine models of the oracle ("micro": FC2 in one fp32 accumulation; "static":
bf16 partials per 256) and the ratio |y - nominal| / bound is recorded for both. A defect of the kinds the census found
(a whole segment wrong, a dropped column group, the wrong half of w1) sits orders of magnitude above either bound; a
rounding-model difference sits near it. Routing is chosen so that most expert segments start off a 128-row boundary
(the grouped block-scale defect's trigger), and with top-1 each output row belongs to one expert, so errors can be
attributed to segment position.

    PYTHONPATH=/path/to/sm120-fp4 python scripts/flashinfer_cutlass_fused_moe_nvfp4_probe.py --out reports/...json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sm120fp4 import moe_oracle_w4a4 as W
from sm120fp4.layouts import to_128x4
from sm120fp4.reference import quantize_nvfp4


def routing(m: int, e: int, k: int, mode: str, gen: torch.Generator):
    if mode == "random":
        logits = torch.randn(m, e, generator=gen)
        wts, ids = torch.topk(torch.softmax(logits, -1), k, dim=-1)
    else:  # "skewed": expert j gets a share proportional to (j + 1), so segment lengths differ and start mid-tile
        p = torch.arange(1, e + 1, dtype=torch.float64)
        ids = torch.stack([torch.multinomial(p, k, replacement=False, generator=gen) for _ in range(m)])
        wts = torch.rand(m, k, generator=gen) + 0.1
    wts = (wts / wts.sum(-1, keepdim=True)).float()
    return wts, ids.to(torch.int32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--inter", type=int, default=512)
    ap.add_argument("--cases", type=str, default="1:1:random,4:2:random,16:2:random,64:1:skewed,200:1:skewed,200:2:random,700:1:skewed,1000:2:random")
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args(argv)
    import flashinfer
    from flashinfer import fused_moe as fm
    dev = torch.device("cuda")
    e, h, i = a.experts, a.hidden, a.inter
    one = torch.ones(1)
    gen = torch.Generator().manual_seed(20261011)
    w13 = [quantize_nvfp4(torch.randn(2 * i, h, generator=gen) * 0.1, one) for _ in range(e)]
    w2 = [quantize_nvfp4(torch.randn(h, i, generator=gen) * 0.1, one) for _ in range(e)]
    w13_p, w13_s = torch.stack([q[0] for q in w13]), torch.stack([q[1] for q in w13])
    w2_p, w2_s = torch.stack([q[0] for q in w2]), torch.stack([q[1] for q in w2])
    w13_g, w2_g = torch.ones(e), torch.ones(e)
    sw13 = torch.stack([to_128x4(w13_s[j]) for j in range(e)]).view(torch.float8_e4m3fn).to(dev)
    sw2 = torch.stack([to_128x4(w2_s[j]) for j in range(e)]).view(torch.float8_e4m3fn).to(dev)
    a1 = torch.tensor(1.0, device=dev)
    a2 = torch.tensor(1.0, device=dev)
    ones_e = torch.ones(e, device=dev)
    quant_scales = [a1, sw13.view(torch.int32), 1.0 / (a1 * ones_e), a2, sw2.view(torch.int32), 1.0 / (a2 * ones_e)]
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "flashinfer": getattr(flashinfer, "__version__", "editable"),
              "shape": {"experts": e, "hidden": h, "inter": i}, "runs": a.runs, "cases": []}
    for spec in a.cases.split(","):
        m, k, mode = spec.split(":")
        m, k = int(m), int(k)
        g = torch.Generator().manual_seed(1000 + m * 10 + k)
        x = torch.randn(m, h, generator=g).to(torch.bfloat16)
        wts, ids = routing(m, e, k, mode, g)
        counts = torch.bincount(ids.reshape(-1).long(), minlength=e)
        starts = torch.cumsum(counts, 0) - counts
        rec = {"tokens": m, "topk": k, "routing": mode, "segment_lengths": counts.tolist(),
               "segments_starting_off_128": int(((starts % 128) != 0)[counts > 0].sum())}
        try:
            outs = []
            for _ in range(a.runs):
                y = torch.zeros(m, h, dtype=torch.bfloat16, device=dev)
                fm.cutlass_fused_moe(x.to(dev), ids.to(dev), wts.to(dev), w13_p.to(dev).contiguous().view(torch.long),
                                     w2_p.to(dev).contiguous().view(torch.long), torch.bfloat16, quant_scales=quant_scales,
                                     output=y)
                torch.cuda.synchronize()
                outs.append(y.double().cpu())
            rec["distinct_outputs"] = len({y.to(torch.bfloat16).view(torch.int16).numpy().tobytes() for y in outs})
            for combine in ("micro", "static"):
                o = W.fused_moe_w4a4_oracle(x, w13_p, w13_s, w13_g, w2_p, w2_s, w2_g, ids, wts, gs_in=1.0, gs_mid=1.0, combine=combine)
                b, nom = o["bound"], o["nominal"]
                er = (outs[0] - nom).abs()
                r = er / b
                rel = er.norm() / nom.norm()
                row_max = r.max(dim=1).values
                rec[combine] = {"outside": int((er > b).sum()), "elements": int(b.numel()), "max_err_over_bound": float(r.max()),
                                "median_err_over_bound": float(r.median()), "rel_err_normwise": float(rel),
                                "rows_with_err_over_10x_bound": int((row_max > 10).sum())}
                if k == 1:
                    # attribute each row to its expert segment and whether that segment starts off a 128-row boundary
                    off = ((starts % 128) != 0)[ids[:, 0].long()]
                    rec[combine]["max_ratio_rows_in_segments_off_128"] = float(row_max[off].max()) if off.any() else None
                    rec[combine]["max_ratio_rows_in_segments_on_128"] = float(row_max[~off].max()) if (~off).any() else None
        except Exception as ex:  # noqa: BLE001
            rec["error"] = f"{type(ex).__name__}: {str(ex)[:400]}"
        report["cases"].append(rec)
        print(json.dumps(rec), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
