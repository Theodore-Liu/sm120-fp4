#!/usr/bin/env python3
"""FlashInfer's b12x fused MoE in W4A16 mode (bf16 activations, NVFP4 expert weights, gated SiLU) against the interval
oracle (`sm120fp4/moe_oracle.py`), on the RTX 5090.

Weights come from the repository's reference quantiser (per-expert global scale, E4M3 block scales); the block scales are
handed to the kernel in the 128x4 swizzled layout the W4A16 preparation documents ("the normal NVFP4 K/16 scale grid in
swizzled storage"), and the alphas are the multiplicative global scales (1 / gs in the quantiser's convention: the kernel
multiplies its accumulator by alpha and the routing weight). A sanity line per shape prints the relative error against a
float reference first, so a wrong operand contract shows up as an error of order one, not as a verdict.

Each token count runs the kernel several times: the oracle verdict is per run, and the number of distinct outputs across
runs is recorded (the W4A4 path was non-deterministic at small M). The operand contract found by sweeping the alternatives
on 2026-10-11 (scratch, not kept): 128x4 scales, alpha = 1 / gs, and the SECOND half of w13 is the gate (relative error
against the float reference 0.0047 with that order, 0.92 with the gate first).

    PYTHONPATH=/path/to/sm120-fp4 python scripts/flashinfer_b12x_w4a16_moe_probe.py --out reports/...json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from sm120fp4 import moe_oracle
from sm120fp4.layouts import to_128x4
from sm120fp4.reference import dequantize_nvfp4, quantize_nvfp4


def float_reference(x, w13_p, w13_s, w13_g, w2_p, w2_s, w2_g, ids, wts, inter):
    e_count = w13_p.shape[0]
    out = torch.zeros(x.shape[0], w2_p.shape[1], dtype=torch.float64)
    xf = x.cpu().double()
    for e in range(e_count):
        tok, slot = (ids.cpu() == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        w13 = dequantize_nvfp4(w13_p[e], w13_s[e], w13_g[e]).double()
        w2 = dequantize_nvfp4(w2_p[e], w2_s[e], w2_g[e]).double()
        fc1 = xf[tok] @ w13.t()
        h = F.silu(fc1[:, inter:]) * fc1[:, :inter]          # the SM120 W4A16 path reads the second half of w13 as the gate
        out.index_add_(0, tok, wts.cpu().double()[tok, slot][:, None] * (h @ w2.t()))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports") / f"flashinfer-b12x-w4a16-moe-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--inter", type=int, default=512)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--tokens", type=str, default="1,4,16,64")
    ap.add_argument("--runs", type=int, default=5)
    a = ap.parse_args(argv)
    import flashinfer
    from flashinfer import fused_moe as fm
    dev = torch.device("cuda")
    e, h, i, k = a.experts, a.hidden, a.inter, a.topk
    gen = torch.Generator().manual_seed(20261011)
    w13 = [quantize_nvfp4(torch.randn(2 * i, h, generator=gen) * 0.05) for _ in range(e)]
    w2 = [quantize_nvfp4(torch.randn(h, i, generator=gen) * 0.05) for _ in range(e)]
    w13_p, w13_s, w13_g = torch.stack([q[0] for q in w13]), torch.stack([q[1] for q in w13]), torch.cat([q[2] for q in w13])
    w2_p, w2_s, w2_g = torch.stack([q[0] for q in w2]), torch.stack([q[1] for q in w2]), torch.cat([q[2] for q in w2])
    w13_s_sw = torch.stack([to_128x4(w13_s[j]) for j in range(e)]).view(torch.float8_e4m3fn).to(dev)
    w2_s_sw = torch.stack([to_128x4(w2_s[j]) for j in range(e)]).view(torch.float8_e4m3fn).to(dev)
    alpha13 = (1.0 / w13_g.double()).float().to(dev)
    alpha2 = (1.0 / w2_g.double()).float().to(dev)
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "flashinfer": flashinfer.__version__,
              "shape": {"experts": e, "hidden": h, "inter": i, "topk": k}, "runs": a.runs, "cases": []}
    for m in [int(t) for t in a.tokens.split(",")]:
        g = torch.Generator().manual_seed(1000 + m)
        x = torch.randn(m, h, generator=g).to(torch.bfloat16)
        wts, ids = torch.topk(F.softmax(torch.randn(m, e, generator=g), dim=-1), k, dim=-1)
        wts = (wts / wts.sum(-1, keepdim=True)).float()
        ids = ids.to(torch.int32)
        rec = {"tokens": m}
        try:
            o = moe_oracle.fused_moe_w4a16_oracle(x, w13_p, w13_s, w13_g, w2_p, w2_s, w2_g, ids, wts, gate_first=False)
            ref = float_reference(x, w13_p, w13_s, w13_g, w2_p, w2_s, w2_g, ids, wts, i)
            outs = []
            for _ in range(a.runs):
                y = fm.b12x_fused_moe(x.to(dev), w13_p.to(dev), w13_s_sw, w2_p.to(dev), w2_s_sw, ids.to(dev), wts.to(dev), e, k,
                                      w1_alpha=alpha13, w2_alpha=alpha2, activation_precision="bf16")
                torch.cuda.synchronize()
                outs.append(y.double().cpu())
            rec["rel_err_vs_float_reference"] = [float((y - ref).norm() / ref.norm()) for y in outs]
            rec["distinct_outputs"] = len({y.to(torch.bfloat16).view(torch.int16).numpy().tobytes() for y in outs})
            b = o["bound"]
            err = [(y - o["nominal"]).abs() for y in outs]
            rec["outside_per_run"] = [int((er > b).sum()) for er in err]
            rec["elements"] = int(b.numel())
            rec["max_err_over_bound_per_run"] = [float((er / b).max()) for er in err]
            rec["conformant_every_run"] = bool(all(not (er > b).any() for er in err))
            rec["tie_elements"] = o["tie_elements"]
            rec["mean_bound_over_abs_contrib"] = float((b / o["abs_contrib"].clamp(min=1e-30)).mean())
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        report["cases"].append(rec)
        print(json.dumps(rec)[:700])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
