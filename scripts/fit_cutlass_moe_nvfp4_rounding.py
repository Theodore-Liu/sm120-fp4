#!/usr/bin/env python3
"""Fit the rounding points of FlashInfer's `cutlass_fused_moe` NVFP4 path (TensorRT-LLM's CUTLASS MoE runner) between FC1 and
FC2, as was done for the b12x paths: run the kernel once, then emulate the layer under several candidate models of the middle
stage and count, per model, how many output elements land within a small fp32-accumulation allowance of the kernel's.

The first read (`reports/cutlass-moe-nvfp4-pro6000/`) put 0 to 0.3 percent of elements outside the oracle's bound, which models
the activation as written to bf16 before the mid quantiser. A model that misplaces one rounding point flips a block's E4M3 scale
or an E2M1 code now and then, and each flip moves a whole output row by about one code step: the signature seen. The
candidates (top-1, routing weight 1, so the finalize is one multiply by 1 and one bf16 rounding):

  oracle   gate, up in fp32; act = silu(gate) * up; act rounded to bf16; mid quantiser on the bf16 values
  g16_a32  gate, up rounded to bf16 (FC1 written as bf16); act from those; mid quantiser on the fp32 act
  g16_a16  gate, up rounded to bf16; act rounded to bf16; mid quantiser on the bf16 act
  f32      no intermediate rounding: mid quantiser on the fp32 act from fp32 gate, up

Each emulation: input quantiser exact (the fast quantiser on exact bf16 x, gs 1), FC1 and FC2 as exact float64 products of the
dequantised operands, silu in float64 then rounded to fp32, the mid quantiser by `quantize_block_fast_exact`, output rounded
to bf16. "Match" means |kernel - emulation| <= 2 bf16 ulp of the output (the fp32 accumulation order of FC2 is not modelled);
a code or scale flip moves an element by far more than that.

    PYTHONPATH=/path/to/sm120-fp4 python scripts/fit_cutlass_moe_nvfp4_rounding.py --out reports/...json
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


def bf16(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.float32).to(torch.bfloat16).double()


def f32(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.float32).double()


def ulp_bf16(t: torch.Tensor) -> torch.Tensor:
    a = t.abs().to(torch.float32).clamp(min=2.0 ** -126)
    return (2.0 ** (torch.floor(torch.log2(a)) - 7)).double()


def emulate(model: str, x, w13d, w2d, ids, inter):
    xq, _ = W.quantize_block_fast_exact(x.float(), 1.0)
    out = torch.zeros(x.shape[0], w2d.shape[1], dtype=torch.float64)
    for e in range(w13d.shape[0]):
        tok = (ids[:, 0] == e).nonzero(as_tuple=True)[0]
        if tok.numel() == 0:
            continue
        f = xq[tok] @ w13d[e].T                          # exact FC1
        up, gate = f[:, :inter], f[:, inter:]           # the gate is the second half (FlashInfer's test)
        if model in ("g16_a32", "g16_a16"):
            up, gate = bf16(up), bf16(gate)
        else:
            up, gate = f32(up), f32(gate)
        act = f32(gate * torch.sigmoid(gate) * up)
        if model in ("oracle", "g16_a16"):
            act = bf16(act)
        q, _ = W.quantize_block_fast_exact(act.float(), 1.0)
        out[tok] = q @ w2d[e].T                          # exact FC2, alpha 1, weight 1
    return bf16(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--inter", type=int, default=512)
    ap.add_argument("--tokens", type=int, default=1024)
    a = ap.parse_args(argv)
    import flashinfer
    from flashinfer import fused_moe as fm
    dev = torch.device("cuda")
    e, h, i, m = a.experts, a.hidden, a.inter, a.tokens
    one = torch.ones(1)
    gen = torch.Generator().manual_seed(20261012)
    w13 = [quantize_nvfp4(torch.randn(2 * i, h, generator=gen) * 0.1, one) for _ in range(e)]
    w2 = [quantize_nvfp4(torch.randn(h, i, generator=gen) * 0.1, one) for _ in range(e)]
    w13_p, w13_s = torch.stack([q[0] for q in w13]), torch.stack([q[1] for q in w13])
    w2_p, w2_s = torch.stack([q[0] for q in w2]), torch.stack([q[1] for q in w2])
    w13d = torch.stack([W._dequant_w(w13_p[j], w13_s[j], 1.0, 16) for j in range(e)])
    w2d = torch.stack([W._dequant_w(w2_p[j], w2_s[j], 1.0, 16) for j in range(e)])
    sw13 = torch.stack([to_128x4(w13_s[j]) for j in range(e)]).view(torch.float8_e4m3fn).to(dev)
    sw2 = torch.stack([to_128x4(w2_s[j]) for j in range(e)]).view(torch.float8_e4m3fn).to(dev)
    a1 = torch.tensor(1.0, device=dev)
    ones_e = torch.ones(e, device=dev)
    qs = [a1, sw13.view(torch.int32), 1.0 / (a1 * ones_e), a1, sw2.view(torch.int32), 1.0 / (a1 * ones_e)]
    x = torch.randn(m, h, generator=gen).to(torch.bfloat16)
    ids = torch.randint(0, e, (m, 1), generator=gen).to(torch.int32)
    wts = torch.ones(m, 1)
    y = torch.zeros(m, h, dtype=torch.bfloat16, device=dev)
    fm.cutlass_fused_moe(x.to(dev), ids.to(dev), wts.to(dev), w13_p.to(dev).contiguous().view(torch.long),
                         w2_p.to(dev).contiguous().view(torch.long), torch.bfloat16, quant_scales=qs, output=y)
    torch.cuda.synchronize()
    y = y.double().cpu()
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "flashinfer": getattr(flashinfer, "__version__", "editable"),
              "shape": {"experts": e, "hidden": h, "inter": i, "tokens": m, "topk": 1}, "models": {}}
    for model in ("oracle", "g16_a32", "g16_a16", "f32"):
        em = emulate(model, x, w13d, w2d, ids.long(), i)
        d = (y - em).abs()
        u = ulp_bf16(torch.maximum(y.abs(), em.abs()))
        match = d <= 2 * u
        rows_all = match.all(dim=1)
        rec = {"elements_matching": int(match.sum()), "elements": int(match.numel()),
               "fraction_matching": round(float(match.double().mean()), 6),
               "rows_fully_matching": int(rows_all.sum()), "rows": m,
               "exact_bitwise": int((d == 0).sum()),
               "max_abs_diff": float(d.max()), "rel_err_normwise": float(d.norm() / em.norm())}
        report["models"][model] = rec
        print(model, json.dumps(rec), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
