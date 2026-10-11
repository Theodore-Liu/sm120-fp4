#!/usr/bin/env python3
"""FlashInfer's b12x fused MoE in W4A4 mode (NVFP4 activations quantised in the kernel, NVFP4 weights, gated SiLU) against
the W4A4 fused-MoE oracle (`sm120fp4/moe_oracle_w4a4.py`) on the RTX 5090.

Call shape as validated by `scripts/diag_b12x_nondeterminism.py`: weights quantised with global scale 1 (so every alpha is 1
and `fc2_input_scale` is 1), block scales in the 128x4 layout converted with `convert_sf_to_mma_layout`, `quant_mode="nvfp4"`.
The backend the dispatcher picks depends on the token count (micro for at most 8 tokens and 40 routed rows with top-k > 1,
static beyond, dynamic past the static cutover); the oracle's combine model follows: "micro" (bf16 atomic scatter) for the
micro range, "static" (bf16x2 weight multiply, fp32 finalize) otherwise. Each token count runs several times; conformance is
per run and the distinct-output count across runs is recorded beside it.

    PYTHONPATH=/path/to/sm120-fp4 python scripts/flashinfer_b12x_w4a4_moe_probe.py --out reports/...json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from sm120fp4 import moe_oracle_w4a4 as W
from sm120fp4.layouts import to_128x4
from sm120fp4.reference import quantize_nvfp4

MICRO_MAX_TOKENS, MICRO_ROWS_TOPK, MICRO_ROWS_MULTI = 8, 20, 40   # moe_dispatch.py constants, 0.7.1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports") / f"flashinfer-b12x-w4a4-moe-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--inter", type=int, default=512)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--tokens", type=str, default="1,4,16,64")
    ap.add_argument("--runs", type=int, default=5)
    a = ap.parse_args(argv)
    import flashinfer
    from flashinfer import fused_moe as fm
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
    dev = torch.device("cuda")
    e, h, i, k = a.experts, a.hidden, a.inter, a.topk
    one = torch.ones(1)
    gen = torch.Generator().manual_seed(20261011)
    w13 = [quantize_nvfp4(torch.randn(2 * i, h, generator=gen) * 0.1, one) for _ in range(e)]
    w2 = [quantize_nvfp4(torch.randn(h, i, generator=gen) * 0.1, one) for _ in range(e)]
    w13_p, w13_s = torch.stack([q[0] for q in w13]), torch.stack([q[1] for q in w13])
    w2_p, w2_s = torch.stack([q[0] for q in w2]), torch.stack([q[1] for q in w2])
    w13_g, w2_g = torch.ones(e), torch.ones(e)
    s13 = convert_sf_to_mma_layout(to_128x4(w13_s.reshape(e * 2 * i, h // 16)).to(dev), m=2 * i, k=h, num_groups=e, sf_vec_size=16)
    s2 = convert_sf_to_mma_layout(to_128x4(w2_s.reshape(e * h, i // 16)).to(dev), m=h, k=i, num_groups=e, sf_vec_size=16)
    ones_e = torch.ones(e, device=dev)
    one_d = torch.ones(1, device=dev)
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "flashinfer": flashinfer.__version__,
              "shape": {"experts": e, "hidden": h, "inter": i, "topk": k}, "runs": a.runs, "cases": []}
    for m in [int(t) for t in a.tokens.split(",")]:
        g = torch.Generator().manual_seed(1000 + m)
        x = torch.randn(m, h, generator=g).to(torch.bfloat16)
        wts, ids = torch.topk(F.softmax(torch.randn(m, e, generator=g), dim=-1), k, dim=-1)
        wts = (wts / wts.sum(-1, keepdim=True)).float()
        ids = ids.to(torch.int32)
        routed = m * k
        micro = m <= MICRO_MAX_TOKENS and routed <= (MICRO_ROWS_MULTI if k > 1 else MICRO_ROWS_TOPK)
        combine = "micro" if micro else "static"
        rec = {"tokens": m, "routed_rows": routed, "combine_model": combine}
        try:
            o = W.fused_moe_w4a4_oracle(x, w13_p, w13_s, w13_g, w2_p, w2_s, w2_g, ids, wts, gs_in=1.0, gs_mid=1.0, combine=combine)
            outs = []
            for _ in range(a.runs):
                y = fm.b12x_fused_moe(x.to(dev), w13_p.to(dev), s13, w2_p.to(dev), s2, ids.to(dev), wts.to(dev), e, k,
                                      w1_alpha=ones_e, w2_alpha=ones_e, fc2_input_scale=one_d, quant_mode="nvfp4")
                torch.cuda.synchronize()
                outs.append(y.double().cpu())
            b = o["bound"]
            err = [(y - o["nominal"]).abs() for y in outs]
            rec["distinct_outputs"] = len({y.to(torch.bfloat16).view(torch.int16).numpy().tobytes() for y in outs})
            rec["outside_per_run"] = [int((er > b).sum()) for er in err]
            rec["elements"] = int(b.numel())
            rec["max_err_over_bound_per_run"] = [float((er / b).max()) for er in err]
            rec["conformant_every_run"] = bool(all(not (er > b).any() for er in err))
            rec["ambiguous_blocks_in"] = o["ambiguous_blocks_in"]
            rec["ambiguous_blocks_mid"] = o["ambiguous_blocks_mid"]
            rec["tie_elements"] = o["tie_elements"]
            rec["mean_bound_over_abs_contrib"] = float((b / o["abs_contrib"].clamp(min=1e-30)).mean())
            rec["rel_err_vs_nominal"] = [float((y - o["nominal"]).norm() / o["nominal"].norm()) for y in outs]
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
