#!/usr/bin/env python3
"""Does `triton_scaled_mm` (vLLM 0.28's fallback for `cutlass_scaled_mm` when b's dimensions are not multiples of 16) match the
two-rounding model its source states ("Add bias, it's already in output format, so add it after conversion":
c = out_dtype(scale_a * scale_b * acc), then c += bias in out_dtype)? Same operands as `vllm_edge_probe.py`'s Triton-path
per-row-with-bias cases; for each output element, compare the kernel with
  one-rounding:  bf16(p + bias)            (what the CUTLASS path and the oracle assume)
  two-rounding:  bf16(bf16(p) + bias)      (the source's statement)
where p is the exact scaled product. An element "matches" a model when the kernel equals it or is one bf16 ulp from it (the fp32
accumulation can move p across a rounding boundary).

    PYTHONPATH=<sm120-fp4> python scripts/fit_vllm_triton_bias.py --out reports/<file>.json
"""
import argparse
import json
import time
from pathlib import Path

import torch

M_LIST = (1, 3, 7, 65, 129, 300)
NK = ((8, 16), (24, 144), (1000, 1040), (16, 24), (136, 40))


def bf16(t):
    return t.to(torch.float32).to(torch.bfloat16).double()


def ulp(t):
    a = t.abs().to(torch.float32).clamp(min=2.0 ** -126)
    return (2.0 ** (torch.floor(torch.log2(a)) - 7)).double()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import vllm
    import vllm._custom_ops as ops
    dev = torch.device("cuda")
    e4m3 = torch.float8_e4m3fn
    rep = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
           "torch": torch.__version__, "vllm": vllm.__version__, "cases": []}
    tot = {"elements": 0, "exact_one": 0, "exact_two": 0, "near_one": 0, "near_two": 0}
    for n, k in NK:
        for m in M_LIST:
            g = torch.Generator().manual_seed(20261014 + 13 * m + 5 * n + k)       # the edge probe's stream for these operands
            a_q = (torch.randn(m, k, generator=g) * 2).to(e4m3)
            w_q = torch.randn(n, k, generator=g).to(e4m3)
            _ = (torch.rand(1, 1, generator=g), torch.rand(1, 1, generator=g))
            sr_a = (torch.rand(m, 1, generator=g) * 0.02 + 0.005).float()
            sr_b = (torch.rand(n, 1, generator=g) * 0.02 + 0.005).float()
            bias = (torch.randn(n, generator=g) * 0.1).to(torch.bfloat16)
            p = (a_q.double() @ w_q.double().T) * sr_a.double() * sr_b.double().reshape(1, -1)
            y = ops.cutlass_scaled_mm(a_q.to(dev), w_q.to(dev).t(), sr_a.to(dev), sr_b.to(dev), torch.bfloat16, bias=bias.to(dev))
            torch.cuda.synchronize()
            y = y.double().cpu()
            one = bf16(p + bias.double())
            two = bf16(bf16(p) + bias.double())
            u = ulp(torch.maximum(y.abs(), two.abs()))
            rec = {"m": m, "n": n, "k": k, "elements": y.numel(),
                   "exact_one_rounding": int((y == one).sum()), "exact_two_rounding": int((y == two).sum()),
                   "within_1ulp_one": int(((y - one).abs() <= u).sum()), "within_1ulp_two": int(((y - two).abs() <= u).sum())}
            rep["cases"].append(rec)
            tot["elements"] += rec["elements"]; tot["exact_one"] += rec["exact_one_rounding"]; tot["exact_two"] += rec["exact_two_rounding"]
            tot["near_one"] += rec["within_1ulp_one"]; tot["near_two"] += rec["within_1ulp_two"]
    rep["totals"] = tot
    print(json.dumps(tot))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
