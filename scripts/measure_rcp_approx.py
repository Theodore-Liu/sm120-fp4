#!/usr/bin/env python3
"""Measure `rcp.approx.ftz.f32` on this GPU for the inputs the b12x fast quantiser feeds it: the eight E4M3 mantissas
(1, 1.125, ..., 1.875, every E4M3 value is one of them times a power of two) and the constant 6 (the E2M1 maximum).

The fast quantiser (`quantize_block_fp4_fast`) computes the block scale as gs * (max_abs * rcp(6)) and the codes as
value * (rcp(scale) * gs); `rcp.approx` is within one fp32 ulp of the true reciprocal but which side it lands on decides
every exact tie, and bf16 inputs produce exact ties often (a bf16 maximum divided by 6 lands on an E4M3 midpoint for about
one block in nine on random data). The oracle therefore emulates the fast path with the measured constants instead of
treating the last bit as unknown. The report also checks that the approximation scales exactly with powers of two (so the
eight constants cover every E4M3 value) and records the fp32 round-to-nearest reciprocal beside each for comparison.

    PYTHONPATH=/path/to/sm120-fp4 python scripts/measure_rcp_approx.py --out reports/rcp-approx-<device>.json
"""
from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def _rcp_kernel(X, Y, N: tl.constexpr):
    idx = tl.arange(0, N)
    x = tl.load(X + idx)
    y = tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1)
    tl.store(Y + idx, y)


def rcp_approx(x: torch.Tensor) -> torch.Tensor:
    n = 1
    while n < x.numel():
        n *= 2
    xp = torch.zeros(n, dtype=torch.float32, device="cuda")
    xp[: x.numel()] = x.float().cuda()
    y = torch.empty_like(xp)
    _rcp_kernel[(1,)](xp, y, N=n)
    torch.cuda.synchronize()
    return y[: x.numel()].cpu()


def bits(v: float) -> str:
    return "0x%08x" % struct.unpack("<I", struct.pack("<f", v))[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports") / f"rcp-approx-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    a = ap.parse_args(argv)
    mant = [1.0 + k / 8.0 for k in range(8)]
    probes = mant + [6.0]
    x = torch.tensor(probes, dtype=torch.float32)
    y = rcp_approx(x)
    exact = 1.0 / x.double()
    rn = (1.0 / x).float()                                    # fp32 round-to-nearest reciprocal
    rows = []
    for xv, yv, ev, rv in zip(probes, y.tolist(), exact.tolist(), rn.tolist()):
        rows.append({"x": xv, "rcp_approx": yv, "rcp_approx_bits": bits(yv), "rn_reciprocal": rv, "rn_bits": bits(rv),
                     "approx_minus_exact_ulps": (yv - ev) / (2.0 ** (torch.frexp(torch.tensor(ev))[1].item() - 24)),
                     "approx_equals_rn": yv == rv, "side": "above" if yv > ev else ("below" if yv < ev else "exact")})
    # power-of-two scaling: rcp(m * 2^e) must equal rcp(m) * 2^-e bit for bit
    scaled_ok = True
    for e in (-9, -5, -1, 3, 8):
        xs = torch.tensor([m * (2.0 ** e) for m in mant], dtype=torch.float32)
        ys = rcp_approx(xs)
        expect = y[:8] * (2.0 ** -e)
        scaled_ok &= bool(torch.equal(ys, expect))
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "triton": triton.__version__, "instruction": "rcp.approx.ftz.f32",
              "scales_exactly_with_powers_of_two": scaled_ok, "rows": rows}
    for r in rows:
        print(f"x={r['x']:<6} rcp.approx={r['rcp_approx']:.9f} ({r['rcp_approx_bits']}) rn={r['rn_reciprocal']:.9f} "
              f"equal_rn={r['approx_equals_rn']} side={r['side']} ulps={r['approx_minus_exact_ulps']:+.3f}")
    print("scales exactly with powers of two:", scaled_ok)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
