#!/usr/bin/env python3
"""Where does the fp32 accumulator add a small product, relative to two large ones that cancel? (R1, reduction-order column)

A GEMM row's K products are summed in some order the documentation does not state: within one tensor-core instruction
(32 fp8 products for mma.sync m16n8k32), across k-tiles, and possibly across split-K partials. Yang et al. (arXiv
2609.11356) reconstruct that order for cuBLAS with known-value inputs. This probe does the smallest version of it for the
FP8 GEMMs the census read on SM120.

The pattern. Three non-zero products in one output element, all others zero:
    P0 = +2^16 at position p0,  P1 = 3 * 2^-9 (0.75 ulp of 2^16 in fp32) at position p1,  P2 = -2^16 at position p2.
Exact sum: 3 * 2^-9. What a kernel returns tells where P1 was added:
    3 * 2^-9   P1 was added after P0 and P2 had cancelled (or in a partial of its own that was added to an exact zero)
    2^-7       P1 was added to a running sum holding +-2^16 under round-to-nearest (0.75 ulp rounds up to one ulp)
    0          P1 was added to a running sum holding +-2^16 under truncation (align to the larger exponent, drop bits)
    other      a wider internal accumulator, or an order this pattern does not describe
Every outcome is exactly representable in bf16 and fp32, so the output dtype does not round it.

The sweep. K = 4096 in 32-wide tiles (the mma.sync k-extent for fp8). p0 and p2 are fixed in tiles t0 < t2; p1 moves over
tiles t1 (before t0, between, after t2, and inside t0's or t2's own tile). A sequential tile order reads 2^-7 or 0 for
t0 < t1 < t2 and 3 * 2^-9 for t1 > t2; a split-K or tree order breaks that pattern at the split points.

    python scripts/fp8_reduction_order_probe.py --out reports/fp8-reduction-order-rtx5090/probe.json

Kernels: vLLM ops.cutlass_scaled_mm (FP8, per-tensor scales 1.0, bf16 out) and torch._scaled_mm (cuBLASLt, fp32 out).
Reads only; nothing is timed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

FP8 = torch.float8_e4m3fn
BIG = 2.0 ** 8          # a = b = 2^8 -> product 2^16 (e4m3 max is 448, so 2^8 is the largest power of two)
SMALL_A = 2.0 ** -3     # 0.125
SMALL_B = 1.5 * 2.0 ** -5   # 0.046875 -> product 3 * 2^-9
EXPECT = {3 * 2.0 ** -9: "exact (P1 added after the cancellation)", 2.0 ** -7: "RN, P1 added to +-2^16", 0.0: "truncation, P1 added to +-2^16"}


def fp8_exact(x: torch.Tensor) -> torch.Tensor:
    y = x.to(FP8)
    assert torch.equal(y.float(), x.float()), "a probe value is not representable in e4m3"
    return y


def operands(K: int, p0: int, p1: int, p2: int, M: int = 16, N: int = 16):
    a = torch.zeros(M, K); b = torch.zeros(N, K)
    a[0, p0] = BIG; b[0, p0] = BIG
    a[0, p1] = SMALL_A; b[0, p1] = SMALL_B
    a[0, p2] = BIG; b[0, p2] = -BIG
    return fp8_exact(a).cuda(), fp8_exact(b).cuda()


def run_cutlass(a8, b8):
    from vllm import _custom_ops as ops
    sa = torch.ones(a8.shape[0], 1, device="cuda"); sb = torch.ones(b8.shape[0], 1, device="cuda")
    return float(ops.cutlass_scaled_mm(a8, b8.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)[0, 0])


def run_cublaslt(a8, b8):
    s = torch.ones((), device="cuda")
    return float(torch._scaled_mm(a8, b8.t(), scale_a=s, scale_b=s, out_dtype=torch.float32)[0, 0])


def classify(v: float) -> str:
    for k, name in EXPECT.items():
        if v == k:
            return name
    return f"other ({v!r})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--tile", type=int, default=32)
    a = ap.parse_args()
    T = a.K // a.tile
    kernels = {"cutlass_scaled_mm": run_cutlass, "torch._scaled_mm": run_cublaslt}
    rows = []
    # cases: (t0, t1, t2) tile indices; within-tile variants use offsets inside the tile
    anchors = [(0, T - 1), (0, T // 2), (T // 4, 3 * T // 4), (T // 2 - 1, T // 2), (T - 2, T - 1)]
    for (t0, t2) in anchors:
        t1s = sorted({0, t0 - 1, t0, t0 + 1, (t0 + t2) // 2, t2 - 1, t2, t2 + 1, T - 1} & set(range(T)))
        for t1 in t1s:
            p0, p2 = t0 * a.tile + 3, t2 * a.tile + 5
            p1 = t1 * a.tile + (9 if t1 not in (t0, t2) else 17)
            a8, b8 = operands(a.K, p0, p1, p2)
            for name, fn in kernels.items():
                try:
                    v = fn(a8, b8)
                    rows.append({"kernel": name, "t0": t0, "t1": t1, "t2": t2, "p0": p0, "p1": p1, "p2": p2, "value": v, "read": classify(v)})
                except Exception as e:  # noqa: BLE001
                    rows.append({"kernel": name, "t0": t0, "t1": t1, "t2": t2, "error": repr(e)[:200]})
    # the exact sum is 3 * 2^-9; the all-zero control must read 0
    z = torch.zeros(16, a.K)
    za, zb = fp8_exact(z).cuda(), fp8_exact(z).cuda()
    controls = {name: fn(za, zb) for name, fn in kernels.items()}
    rep = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
           "torch": torch.__version__, "K": a.K, "tile": a.tile, "pattern": {"P0": BIG * BIG, "P1": SMALL_A * SMALL_B, "P2": -BIG * BIG},
           "zero_controls": controls, "rows": rows}
    try:
        import vllm
        rep["vllm"] = vllm.__version__
    except Exception:  # noqa: BLE001
        pass
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    from collections import Counter
    for name in kernels:
        c = Counter(r["read"] for r in rows if r["kernel"] == name and "read" in r)
        print(name, dict(c))
    print("zero controls:", controls)
    for r in rows:
        if "read" in r:
            print(f"{r['kernel']:18s} t0={r['t0']:3d} t1={r['t1']:3d} t2={r['t2']:3d}  {r['read']}")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
