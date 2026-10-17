#!/usr/bin/env python3
"""Known-value reduction-order probe for the SM120 NVFP4 GEMMs (companion of fp8_reduction_order_probe.py).

Same pattern, built from e2m1 codes and E4M3 block scales (block 16; per-tensor scales 1.0, alpha 1.0):
    P0 = +2^16   a code 4.0, scale 2^6   x  b code 4.0, scale 2^6
    P1 = 3*2^-9  a code 1.5, scale 2^-6  x  b code 1.0, scale 2^-2
    P2 = -2^16   a code 4.0, scale 2^6   x  b code -4.0, scale 2^6
Every other code is 0 (scale 1.0). The three products sit in distinct 16-blocks; K = 4096 in 64-wide tiles (the mma.sync
k-extent for e2m1). Exact sum 3 * 2^-9; the kernel's bf16 output is 3 * 2^-9, 2^-7 or 0 as in the FP8 probe, and the
descriptor is read with scripts/fp8_reduction_order_fit.py on the written JSON.

    --lib flashinfer   mm_fp4 backends cutlass and b12x        (fi071 venv)
    --lib trtllm       nvfp4_gemm_cutlass / cublaslt / default   (trtllm venv)
    --lib vllm         ops.cutlass_scaled_fp4_mm                 (vllm028 venv)
    python scripts/nvfp4_reduction_order_probe.py --lib vllm --out reports/nvfp4-reduction-order-rtx5090/vllm-k4096.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sm120fp4.layouts import to_128x4  # noqa: E402
from sm120fp4.reference import pack_e2m1  # noqa: E402

BLOCK = 16
CODE = {0.0: 0, 1.0: 2, 1.5: 3, 4.0: 6}      # e2m1 grid index; sign bit is 8
EXPECT = {3 * 2.0 ** -9: "exact (P1 added after the cancellation)", 2.0 ** -7: "2^-7 (one of several orders)", 0.0: "truncation, P1 added to +-2^16"}


def e4m3_bits(v: float) -> int:
    t = torch.tensor([v]).to(torch.float8_e4m3fn)
    assert float(t.float()[0]) == v, f"{v} is not an E4M3 value"
    return int(t.view(torch.uint8)[0])


def operands(K: int, p0: int, p1: int, p2: int, M: int, N: int):
    """Returns packed codes [M, K/2], [N, K/2] (uint8) and row-major scale bits [M, K/16], [N, K/16] (uint8)."""
    assert len({p0 // BLOCK, p1 // BLOCK, p2 // BLOCK}) == 3, "the three products must sit in distinct blocks"
    ca = torch.zeros(M, K, dtype=torch.uint8); cb = torch.zeros(N, K, dtype=torch.uint8)
    sa = torch.full((M, K // BLOCK), e4m3_bits(1.0), dtype=torch.uint8); sb = sa[:N].clone() if N <= M else torch.full((N, K // BLOCK), e4m3_bits(1.0), dtype=torch.uint8)
    ca[0, p0] = CODE[4.0]; cb[0, p0] = CODE[4.0]; sa[0, p0 // BLOCK] = e4m3_bits(64.0); sb[0, p0 // BLOCK] = e4m3_bits(64.0)
    ca[0, p1] = CODE[1.5]; cb[0, p1] = CODE[1.0]; sa[0, p1 // BLOCK] = e4m3_bits(2.0 ** -6); sb[0, p1 // BLOCK] = e4m3_bits(0.25)
    ca[0, p2] = CODE[4.0]; cb[0, p2] = CODE[4.0] | 8; sa[0, p2 // BLOCK] = e4m3_bits(64.0); sb[0, p2 // BLOCK] = e4m3_bits(64.0)
    return pack_e2m1(ca), pack_e2m1(cb), sa, sb


def kernels(lib: str):
    dev = torch.device("cuda")
    alpha = torch.ones(1, device=dev)
    if lib == "flashinfer":
        import flashinfer
        ver = flashinfer.__version__
        def mk(be):
            def f(ap, bp, asf, bsf):
                return flashinfer.mm_fp4(ap, bp.t(), asf.view(torch.float8_e4m3fn), bsf.view(torch.float8_e4m3fn), alpha, torch.bfloat16, None, 16, False, be)
            return f
        return ver, {f"mm_fp4 {be}": mk(be) for be in ("cutlass", "b12x")}
    if lib == "trtllm":
        import tensorrt_llm
        ops = torch.ops.trtllm
        fns = {"nvfp4_gemm cutlass": lambda ap, bp, asf, bsf: ops.nvfp4_gemm_cutlass(ap, bp, asf, bsf, alpha, torch.bfloat16, False),
               "nvfp4_gemm cublaslt": lambda ap, bp, asf, bsf: ops.nvfp4_gemm_cublaslt(ap, bp, asf, bsf, alpha, torch.bfloat16, False)}
        return tensorrt_llm.__version__, fns
    if lib == "vllm":
        import vllm
        from vllm import _custom_ops as ops
        return vllm.__version__, {"cutlass_scaled_fp4_mm": lambda ap, bp, asf, bsf: ops.cutlass_scaled_fp4_mm(ap, bp, asf.view(torch.float8_e4m3fn), bsf.view(torch.float8_e4m3fn), alpha, torch.bfloat16)}
    raise SystemExit(lib)


def classify(v: float) -> str:
    for k, name in EXPECT.items():
        if v == k:
            return name
    return f"other ({v!r})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lib", required=True, choices=["flashinfer", "trtllm", "vllm"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--tile", type=int, default=64)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--boundary-sweep", action="store_true",
                    help="P0 and P2 in adjacent tiles (t, t+1) for every t, P1 in tile t+2: a split-K chunk boundary between t and t+1 "
                         "reads 2^-7 (the -2^16 chunk truncates the small term before the chunks combine); the same chunk reads exact. "
                         "The set of t that read 2^-7 is the set of chunk boundaries.")
    a = ap.parse_args()
    dev = torch.device("cuda")
    ver, fns = kernels(a.lib)
    T = a.K // a.tile
    anchors = [(0, T - 1), (0, T // 2), (T // 4, 3 * T // 4), (T // 2 - 1, T // 2), (T - 2, T - 1)]
    cases = [(t0, t1, t2) for (t0, t2) in anchors
             for t1 in sorted({0, t0 - 1, t0, t0 + 1, (t0 + t2) // 2, t2 - 1, t2, t2 + 1, T - 1} & set(range(T)))]
    if a.boundary_sweep:
        cases = [(t, t + 2, t + 1) for t in range(T - 2)]
    rows = []
    for (t0, t1, t2) in cases:
        if True:
            p0, p2 = t0 * a.tile + 3, t2 * a.tile + 5           # blocks 0 of their tiles (t0 < t2, so distinct)
            p1 = t1 * a.tile + (9 if t1 not in (t0, t2) else 17)   # block 0, or block 1 when it shares a tile
            ap_, bp_, asf, bsf = operands(a.K, p0, p1, p2, a.M, a.N)
            asw = to_128x4(asf).contiguous().to(dev); bsw = to_128x4(bsf).contiguous().to(dev)
            for name, fn in fns.items():
                try:
                    v = float(fn(ap_.to(dev), bp_.to(dev), asw, bsw)[0, 0]); torch.cuda.synchronize()
                    rows.append({"kernel": name, "t0": t0, "t1": t1, "t2": t2, "p0": p0, "p1": p1, "p2": p2, "value": v, "read": classify(v)})
                except Exception as e:  # noqa: BLE001
                    rows.append({"kernel": name, "t0": t0, "t1": t1, "t2": t2, "error": repr(e)[:200]})
    # control: the zero pattern must read 0, and a single P0 must read 2^16 (the scales are being applied as intended)
    controls = {}
    zp_a, zp_b, zs_a, zs_b = operands(a.K, 3, 21, 37, a.M, a.N)
    for name, fn in fns.items():
        ca = torch.zeros(a.M, a.K, dtype=torch.uint8); cb = torch.zeros(a.N, a.K, dtype=torch.uint8)
        sa = torch.full((a.M, a.K // BLOCK), e4m3_bits(1.0), dtype=torch.uint8); sb = torch.full((a.N, a.K // BLOCK), e4m3_bits(1.0), dtype=torch.uint8)
        ca[0, 3] = CODE[4.0]; cb[0, 3] = CODE[4.0]; sa[0, 0] = e4m3_bits(64.0); sb[0, 0] = e4m3_bits(64.0)
        try:
            controls[name] = {"single_P0": float(fn(pack_e2m1(ca).to(dev), pack_e2m1(cb).to(dev), to_128x4(sa).contiguous().to(dev), to_128x4(sb).contiguous().to(dev))[0, 0])}
        except Exception as e:  # noqa: BLE001
            controls[name] = {"error": repr(e)[:200]}
    rep = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "lib": a.lib, "version": ver, "K": a.K, "tile": a.tile, "M": a.M, "N": a.N, "pattern": {"P0": 65536.0, "P1": 3 * 2.0 ** -9, "P2": -65536.0},
           "controls": controls, "rows": rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    from collections import Counter
    for name in fns:
        print(name, dict(Counter(r.get("read", "error") for r in rows if r["kernel"] == name)))
    print("controls:", controls)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
