#!/usr/bin/env python3
"""Fit a reduction-order descriptor to the known-value probe's readings (reports/fp8-reduction-order-rtx5090/*.json).

Candidate descriptors, in the spirit of arXiv 2609.11356: K is cut into T tiles; tile t goes to partial (t mod S)
("interleaved") or to partial (t // (T/S)) ("contiguous"); inside a partial the tiles are added in index order into an
fp32 accumulator with tensor-core semantics (the tile's products are summed exactly, then the sum enters the accumulator
by alignment to the larger exponent and truncation toward zero); the S partials are then combined by fp32 adds that are
either round-to-nearest-even (CUDA cores) or truncating (tensor cores), in index order or as a balanced tree. S = 1 is
"sequential, no split". The descriptor that predicts every reading is reported; ties are listed.

    python scripts/fp8_reduction_order_fit.py reports/fp8-reduction-order-rtx5090/probe-k4096-m1-n4096.json
"""
from __future__ import annotations

import json
import math
import sys
from fractions import Fraction
from pathlib import Path


def fp32_round(x: Fraction, mode: str) -> Fraction:
    """Round an exact value to fp32 (24-bit significand; no subnormal handling needed for these magnitudes)."""
    if x == 0:
        return x
    s = -1 if x < 0 else 1
    m = abs(x)
    e = math.floor(math.log2(m))
    # guard against log2 rounding at powers of two
    while Fraction(2) ** e > m:
        e -= 1
    while Fraction(2) ** (e + 1) <= m:
        e += 1
    ulp = Fraction(2) ** (e - 23)
    q = m / ulp
    if mode == "trunc":
        n = math.floor(q)
    else:  # round to nearest, ties to even
        fl = math.floor(q)
        r = q - fl
        n = fl + 1 if r > Fraction(1, 2) or (r == Fraction(1, 2) and fl % 2 == 1) else fl
    return s * n * ulp


def predict(case: dict, T: int, S: int, assign: str, combine: str, order: str) -> Fraction:
    P = {case["t0"]: Fraction(2) ** 16, case["t2"]: -(Fraction(2) ** 16)}
    small = Fraction(3, 2 ** 9)
    # products in the same tile are summed exactly
    tile_sum = {}
    for t, v in P.items():
        tile_sum[t] = tile_sum.get(t, Fraction(0)) + v
    tile_sum[case["t1"]] = tile_sum.get(case["t1"], Fraction(0)) + small
    parts = [Fraction(0)] * S
    for t in sorted(tile_sum):
        j = t % S if assign == "interleaved" else min(t * S // T, S - 1)
        parts[j] = fp32_round(parts[j] + tile_sum[t], "trunc")
    if order == "sequential":
        acc = Fraction(0)
        for p in parts:
            acc = fp32_round(acc + p, combine)
        return acc
    # balanced tree
    level = parts[:]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(fp32_round(level[i] + level[i + 1], combine))
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return level[0]


def main() -> int:
    path = Path(sys.argv[1])
    d = json.loads(path.read_text(encoding="utf-8"))
    T = d["K"] // d["tile"]
    for kernel in sorted({r["kernel"] for r in d["rows"]}):
        cases = [r for r in d["rows"] if r["kernel"] == kernel and "value" in r]
        results = []
        for S in [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 32, 64, 128]:
            if S > T:
                continue
            for assign in ("interleaved", "contiguous"):
                for combine in ("rn", "trunc"):
                    for order in ("sequential", "tree"):
                        if S == 1 and (assign != "interleaved" or combine != "rn" or order != "sequential"):
                            continue
                        hits = sum(1 for c in cases if Fraction(c["value"]) == predict(c, T, S, assign, combine, order))
                        results.append((hits, S, assign, combine, order))
        results.sort(reverse=True)
        best = results[0][0]
        print(f"{kernel}: {len(cases)} cases; best descriptor(s) match {best}:")
        for hits, S, assign, combine, order in results:
            if hits == best:
                print(f"    S={S} {assign} partials, combined {order} with {'round-to-nearest' if combine == 'rn' else 'truncation'}")
        if best < len(cases):
            print("    (no candidate explains every reading; the misses are:)")
            hits, S, assign, combine, order = results[0]
            for c in cases:
                p = predict(c, T, S, assign, combine, order)
                if Fraction(c["value"]) != p:
                    print(f"      t0={c['t0']} t1={c['t1']} t2={c['t2']}: read {c['value']}, predicted {float(p)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
