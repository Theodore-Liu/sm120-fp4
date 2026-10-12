#!/usr/bin/env python3
"""Compare llama.cpp test-backend-ops perf output (console mode, "us/run") of a baseline and a patched build, case by case.

    python scripts/compare_tbo_perf.py [report dir]     (default: reports/llamacpp-dupfix-pro6000)

Expects perf-base-<op>.txt and perf-fix-<op>.txt for op in MUL_MAT_ID and MUL_MAT-f16.
"""
import os
import re
import statistics
import sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports", "llamacpp-dupfix-pro6000")
PAT = re.compile(r"^\s*(MUL_MAT(?:_ID)?)\((.*?)\):.*?([\d.]+) us/run")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def load(name):
    out = {}
    with open(os.path.join(D, name), encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = PAT.search(ANSI.sub("", line))
            if m:
                out[(m.group(1), m.group(2))] = float(m.group(3))
    return out


for op in ("MUL_MAT_ID", "MUL_MAT-f16"):
    b, f = load(f"perf-base-{op}.txt"), load(f"perf-fix-{op}.txt")
    keys = [k for k in b if k in f]
    ratios = sorted((f[k] / b[k], k) for k in keys)
    rs = [r for r, _ in ratios]
    print(f"== {op}: {len(keys)} common cases (base {len(b)}, fix {len(f)})")
    print(f"   ratio fix/base: median {statistics.median(rs):.3f}, min {rs[0]:.3f}, max {rs[-1]:.3f}")
    for r, k in ratios[-8:]:
        print(f"   {r:.3f}  base {b[k]:9.2f} us  fix {f[k]:9.2f} us  {k[1][:110]}")
