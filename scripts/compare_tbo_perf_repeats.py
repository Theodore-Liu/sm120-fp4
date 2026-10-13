#!/usr/bin/env python3
"""Baseline against patched llama.cpp test-backend-ops perf with repeats: per case, the median of the patched runs over the
median of the baseline runs, next to the run-to-run spread of the baseline itself (baseline run i over baseline run j).

    python scripts/compare_tbo_perf_repeats.py <report dir>

Expects perf-base-MUL_MAT_ID.txt, perf-base-MUL_MAT_ID-r2.txt, ... and the same for fix (console output, "us/run").
"""
import glob
import itertools
import os
import re
import statistics
import sys

D = sys.argv[1]
PAT = re.compile(r"^\s*(MUL_MAT(?:_ID)?)\((.*?)\):.*?([\d.]+) us/run")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def load(path):
    out = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = PAT.search(ANSI.sub("", line))
            if m:
                out[m.group(2)] = float(m.group(3))
    return out


runs = {v: [load(p) for p in sorted(glob.glob(os.path.join(D, f"perf-{v}-MUL_MAT_ID*.txt")))] for v in ("base", "fix")}
print("runs:", {v: len(r) for v, r in runs.items()})
keys = sorted(set.intersection(*(set(r) for v in runs for r in runs[v])))
eff, noise = [], []
for k in keys:
    b = [r[k] for r in runs["base"]]
    f = [r[k] for r in runs["fix"]]
    eff.append((statistics.median(f) / statistics.median(b), k, b, f))
    for x, y in itertools.combinations(b, 2):
        noise.append(max(x, y) / min(x, y))
eff.sort()
rs = [e[0] for e in eff]
print(f"cases {len(keys)}")
print(f"effect, median(fix)/median(base): median {statistics.median(rs):.3f}, min {rs[0]:.3f}, max {rs[-1]:.3f}")
if noise:
    print(f"baseline run-to-run spread (max/min of a pair): median {statistics.median(noise):.3f}, 90th pct "
          f"{sorted(noise)[int(0.9 * len(noise))]:.3f}, max {max(noise):.3f}")
else:
    print("baseline run-to-run spread: needs at least two baseline runs")
print("largest effects:")
for r, k, b, f in eff[-10:]:
    print(f"  {r:.3f}  base {[round(x, 1) for x in b]}  fix {[round(x, 1) for x in f]}  {k[:100]}")
print("smallest effects:")
for r, k, b, f in eff[:3]:
    print(f"  {r:.3f}  base {[round(x, 1) for x in b]}  fix {[round(x, 1) for x in f]}  {k[:100]}")
