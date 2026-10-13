#!/usr/bin/env python3
"""Attribute a base-against-fix time difference to kernels: reads `nsys stats -r cuda_gpu_kern_sum -f csv` exports.

    python scripts/compare_nsys_kernel_times.py <dir> <fix-name>

Expects <case>-base.kern.csv and <case>-<fix-name>.kern.csv per case. Kernel names are shortened to the template name.
For each case prints total GPU kernel time per launch of the op (the run count is taken from the most launched kernel
present in both) and per-kernel average time, base against fix.
"""
import csv
import glob
import os
import re
import sys

D, FIX = sys.argv[1], sys.argv[2]


def short(name):
    name = re.sub(r"^void ", "", name)
    return re.split(r"[<(]", name, 1)[0]


def load(path):
    out = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        rows = [r for r in csv.reader(fh) if r]
    rows = rows[next(i for i, r in enumerate(rows) if r[0] == "Time (%)"):]  # nsys prints progress lines first
    head = rows[0]
    i_tot, i_n, i_name = head.index("Total Time (ns)"), head.index("Instances"), head.index("Name")
    for r in rows[1:]:
        k = short(r[i_name])
        tot, n = float(r[i_tot]), int(r[i_n])
        t0, n0 = out.get(k, (0.0, 0))
        out[k] = (t0 + tot, n0 + n)
    return out


for bpath in sorted(glob.glob(os.path.join(D, "*-base.kern.csv"))):
    case = os.path.basename(bpath)[: -len("-base.kern.csv")]
    fpath = os.path.join(D, f"{case}-{FIX}.kern.csv")
    if not os.path.exists(fpath):
        continue
    b, f = load(bpath), load(fpath)
    print(f"== {case}")
    for k in sorted(set(b) | set(f), key=lambda k: -max(b.get(k, (0, 0))[0], f.get(k, (0, 0))[0])):
        tb, nb = b.get(k, (0.0, 0))
        tf, nf = f.get(k, (0.0, 0))
        ab = tb / nb if nb else float("nan")
        af = tf / nf if nf else float("nan")
        ratio = af / ab if nb and nf else float("nan")
        print(f"  {k[:48]:48s} base {nb:6d} x {ab / 1000:8.2f} us   fix {nf:6d} x {af / 1000:8.2f} us   {ratio:6.3f}")
