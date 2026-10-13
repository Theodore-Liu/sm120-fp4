#!/usr/bin/env python3
"""Compare two `cuobjdump --dump-resource-usage` dumps kernel by kernel (registers and stack, the spill indicator).

    python scripts/compare_cuobjdump_resources.py <base.txt> <other.txt> [name regex ...]

Functions are matched by mangled name (with the per-build _INTERNAL_<hash> of file-local symbols removed) and by their
order of appearance (one dump holds a copy per embedded architecture). Prints, per name filter, how many instances
differ and the median and largest REG and STACK changes (the median also over the differing ones: instances compiled
as stubs for unsupported configurations are identical in both and pull the overall median to zero).
"""
import collections
import re
import statistics
import sys

FN = re.compile(r"^\s*Function (\S+):")
INTERNAL = re.compile(r"_INTERNAL_[0-9a-f]+_\d+_[A-Za-z0-9_]+?_[0-9a-f]{8}")
RES = re.compile(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)")


def load(path):
    out, seen, cur = {}, collections.Counter(), None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = FN.match(line)
            if m:
                cur = INTERNAL.sub("_INTERNAL_", m.group(1))
                continue
            m = RES.search(line)
            if m and cur:
                seen[cur] += 1
                out[(cur, seen[cur])] = tuple(int(g) for g in m.groups())
                cur = None
    return out


a, b = load(sys.argv[1]), load(sys.argv[2])
filters = sys.argv[3:] or [r"^_Z9mul_mat_qI", r"^_Z9mul_mat_fI", r"^_Z13mul_mat_f_idsI"]
print(f"functions: base {len(a)}, other {len(b)}, only in base {len(set(a) - set(b))}, only in other {len(set(b) - set(a))}")
for flt in filters:
    keys = [k for k in a if k in b and re.search(flt, k[0])]
    dreg = [b[k][0] - a[k][0] for k in keys]
    dstk = [b[k][1] - a[k][1] for k in keys]
    diff = [k for k in keys if a[k] != b[k]]
    spill_new = sum(1 for k in keys if b[k][1] > a[k][1])
    if not keys:
        print(f"{flt}: no common instances")
        continue
    print(f"{flt}: {len(keys)} common instances, {len(diff)} differ; REG change median {statistics.median(dreg):+g} "
          f"(over differing {statistics.median([b[k][0] - a[k][0] for k in diff]) if diff else 0:+g}) "
          f"min {min(dreg):+d} max {max(dreg):+d}; STACK grew in {spill_new} (max {max(dstk):+d} bytes), "
          f"base instances with STACK>0 {sum(1 for k in keys if a[k][1] > 0)}, other {sum(1 for k in keys if b[k][1] > 0)}")
new = sorted(k[0] for k in set(b) - set(a))
for name in new[:12]:
    print(f"  only in other: {name[:110]} {b[(name, 1)] if (name, 1) in b else ''}")
