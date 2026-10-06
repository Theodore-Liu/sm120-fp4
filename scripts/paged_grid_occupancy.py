"""How full the paged indexer's (row block, page) grid is under the benchmark's context lengths, computed on the CPU from the lengths alone.

The paged kernels launch a grid of (ceil(S / 8), max_pages) blocks of 8 warps, one warp per (row, page); a warp whose page starts at or past its row's
context length returns at once. A block holds its block slot (shared memory, registers) until its last live warp finishes, so a block with few live
warps keeps an SM slot at a fraction of its warps. This script regenerates the benchmark's context lengths (run_paged_case: seed 200 + S, lengths drawn
uniformly from [1, N] with seed + 7) and reports, per shape: max_pages (checked against the grid Nsight Compute printed for S 64 N 32768, 510), the share
of blocks with no live warp, the distribution of live warps over the other blocks, and the mean live warps per non-empty block as a share of 8, the
achieved-over-theoretical occupancy those blocks allow if every live warp ran equally long.

    ~/mlsys-5090-runtime/vllm028/.venv/bin/python scripts/paged_grid_occupancy.py --out reports/paged-grid-occupancy-20261006.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import torch

PAGE, WARPS = 64, 8
SHAPES = ((32, 8192, 8), (64, 32768, 8), (32, 65536, 16))
NCU_GRID_Y = {(64, 32768): 510}      # reports/ncu-paged-v6i-rtx5090-20261005.txt: grid (8, 510, 1)


def lengths(S: int, N: int) -> list[int]:
    g = torch.Generator().manual_seed(200 + S + 7)
    return torch.randint(1, N + 1, (S,), generator=g).tolist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.out is not None and a.out.exists():
        print(f"refusing to overwrite {a.out}", file=sys.stderr)
        return 2
    rows, ok = [], True
    for S, N, H in SHAPES:
        ctx = lengths(S, N)
        max_pages = -(-max(ctx) // PAGE)
        if (S, N) in NCU_GRID_Y and NCU_GRID_Y[(S, N)] != max_pages:
            print(f"S={S} N={N}: max_pages {max_pages} but Nsight Compute launched {NCU_GRID_Y[(S, N)]}; the lengths are not the benchmark's", file=sys.stderr)
            ok = False
        nblk_x = -(-S // WARPS)
        hist = collections.Counter()
        for bx in range(nblk_x):
            rs = ctx[bx * WARPS:(bx + 1) * WARPS]
            for p in range(max_pages):
                hist[sum(1 for c in rs if c > p * PAGE)] += 1
        blocks = nblk_x * max_pages
        nonempty = blocks - hist[0]
        live = sum(k * v for k, v in hist.items())
        r = {"S": S, "N": N, "H": H, "max_pages": max_pages, "blocks": blocks, "empty_blocks": hist[0], "empty_share": hist[0] / blocks,
             "live_warps": live, "warps_launched": blocks * WARPS, "live_share_of_launched": live / (blocks * WARPS),
             "mean_live_per_nonempty_block": live / nonempty, "est_achieved_over_theoretical": live / nonempty / WARPS,
             "live_warps_histogram": {str(k): hist[k] for k in sorted(hist)}}
        rows.append(r)
        print(f"S={S} N={N} H={H}: max_pages {max_pages}, {blocks} blocks, {hist[0]} empty ({hist[0] / blocks:.1%}); "
              f"live warps per non-empty block {live / nonempty:.2f} of 8 -> achieved/theoretical about {live / nonempty / WARPS:.3f}; "
              f"histogram {dict(sorted(hist.items()))}", flush=True)
    if a.out is not None:
        a.out.write_text(json.dumps({"note": __doc__.strip().split(chr(10))[0], "rows": rows}, indent=1) + "\n", encoding="utf-8")
        print("->", a.out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
