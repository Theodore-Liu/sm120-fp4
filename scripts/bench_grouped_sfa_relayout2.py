#!/usr/bin/env python3
"""Second look at the cost of the per-group token-scale re-layout: (1) how much of the measured time is GPU kernel time versus
eager launch gaps (torch.profiler kernel totals), (2) a leaner variant that gathers 32-bit words (four scale bytes, one row of one
128x4 atom column) directly from the source swizzle into the destination swizzle in one index_select, checked bit-exact against
the worktree helper, and (3) both under CUDA graph replay, the way a serving engine runs decode.

    PYTHONPATH=. python scripts/bench_grouped_sfa_relayout2.py --worktree-core <path> --out reports/<file>.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

CASES = ((8, 128, 4096), (8, 1024, 4096), (64, 128, 7168), (128, 64, 7168), (128, 512, 4096))  # (experts, tokens/expert, k)


def lean_relayout(a_descale: torch.Tensor, m_indptr: torch.Tensor, cum_m: int, k_blocks: int) -> torch.Tensor:
    """Same output as the worktree helper. In the F8_128x4 layout the four scales of one row in one 4-column group are one
    contiguous 32-bit word at index ((r // 128) * C4 + c4) * 128 + (r % 32) * 4 + (r // 32) % 4, so the re-layout is a single
    gather of words: for every destination word, the source row follows from m_indptr and the column group is unchanged."""
    cols = ((k_blocks + 3) // 4) * 4
    c4 = cols // 4
    rows = a_descale.numel() // cols
    num_groups = m_indptr.numel() - 1
    active = min(cum_m, num_groups)
    cap = 128 * (active + (cum_m - active) // 128)
    src_words = a_descale.contiguous().view(torch.uint8).view(torch.int32).reshape(-1)
    indptr = m_indptr.to(torch.int64)
    counts = indptr[1:] - indptr[:-1]
    padded_end = torch.cumsum((counts + 127) // 128 * 128, 0)
    dev = a_descale.device
    # destination word grid in swizzled order: (block, c4, r32, r4) -> destination row d = block*128 + r4*32 + r32
    blk = torch.arange(cap // 128, device=dev).view(-1, 1, 1, 1)
    cc = torch.arange(c4, device=dev).view(1, -1, 1, 1)
    r32 = torch.arange(32, device=dev).view(1, 1, -1, 1)
    r4 = torch.arange(4, device=dev).view(1, 1, 1, -1)
    d = blk * 128 + r4 * 32 + r32                                    # (B, 1, 32, 4)
    group = torch.searchsorted(padded_end, d.reshape(-1), right=True).clamp_(max=num_groups - 1).view(d.shape)
    local = d - (padded_end - (counts + 127) // 128 * 128)[group]
    valid = (d < padded_end[-1]) & (local < counts[group])
    s = (indptr[:-1][group] + local).clamp_(0, rows - 1)           # source row
    src_idx = ((s // 128) * c4 + cc) * 128 + (s % 32) * 4 + (s // 32) % 4   # (B, C4, 32, 4)
    out = torch.where(valid, src_words[src_idx], torch.zeros((), dtype=torch.int32, device=dev))
    return out.reshape(cap, cols // 4).view(torch.uint8).reshape(cap, cols).view(a_descale.dtype)


def median_ms(fn, warm=10, reps=50):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def graph_ms(fn, reps=50):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return median_ms(g.replay, reps=reps)


def kernel_us(fn):
    from torch.profiler import ProfilerActivity, profile
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
    tot = sum(e.device_time_total for e in prof.key_averages() if e.device_time_total > 0 and e.key and not e.key.startswith("cuda"))
    return tot / 20.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree-core", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    src = open(a.worktree_core, encoding="utf-8").read()
    ns = {"torch": torch}
    exec(src[src.index("def _segmented_token_scale_capacity_rows("):src.index("def _build_cudnn_moe_block_scale_grouped_gemm_graph(")], ns)
    ref = ns["_segment_token_descale_per_group"]
    dev = torch.device("cuda")
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "cases": []}
    for e_count, tpe, k in CASES:
        g = torch.Generator(device=dev).manual_seed(2)
        cum_m = e_count * tpe
        kb = k // 16
        rows = (cum_m + 127) // 128 * 128
        a_sf = torch.randint(0, 255, (rows, kb), generator=g, device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
        # ragged routing: random segment lengths summing to cum_m
        cuts = torch.sort(torch.randint(0, cum_m + 1, (e_count - 1,), generator=g, device=dev)).values
        m_indptr = torch.cat([torch.zeros(1, device=dev, dtype=torch.int64), cuts, torch.full((1,), cum_m, device=dev)]).to(torch.int32)
        r1, r2 = ref(a_sf, m_indptr, cum_m, kb), lean_relayout(a_sf, m_indptr, cum_m, kb)
        same = bool(torch.equal(r1.view(torch.uint8), r2.view(torch.uint8)))
        f_ref = lambda: ref(a_sf, m_indptr, cum_m, kb)  # noqa: E731
        f_lean = lambda: lean_relayout(a_sf, m_indptr, cum_m, kb)  # noqa: E731
        rec = {"experts": e_count, "tokens_per_expert": tpe, "k": k, "lean_bit_exact_vs_worktree": same,
               "worktree_eager_ms": round(median_ms(f_ref), 4), "worktree_kernel_us": round(kernel_us(f_ref), 1),
               "worktree_graph_ms": round(graph_ms(f_ref), 4),
               "lean_eager_ms": round(median_ms(f_lean), 4), "lean_kernel_us": round(kernel_us(f_lean), 1),
               "lean_graph_ms": round(graph_ms(f_lean), 4)}
        report["cases"].append(rec)
        print(json.dumps(rec))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
