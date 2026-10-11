#!/usr/bin/env python3
"""Cost of the per-group token-scale re-layout (the FlashInfer fix for the grouped block-scaled GEMM) against the grouped GEMM it
precedes, on the local GPU.

The helper is taken verbatim from the FlashInfer worktree's `flashinfer/grouped_mm/cudnn/core.py` (the text between
`def _segmented_token_scale_capacity_rows` and `def _build_cudnn_moe_block_scale_grouped_gemm_graph`), executed in isolation, and
timed with CUDA events against the unpatched installed `grouped_mm_fp4` on the same operands (expert segments 128-aligned, so the
unpatched GEMM is correct and is the right denominator). Median of 50 timed calls after 10 warm-up calls.

    PYTHONPATH=. python scripts/bench_grouped_sfa_relayout.py --worktree-core <path> --out reports/<file>.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

CASES = (  # (experts, tokens per expert, n, k)
    (8, 128, 4096, 4096),
    (8, 1024, 4096, 4096),
    (64, 128, 2048, 7168),
    (128, 64, 2048, 7168),
    (128, 512, 1536, 4096),
)


def median_ms(fn, warm=10, reps=50):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree-core", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    src = open(a.worktree_core, encoding="utf-8").read()
    helper = src[src.index("def _segmented_token_scale_capacity_rows("):src.index("def _build_cudnn_moe_block_scale_grouped_gemm_graph(")]
    ns = {"torch": torch}
    exec(helper, ns)
    relayout = ns["_segment_token_descale_per_group"]
    import flashinfer
    from flashinfer import SfLayout
    from flashinfer.fp4_quantization import nvfp4_quantize
    dev = torch.device("cuda")
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "flashinfer": flashinfer.__version__, "cases": []}
    for e_count, tpe, n, k in CASES:
        cum_m = e_count * tpe
        g = torch.Generator(device=dev).manual_seed(1)
        a_bf = torch.randn(cum_m, k, generator=g, device=dev, dtype=torch.bfloat16)
        b_bf = torch.randn(e_count * n, k, generator=g, device=dev, dtype=torch.bfloat16)
        gsa = (448 * 6) / a_bf.float().abs().max(); gsb = (448 * 6) / b_bf.float().abs().max()
        a_fp4, a_sf = nvfp4_quantize(a_bf, gsa, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
        b_fp4, b_sf = nvfp4_quantize(b_bf, gsb, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
        a_sf = a_sf.view(torch.float8_e4m3fn).reshape(-1, k // 16)
        b_fp4 = b_fp4.reshape(e_count, n, k // 2)
        b_sf = b_sf.view(torch.float8_e4m3fn).reshape(e_count, -1, k // 16)
        m_indptr = (torch.arange(e_count + 1, device=dev) * tpe).to(torch.int32)
        alpha = torch.tensor([1.0 / float(gsa * gsb)], device=dev)
        gemm = lambda: flashinfer.grouped_mm_fp4(a_fp4, b_fp4, a_sf, b_sf, m_indptr, alpha=alpha, out_dtype=torch.bfloat16, block_size=16)  # noqa: E731
        rel = lambda: relayout(a_sf, m_indptr, cum_m, k // 16)  # noqa: E731
        t_gemm, t_rel = median_ms(gemm), median_ms(rel)
        rec = {"experts": e_count, "tokens_per_expert": tpe, "n": n, "k": k, "grouped_mm_fp4_ms": round(t_gemm, 4),
               "relayout_ms": round(t_rel, 4), "relayout_over_gemm_percent": round(100 * t_rel / t_gemm, 2)}
        report["cases"].append(rec)
        print(json.dumps(rec))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
