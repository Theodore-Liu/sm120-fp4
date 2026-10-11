#!/usr/bin/env python3
"""Prototype of a fix for FlashInfer's grouped block-scaled GEMM (`grouped_mm_fp4`, `grouped_mm_mxfp8`, cuDNN backend) on SM120:
repack A's block scales from the layout the API documents (one F8_128x4 swizzle over all `cum_m` tokens) into the layout cuDNN's
SM120 MoE kernel consumes (each group's rows padded to whole 128-row blocks and swizzled on their own, groups concatenated in
order, the blob sized to cuDNN's static capacity `min(S, G) + (S - min(S, G)) // 128` blocks), then call the unchanged runner.

The repack is device-side with no host synchronisation: the destination row of every source row follows from `m_indptr` alone.
`--check` runs the alignment sweep of `flashinfer_grouped_fp4_alignment.py` with the patch in place and reports conformance.

    PYTHONPATH=. python scripts/fix_proto_grouped_sfa.py --out reports/<file>.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def segmented_capacity_rows(total_rows: int, num_groups: int) -> int:
    """cuDNN's `segmented_row_scale_capacity_rows` (fusion_ir.py): rows of the worst-case per-group padded blob."""
    active = min(total_rows, num_groups)
    return 128 * (active + (total_rows - active) // 128)


def unswizzle_128x4(sf: torch.Tensor, rows_pad: int, cols_pad: int) -> torch.Tensor:
    """F8_128x4 blob -> linear (rows_pad, cols_pad), byte view (inverse of the swizzle FlashInfer's quantisers write)."""
    return sf.view(torch.uint8).reshape(rows_pad // 128, cols_pad // 4, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(rows_pad, cols_pad)


def swizzle_128x4(lin: torch.Tensor) -> torch.Tensor:
    rows, cols = lin.shape
    return lin.reshape(rows // 128, 4, 32, cols // 4, 4).permute(0, 3, 2, 1, 4).reshape(rows, cols)


def segment_token_scales(a_descale: torch.Tensor, m_indptr: torch.Tensor, cum_m: int, k_blocks: int) -> torch.Tensor:
    """Global F8_128x4 token scales over `cum_m` rows -> cuDNN's per-group segmented blob (same dtype, dense)."""
    rows_pad = ((cum_m + 127) // 128) * 128
    cols_pad = ((k_blocks + 3) // 4) * 4
    lin = unswizzle_128x4(a_descale.contiguous(), rows_pad, cols_pad)
    num_groups = m_indptr.numel() - 1
    cap = segmented_capacity_rows(cum_m, num_groups)
    ind = m_indptr.to(torch.int64)
    counts = ind[1:] - ind[:-1]
    padded = ((counts + 127) // 128) * 128
    pend = torch.cumsum(padded, 0)                  # end row of each group's padded region
    pstart = pend - padded
    d = torch.arange(cap, device=a_descale.device, dtype=torch.int64)
    g = torch.searchsorted(pend, d, right=True).clamp(max=num_groups - 1)
    local = d - pstart[g]
    valid = (d < pend[-1]) & (local < counts[g])
    src = (ind[:-1][g] + local).clamp(0, rows_pad - 1)
    seg = torch.where(valid[:, None], lin[src], torch.zeros((), dtype=lin.dtype, device=lin.device))
    return swizzle_128x4(seg).contiguous().view(a_descale.dtype)


_KEEP: list = []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    import flashinfer
    import scripts.flashinfer_grouped_fp4_alignment as sweep
    import flashinfer.grouped_mm.cudnn.core as cc
    orig = cc._run_cudnn_moe_block_scale_grouped_gemm_fp4

    def patched(a_, b_, a_descale, b_descale, m_indptr, *args, block_size=16, **kw):
        cum_m, k_packed = a_.shape
        seg = segment_token_scales(a_descale, m_indptr, cum_m, (k_packed * 2) // block_size)
        _KEEP.append(seg)  # the logical view below does not own the storage; keep the blob alive through the call
        logical = seg.reshape(-1)[: a_descale.numel()].view(a_descale.shape)
        return orig(a_, b_, logical, b_descale, m_indptr, *args, block_size=block_size, **kw)

    cc._run_cudnn_moe_block_scale_grouped_gemm_fp4 = patched
    import flashinfer.grouped_mm.core as gc
    gc._run_cudnn_moe_block_scale_grouped_gemm_fp4 = patched
    dev = torch.device("cuda")
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "flashinfer": flashinfer.__version__, "patched": "segment_token_scales", "cases": []}
    gen = torch.Generator().manual_seed(20261010)
    for name, segs in sweep.SEGMENTS.items():
        rec = {"name": name, "segments": segs}
        try:
            rec.update(sweep.run(flashinfer, segs, 256, 512, gen, dev))
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        report["cases"].append(rec)
        print(json.dumps({k: rec.get(k) for k in ("name", "outside_bound_per_segment", "nan_per_segment", "conformant", "error")})[:300])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
