#!/usr/bin/env python3
"""Which activation scales did FlashInfer `grouped_mm_fp4` (cuDNN) actually apply to a mid-tile segment's rows?

For a row i of a wrong segment, out[i, :] = sum_b sf_used[i, b] * P[i, b, :] with P[i, b, :] = sum_{k in block b} a[i, k] * B[e][:, k] * alpha
(the kernel's inner product with the scale factored out). P is known exactly, out is observed, so the K/16 unknown scales a row was
multiplied by follow from a least-squares solve over the n outputs; each solved scale is then matched to the E4M3 grid and to the
rows of the swizzled scale array that carry that value, which names the scale row the kernel read.

    PYTHONPATH=/path/to/sm120-fp4 python scripts/flashinfer_grouped_fp4_solve_scales.py
"""
from __future__ import annotations

import json
import sys

import torch

from sm120fp4.layouts import to_128x4
from sm120fp4.reference import e2m1_decode, global_scale_for, quantize_nvfp4, unpack_e2m1


def main() -> int:
    import flashinfer
    dev = torch.device("cuda")
    segs, n, k = [64, 64], 256, 512
    indptr = [0, 64, 128]
    gen = torch.Generator().manual_seed(20261010)
    x = torch.randn(128, k, generator=gen).to(torch.bfloat16).float()
    w = torch.randn(2, n, k, generator=gen).to(torch.bfloat16).float()
    gs_a, gs_b = global_scale_for(x), global_scale_for(w)
    ap_, asf, _ = quantize_nvfp4(x, gs_a)
    bq = [quantize_nvfp4(w[e], gs_b) for e in range(2)]
    bp = torch.stack([b[0] for b in bq]); bsf = torch.stack([b[1] for b in bq])
    a_sw = to_128x4(asf).view(torch.float8_e4m3fn).to(dev)
    b_sw = torch.stack([to_128x4(bsf[e]) for e in range(2)]).view(torch.float8_e4m3fn).to(dev)
    alpha = (1.0 / (gs_a.double() * gs_b.double())).float().reshape(1).to(dev)
    out = flashinfer.grouped_mm_fp4(ap_.to(dev), bp.to(dev), a_sw, b_sw, torch.tensor(indptr, dtype=torch.int32, device=dev),
                                    alpha=alpha, out_dtype=torch.bfloat16, block_size=16, backend="cudnn")
    torch.cuda.synchronize()
    out = out.double().cpu()
    e = 1
    a_val = e2m1_decode(unpack_e2m1(ap_)).double()                          # [128, k] codes as values
    b_val = e2m1_decode(unpack_e2m1(bp[e])).double() * bsf[e].view(torch.float8_e4m3fn).double().repeat_interleave(16, dim=1)  # [n, k] dequant without gs
    sf_true = asf.view(torch.float8_e4m3fn).double()                       # [128, k/16]
    grid = torch.arange(0, 256, dtype=torch.uint8).view(torch.float8_e4m3fn).double()
    grid = grid[torch.isfinite(grid)].unique()   # NaN codes would win every argmin
    # H6: the kernel advances the scale pointer by the segment start in linear elements (s * k/16) and then addresses the
    # buffer with the 128x4 swizzle of the segment-local row (i - s): used[i, b] = flat[s * (k/16) + idx128x4(i - s, b)]
    from sm120fp4.layouts import linear_to_128x4_index
    flat = to_128x4(asf).view(torch.uint8).reshape(-1)
    sfc = k // 16
    idx_local = linear_to_128x4_index(128, sfc).reshape(128, sfc)           # flat position of (row, col) in a 128-row swizzle
    s = indptr[e]
    h6 = torch.zeros_like(sf_true)
    for i in range(s, indptr[e + 1]):
        pos = (s * sfc + idx_local[i - s]).clamp(max=flat.numel() - 1)
        h6[i] = flat[pos].view(torch.float8_e4m3fn).double()
    report = []
    for i in range(64, 72):
        P = torch.stack([(a_val[i, b * 16:(b + 1) * 16][None, :] * b_val[:, b * 16:(b + 1) * 16]).sum(dim=1) for b in range(k // 16)], dim=1)  # [n, k/16]
        P = P * float(alpha.double().cpu())
        sol = torch.linalg.lstsq(P, out[i][:, None]).solution.squeeze(1)      # [k/16] solved scales
        resid = float((P @ sol - out[i]).abs().max())
        snapped = grid[(grid[None, :] - sol[:, None]).abs().argmin(dim=1)]
        # which rows of the true scale array carry exactly this scale vector (or most of it)?
        agree = (sf_true == snapped[None, :]).sum(dim=1)                       # [128] blocks agreeing per candidate row
        best_row = int(agree.argmax()); best_agree = int(agree.max())
        h6_agree = int((snapped == h6[i]).sum())
        report.append({"row": i, "lstsq_resid_max": round(resid, 4), "H6_blocks_agreeing": h6_agree, "solved_first6": [round(float(v), 3) for v in sol[:6]], "H6_first6": [float(v) for v in h6[i, :6]],
                       "true_first6": [float(v) for v in sf_true[i, :6]], "best_matching_scale_row": best_row, "blocks_agreeing": best_agree, "of": k // 16,
                       "own_row_agreeing": int(agree[i])})
        print(json.dumps(report[-1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
