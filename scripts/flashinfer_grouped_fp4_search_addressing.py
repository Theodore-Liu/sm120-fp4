#!/usr/bin/env python3
"""Search for the addressing rule behind the scales FlashInfer `grouped_mm_fp4` (cuDNN) applies to a mid-tile segment.

Solves, for every row of the wrong segment, the K/16 scales the kernel multiplied by (least squares over the n outputs, snapped to
the E4M3 grid), then searches a family of addressing rules for the one that reproduces them: the scale buffer read as the 128x4
swizzle (global row i or segment-local row i - s), as the 8x4 swizzle, or as plain row-major, each with a constant element
offset d in [-8192, 8192]. Reports the best rule by the fraction of (row, block) scales it reproduces.

    PYTHONPATH=/path/to/sm120-fp4 python scripts/flashinfer_grouped_fp4_search_addressing.py
"""
from __future__ import annotations

import json

import torch

from sm120fp4.layouts import linear_to_128x4_index, linear_to_8x4_index, to_128x4, to_8x4
from sm120fp4.reference import e2m1_decode, global_scale_for, quantize_nvfp4, unpack_e2m1


def main() -> int:
    import flashinfer
    dev = torch.device("cuda")
    n, k = 256, 512
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
    e, s, t = 1, 64, 128
    sfc = k // 16
    a_val = e2m1_decode(unpack_e2m1(ap_)).double()
    b_val = e2m1_decode(unpack_e2m1(bp[e])).double() * bsf[e].view(torch.float8_e4m3fn).double().repeat_interleave(16, dim=1)
    grid = torch.arange(0, 256, dtype=torch.uint8).view(torch.float8_e4m3fn).double()
    grid = grid[torch.isfinite(grid)].unique()   # NaN codes would win every argmin
    used = torch.zeros(t - s, sfc, dtype=torch.float64)
    resid = []
    al = float(alpha.double().cpu())
    for i in range(s, t):
        P = torch.stack([(a_val[i, b * 16:(b + 1) * 16][None, :] * b_val[:, b * 16:(b + 1) * 16]).sum(dim=1) for b in range(sfc)], dim=1) * al
        sol = torch.linalg.lstsq(P, out[i][:, None]).solution.squeeze(1)
        resid.append(float((P @ sol - out[i]).abs().max()))
        used[i - s] = grid[(grid[None, :] - sol[:, None]).abs().argmin(dim=1)]
    print(json.dumps({"lstsq_resid_max": max(resid), "rows": t - s, "blocks_per_row": sfc}))
    sf_true = asf.view(torch.float8_e4m3fn).double()
    flat128 = to_128x4(asf).view(torch.uint8).reshape(-1).view(torch.float8_e4m3fn).double()
    flat8 = to_8x4(asf).view(torch.uint8).reshape(-1).view(torch.float8_e4m3fn).double()
    flatlin = sf_true.reshape(-1)
    rows_i = torch.arange(s, t)
    rules = {
        "128x4 global row": (flat128, linear_to_128x4_index(128, sfc).reshape(128, sfc)[rows_i]),
        "128x4 local row": (flat128, linear_to_128x4_index(128, sfc).reshape(128, sfc)[rows_i - s]),
        "8x4 global row": (flat8, linear_to_8x4_index(128, sfc).reshape(128, sfc)[rows_i]),
        "8x4 local row": (flat8, linear_to_8x4_index(128, sfc).reshape(128, sfc)[rows_i - s]),
        "row-major global": (flatlin, (rows_i[:, None] * sfc + torch.arange(sfc)[None, :])),
        "row-major local": (flatlin, ((rows_i - s)[:, None] * sfc + torch.arange(sfc)[None, :])),
        "128x4 buffer read row-major": (flat128, (rows_i[:, None] * sfc + torch.arange(sfc)[None, :])),
        "row-major buffer read as 128x4": (flatlin, linear_to_128x4_index(128, sfc).reshape(128, sfc)[rows_i]),
    }
    results = []
    for name, (buf, base) in rules.items():
        best = (0.0, None)
        for d in range(-8192, 8193):
            pos = base + d
            ok = (pos >= 0) & (pos < buf.numel())
            if ok.sum() == 0:
                continue
            pred = torch.where(ok, buf[pos.clamp(0, buf.numel() - 1)], torch.full_like(used, -1.0))
            frac = float((pred == used).double().mean())
            if frac > best[0]:
                best = (frac, d)
        results.append({"rule": name, "best_fraction": round(best[0], 4), "offset": best[1]})
        print(json.dumps(results[-1]))
    # the chance level: how often a random position carries the same E4M3 value
    vals, counts = torch.unique(sf_true.reshape(-1), return_counts=True)
    chance = float(((counts.double() / counts.sum()) ** 2).sum())
    print(json.dumps({"chance_agreement": round(chance, 4), "distinct_scale_values": int(vals.numel())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
