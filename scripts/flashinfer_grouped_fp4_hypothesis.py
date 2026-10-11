#!/usr/bin/env python3
"""Where does FlashInfer `grouped_mm_fp4` (cuDNN) read the activation scales of an expert segment that starts mid-tile?

The alignment sweep showed every element of a segment whose first row is not a multiple of 128 outside the oracle's bound, and
every aligned segment conformant. Two readings of the kernel's scale indexing are tested against the exact oracle computed with
the scales a kernel would have used under each:

  H1  scale row for token i of the segment = segment_tile_base + (i - start), with segment_tile_base = (start // 128) * 128
      (the kernel indexes scales relative to the tile the segment starts in, as if the segment began at that tile);
  H2  scale row = i - start (the kernel indexes scales from row 0, as if each segment were its own tensor);
  H0  the documented reading, scale row = i (which the sweep already shows the kernel does not use for a mid-tile start).

Also runs the same case twice to record whether the wrong output is deterministic.

    PYTHONPATH=/path/to/sm120-fp4 python scripts/flashinfer_grouped_fp4_hypothesis.py --out reports/...json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sm120fp4 import oracle
from sm120fp4.layouts import to_128x4
from sm120fp4.reference import global_scale_for, quantize_nvfp4

CASES = {"64,64": [64, 64], "192,64": [192, 64], "16,48,64": [16, 48, 64], "120,136": [120, 136]}


def permuted_scales(asf: torch.Tensor, indptr: list[int], rule: str) -> torch.Tensor:
    rows = asf.shape[0]
    idx = torch.arange(rows)
    for e in range(len(indptr) - 1):
        s, t = indptr[e], indptr[e + 1]
        if t <= s:
            continue
        local = torch.arange(t - s)
        if rule == "H1":
            src = (s // 128) * 128 + local
        elif rule == "H2":
            src = local
        else:
            src = s + local
        idx[s:t] = src.clamp(max=rows - 1)
    return asf[idx]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports") / f"flashinfer-grouped-fp4-hypothesis-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--k", type=int, default=512)
    a = ap.parse_args(argv)
    import flashinfer
    dev = torch.device("cuda")
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "flashinfer": flashinfer.__version__, "cudnn": torch.backends.cudnn.version(), "cases": []}
    gen = torch.Generator().manual_seed(20261010)
    for name, segs in CASES.items():
        indptr = [0]
        for s in segs:
            indptr.append(indptr[-1] + s)
        m_total, e_count = indptr[-1], len(segs)
        x = torch.randn(m_total, a.k, generator=gen).to(torch.bfloat16).float()
        w = torch.randn(e_count, a.n, a.k, generator=gen).to(torch.bfloat16).float()
        gs_a, gs_b = global_scale_for(x), global_scale_for(w)
        ap_, asf, _ = quantize_nvfp4(x, gs_a)
        bq = [quantize_nvfp4(w[e], gs_b) for e in range(e_count)]
        bp = torch.stack([b[0] for b in bq]); bsf = torch.stack([b[1] for b in bq])
        a_sw = to_128x4(asf).view(torch.float8_e4m3fn).to(dev)
        b_sw = torch.stack([to_128x4(bsf[e]) for e in range(e_count)]).view(torch.float8_e4m3fn).to(dev)
        alpha = (1.0 / (gs_a.double() * gs_b.double())).float().reshape(1).to(dev)
        rec = {"name": name, "segments": segs, "m_indptr": indptr}
        try:
            outs = []
            for _ in range(2):
                o = flashinfer.grouped_mm_fp4(ap_.to(dev), bp.to(dev), a_sw, b_sw, torch.tensor(indptr, dtype=torch.int32, device=dev),
                                              alpha=alpha, out_dtype=torch.bfloat16, block_size=16, backend="cudnn")
                torch.cuda.synchronize()
                outs.append(o.float().cpu().clone())
            rec["deterministic"] = bool(torch.equal(outs[0], outs[1]))
            out = outs[0]
            for rule in ("H0", "H1", "H2", "H3"):
                if rule == "H3":   # codes and scales both read from the segment's tile base
                    idx = torch.arange(m_total)
                    for e in range(e_count):
                        s, t_ = indptr[e], indptr[e + 1]
                        if t_ > s:
                            idx[s:t_] = ((s // 128) * 128 + torch.arange(t_ - s)).clamp(max=m_total - 1)
                    exact, bound = oracle.exact_grouped_gemm_nvfp4(ap_[idx], asf[idx], gs_a, bp, bsf, gs_b, torch.tensor(indptr))
                else:
                    sf_rule = permuted_scales(asf, indptr, rule)
                    exact, bound = oracle.exact_grouped_gemm_nvfp4(ap_, sf_rule, gs_a, bp, bsf, gs_b, torch.tensor(indptr))
                v = oracle.conformance(out, exact, bound)
                err = (out.double() - exact).abs() > bound
                v["outside_bound_per_segment"] = [int(err[indptr[e]:indptr[e + 1]].sum()) for e in range(e_count)]
                rec[rule] = v
            # row matching: for each row of a bad segment, which token's exact product against that expert does the kernel's row equal?
            match = {}
            for e in range(e_count):
                s, t_ = indptr[e], indptr[e + 1]
                if t_ <= s or rec["H0"]["outside_bound_per_segment"][e] == 0:
                    continue
                full, fb = oracle.exact_gemm_nvfp4(ap_, asf, gs_a, bp[e], bsf[e], gs_b.reshape(1))   # every token against expert e
                seg = out[s:t_].double()
                d = (seg[:, None, :] - full[None, :, :]).abs()                     # [len, m_total, n]
                within = (d <= fb[None, :, :]).all(dim=-1)                          # [len, m_total]
                best = d.amax(dim=-1).argmin(dim=1)                                  # nearest token row by max error
                hits = within.any(dim=1)
                match[f"segment{e}"] = {"rows": t_ - s, "rows_matching_some_token": int(hits.sum()),
                                        "matched_token_minus_row": [int(best[i]) - (s + i) for i in range(min(t_ - s, 8))],
                                        "nearest_token_rows_first8": best[:8].tolist(), "max_err_to_nearest_first8": [round(float(d[i, best[i]].max()), 3) for i in range(min(t_ - s, 8))]}
            rec["row_matching"] = match
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        report["cases"].append(rec)
        print(name, {r: rec[r]["outside_bound_per_segment"] for r in ("H0", "H1", "H2", "H3") if r in rec}, "deterministic", rec.get("deterministic"), rec.get("error", "")); print("   ", json.dumps(rec.get("row_matching", {}))[:600])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
