#!/usr/bin/env python3
"""FlashInfer `grouped_mm_fp4` (the cuDNN MoE grouped GEMM) on this SM120 device, judged by the exact grouped oracle.

Operands: tokens quantised by the reference quantiser (one global scale for all tokens), E expert weight matrices each with its
own global scale, scales swizzled 128x4 per expert (the 3D layout the op documents), routing given as m_indptr. The oracle is
`sm120fp4.oracle.exact_grouped_gemm_nvfp4`; the op's alpha is a single scalar, so this probe uses one shared global scale for the
experts (alpha = 1 / (gs_a * gs_b)); per-expert alphas are outside what the op takes. A refusal is a result (stage 1 recorded cuDNN
refusing mm_fp4 on SM120).

    PYTHONPATH=/path/to/sm120-fp4 python scripts/flashinfer_grouped_fp4_probe.py --out reports/flashinfer-grouped-fp4-<device>-<date>.json
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

CASES = ((4, 64, 512, 1024, (0, 16, 16, 40, 64)), (8, 128, 1024, 2048, None), (2, 32, 256, 512, (0, 0, 32)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports") / f"flashinfer-grouped-fp4-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    a = ap.parse_args(argv)
    import flashinfer
    dev = torch.device("cuda")
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "capability": list(torch.cuda.get_device_capability()), "torch": torch.__version__, "flashinfer": flashinfer.__version__, "cases": []}
    gen = torch.Generator().manual_seed(20261010)
    for e_count, m_total, n, k, indptr in CASES:
        if indptr is None:
            cuts = sorted(torch.randint(0, m_total + 1, (e_count - 1,), generator=gen).tolist())
            indptr = tuple([0] + cuts + [m_total])
        rec = {"experts": e_count, "tokens": m_total, "n": n, "k": k, "m_indptr": list(indptr)}
        try:
            x = torch.randn(m_total, k, generator=gen).to(torch.bfloat16).float()
            w = torch.randn(e_count, n, k, generator=gen).to(torch.bfloat16).float()
            gs_a = global_scale_for(x)
            gs_b = global_scale_for(w)                      # one shared scale: the op takes one alpha
            ap_, asf, _ = quantize_nvfp4(x, gs_a)
            bq = [quantize_nvfp4(w[e], gs_b) for e in range(e_count)]
            bp = torch.stack([b[0] for b in bq]); bsf = torch.stack([b[1] for b in bq])
            exact, bound = oracle.exact_grouped_gemm_nvfp4(ap_, asf, gs_a, bp, bsf, gs_b, torch.tensor(indptr))
            a_sw = to_128x4(asf).view(torch.float8_e4m3fn).to(dev)
            b_sw = torch.stack([to_128x4(bsf[e]) for e in range(e_count)]).view(torch.float8_e4m3fn).to(dev)
            alpha = (1.0 / (gs_a.double() * gs_b.double())).float().reshape(1).to(dev)
            out = flashinfer.grouped_mm_fp4(ap_.to(dev), bp.to(dev), a_sw, b_sw, torch.tensor(indptr, dtype=torch.int32, device=dev),
                                            alpha=alpha, out_dtype=torch.bfloat16, block_size=16, backend="cudnn")
            torch.cuda.synchronize()
            rec.update(oracle.conformance(out.float(), exact, bound))
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
        report["cases"].append(rec)
        print(json.dumps(rec)[:500])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
