#!/usr/bin/env python3
"""Check the Triton re-layout kernel of the FlashInfer grouped block-scale fix against the PyTorch reference (bit for bit, many
random ragged routings including empty groups, NVFP4 and MXFP8 column counts), then time it in eager mode and under CUDA graph
replay against `grouped_mm_fp4` at MoE-like sizes. Run with the fix overlaid on the installed FlashInfer.

    PYTHONPATH=. python scripts/check_grouped_sfa_kernel.py --reference <torch_reference_helper.py> --out reports/<file>.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from scripts.bench_grouped_sfa_relayout2 import graph_ms, kernel_us, median_ms


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    ns = {}
    exec(open(a.reference, encoding="utf-8").read(), ns)
    ref = ns["_segment_token_descale_per_group"]
    import flashinfer
    from flashinfer import SfLayout
    from flashinfer.fp4_quantization import nvfp4_quantize
    from flashinfer.grouped_mm.cudnn.core import _segment_token_descale_per_group as kern
    dev = torch.device("cuda")
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "flashinfer": flashinfer.__version__}
    g = torch.Generator(device=dev).manual_seed(7)
    checked, mismatched = 0, 0
    for trial in range(300):
        num_groups = int(torch.randint(1, 300, (1,), generator=g, device=dev))
        cum_m = int(torch.randint(1, 6000, (1,), generator=g, device=dev))
        k_blocks = int([16, 32, 128, 256, 448][trial % 5])
        rows = (cum_m + 127) // 128 * 128
        cols = (k_blocks + 3) // 4 * 4
        a_sf = torch.randint(0, 255, (rows, cols), generator=g, device=dev, dtype=torch.uint8)
        dtype = torch.float8_e4m3fn if trial % 2 else torch.uint8
        a_sf = a_sf.view(dtype)
        cuts = torch.sort(torch.randint(0, cum_m + 1, (num_groups - 1,), generator=g, device=dev)).values
        m_indptr = torch.cat([torch.zeros(1, device=dev, dtype=torch.int64), cuts,
                              torch.full((1,), cum_m, device=dev, dtype=torch.int64)]).to(torch.int32)
        r1 = ref(a_sf, m_indptr, cum_m, k_blocks).view(torch.uint8)
        r2 = kern(a_sf, m_indptr, cum_m, k_blocks).view(torch.uint8)
        checked += 1
        if r1.shape != r2.shape or not torch.equal(r1, r2):
            mismatched += 1
            if mismatched <= 3:
                print("MISMATCH", num_groups, cum_m, k_blocks, tuple(r1.shape), tuple(r2.shape))
    report["bit_exact"] = {"trials": checked, "mismatched": mismatched}
    print(json.dumps(report["bit_exact"]))
    cases = []
    for e_count, tpe, n, k in ((8, 128, 4096, 4096), (8, 1024, 4096, 4096), (64, 128, 2048, 7168), (128, 64, 2048, 7168), (128, 512, 1536, 4096)):
        cum_m = e_count * tpe
        a_bf = torch.randn(cum_m, k, generator=g, device=dev, dtype=torch.bfloat16)
        b_bf = torch.randn(e_count * n, k, generator=g, device=dev, dtype=torch.bfloat16)
        gsa = (448 * 6) / a_bf.float().abs().max(); gsb = (448 * 6) / b_bf.float().abs().max()
        a_fp4, a_sf = nvfp4_quantize(a_bf, gsa, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
        b_fp4, b_sf = nvfp4_quantize(b_bf, gsb, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
        a_sf = a_sf.view(torch.float8_e4m3fn).reshape(-1, k // 16)
        b_fp4 = b_fp4.reshape(e_count, n, k // 2); b_sf = b_sf.view(torch.float8_e4m3fn).reshape(e_count, -1, k // 16)
        m_indptr = (torch.arange(e_count + 1, device=dev) * tpe).to(torch.int32)
        alpha = torch.tensor([1.0 / float(gsa * gsb)], device=dev)
        gemm = lambda: flashinfer.grouped_mm_fp4(a_fp4, b_fp4, a_sf, b_sf, m_indptr, alpha=alpha, out_dtype=torch.bfloat16, block_size=16)  # noqa: E731
        rel = lambda: kern(a_sf, m_indptr, cum_m, k // 16)  # noqa: E731
        rec = {"experts": e_count, "tokens_per_expert": tpe, "n": n, "k": k,
               "grouped_mm_fp4_with_fix_ms": round(median_ms(gemm), 4),
               "relayout_eager_ms": round(median_ms(rel), 4), "relayout_kernel_us": round(kernel_us(rel), 1),
               "relayout_graph_ms": round(graph_ms(rel), 4)}
        cases.append(rec)
        print(json.dumps(rec))
    report["cost"] = cases
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0 if mismatched == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
