#!/usr/bin/env python3
"""vLLM 0.28's dense SM120 GEMMs at shapes the contract allows but earlier census rows did not reach, against the exact oracle:
`cutlass_scaled_fp4_mm` (NVFP4) and `cutlass_scaled_mm` (FP8, per-tensor / per-row / 128-block scales, optional bias).

The contract, read at the v0.28.0 tag:
  - `cutlass_scaled_fp4_mm` (`csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu`): k % 32 == 0 and
    n % 32 == 0, any m; scales padded and swizzled to (round_up(m, 128), round_up(k/16, 4)) and the same for n.
  - `cutlass_scaled_mm` (`vllm/_custom_ops.py`): when b's two dimensions are multiples of 16 the CUTLASS kernel runs (row-major
    a, column-major b, c and b strides 16-element aligned: `scaled_mm_entry.cu`); otherwise the call falls back to
    `triton_scaled_mm` (per-tensor or per-row scales only). Block scales: scale_a.shape * [1, 128] == a.shape and
    scale_b.shape * [128, 128] == b.shape, so k and n multiples of 128; vLLM's own caller passes scale_a column-major and
    scale_b as the transposed view of (n/128, k/128) (the census's block row reads it that way). Bias: n elements, out dtype.
Cases: NVFP4 m in {1, 3, 7, 65, 129, 300} x n, k in {32, 96, 160, 1056, 4128} with the census's reference quantiser; FP8 with
m in the same list and (n, k) off 128 on the CUTLASS path, (n, k) off 16 on the Triton path, block scales at (n, k) in
{(128, 128), (384, 1152)}; per-row with bias on a subset. Each valid element against the exact product (float64) under the
census bound; two runs per case; a marker line before every call.

    PYTHONPATH=<sm120-fp4> python scripts/vllm_edge_probe.py --log <marker file> --out reports/<file>.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

M_LIST = (1, 3, 7, 65, 129, 300)
FP4_NK = (32, 96, 160, 1056, 4128)
FP8_CUTLASS_NK = ((16, 16), (48, 144), (144, 48), (1040, 1040), (4112, 144))
FP8_TRITON_NK = ((8, 16), (24, 144), (1000, 1040), (16, 24), (136, 40))
FP8_BLOCK_NK = ((128, 128), (384, 1152))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    def mark(s: str) -> None:
        with open(a.log, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} vllm-edge {s}\n")

    import vllm
    import vllm._custom_ops as ops
    from sm120fp4 import oracle
    from sm120fp4.layouts import to_128x4
    from sm120fp4.reference import quantize_nvfp4
    dev = torch.device("cuda")
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "vllm": vllm.__version__, "cases": []}

    def run(rec, fn, exact, bound):
        try:
            outs = []
            for _ in range(2):
                mark(f"{rec['op']} {rec.get('scales', '')} m={rec['m']} n={rec['n']} k={rec['k']}")
                o = fn()
                torch.cuda.synchronize()
                outs.append(o.double().cpu())
            err = (outs[0] - exact).abs()
            rec.update({"elements": int(err.numel()), "outside": int((err > bound).sum()), "nan": int(torch.isnan(outs[0]).sum()),
                        "max_err_over_bound": float((err / bound.clamp(min=1e-300)).max()),
                        "distinct_outputs": len({t.numpy().tobytes() for t in outs})})
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        report["cases"].append(rec)
        if rec.get("outside") or rec.get("nan") or rec.get("distinct_outputs", 1) != 1 or "error" in rec:
            print(json.dumps(rec)[:400], flush=True)

    def fp8_bound(sabs, ex, k):
        pre = k * oracle.U32 * sabs + 2 * oracle.U32 * ex.abs() + 2 * oracle.U64 * sabs
        return pre + 0.5 * oracle._ulp(ex.abs() + pre, torch.bfloat16)

    # ---------------------------------------------------------------- NVFP4
    for m in M_LIST:
        for n in FP4_NK:
            for k in FP4_NK:
                g = torch.Generator().manual_seed(20261012 + 11 * m + 3 * n + k)
                aq, asf, ags = quantize_nvfp4(torch.randn(m, k, generator=g).to(torch.bfloat16).float())
                bq, bsf, bgs = quantize_nvfp4(torch.randn(n, k, generator=g).to(torch.bfloat16).float())
                ex, bd = oracle.exact_gemm_nvfp4(aq, asf, ags, bq, bsf, bgs, out_dtype=torch.bfloat16)
                pre = bd - 0.5 * oracle._ulp(ex, torch.bfloat16)
                bd = pre + 0.5 * oracle._ulp(ex.abs() + pre, torch.bfloat16)        # half ulp at the top of the interval
                alpha = (1.0 / (ags.double() * bgs.double())).float().reshape(1).to(dev)
                a_d, b_d = aq.to(dev), bq.to(dev)
                sa = to_128x4(asf).view(torch.float8_e4m3fn).to(dev)
                sb = to_128x4(bsf).view(torch.float8_e4m3fn).to(dev)
                run({"op": "cutlass_scaled_fp4_mm", "m": m, "n": n, "k": k},
                    lambda: ops.cutlass_scaled_fp4_mm(a_d, b_d, sa, sb, alpha, torch.bfloat16), ex, bd)

    # ---------------------------------------------------------------- FP8
    e4m3 = torch.float8_e4m3fn
    for path, nk_list in (("cutlass", FP8_CUTLASS_NK), ("triton-fallback", FP8_TRITON_NK)):
        for n, k in nk_list:
            for m in M_LIST:
                g = torch.Generator().manual_seed(20261014 + 13 * m + 5 * n + k)
                a_q = (torch.randn(m, k, generator=g) * 2).to(e4m3)
                w_q = torch.randn(n, k, generator=g).to(e4m3)
                ad, wd = a_q.double(), w_q.double()
                p = ad @ wd.T
                pabs = ad.abs() @ wd.abs().T
                a_d, w_d = a_q.to(dev), w_q.to(dev)
                st_a = (torch.rand(1, 1, generator=g) * 0.02 + 0.005).float()
                st_b = (torch.rand(1, 1, generator=g) * 0.02 + 0.005).float()
                sr_a = (torch.rand(m, 1, generator=g) * 0.02 + 0.005).float()
                sr_b = (torch.rand(n, 1, generator=g) * 0.02 + 0.005).float()
                bias = (torch.randn(n, generator=g) * 0.1).to(torch.bfloat16)
                for scales in ("tensor", "row", "row+bias"):
                    if scales == "tensor":
                        f = st_a.double().reshape(()) * st_b.double().reshape(())
                        ex, sabs = p * f, pabs * f
                        fn = lambda: ops.cutlass_scaled_mm(a_d, w_d.t(), st_a.to(dev), st_b.to(dev), torch.bfloat16)  # noqa: E731
                    else:
                        f = sr_a.double() * sr_b.double().reshape(1, -1)
                        ex, sabs = p * f, pabs * f
                        if scales == "row+bias":
                            ex = ex + bias.double().reshape(1, -1)
                            sabs = sabs + bias.double().abs().reshape(1, -1)
                            fn = lambda: ops.cutlass_scaled_mm(a_d, w_d.t(), sr_a.to(dev), sr_b.to(dev), torch.bfloat16,  # noqa: E731
                                                               bias=bias.to(dev))
                        else:
                            fn = lambda: ops.cutlass_scaled_mm(a_d, w_d.t(), sr_a.to(dev), sr_b.to(dev), torch.bfloat16)  # noqa: E731
                    run({"op": "cutlass_scaled_mm", "path": path, "scales": scales, "m": m, "n": n, "k": k}, fn, ex,
                        fp8_bound(sabs, ex, k))
    for n, k in FP8_BLOCK_NK:
        for m in M_LIST:
            g = torch.Generator().manual_seed(20261015 + 13 * m + 5 * n + k)
            a_q = (torch.randn(m, k, generator=g) * 2).to(e4m3)
            w_q = torch.randn(n, k, generator=g).to(e4m3)
            sa_blk = (torch.rand(m, k // 128, generator=g) * 0.02 + 0.005).float()
            sw_blk = (torch.rand(n // 128, k // 128, generator=g) * 0.02 + 0.005).float()
            sa = sa_blk.repeat_interleave(128, dim=1)
            sw = sw_blk.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
            ad, wd = a_q.double() * sa.double(), w_q.double() * sw.double()
            ex = ad @ wd.T
            sabs = ad.abs() @ wd.abs().T
            a_d, w_d = a_q.to(dev), w_q.to(dev)
            sa_d = sa_blk.t().contiguous().t().to(dev)          # column-major, as vLLM's QuantFP8 emits it
            sw_d = sw_blk.to(dev).t()                           # the transposed view of (n/128, k/128)
            run({"op": "cutlass_scaled_mm", "path": "cutlass", "scales": "block128", "m": m, "n": n, "k": k},
                lambda: ops.cutlass_scaled_mm(a_d, w_d.t(), sa_d, sw_d, torch.bfloat16), ex, fp8_bound(sabs, ex, k))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    by = {}
    for c in report["cases"]:
        key = (c["op"], c.get("path", "-"), c.get("scales", "-"))
        s = by.setdefault(key, {"judged": 0, "bad_cases": 0, "errors": 0})
        if "outside" in c:
            s["judged"] += 1
            s["bad_cases"] += int(c["outside"] > 0 or c["nan"] > 0 or c["distinct_outputs"] != 1)
        else:
            s["errors"] += 1
    for key, s in sorted(by.items()):
        print(key, s)
    print(f"cases {len(report['cases'])}; wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
