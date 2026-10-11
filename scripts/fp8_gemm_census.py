#!/usr/bin/env python3
"""Dense FP8 (e4m3) GEMM paths on SM120 against an exact oracle: torch's `_scaled_mm` (cuBLASLt), vLLM's `cutlass_scaled_mm`
(CUTLASS, per-tensor, per-token x per-channel and 128-blockwise scales), FlashInfer's `mm_fp8` (TRT-LLM low-latency) and
FlashInfer's `bmm_fp8` on each backend it names (cublas, cudnn, cutlass, cute-dsl), per-tensor scales.

The oracle is exact: an e4m3 value is exact in fp64, so the product of the quantised operands summed in fp64 and multiplied by
the fp32 scales (exact in fp64) is the value a correct kernel approximates. The bound is the fp32-accumulation bound used across
the census (K * 2^-24 * sum |a b| scaled), plus two fp32 roundings for the epilogue's scale multiplies, plus half an ulp of the
output dtype, plus the fp64 arithmetic's own slack. Every element is judged; no tolerance.

Operands are drawn on the CPU with a fixed seed: a (m, k) e4m3, w (n, k) e4m3, scales positive fp32. The scale kinds:
  tensor   one scale per operand
  rowwise  a: one scale per row (m, 1); w: one scale per output column (1, n)
  block    a: one scale per (1, 128) group, column-major as vLLM's QuantFP8 writes it; w: one scale per (128, 128) block (vLLM only; DeepSeek-V3 style)

    PYTHONPATH=/path/to/sm120-fp4 python scripts/fp8_gemm_census.py --out reports/fp8-gemm-<device>-<env>-<date>.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sm120fp4 import oracle

SHAPES = ((1, 4096, 4096), (16, 2048, 7168), (128, 4096, 1024), (256, 1024, 2048), (512, 512, 512))  # (m, n, k)


def exact_fp8(a_q: torch.Tensor, w_q: torch.Tensor, sa: torch.Tensor, sw: torch.Tensor, k: int, out_dtype: torch.dtype, kind: str):
    """a_q (m, k), w_q (n, k) e4m3. kind 'tensor': sa, sw one element; 'rowwise': sa (m, 1), sw (1, n); 'elementwise': sa (m, k), sw (n, k).
    fp64 throughout (e4m3 values and fp32 scales are exact in fp64)."""
    ad, wd = a_q.double(), w_q.double()
    sad, swd = sa.double(), sw.double()
    if kind == "elementwise":
        ad = ad * sad
        wd = wd * swd
        ex = ad @ wd.t()
        sabs = ad.abs() @ wd.abs().t()
    elif kind == "rowwise":
        ex = (ad @ wd.t()) * sad.reshape(-1, 1) * swd.reshape(1, -1)
        sabs = (ad.abs() @ wd.abs().t()) * sad.reshape(-1, 1) * swd.reshape(1, -1)
    elif kind == "tensor":
        ex = (ad @ wd.t()) * sad.reshape(()) * swd.reshape(())
        sabs = (ad.abs() @ wd.abs().t()) * sad.reshape(()) * swd.reshape(())
    else:
        raise ValueError(kind)
    bound = k * oracle.U32 * sabs + 2 * oracle.U32 * ex.abs() + 0.5 * oracle._ulp(ex, out_dtype) + 2 * oracle.U64 * ex.abs()
    return ex, bound


def expand_block_scales(sa_blk: torch.Tensor, sw_blk: torch.Tensor, m: int, n: int, k: int, g: int = 128):
    sa = sa_blk.repeat_interleave(g, dim=1)[:, :k]                       # (m, k)
    sw = sw_blk.repeat_interleave(g, dim=0)[:n].repeat_interleave(g, dim=1)[:, :k]  # (n, k)
    return sa, sw

def flashinfer_case(m: int, n: int, k: int, out_path: str) -> None:
    """One mm_fp8 call with the same seeded operands as the parent; the output is saved for the parent to judge."""
    import flashinfer
    dev = torch.device("cuda")
    a_q, w_q, sa_t, sw_t = operands(m, n, k)
    cache = {}
    w_prep = flashinfer.prepare_low_latency_gemm_weights(w_q.to(dev), cache)
    alpha = (sa_t.double() * sw_t.double()).float().to(dev)
    out = flashinfer.mm_fp8(a_q.to(dev), w_prep, alpha=alpha, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    torch.save(out.cpu(), out_path)


def flashinfer_child(m: int, n: int, k: int, exact: torch.Tensor, bound: torch.Tensor) -> dict:
    import subprocess
    import sys
    import tempfile
    rec = {"path": "flashinfer mm_fp8 trtllm_low_latency", "m": m, "n": n, "k": k}
    with tempfile.TemporaryDirectory() as d:
        out_path = str(Path(d) / "out.pt")
        proc = subprocess.run([sys.executable, __file__, "--child-flashinfer", f"{m},{n},{k},{out_path}"], capture_output=True, text=True, timeout=1800)
        rec["child_returncode"] = proc.returncode
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout)[-600:]
            rec["error"] = f"child exited {proc.returncode} (-11 is SIGSEGV): {tail}"
        else:
            out = torch.load(out_path)
            rec.update(oracle.conformance(out.float(), exact, bound))
    print(json.dumps(rec)[:300])
    return rec


def operands(m: int, n: int, k: int):
    """The seeded operands of one shape; the seed is per shape so the child process draws the same ones."""
    gen = torch.Generator().manual_seed(20261011 + 1000003 * m + 1009 * n + k)
    e4m3 = torch.float8_e4m3fn
    a_q = (torch.randn(m, k, generator=gen) * 2).to(e4m3)
    w_q = torch.randn(n, k, generator=gen).to(e4m3)
    sa_t = (torch.rand(1, generator=gen) * 0.02 + 0.005).float()
    sw_t = (torch.rand(1, generator=gen) * 0.02 + 0.005).float()
    return a_q, w_q, sa_t, sw_t


def run_case(fn, name, rec_list, m, n, k, out_dtype, exact, bound):
    rec = {"path": name, "m": m, "n": n, "k": k}
    try:
        out = fn()
        torch.cuda.synchronize()
        rec.update(oracle.conformance(out.float(), exact, bound))
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    rec_list.append(rec)
    print(json.dumps(rec)[:300])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-flashinfer", action="store_true", help="skip the mm_fp8 child (segfaults on 0.6.16)")
    ap.add_argument("--bmm-backends", default="cublas,cudnn,cute-dsl",
                    help="bmm_fp8 backends to read; 'cutlass' is left out by default because on the RTX 5090 under WSL2 it took the "
                         "WSL virtual machine down on every call (2026-10-11, scripts/bmm_fp8_isolate.py), which no in-process guard can catch")
    ap.add_argument("--child-flashinfer", type=str, default=None, help="internal: run one mm_fp8 case and print its output tensor path")
    ap.add_argument("--out", type=Path, default=Path("reports") / f"fp8-gemm-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    a = ap.parse_args(argv)
    if a.child_flashinfer:
        m_, n_, k_, path_ = a.child_flashinfer.split(",", 3)
        flashinfer_case(int(m_), int(n_), int(k_), path_)
        return 0
    dev = torch.device("cuda")
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "cases": []}
    try:
        import vllm
        import vllm._custom_ops as ops
        report["vllm"] = vllm.__version__
    except Exception as exc:  # noqa: BLE001
        ops = None
        report["vllm"] = f"unavailable: {type(exc).__name__}"
    try:
        import flashinfer
        report["flashinfer"] = flashinfer.__version__
    except Exception as exc:  # noqa: BLE001
        flashinfer = None
        report["flashinfer"] = f"unavailable: {type(exc).__name__}"
    for m, n, k in SHAPES:
        a_q, w_q, sa_t, sw_t = operands(m, n, k)
        gen = torch.Generator().manual_seed(7 + m + n + k)
        sa_r = (torch.rand(m, 1, generator=gen) * 0.02 + 0.005).float()
        sw_r = (torch.rand(1, n, generator=gen) * 0.02 + 0.005).float()
        a_d, w_d = a_q.to(dev), w_q.to(dev)
        # torch._scaled_mm: b is (k, n) column-major, i.e. the transposed view of a row-major (n, k)
        ex_t, bd_t = exact_fp8(a_q, w_q, sa_t, sw_t, k, torch.bfloat16, "tensor")
        run_case(lambda: torch._scaled_mm(a_d, w_d.t(), scale_a=sa_t.to(dev), scale_b=sw_t.to(dev), out_dtype=torch.bfloat16),
                 "torch._scaled_mm tensor", report["cases"], m, n, k, torch.bfloat16, ex_t, bd_t)
        ex_r, bd_r = exact_fp8(a_q, w_q, sa_r, sw_r, k, torch.bfloat16, "rowwise")
        run_case(lambda: torch._scaled_mm(a_d, w_d.t(), scale_a=sa_r.to(dev), scale_b=sw_r.to(dev), out_dtype=torch.bfloat16),
                 "torch._scaled_mm rowwise", report["cases"], m, n, k, torch.bfloat16, ex_r, bd_r)
        if ops is not None:
            run_case(lambda: ops.cutlass_scaled_mm(a_d, w_d.t(), sa_t.reshape(1, 1).to(dev), sw_t.reshape(1, 1).to(dev), torch.bfloat16),
                     "vllm cutlass_scaled_mm tensor", report["cases"], m, n, k, torch.bfloat16, ex_t, bd_t)
            run_case(lambda: ops.cutlass_scaled_mm(a_d, w_d.t(), sa_r.to(dev), sw_r.to(dev), torch.bfloat16),
                     "vllm cutlass_scaled_mm rowwise", report["cases"], m, n, k, torch.bfloat16, ex_r, bd_r)
            if k % 128 == 0 and n % 128 == 0:
                sa_blk = (torch.rand(m, k // 128, generator=gen) * 0.02 + 0.005).float()
                sw_blk = (torch.rand(n // 128, k // 128, generator=gen) * 0.02 + 0.005).float()
                sa_full, sw_full = expand_block_scales(sa_blk, sw_blk, m, n, k)
                ex_b, bd_b = exact_fp8(a_q, w_q, sa_full, sw_full, k, torch.bfloat16, "elementwise")
                # vLLM's own call (CutlassFp8BlockScaledMMKernel): scale_a (m, k/128) COLUMN-major (QuantFP8 column_major_scales=True), scale_b = Bs.T, the transposed view of (n/128, k/128); a row-major scale_a is read wrong for every m > 1
                run_case(lambda: ops.cutlass_scaled_mm(a_d, w_d.t(), sa_blk.t().contiguous().t().to(dev), sw_blk.to(dev).t(), torch.bfloat16),
                         "vllm cutlass_scaled_mm block128", report["cases"], m, n, k, torch.bfloat16, ex_b, bd_b)
        if flashinfer is not None and not a.no_flashinfer:
            report["cases"].append(flashinfer_child(m, n, k, ex_t, bd_t))
        if flashinfer is not None and hasattr(flashinfer, "bmm_fp8"):
            for backend in [b for b in a.bmm_backends.split(",") if b]:
                def bmm(backend=backend):
                    # bmm_fp8: A (b, m, k) row-major, B (b, k, n) column-major, per-tensor float scales, bf16 out
                    return flashinfer.bmm_fp8(a_d[None], w_d.t()[None], sa_t.to(dev), sw_t.to(dev), torch.bfloat16, backend=backend)[0]
                run_case(bmm, f"flashinfer bmm_fp8 {backend}", report["cases"], m, n, k, torch.bfloat16, ex_t, bd_t)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
