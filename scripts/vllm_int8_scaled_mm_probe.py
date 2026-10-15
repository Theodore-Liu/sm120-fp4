#!/usr/bin/env python3
"""vLLM 0.28.0's INT8 W8A8 path on an SM120 device: (1) what the kernel chooser selects at compute capability 120 and
what the selected CUTLASS op does when called; (2) the Triton kernel next in its list (`triton_scaled_mm`, the function
`TritonInt8ScaledMMLinearKernel` calls) swept at shapes outside its test enumeration against the exact oracle.

Registered before any sweep case ran: mlsys-5090-research docs/d1-vllm-int8-scaled-mm-registration-2026-10-15.md.

    PYTHONPATH=<sm120-fp4> python vllm_int8_scaled_mm_probe.py --log <marker> --out <json>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

U32 = 2.0 ** -23


def ulp(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The output dtype's spacing at |x|, floored at the subnormal spacing (fp16: 2^-24 below 2^-14; bf16: 2^-133).
    The first run of this sweep (2026-10-15) had no floor and so read 152 fp16 cases with outputs in (1e-5, 6e-5),
    the fp16 subnormal range, as outside a bound that was half the true spacing; an instrument fault, amended."""
    mant = 7 if dtype == torch.bfloat16 else 10
    min_exp = -126 if dtype == torch.bfloat16 else -14
    e = torch.floor(torch.log2(x.clamp(min=2.0 ** -140))).clamp(min=min_exp)
    return torch.ldexp(torch.ones_like(x), e - mant)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-cases", type=int, default=0)
    a = ap.parse_args(argv)

    def mark(s: str) -> None:
        with open(a.log, "a", encoding="utf-8") as f:
            f.write(time.strftime("%H:%M:%S") + " int8 " + s + "\n")

    import vllm
    from vllm import _custom_ops as ops
    import vllm.model_executor.kernels.linear as kl
    from vllm.model_executor.kernels.linear import Int8ScaledMMLinearLayerConfig, choose_scaled_mm_linear_kernel
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import triton_scaled_mm

    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "capability": list(torch.cuda.get_device_capability()), "torch": torch.__version__, "vllm": vllm.__version__,
              "chooser": [], "cutlass_calls": {}, "cases": []}
    gen = torch.Generator().manual_seed(20261015)

    # ---- (1) the chooser at compute capability 120, and the chosen op called on this device
    report["chooser_candidates"] = [k.__name__ for k in kl._POSSIBLE_INT8_KERNELS[kl.current_platform._enum]]
    for ch, st, sym in ((True, False, True), (True, False, False), (False, True, True), (False, True, False)):
        c = Int8ScaledMMLinearLayerConfig(is_channelwise=ch, is_static_input_scheme=st, input_symmetric=sym)
        k = choose_scaled_mm_linear_kernel(c, kl._POSSIBLE_INT8_KERNELS, compute_capability=120)
        report["chooser"].append({"is_channelwise": ch, "is_static_input_scheme": st, "input_symmetric": sym, "chosen": k.__name__})
    M0, K0, N0 = 16, 64, 128
    a0 = torch.randint(-128, 128, (M0, K0), generator=gen).to(torch.int8).cuda()
    w0 = torch.randint(-128, 128, (N0, K0), generator=gen).to(torch.int8).cuda().t()
    sa0 = (torch.rand((M0, 1), generator=gen) * 0.01).cuda(); sb0 = (torch.rand((N0, 1), generator=gen) * 0.01).cuda()
    try:
        ops.cutlass_scaled_mm(a0, w0, scale_a=sa0, scale_b=sb0, out_dtype=torch.bfloat16)
        report["cutlass_calls"]["cutlass_scaled_mm"] = "served"
    except Exception as exc:  # noqa: BLE001
        report["cutlass_calls"]["cutlass_scaled_mm"] = type(exc).__name__ + ": " + str(exc)[:300]
    try:
        azp_adj = w0.sum(dim=0, keepdim=True, dtype=torch.int32)
        azp = torch.randint(-128, 128, (M0, 1), generator=gen).to(torch.int32).cuda()
        ops.cutlass_scaled_mm_azp(a0, w0, scale_a=sa0, scale_b=sb0, out_dtype=torch.bfloat16, azp_adj=azp_adj, azp=azp)
        report["cutlass_calls"]["cutlass_scaled_mm_azp"] = "served"
    except Exception as exc:  # noqa: BLE001
        report["cutlass_calls"]["cutlass_scaled_mm_azp"] = type(exc).__name__ + ": " + str(exc)[:300]
    print(json.dumps({"chooser": report["chooser"], "cutlass_calls": report["cutlass_calls"]}), flush=True)

    # ---- (2) the Triton kernel against the exact oracle
    def run_case(label, dtype, scale_kind, form, with_bias, m, n, k, use_td=None, runs=2):
        rec = {"label": label, "dtype": str(dtype).split(".")[-1], "scales": scale_kind, "weight_form": form, "bias": with_bias,
               "m": m, "n": n, "k": k, "use_td": use_td}
        try:
            A = torch.randint(-128, 128, (m, k), generator=gen).to(torch.int8).cuda()
            if form == "kn-contiguous":
                W = torch.randint(-128, 128, (k, n), generator=gen).to(torch.int8).cuda()
            else:                                   # the layer's form: the transposed view of an [N, K] buffer
                W = torch.randint(-128, 128, (n, k), generator=gen).to(torch.int8).cuda().t()
            if scale_kind == "tensor-tensor":
                sa = (torch.rand((1, 1), generator=gen) * 0.02 + 1e-4).cuda(); sb = (torch.rand((1, 1), generator=gen) * 0.02 + 1e-4).cuda()
            else:
                sa = (torch.rand((m, 1), generator=gen) * 0.02 + 1e-4).cuda(); sb = (torch.rand((n, 1), generator=gen) * 0.02 + 1e-4).cuda()
            bias = (torch.randn((n,), generator=gen) * 0.5).to(dtype).cuda() if with_bias else None
            # exact oracle in float64 (the integer sum is exact: |acc| < 2^27)
            acc = A.double() @ W.double()
            v0 = acc * sa.double() * sb.double().T
            v = v0 + (bias.double()[None, :] if with_bias else 0.0)
            slack = 3 * U32 * v0.abs()
            bnd = slack + 0.5 * ulp((v0.abs() + slack).float(), dtype).double()
            if with_bias:
                bnd = bnd + 0.5 * ulp(v.abs().float(), dtype).double()
            # the epilogue model read from source: fp32(acc) * sa, * sb, cast, then + bias in the output dtype
            model = ((acc.float() * sa) * sb.T).to(dtype)
            if with_bias:
                model = model + bias[None, :]
            model = model.double()
            outs = []
            for _ in range(runs):
                mark(label + " " + rec["dtype"] + " " + scale_kind + " " + form + " bias=" + str(with_bias) + " m=" + str(m) + " n=" + str(n) + " k=" + str(k))
                kw = {} if use_td is None else {"use_td": use_td}
                out = triton_scaled_mm(A, W, sa, sb, dtype, bias, **kw)
                torch.cuda.synchronize()
                outs.append(out.double())
            err = (outs[0] - v).abs()
            rec["elements"] = int(err.numel()); rec["outside"] = int((err > bnd).sum()); rec["nan"] = int(torch.isnan(outs[0]).sum())
            rec["max_err_over_bound"] = float((err / bnd).max()); rec["max_abs_err"] = float(err.max())
            rec["bit_exact_vs_model"] = round(float(((outs[0] - model) == 0).double().mean()), 5)
            rec["distinct_outputs"] = len({o.cpu().numpy().tobytes() for o in outs})
            rec["max_abs_acc"] = float(acc.abs().max())
            if rec["outside"]:
                bad = (err > bnd).nonzero()[:4].tolist()
                rec["examples"] = [{"row": r, "col": c, "got": float(outs[0][r, c]), "exact": float(v[r, c]), "model": float(model[r, c])} for r, c in bad]
        except Exception as exc:  # noqa: BLE001
            rec["error"] = type(exc).__name__ + ": " + str(exc)[:300]
        return rec

    # instrument check: the test's own shape, and the oracle's exactness against an int64 CPU matmul
    Ai = torch.randint(-128, 128, (32, 4224), generator=gen).to(torch.int8); Wi = torch.randint(-128, 128, (4224, 128), generator=gen).to(torch.int8)
    exact_int = Ai.long() @ Wi.long(); exact_f64 = Ai.double() @ Wi.double()
    report["oracle_exact_vs_int64"] = bool((exact_int.double() == exact_f64).all())
    chk = run_case("test-shape", torch.bfloat16, "token-channel", "kn-contiguous", False, 32, 128, 256)
    report["instrument_check"] = chk
    print(json.dumps(chk)[:400], flush=True)
    if not report["oracle_exact_vs_int64"] or "error" in chk or chk["outside"]:
        raise SystemExit("instrument check failed on the test's own shape; no verdict")

    plan = []
    for dtype in (torch.bfloat16, torch.float16):
        for scale_kind in ("tensor-tensor", "token-channel"):
            for form in ("kn-contiguous", "nk-transposed"):
                for with_bias in (False, True):
                    for m in (1, 7, 33, 129, 1000):
                        for n in (8, 24, 40, 1000, 4160, 8200):
                            for k in (16, 40, 96, 1152, 4224):
                                plan.append(("edge", dtype, scale_kind, form, with_bias, m, n, k, None))
    for form in ("kn-contiguous", "nk-transposed"):
        for k in (1152, 4224):
            plan.append(("no-td", torch.bfloat16, "token-channel", form, True, 33, 4160, k, False))

    for i, p in enumerate(plan):
        if a.max_cases and i >= a.max_cases:
            break
        rec = run_case(*p)
        report["cases"].append(rec)
        flag = rec.get("outside") or rec.get("nan") or "error" in rec or rec.get("distinct_outputs", 1) != 1
        if flag or (i + 1) % 200 == 0:
            print(json.dumps(rec)[:420], flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    judged = [c for c in report["cases"] if "outside" in c]
    print("cases " + str(len(report["cases"])) + ", judged " + str(len(judged)) + ", outside in " + str(sum(1 for c in judged if c["outside"]))
          + ", nan in " + str(sum(1 for c in judged if c["nan"])) + ", nondeterministic " + str(sum(1 for c in judged if c.get("distinct_outputs", 1) != 1))
          + ", refused/errors " + str(sum(1 for c in report["cases"] if "error" in c)) + ", elements " + str(sum(c["elements"] for c in judged))
          + ", min bit-exact vs model " + str(min((c["bit_exact_vs_model"] for c in judged), default=None)) + "; wrote " + str(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
