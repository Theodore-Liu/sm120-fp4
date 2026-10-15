#!/usr/bin/env python3
"""vLLM's FP8 activation quantiser (`ops.scaled_fp8_quant`) on an SM120 device at hidden sizes, token counts, dtypes,
strides and row contents its test does not enumerate, judged bit for bit against a step model of the kernel's arithmetic
(read from csrc/quantization/w8a8/fp8/common.cu and common.cuh).

Registered before any case ran: mlsys-5090-research docs/d1-vllm-scaled-fp8-quant-registration-2026-10-15.md.

    PYTHONPATH=<sm120-fp4> python vllm_scaled_fp8_quant_probe.py --log <marker> --out <json>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
MIN_SCALE = 1.0 / (448.0 * 512.0)


def e4m3_rne_reference(x: torch.Tensor) -> torch.Tensor:
    """Hand-rolled round-to-nearest-even into e4m3fn for |x| <= 448 (fp32 in, uint8 bit pattern out)."""
    xd = x.double()
    sign = torch.signbit(xd).to(torch.uint8)       # -0.0 keeps its sign bit (0x80), as the hardware cvt does
    a = xd.abs()
    # e4m3fn: bias 7, 3 mantissa bits; min normal 2^-6; subnormal spacing 2^-9
    def exp2_floor(v: torch.Tensor) -> torch.Tensor:
        # exact floor(log2 v) through frexp (log2 of an exact power of two can land a hair below the integer)
        _, ex = torch.frexp(v.clamp(min=2.0 ** -20))
        return (ex - 1).to(torch.float64)
    e = exp2_floor(a).clamp(min=-6.0)
    spacing = torch.ldexp(torch.ones_like(e), (e - 3).to(torch.int32))   # exact; CUDA's double pow(2, -4) came out inexact
    q = torch.round(a / spacing) * spacing           # RNE on the quantised grid (torch.round is half to even)
    # re-derive exponent after rounding (a round-up can carry into the next binade)
    e2 = exp2_floor(q).clamp(min=-6.0)
    normal = q >= 2.0 ** -6
    mant = torch.where(normal, q / torch.ldexp(torch.ones_like(e2), e2.to(torch.int32)) - 1.0, q / (2.0 ** -6))   # subnormal: exponent field 0
    mbits = torch.round(mant * 8).to(torch.int64)
    ebits = torch.where(normal, (e2 + 7).to(torch.int64), torch.zeros_like(mbits))
    bits = (sign.to(torch.int64) << 7) | (ebits << 3) | mbits
    bits = torch.where(a == 0, sign.to(torch.int64) << 7, bits)
    return bits.to(torch.uint8)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-cases", type=int, default=0)
    a = ap.parse_args(argv)

    def mark(s: str) -> None:
        with open(a.log, "a", encoding="utf-8") as f:
            f.write(time.strftime("%H:%M:%S") + " fp8quant " + s + "\n")

    import vllm
    from vllm import _custom_ops as ops

    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "capability": list(torch.cuda.get_device_capability()), "torch": torch.__version__, "vllm": vllm.__version__, "cases": []}
    gen = torch.Generator().manual_seed(20261015)
    dev = "cuda"
    f32 = lambda v: torch.tensor(v, dtype=torch.float32, device=dev)  # noqa: E731

    # ---- cast check: torch's e4m3fn cast against the hand-rolled RNE on 2^20 values in (-448, 448)
    xs = (torch.rand((1 << 20,), generator=gen) * 2 - 1).cuda() * 448.0
    xs = torch.cat([xs, torch.tensor([0.0, -0.0, 2.0 ** -9, 2.0 ** -9 * 1.5, 2.0 ** -6, 448.0, -448.0, 0.0625 * 1.0625, 1.0 + 1.0 / 16], device=dev)])
    tcast = xs.to(FP8).view(torch.uint8)
    ref = e4m3_rne_reference(xs)
    report["cast_check"] = {"values": int(xs.numel()), "mismatches": int((tcast != ref).sum())}
    if report["cast_check"]["mismatches"]:
        bad = (tcast != ref).nonzero().flatten()[:5]
        report["cast_check"]["examples"] = [{"x": float(xs[i]), "torch": int(tcast[i]), "ref": int(ref[i])} for i in bad]
    print(json.dumps(report["cast_check"]), flush=True)

    def convert(xf: torch.Tensor, inv: torch.Tensor, divide: bool = False) -> torch.Tensor:
        """scaled_fp8_conversion: x * inv (is_scale_inverted, the per-tensor dynamic and static kernels) or x / scale
        (the per-token kernel calls the <false> instantiation with token_scale), clamp to +-448, cvt rne satfinite.
        Returns uint8 bit patterns. The first run modelled every kernel as the reciprocal product and the per-token
        kernel read one step off on 5,665 of 2.4e8 diag elements; the division fits it with 0 off (diag, 2026-10-15)."""
        x = xf / inv if divide else xf * inv
        # fminf(NaN, 448) is 448 and fmaxf(448, -448) is 448: a NaN product (0 * inf, the all-zero tensor under the
        # per-tensor dynamic scale 0) leaves the clamp as +448, the format's largest finite value, not as NaN
        r = torch.where(torch.isnan(x), f32(FP8_MAX).expand_as(x), torch.maximum(torch.minimum(x, f32(FP8_MAX)), f32(-FP8_MAX)))
        return r.to(FP8).view(torch.uint8)

    def model(x: torch.Tensor, mode: str, scale=None, scale_ub=None, group=None):
        xf = x.float()
        if mode == "dyn-tensor":
            s = (xf.abs().amax() / f32(FP8_MAX)).reshape(1)
            inv = f32(1.0) / s
            return convert(xf, inv), s
        if mode in ("dyn-token", "dyn-token-ub"):
            am = xf.abs().amax(dim=-1, keepdim=True)
            if scale_ub is not None:
                am = torch.minimum(am, scale_ub.float())
            s = torch.maximum(am / f32(FP8_MAX), f32(MIN_SCALE))
            return convert(xf, s, divide=True), s
        if mode == "static-tensor":
            inv = f32(1.0) / scale.float()
            return convert(xf, inv), scale
        if mode == "static-group":
            gm, gn = group
            inv = f32(1.0) / scale.float()                       # [T/gm, H/gn]
            inv_full = inv.repeat_interleave(gm, 0).repeat_interleave(gn, 1)
            return convert(xf, inv_full), scale
        raise ValueError(mode)

    def make_rows(tokens, hidden, dtype, special):
        x = (torch.randn((tokens, hidden), generator=gen) * 3).to(dtype)
        if special:
            rows = [0, min(1, tokens - 1), tokens - 1]
            kinds = ["zero", "single", "max-last"]
            for i, r in enumerate(rows):
                k = kinds[(i + hidden) % 3]
                if k == "zero":
                    x[r] = 0
                elif k == "single":
                    x[r] = 0; x[r, hidden // 2] = torch.tensor(2.5, dtype=dtype)
                else:
                    x[r, -1] = torch.tensor(600.0, dtype=dtype)   # beyond 448 before scaling
        return x

    def run_case(label, dtype, tokens, hidden, mode, scale_v=None, ub_v=None, group=None, strided=False, pad=None, special=True, runs=2):
        rec = {"label": label, "dtype": str(dtype).split(".")[-1], "tokens": tokens, "hidden": hidden, "mode": mode, "scale": scale_v,
               "scale_ub": ub_v, "group": group, "strided": strided, "num_token_padding": pad, "special_rows": special}
        try:
            x = make_rows(tokens, hidden, dtype, special)
            if strided:
                buf = torch.zeros((tokens, hidden + 8), dtype=dtype)
                buf[:, :hidden] = x
                x = buf.cuda()[:, :hidden]
            else:
                x = x.cuda()
            scale = None; ub = None
            if mode == "static-tensor":
                scale = torch.tensor([scale_v], dtype=torch.float32, device=dev)
            if mode == "static-group":
                gm, gn = group
                scale = (torch.rand((tokens // gm, hidden // gn), generator=gen) * 0.05 + 1e-3).cuda()
            if mode == "dyn-token-ub":
                ub = torch.tensor([ub_v], dtype=torch.float32, device=dev)
            kw = {"scale": scale, "use_per_token_if_dynamic": mode.startswith("dyn-token"), "scale_ub": ub}
            if mode == "static-group":
                kw["group_shape"] = tuple(group)
            if pad:
                kw["num_token_padding"] = pad
            outs = []
            for _ in range(runs):
                mark(label + " " + rec["dtype"] + " t=" + str(tokens) + " h=" + str(hidden) + " " + mode)
                q, s_out = ops.scaled_fp8_quant(x, **kw)
                torch.cuda.synchronize()
                outs.append((q[:tokens].clone(), (s_out[:tokens] if s_out.dim() == 2 and s_out.shape[0] > tokens else s_out).clone()))
            q, s_out = outs[0]
            mq, ms = model(x, mode, scale, ub, group)
            qb = q.view(torch.uint8)
            rec["elements"] = int(qb.numel()); rec["codes_off_model"] = int((qb != mq).sum())
            rec["nan_codes"] = int((qb == 0x7F).sum() + (qb == 0xFF).sum())
            rec["scales_off_model"] = int((s_out.reshape(-1) != ms.reshape(-1)).sum()) if mode.startswith("dyn") else 0
            rec["scale_floor_rows"] = int((s_out.reshape(-1) == f32(MIN_SCALE)).sum()) if mode.startswith("dyn-token") else None
            rec["distinct_outputs"] = len({(o[0].view(torch.uint8).cpu().numpy().tobytes(), o[1].cpu().numpy().tobytes()) for o in outs})
            if rec["codes_off_model"] or rec["scales_off_model"]:
                bad = (qb != mq).nonzero()[:4].tolist()
                rec["examples"] = [{"row": r, "col": c, "x": float(x[r, c]), "got": int(qb[r, c]), "model": int(mq[r, c])} for r, c in bad]
                if rec["scales_off_model"]:
                    i = (s_out.reshape(-1) != ms.reshape(-1)).nonzero().flatten()[0]
                    rec["scale_example"] = {"index": int(i), "got": float(s_out.reshape(-1)[i]), "model": float(ms.reshape(-1)[i])}
        except Exception as exc:  # noqa: BLE001
            rec["error"] = type(exc).__name__ + ": " + str(exc)[:300]
        return rec

    chk = run_case("test-case", torch.bfloat16, 7, 1025, "dyn-token", special=False)
    report["instrument_check"] = chk
    print(json.dumps(chk)[:400], flush=True)
    if report["cast_check"]["mismatches"] or "error" in chk or chk["codes_off_model"] or chk["scales_off_model"]:
        raise SystemExit("instrument check failed (cast or the test's own case); no verdict")

    plan = []
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for tokens in (1, 3, 1000, 4097):
            for hidden in (1, 2, 8, 15, 17, 24, 40, 1000, 8200, 32768):
                plan.append(("edge", dtype, tokens, hidden, "dyn-tensor"))
                plan.append(("edge", dtype, tokens, hidden, "dyn-token"))
                plan.append(("edge", dtype, tokens, hidden, "dyn-token-ub", None, 0.5))
                for sc in (0.01, 1.0, 1e-5):
                    plan.append(("edge", dtype, tokens, hidden, "static-tensor", sc))
        for tokens in (1, 3, 1000):
            for hidden in (1024, 8192):
                plan.append(("group", dtype, tokens, hidden, "static-group", None, None, (1, 128)))
        for hidden in (1000, 8200):
            plan.append(("strided", dtype, 1000, hidden, "dyn-tensor", None, None, None, True))
            plan.append(("strided", dtype, 1000, hidden, "dyn-token", None, None, None, True))
        plan.append(("padded", dtype, 4097, 1000, "dyn-token", None, None, None, False, 4100))
        plan.append(("padded", dtype, 4097, 1000, "dyn-tensor", None, None, None, False, 4100))

    for i, p in enumerate(plan):
        if a.max_cases and i >= a.max_cases:
            break
        rec = run_case(*p)
        report["cases"].append(rec)
        flag = rec.get("codes_off_model") or rec.get("scales_off_model") or "error" in rec or rec.get("distinct_outputs", 1) != 1
        if flag or (i + 1) % 100 == 0:
            print(json.dumps(rec)[:420], flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    judged = [c for c in report["cases"] if "codes_off_model" in c]
    print("cases " + str(len(report["cases"])) + ", judged " + str(len(judged)) + ", codes off model in " + str(sum(1 for c in judged if c["codes_off_model"]))
          + " cases, scales off model in " + str(sum(1 for c in judged if c.get("scales_off_model"))) + ", nan codes in " + str(sum(1 for c in judged if c.get("nan_codes")))
          + ", nondeterministic " + str(sum(1 for c in judged if c.get("distinct_outputs", 1) != 1)) + ", errors " + str(sum(1 for c in report["cases"] if "error" in c))
          + ", elements " + str(sum(c["elements"] for c in judged)) + "; wrote " + str(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
