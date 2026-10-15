#!/usr/bin/env python3
"""vLLM's INT8 activation quantiser (`ops.scaled_int8_quant`) on an SM120 device at hidden sizes, token counts, dtypes
and row contents its test does not enumerate, judged bit for bit against a step-by-step fp32 model of the kernel's own
arithmetic (read from csrc/quantization/w8a8/int8/scaled_quant.cu), with the ideal quantiser counted beside it.

Registered before any case ran: mlsys-5090-research docs/d1-vllm-scaled-int8-quant-registration-2026-10-15.md.

    PYTHONPATH=<sm120-fp4> python vllm_scaled_int8_quant_probe.py --log <marker> --out <json>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def model_codes(x: torch.Tensor, mode: str, scale=None, azp=None):
    """The kernel's arithmetic, one correctly rounded fp32 operation per step. Returns (codes int8, scale fp32 [T,1], azp int32 [T,1] or None)."""
    xf = x.float()
    if mode == "dynamic-sym":
        absmax = xf.abs().amax(dim=-1, keepdim=True)
        s = absmax / torch.tensor(127.0, dtype=torch.float32, device=x.device)
        inv = torch.where(absmax == 0, torch.zeros_like(absmax), torch.tensor(127.0, dtype=torch.float32, device=x.device) / absmax)
        q = torch.round(xf * inv).clamp(-128, 127)
        return q.to(torch.int8), s, None
    if mode == "dynamic-asym":
        mn = xf.amin(dim=-1, keepdim=True); mx = xf.amax(dim=-1, keepdim=True)
        s = (mx - mn) / torch.tensor(255.0, dtype=torch.float32, device=x.device)
        zp_f = torch.round(torch.tensor(-128.0, dtype=torch.float32, device=x.device) - mn / s)
        inv = torch.tensor(1.0, dtype=torch.float32, device=x.device) / s
        v = xf * inv
        # cvt.rni.sat.s32.f32: NaN -> 0, +-inf saturate
        r = torch.round(v)
        r = torch.where(torch.isnan(r), torch.zeros_like(r), r).clamp(-2147483648.0, 2147483647.0)
        zp_i = torch.where(torch.isnan(zp_f), torch.zeros_like(zp_f), zp_f).clamp(-2147483648.0, 2147483647.0)
        q = (r.double() + zp_i.double()).clamp(-128, 127)
        return q.to(torch.int8), s, zp_i.to(torch.int32)
    if mode == "static-sym":
        q = torch.round(xf / scale.float()).clamp(-128, 127)
        return q.to(torch.int8), scale, None
    if mode == "static-asym":
        r = torch.round(xf / scale.float()).clamp(-2147483648.0, 2147483647.0)
        q = (r.double() + float(azp)).clamp(-128, 127)
        return q.to(torch.int8), scale, azp
    raise ValueError(mode)


def ideal_codes(x: torch.Tensor, scale: torch.Tensor, azp):
    """The ideal quantiser in float64 from the scale (and zero point) the kernel returned."""
    v = x.double() / scale.double()
    if azp is not None:
        v = v + azp.double()
    return torch.round(v).clamp(-128, 127).to(torch.int8)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-cases", type=int, default=0)
    a = ap.parse_args(argv)

    def mark(s: str) -> None:
        with open(a.log, "a", encoding="utf-8") as f:
            f.write(time.strftime("%H:%M:%S") + " int8quant " + s + "\n")

    import vllm
    from vllm import _custom_ops as ops

    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "capability": list(torch.cuda.get_device_capability()), "torch": torch.__version__, "vllm": vllm.__version__, "cases": []}
    gen = torch.Generator().manual_seed(20261015)

    def make_rows(tokens, hidden, dtype, special):
        x = (torch.randn((tokens, hidden), generator=gen) * 3).to(dtype)
        if special and tokens >= 1:
            rows = [0, min(1, tokens - 1), tokens - 1]
            kinds = ["zero", "single", "positive", "negative", "max-last"]
            for i, r in enumerate(rows):
                k = kinds[(i + hidden) % len(kinds)]
                if k == "zero":
                    x[r] = 0
                elif k == "single":
                    x[r] = 0; x[r, hidden // 2] = torch.tensor(2.5, dtype=dtype)
                elif k == "positive":
                    x[r] = x[r].abs() + torch.tensor(0.25, dtype=dtype)
                elif k == "negative":
                    x[r] = -(x[r].abs()) - torch.tensor(0.25, dtype=dtype)
                elif k == "max-last":
                    x[r, -1] = torch.tensor(50.0, dtype=dtype)
        return x.cuda()

    def run_case(label, dtype, tokens, hidden, mode, scale_v=None, azp_v=None, special=True, runs=2):
        rec = {"label": label, "dtype": str(dtype).split(".")[-1], "tokens": tokens, "hidden": hidden, "mode": mode,
               "scale": scale_v, "azp": azp_v, "special_rows": special}
        try:
            x = make_rows(tokens, hidden, dtype, special)
            scale = torch.tensor([scale_v], dtype=torch.float32, device="cuda") if scale_v is not None else None
            azp = torch.tensor([azp_v], dtype=torch.int32, device="cuda") if azp_v is not None else None
            symmetric = mode.endswith("sym") and not mode.endswith("asym")
            outs = []
            for _ in range(runs):
                mark(label + " " + rec["dtype"] + " t=" + str(tokens) + " h=" + str(hidden) + " " + mode)
                q, s_out, zp_out = ops.scaled_int8_quant(x, scale, azp, symmetric=symmetric)
                torch.cuda.synchronize()
                outs.append((q.clone(), s_out.clone() if s_out is not None else None, zp_out.clone() if zp_out is not None else None))
            q, s_out, zp_out = outs[0]
            mq, ms, mzp = model_codes(x, mode, scale, azp)
            rec["elements"] = int(q.numel())
            rec["codes_off_model"] = int((q != mq).sum())
            rec["max_code_diff_model"] = int((q.int() - mq.int()).abs().max())
            if mode.startswith("dynamic"):
                rec["scales_off_model"] = int((s_out.view(-1) != ms.view(-1)).sum())
                if mzp is not None:
                    rec["azp_off_model"] = int((zp_out.view(-1) != mzp.view(-1)).sum())
                rec["constant_rows"] = int((x.float().amax(-1) == x.float().amin(-1)).sum())
            s_used = s_out.view(-1, 1) if s_out.numel() > 1 else s_out.view(1, 1)
            ideal = ideal_codes(x, s_used, zp_out.view(-1, 1) if (zp_out is not None and zp_out.numel() > 1) else zp_out)
            rec["codes_off_ideal"] = int((q != ideal).sum())
            rec["max_code_diff_ideal"] = int((q.int() - ideal.int()).abs().max())
            rec["distinct_outputs"] = len({(o[0].cpu().numpy().tobytes(), o[1].cpu().numpy().tobytes() if o[1] is not None else b"",
                                            o[2].cpu().numpy().tobytes() if o[2] is not None else b"") for o in outs})
            if rec["codes_off_model"]:
                bad = (q != mq).nonzero()[:4].tolist()
                rec["examples"] = [{"row": r, "col": c, "x": float(x[r, c]), "got": int(q[r, c]), "model": int(mq[r, c]),
                                    "scale_got": float(s_used[r if s_used.shape[0] > 1 else 0, 0]), "scale_model": float(ms.view(-1)[r if ms.numel() > 1 else 0])}
                                   for r, c in bad]
        except Exception as exc:  # noqa: BLE001
            rec["error"] = type(exc).__name__ + ": " + str(exc)[:300]
        return rec

    # instrument check: the test's own case, bit-exact against the model, and the model within one code of the test's reference
    chk = run_case("test-case", torch.bfloat16, 7, 1025, "dynamic-sym", special=False)
    report["instrument_check"] = chk
    try:
        from tests.kernels.quant_utils import ref_dynamic_per_token_quant  # vLLM's test reference, if the tests tree is importable
    except Exception:  # noqa: BLE001
        ref_dynamic_per_token_quant = None
    if ref_dynamic_per_token_quant is not None:
        x = (torch.randn((7, 1025), generator=gen) * 3).to(torch.bfloat16).cuda()
        mq, ms, _ = model_codes(x, "dynamic-sym")
        rq, rs = ref_dynamic_per_token_quant(x, torch.int8)
        report["model_vs_test_reference_max_code_diff"] = int((mq.int() - rq.int()).abs().max())
    print(json.dumps(chk)[:400], flush=True)
    if "error" in chk or chk["codes_off_model"] or chk["scales_off_model"]:
        raise SystemExit("instrument check failed on the test's own case; no verdict")

    plan = []
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for tokens in (1, 3, 1000, 4097):
            for hidden in (1, 2, 8, 15, 17, 24, 40, 1000, 8200, 16384, 32768):
                plan.append(("edge", dtype, tokens, hidden, "dynamic-sym", None, None))
                plan.append(("edge", dtype, tokens, hidden, "dynamic-asym", None, None))
                for sc in (0.1, 2.1, 1e-3):
                    plan.append(("edge", dtype, tokens, hidden, "static-sym", sc, None))
                for sc, zp in ((0.1, -255), (2.1, 54), (0.05, 0)):
                    plan.append(("edge", dtype, tokens, hidden, "static-asym", sc, zp))

    for i, p in enumerate(plan):
        if a.max_cases and i >= a.max_cases:
            break
        rec = run_case(*p)
        report["cases"].append(rec)
        flag = rec.get("codes_off_model") or rec.get("scales_off_model") or rec.get("azp_off_model") or "error" in rec or rec.get("distinct_outputs", 1) != 1
        if flag or (i + 1) % 100 == 0:
            print(json.dumps(rec)[:420], flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    judged = [c for c in report["cases"] if "codes_off_model" in c]
    print("cases " + str(len(report["cases"])) + ", judged " + str(len(judged)) + ", codes off model in " + str(sum(1 for c in judged if c["codes_off_model"]))
          + " cases, scales off model in " + str(sum(1 for c in judged if c.get("scales_off_model"))) + ", azp off model in " + str(sum(1 for c in judged if c.get("azp_off_model")))
          + ", off ideal in " + str(sum(1 for c in judged if c["codes_off_ideal"])) + " cases (" + str(sum(c["codes_off_ideal"] for c in judged)) + " codes)"
          + ", nondeterministic " + str(sum(1 for c in judged if c.get("distinct_outputs", 1) != 1)) + ", errors " + str(sum(1 for c in report["cases"] if "error" in c))
          + ", elements " + str(sum(c["elements"] for c in judged)) + "; wrote " + str(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
