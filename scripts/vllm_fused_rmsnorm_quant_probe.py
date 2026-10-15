#!/usr/bin/env python3
"""vLLM 0.28.0's fused RMSNorm + dynamic quantiser (`rms_norm_dynamic_per_token_quant`, `rms_norm_per_block_quant`) on an
SM120 device at shapes and rows its test does not enumerate, judged against (a) a step model of the fused source read at
v0.28.0 (rms taken as any fp32 value within two ulps of the exact rsqrt, since cub's summation order is not reproducible)
and (b) the composition of the unfused kernels run on the same inputs.

Registered before any case ran: mlsys-5090-research docs/d1-vllm-fused-rmsnorm-quant-registration-2026-10-15.md.

    PYTHONPATH=<sm120-fp4> python vllm_fused_rmsnorm_quant_probe.py --log <marker> --out <json>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

FP8 = torch.float8_e4m3fn
EPS = 1e-6
FP8_MIN_SCALE = 1.0 / (448.0 * 512.0)
INT8_MIN_SCALE = 1.1920928955078125e-07   # FLT_EPSILON


def rms_candidates(ss_exact: torch.Tensor, hidden: int) -> list[torch.Tensor]:
    """fp32 values within two ulps of rsqrt(ss / hidden + eps) in float64: five candidates per row."""
    r = torch.rsqrt(ss_exact / hidden + EPS)
    base = r.float()
    out = [base]
    up, down = base, base
    for _ in range(2):
        up = torch.nextafter(up, torch.full_like(up, float("inf")))
        down = torch.nextafter(down, torch.full_like(down, float("-inf")))
        out += [up, down]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-cases", type=int, default=0)
    a = ap.parse_args(argv)

    def mark(s: str) -> None:
        with open(a.log, "a", encoding="utf-8") as f:
            f.write(time.strftime("%H:%M:%S") + " fusednorm " + s + "\n")

    import vllm
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

    dev = "cuda"
    f32 = lambda v: torch.tensor(v, dtype=torch.float32, device=dev)  # noqa: E731
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "capability": list(torch.cuda.get_device_capability()), "torch": torch.__version__, "vllm": vllm.__version__, "cases": []}
    gen = torch.Generator().manual_seed(20261015)

    def make(tokens, hidden, dtype):
        x = (torch.randn((tokens, hidden), generator=gen) * 2).to(dtype)
        res = (torch.randn((tokens, hidden), generator=gen)).to(dtype)
        w = (torch.rand((hidden,), generator=gen) * 1.5 + 0.25).to(dtype)
        rows = [0, min(1, tokens - 1), tokens - 1]
        x[rows[0]] = 0                                                 # all-zero row: rms = rsqrt(eps)
        x[rows[1]] = torch.tensor(1e-4, dtype=dtype)                   # epsilon binds
        x[rows[2]] = 0; x[rows[2], hidden // 2] = torch.tensor(1000.0, dtype=dtype)   # one dominant element
        if tokens >= 3:
            res[rows[0]] = 0; res[rows[1]] = 0; res[rows[2]] = 0      # keep the special rows special under the residual
        return x.cuda(), res.cuda(), w.cuda()

    def norm_values(x, res, w, dtype, rms):
        """y = fp32(scalar_t(fp32(x+res) * rms) * w) with the input dtype's roundings (none for fp32)."""
        xf = x.float() + (res.float() if res is not None else 0.0)
        if dtype == torch.float32:
            return xf * rms[:, None] * w.float(), xf
        t = (xf * rms[:, None]).to(dtype)
        y = (t.float() * w.float()).to(dtype)            # c10 bf16*bf16: float multiply, rounded to the dtype
        return y.float(), xf

    def step_model(x, res, w, dtype, quant, ub, group, rms):
        y, _ = norm_values(x, res, w, dtype, rms)
        tokens, hidden = y.shape
        if group is None:
            am = y.abs().amax(dim=-1, keepdim=True)
            if ub is not None:
                am = torch.minimum(am, ub.float())
            if quant == "fp8":
                s = torch.maximum(am / f32(448.0), f32(FP8_MIN_SCALE))
                q = torch.minimum(torch.maximum(y / s, f32(-448.0)), f32(448.0)).to(FP8).view(torch.uint8)
            else:
                s = torch.maximum(am / f32(127.0), f32(INT8_MIN_SCALE))
                inv = f32(1.0) / s
                q = torch.round(y * inv).clamp(-128, 127).to(torch.int8).view(torch.uint8)
            return q, s
        g = y.reshape(tokens, hidden // group, group)
        am = g.abs().amax(dim=-1)
        if ub is not None:
            am = torch.minimum(am, ub.float())
        s = torch.maximum(am / f32(448.0), f32(FP8_MIN_SCALE))
        # fp8 is launched with is_scale_inverted = false ("exact match with FBGemm"), so the per-block path divides too: the
        # "invert the scale here" branch only runs for int8. The first run modelled the block path as a reciprocal product and
        # read 72 cases off; the division fits with 2 to 12 off per 8e6 to 1.6e7 (rms ties) (diag2, 2026-10-15).
        q = torch.minimum(torch.maximum(g / s[:, :, None], f32(-448.0)), f32(448.0)).reshape(tokens, hidden).to(FP8).view(torch.uint8)
        return q, s

    def composition(x, res, w, dtype, quant, ub, group):
        if res is not None:
            xin = x.clone(); r = res.clone()
            ops.fused_add_rms_norm(xin, r, w, EPS)
            yn = xin
        else:
            yn = torch.empty_like(x)
            ops.rms_norm(yn, x, w, EPS)
        if group is None:
            if quant == "fp8":
                q, s = ops.scaled_fp8_quant(yn, use_per_token_if_dynamic=True, scale_ub=ub)
            else:
                q, s, _ = ops.scaled_int8_quant(yn, None, None, symmetric=True)
            return q.view(torch.uint8), s.reshape(-1, 1)
        q, s = per_token_group_quant_fp8(yn, group, eps=1e-10, column_major_scales=False, use_ue8m0=False)
        return q.view(torch.uint8), s

    def run_case(label, dtype, tokens, hidden, quant, residual, ub_v, group=None, transposed=False, runs=2):
        rec = {"label": label, "dtype": str(dtype).split(".")[-1], "tokens": tokens, "hidden": hidden, "quant": quant,
               "residual": residual, "scale_ub": ub_v, "group": group, "scales_transposed": transposed}
        try:
            x, res, w = make(tokens, hidden, dtype)
            res_in = res if residual else None
            ub = torch.tensor([ub_v], dtype=torch.float32, device=dev) if ub_v is not None else None
            qdt = FP8 if quant == "fp8" else torch.int8
            outs = []
            for _ in range(runs):
                mark(label + " " + rec["dtype"] + " t=" + str(tokens) + " h=" + str(hidden) + " " + quant + " res=" + str(residual) + " g=" + str(group))
                r_copy = res_in.clone() if res_in is not None else None
                if group is None:
                    q, s = ops.rms_norm_dynamic_per_token_quant(x, w, EPS, qdt, scale_ub=ub, residual=r_copy)
                else:
                    q, s = ops.rms_norm_per_block_quant(x, w, EPS, qdt, [1, group], scale_ub=ub, residual=r_copy,
                                                        is_scale_transposed=transposed)
                torch.cuda.synchronize()
                outs.append((q.view(torch.uint8).clone(), s.clone(), r_copy.clone() if r_copy is not None else None))
            q, s, r_out = outs[0]
            s_rm = s.reshape(tokens, -1) if (group is not None and not transposed) else (s.transpose(0, 1).reshape(tokens, -1) if (group is not None and transposed and s.shape[0] != tokens) else s.reshape(tokens, -1))
            # step model over the rms candidates: an element is off only if it matches none
            xf = x.double() + (res.double() if residual else 0.0)
            ss = (xf * xf).sum(dim=-1)
            off = torch.ones_like(q, dtype=torch.bool); s_off = torch.ones_like(s_rm, dtype=torch.bool)
            for rms in rms_candidates(ss, hidden):
                mq, ms = step_model(x, res_in, w, dtype, quant, ub, group, rms)
                off &= (q != mq); s_off &= (s_rm.float() != ms.reshape(tokens, -1))
            rec["elements"] = int(q.numel()); rec["codes_off_model"] = int(off.sum()); rec["scales_off_model"] = int(s_off.sum()); rec["scales"] = int(s_rm.numel())
            if residual:
                rec["residual_off_model"] = int((r_out.float() != (x.float() + res.float()).to(dtype).float()).sum())
            # the composition of the unfused kernels
            cq, cs = composition(x, res_in, w, dtype, quant, ub, group)
            rec["codes_differ_from_composition"] = int((q != cq).sum())
            rec["scales_differ_from_composition"] = int((s_rm.float() != cs.reshape(tokens, -1).float()).sum())
            if quant == "fp8":
                d = (q.view(FP8).float() - cq.view(FP8).float()).abs()
            else:
                d = (q.view(torch.int8).int() - cq.view(torch.int8).int()).abs().float()
            rec["max_abs_diff_vs_composition"] = float(d.max())
            rec["distinct_outputs"] = len({(o[0].cpu().numpy().tobytes(), o[1].cpu().numpy().tobytes()) for o in outs})
            if rec["codes_off_model"]:
                bad = off.nonzero()[:3].tolist()
                rec["examples"] = [{"row": r, "col": c, "x": float(x[r, c]), "got": int(q[r, c])} for r, c in bad]
        except Exception as exc:  # noqa: BLE001
            rec["error"] = type(exc).__name__ + ": " + str(exc)[:300]
        return rec

    chk = run_case("test-shape", torch.bfloat16, 2048, 1024, "fp8", False, None)
    report["instrument_check"] = chk
    print(json.dumps(chk)[:500], flush=True)
    if "error" in chk or chk["codes_off_model"] or chk["scales_off_model"]:
        raise SystemExit("instrument check failed on a test-enumerated configuration; no verdict")

    plan = []
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for tokens in (1, 3, 1000, 4097):
            for hidden in (8, 40, 1000, 1023, 8200, 16384):
                for quant in ("fp8", "int8"):
                    for residual in (False, True):
                        for ub in (None, 0.5):
                            plan.append(("edge", dtype, tokens, hidden, quant, residual, ub))
        for tokens in (1, 3, 1000):
            for hidden in (128, 8192, 16384):
                for group in (64, 128):
                    for residual in (False, True):
                        for transposed in (False, True):
                            plan.append(("block", dtype, tokens, hidden, "fp8", residual, None, group, transposed))

    for i, p in enumerate(plan):
        if a.max_cases and i >= a.max_cases:
            break
        rec = run_case(*p)
        report["cases"].append(rec)
        flag = rec.get("codes_off_model") or rec.get("scales_off_model") or rec.get("residual_off_model") or "error" in rec or rec.get("distinct_outputs", 1) != 1
        if flag or (i + 1) % 100 == 0:
            print(json.dumps(rec)[:420], flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    judged = [c for c in report["cases"] if "codes_off_model" in c]
    print("cases " + str(len(report["cases"])) + ", judged " + str(len(judged)) + ", codes off model in " + str(sum(1 for c in judged if c["codes_off_model"]))
          + " cases, scales off model in " + str(sum(1 for c in judged if c.get("scales_off_model"))) + ", residual off in " + str(sum(1 for c in judged if c.get("residual_off_model")))
          + ", differ from composition in " + str(sum(1 for c in judged if c.get("codes_differ_from_composition"))) + " cases (" + str(sum(c.get("codes_differ_from_composition", 0) for c in judged)) + " codes)"
          + ", nondeterministic " + str(sum(1 for c in judged if c.get("distinct_outputs", 1) != 1)) + ", errors " + str(sum(1 for c in report["cases"] if "error" in c))
          + ", elements " + str(sum(c["elements"] for c in judged)) + "; wrote " + str(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
