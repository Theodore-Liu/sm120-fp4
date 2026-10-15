#!/usr/bin/env python3
"""vLLM's (1, group) FP8 activation quantiser (`per_token_group_quant_fp8`, csrc/quantization/w8a8/fp8/per_token_group_quant.cu),
fp32 and UE8M0 scale arms, row-major and column-major scales, on an SM120 device at shapes and group contents its tests do
not enumerate, judged bit for bit against a step model of the kernel's arithmetic.

Registered before any case ran: mlsys-5090-research docs/d1-vllm-group-quant-fp8-registration-2026-10-15.md.

    PYTHONPATH=<sm120-fp4> python vllm_group_quant_fp8_probe.py --log <marker> --out <json>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-cases", type=int, default=0)
    a = ap.parse_args(argv)

    def mark(s: str) -> None:
        with open(a.log, "a", encoding="utf-8") as f:
            f.write(time.strftime("%H:%M:%S") + " groupquant " + s + "\n")

    import vllm
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

    dev = "cuda"
    f32 = lambda v: torch.tensor(v, dtype=torch.float32, device=dev)  # noqa: E731
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "capability": list(torch.cuda.get_device_capability()), "torch": torch.__version__, "vllm": vllm.__version__, "cases": []}
    gen = torch.Generator().manual_seed(20261015)

    def model(x: torch.Tensor, group: int, eps: float, ue8m0: bool):
        """ComputeGroupScale + QuantizeGroup, one correctly rounded fp32 op per step."""
        xf = x.float()
        t, h = xf.shape
        g = xf.reshape(t, h // group, group)
        absmax = torch.maximum(g.abs().amax(dim=-1), f32(eps))            # local_absmax starts at eps
        y_s = absmax / f32(FP8_MAX)
        if ue8m0:
            y_s = torch.exp2(torch.ceil(torch.log2(torch.maximum(y_s.abs(), f32(1e-10)))))
        # The source writes inv_y = 1.0f / y_s; v * inv_y, but the compiled kernels (register and shared-memory paths, all
        # three input dtypes) fit the division x / y_s with 0 off on 4.2e7 diag elements and the reciprocal product with
        # 766 to 3,559 off per array (scripts/vllm_group_quant_fp8_diag.py, 2026-10-15); the model follows the kernel.
        q = torch.minimum(torch.maximum(g / y_s[:, :, None], f32(-FP8_MAX)), f32(FP8_MAX))
        return q.reshape(t, h).to(FP8).view(torch.uint8), y_s

    def make(tokens, hidden, group, dtype, special):
        x = (torch.randn((tokens, hidden), generator=gen) * 3).to(dtype)
        if special:
            ng = hidden // group
            # group 0 of the first row all zero; group 1 (if any) a single nonzero; the last group of the last row has its
            # absmax exactly 448 * 2^-5 = 14 (an exact power-of-two scale, the UE8M0 ceil boundary); the second-to-last
            # group a hair above it (14.0625) so the ceil must step up
            x[0, :group] = 0
            if ng > 1:
                x[0, group:2 * group] = 0; x[0, group + group // 2] = torch.tensor(2.5, dtype=dtype)
            x[-1, -group:] = (x[-1, -group:].abs() % 10).to(dtype); x[-1, -1] = torch.tensor(14.0, dtype=dtype)
            if ng > 1:
                x[-1, -2 * group:-group] = (x[-1, -2 * group:-group].abs() % 10).to(dtype); x[-1, -group - 1] = torch.tensor(14.0625, dtype=dtype)
        return x.cuda()

    def run_case(label, dtype, tokens, hidden, group, ue8m0, col_major, eps=1e-10, special=True, runs=2):
        rec = {"label": label, "dtype": str(dtype).split(".")[-1], "tokens": tokens, "hidden": hidden, "group": group,
               "ue8m0": ue8m0, "column_major": col_major, "eps": eps, "special_groups": special}
        try:
            x = make(tokens, hidden, group, dtype, special)
            outs = []
            for _ in range(runs):
                mark(label + " " + rec["dtype"] + " t=" + str(tokens) + " h=" + str(hidden) + " g=" + str(group) + " ue8m0=" + str(ue8m0) + " col=" + str(col_major))
                q, s = per_token_group_quant_fp8(x, group, eps=eps, column_major_scales=col_major, use_ue8m0=ue8m0)
                torch.cuda.synchronize()
                outs.append((q.clone(), s.clone()))
            q, s = outs[0]
            mq, ms = model(x, group, eps, ue8m0)
            qb = q.view(torch.uint8)
            # the wrapper returns the scales as a [T, H/g] view in both layouts (column-major is a [H/g, T] buffer permuted),
            # so no transpose here: the first run transposed square [3, 3] cases and misread 6 of them (placement, not value)
            s_rm = s.reshape(tokens, hidden // group) if s.shape == (tokens, hidden // group) else s
            rec["elements"] = int(qb.numel()); rec["codes_off_model"] = int((qb != mq).sum())
            rec["scales_off_model"] = int((s_rm.float() != ms).sum()); rec["scales"] = int(ms.numel())
            rec["scale_is_pow2_fraction"] = round(float((torch.frexp(s_rm.float())[0] == 0.5).double().mean()), 4) if ue8m0 else None
            rec["nan_codes"] = int(((qb & 0x7F) == 0x7F).sum())
            rec["distinct_outputs"] = len({(o[0].view(torch.uint8).cpu().numpy().tobytes(), o[1].cpu().numpy().tobytes()) for o in outs})
            if rec["codes_off_model"] or rec["scales_off_model"]:
                bad = (qb != mq).nonzero()[:3].tolist()
                rec["examples"] = [{"row": r, "col": c, "x": float(x[r, c]), "got": int(qb[r, c]), "model": int(mq[r, c])} for r, c in bad]
                sb = (s_rm.float() != ms).nonzero()[:3].tolist()
                rec["scale_examples"] = [{"row": r, "group": g, "got": float(s_rm[r, g]), "model": float(ms[r, g]),
                                          "absmax": float(x.float().reshape(tokens, hidden // group, group)[r, g].abs().amax())} for r, g in sb]
        except Exception as exc:  # noqa: BLE001
            rec["error"] = type(exc).__name__ + ": " + str(exc)[:300]
        return rec

    # instrument check: a test-enumerated shape (2048 tokens, 1024 hidden, group 128, bf16, fp32 scales, row-major)
    chk = run_case("test-shape", torch.bfloat16, 2048, 1024, 128, False, False, special=False)
    report["instrument_check"] = chk
    print(json.dumps(chk)[:400], flush=True)
    if "error" in chk or chk["codes_off_model"] or chk["scales_off_model"]:
        raise SystemExit("instrument check failed on a test-enumerated shape; no verdict")

    plan = []
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for tokens in (1, 3, 1000, 4097):
            for hidden in (128, 384, 1152, 8320, 16384):
                for group in (64, 128):
                    for ue8m0 in (False, True):
                        for col in (False, True):
                            plan.append(("edge", dtype, tokens, hidden, group, ue8m0, col))
    plan.append(("eps", torch.bfloat16, 1000, 1152, 128, False, False, 1e-4))
    plan.append(("eps", torch.bfloat16, 1000, 1152, 128, True, False, 1e-4))

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
