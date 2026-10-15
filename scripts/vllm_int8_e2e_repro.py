#!/usr/bin/env python3
"""End-to-end reproducer for vLLM's INT8 W8A8 kernel selection on SM120: load a compressed-tensors W8A8 int8 checkpoint
and generate. Run A (default selection) is expected to fail at the first forward with "Int8 not supported on SM120";
run B with VLLM_DISABLED_KERNELS=CutlassInt8ScaledMMLinearKernel (the effect of the is_supported fix on 0.28.0) is
expected to generate text. The caller sets the environment variable; this script runs one configuration.

    python vllm_int8_e2e_repro.py --model RedHatAI/Llama-3.2-1B-Instruct-quantized.w8a8 --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    rec = {"model": a.model, "VLLM_DISABLED_KERNELS": os.environ.get("VLLM_DISABLED_KERNELS", ""), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        import torch
        import vllm
        from vllm import LLM, SamplingParams
        rec["vllm"] = vllm.__version__
        rec["device"] = torch.cuda.get_device_name(0)
        rec["capability"] = list(torch.cuda.get_device_capability())
        llm = LLM(model=a.model, max_model_len=512, gpu_memory_utilization=0.45, enforce_eager=True, seed=0)
        prompts = ["The capital of France is", "List three prime numbers:"]
        outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=24))
        rec["outputs"] = [{"prompt": o.prompt, "text": o.outputs[0].text} for o in outs]
        rec["result"] = "generated"
    except Exception as exc:  # noqa: BLE001
        rec["result"] = "error"
        rec["error_type"] = type(exc).__name__
        rec["error"] = str(exc)[:1200]
        rec["traceback_tail"] = traceback.format_exc()[-2500:]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps({k: rec[k] for k in rec if k not in ("traceback_tail",)})[:1500], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
