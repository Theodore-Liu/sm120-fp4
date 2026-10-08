"""Prefill time inside vLLM: a batch of N prompts of about L tokens, one new token each, stock path against the sm120fp4 backend with the
CUTLASS prefill hand-off (the default) and with the slices (``SM120FP4_PREFILL=slices``).

One engine per run (``max_num_seqs=16``, prefix caching off, the Triton attention backend as in the greedy comparison). For each (N, L) the
engine is handed N distinct prompts at once with ``max_tokens=1``, so the wall time is one prefill of N x L tokens plus one decode step and the
fixed per-call overhead; the report keeps the measured prompt token counts, the median of ``--repeats`` runs after one warm-up at that shape, and
prefill tokens per second as N x L over that time. The per-call overhead is not differenced away here (it is the same across the three runs of one
session), so the ratios between runs are what the report is for. One JSON per mode, never overwritten:

    PYTHONPATH=. python scripts/vllm_prefill_latency.py --out reports/vllm-prefill-stock-<date>.json
    SM120FP4_MOE=1 PYTHONPATH=. python scripts/vllm_prefill_latency.py --out reports/vllm-prefill-sm120-cutlass-<date>.json
    SM120FP4_MOE=1 SM120FP4_PREFILL=slices PYTHONPATH=. python scripts/vllm_prefill_latency.py --out reports/vllm-prefill-sm120-slices-<date>.json
    python scripts/vllm_prefill_latency.py --compare reports/vllm-prefill-stock-<date>.json reports/vllm-prefill-sm120-cutlass-<date>.json

Numbers are from the RTX 5090 with the desktop running, as every table in this repository states.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

MODEL = "nvidia/Qwen3-30B-A3B-NVFP4"
SHAPES = ((1, 512), (4, 512), (16, 512), (1, 1024), (4, 1024))   # (concurrency, about this many prompt tokens)


def prompts(n: int, words: int) -> list[str]:
    """n distinct prompts of about the same token count (distinct so prefix caching, even if on, cannot share them)."""
    base = ("The quarterly report lists revenue, costs and headcount for each regional office, followed by a short note "
            "from the office lead on what changed since the previous quarter and what is planned next. Office")
    return [f"{base} {i:02d}:" + " item" * max(0, words - len(base.split()) - 2) for i in range(n)]


def mode_name() -> str:
    if os.environ.get("SM120FP4_MOE") != "1":
        return "stock"
    return "sm120-prefill-slices" if os.environ.get("SM120FP4_PREFILL") == "slices" else "sm120-prefill-cutlass"


def run(a) -> int:
    if not a.no_wsl_defaults:
        os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    import torch
    from vllm import LLM, SamplingParams
    mode = mode_name()
    if a.out.exists():
        print(f"refusing to overwrite {a.out}", file=sys.stderr)
        return 2
    llm = LLM(model=a.model, max_model_len=2048, gpu_memory_utilization=a.gpu_mem, max_num_seqs=16,
              enforce_eager=a.enforce_eager, seed=0, tensor_parallel_size=a.tp, enable_prefix_caching=False,
              attention_config={"backend": a.attention_backend} if a.attention_backend else None)
    sp = SamplingParams(temperature=0.0, max_tokens=1, min_tokens=1, ignore_eos=True)

    def timed(ps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs = llm.generate(ps, sp, use_tqdm=False)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        toks = [len(o.prompt_token_ids) for o in outs]
        assert all(len(o.outputs[0].token_ids) == 1 for o in outs)
        return dt, toks

    rows = []
    for n, words in SHAPES:
        ps = prompts(n, words)
        timed(ps)  # warm-up at this shape
        times, toks = [], None
        for _ in range(a.repeats):
            dt, toks = timed(ps)
            times.append(dt)
        med = statistics.median(times)
        total = sum(toks)
        rows.append({"concurrency": n, "prompt_words": words, "prompt_tokens": toks, "total_prompt_tokens": total, "wall_s": times,
                     "median_wall_s": med, "prefill_tokens_per_s": total / med})
        print(f"{mode} N={n} L~{total // n}: {med:.3f}s -> {total / med:.0f} prefill tok/s", flush=True)
    report = {"model": a.model, "mode": mode, "device": torch.cuda.get_device_name(0), "vllm": __import__("vllm").__version__,
              "torch": torch.__version__, "attention_backend": a.attention_backend, "enforce_eager": a.enforce_eager, "tensor_parallel_size": a.tp, "repeats": a.repeats,
              "note": "wall time of generate() for N prompts with max_tokens=1: one prefill of the batch plus one decode step and the call overhead",
              "rows": rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("->", a.out)
    return 0


def compare(a) -> int:
    x = json.loads(Path(a.compare[0]).read_text(encoding="utf-8"))
    y = json.loads(Path(a.compare[1]).read_text(encoding="utf-8"))
    key = lambda r: (r["concurrency"], r["prompt_words"])
    bx = {key(r): r for r in x["rows"]}
    by = {key(r): r for r in y["rows"]}
    out = []
    for k in sorted(set(bx) & set(by)):
        out.append({"concurrency": k[0], "prompt_words": k[1], f"wall_s_{x['mode']}": bx[k]["median_wall_s"], f"wall_s_{y['mode']}": by[k]["median_wall_s"],
                    f"{y['mode']}_over_{x['mode']}": by[k]["median_wall_s"] / bx[k]["median_wall_s"]})
    res = {"a": a.compare[0], "b": a.compare[1], "rows": out}
    print(json.dumps(res, indent=1))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--tp", type=int, default=1, help="tensor_parallel_size (step 3 runs DeepSeek-V4-Flash at 2)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--attention-backend", default="TRITON_ATTN")
    ap.add_argument("--no-wsl-defaults", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("A_JSON", "B_JSON"))
    a = ap.parse_args(argv)
    if a.compare:
        return compare(a)
    if a.out is None:
        ap.error("--out is required for a run")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
