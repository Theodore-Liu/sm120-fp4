"""Decode throughput inside vLLM at 1, 2, 4, 8 and 16 concurrent sequences, stock path against the sm120fp4 backend.

One engine per run (``max_num_seqs=16``, prefix caching off, the Triton attention backend as in the greedy comparison).
For each concurrency N the engine is handed N prompts of the same length at once, so the scheduler decodes them as
one batch of N; every sequence is forced to exactly T new tokens (``ignore_eos``). Decode throughput is measured by
differencing two token budgets, T_long and T_short, on the same prompts: the prefill and the fixed per-call overhead
cancel and

    decode tokens/s = N * (T_long - T_short) / (t_long - t_short)

Each (N, T) timing is the median of ``--repeats`` runs after one warm-up at that N. The report also keeps the raw
wall times so a reader can recompute. One JSON per mode, never overwritten:

    PYTHONPATH=. python scripts/vllm_decode_throughput.py --out reports/vllm-decode-stock-<date>.json
    SM120FP4_MOE=1 PYTHONPATH=. python scripts/vllm_decode_throughput.py --out reports/vllm-decode-sm120-<date>.json
    python scripts/vllm_decode_throughput.py --compare reports/vllm-decode-stock-<date>.json reports/vllm-decode-sm120-<date>.json

What this measures and what it does not: the whole engine's decode step (attention, dense layers, sampling, scheduling)
with only the 48 routed-experts layers differing between the two runs, so the ratio is the engine-level remainder of
the per-layer gain in the README table. Prefill is differenced away, so this script cannot see the prefill path (the
CUTLASS hand-off above 16 tokens, the default since 2026-10-07, or the slices); scripts/vllm_prefill_latency.py measures that. Numbers are from the RTX 5090 with the desktop running, as every table in this
repository states.
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
CONCURRENCY = (1, 2, 4, 8, 16)
T_SHORT, T_LONG = 32, 128


def prompts(n: int, words: int = 48) -> list[str]:
    """n distinct prompts of about the same token count (distinct so prefix caching, even if on, cannot share them)."""
    base = ("The quarterly report lists revenue, costs and headcount for each regional office, followed by a short note "
            "from the office lead on what changed since the previous quarter and what is planned next. Office")
    return [f"{base} {i:02d}:" + " item" * max(0, words - len(base.split()) - 2) for i in range(n)]


def run(a) -> int:
    if not a.no_wsl_defaults:
        os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    import torch
    from vllm import LLM, SamplingParams
    mode = "sm120" if os.environ.get("SM120FP4_MOE") == "1" else "stock"
    if a.out.exists():
        print(f"refusing to overwrite {a.out}", file=sys.stderr)
        return 2
    llm = LLM(model=a.model, max_model_len=512, gpu_memory_utilization=a.gpu_mem, max_num_seqs=max(CONCURRENCY),
              enforce_eager=a.enforce_eager, seed=0, tensor_parallel_size=a.tp, enable_prefix_caching=False,
              attention_config={"backend": a.attention_backend} if a.attention_backend else None)

    def timed(ps, t_new):
        sp = SamplingParams(temperature=0.0, max_tokens=t_new, min_tokens=t_new, ignore_eos=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs = llm.generate(ps, sp, use_tqdm=False)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        n_tok = sum(len(o.outputs[0].token_ids) for o in outs)
        assert n_tok == len(ps) * t_new, (n_tok, len(ps), t_new)
        return dt

    rows = []
    for n in CONCURRENCY:
        ps = prompts(n)
        timed(ps, T_SHORT)  # warm-up at this concurrency
        short = [timed(ps, T_SHORT) for _ in range(a.repeats)]
        long = [timed(ps, T_LONG) for _ in range(a.repeats)]
        ms, ml = statistics.median(short), statistics.median(long)
        dec = n * (T_LONG - T_SHORT) / (ml - ms)
        rows.append({"concurrency": n, "t_short_s": short, "t_long_s": long, "median_short_s": ms, "median_long_s": ml,
                     "decode_tokens_per_s": dec, "decode_step_ms": 1000 * (ml - ms) / (T_LONG - T_SHORT)})
        print(f"{mode} N={n}: short {ms:.3f}s long {ml:.3f}s -> {dec:.1f} decode tok/s, {rows[-1]['decode_step_ms']:.2f} ms/step", flush=True)
    report = {"model": a.model, "mode": mode, "device": torch.cuda.get_device_name(0), "vllm": __import__("vllm").__version__,
              "torch": torch.__version__, "attention_backend": a.attention_backend, "enforce_eager": a.enforce_eager, "tensor_parallel_size": a.tp,
              "t_short": T_SHORT, "t_long": T_LONG, "repeats": a.repeats, "prompt_words": 48, "rows": rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("->", a.out)
    return 0


def compare(a) -> int:
    x = json.loads(Path(a.compare[0]).read_text(encoding="utf-8"))
    y = json.loads(Path(a.compare[1]).read_text(encoding="utf-8"))
    bx = {r["concurrency"]: r for r in x["rows"]}
    by = {r["concurrency"]: r for r in y["rows"]}
    out = []
    for n in sorted(set(bx) & set(by)):
        out.append({"concurrency": n, x["mode"]: bx[n]["decode_tokens_per_s"], y["mode"]: by[n]["decode_tokens_per_s"],
                    f"{y['mode']}_over_{x['mode']}": by[n]["decode_tokens_per_s"] / bx[n]["decode_tokens_per_s"],
                    f"step_ms_{x['mode']}": bx[n]["decode_step_ms"], f"step_ms_{y['mode']}": by[n]["decode_step_ms"]})
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
    ap.add_argument("--compare", nargs=2, metavar=("STOCK_JSON", "SM120_JSON"))
    a = ap.parse_args(argv)
    if a.compare:
        return compare(a)
    if a.out is None:
        ap.error("--out is required for a run")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
