"""Model-level comparison of the vLLM backend against vLLM's own path (backlog item 2, notes section 4).

Two runs of the same greedy generation over a fixed prompt set, one with the stock backend and one with
``SM120FP4_MOE=1`` (the plugin re-registers ``modelopt_fp4`` and the routed-experts layers run on scripts/moe_layer.py's
kernels), each written to its own JSON; then ``--compare`` reads two reports and prints the agreement rate and, where the
token sequences differ, the first differing position. The two backends compute the same function in different rounding
order, so identity is not required; the report states the agreement rate the way the layer table states the error.

    # stock
    PYTHONPATH=. python scripts/vllm_model_compare.py run --out reports/vllm-compare-stock-<date>.json
    # ours
    SM120FP4_MOE=1 PYTHONPATH=. python scripts/vllm_model_compare.py run --out reports/vllm-compare-sm120-<date>.json
    python scripts/vllm_model_compare.py compare reports/vllm-compare-stock-<date>.json reports/vllm-compare-sm120-<date>.json

The prompt set is generated here from a fixed seed: 300 retrieval items of the shape the layer benches already use
(a short record list and a question about one entry, the answer an integer) and 50 free-form prompts. Greedy, 64 new
tokens, batch of up to 16 (the backend's decode shape; prefill goes through the 16-token slices, which is why a run on
the plugin is slow and the throughput table is a separate measurement). The report records the engine's resolved
quantization method, the plugin's registration state, the device and the per-prompt token ids.

WSL2 note: this host runs vLLM under WSL2, where the default model runner needs UVA the driver does not expose; the
script sets ``VLLM_USE_V2_MODEL_RUNNER=0`` unless the variable is already set (pass ``--no-wsl-defaults`` elsewhere). The
attention backend defaults to TRITON_ATTN: vLLM picks FLASHINFER here, but the installed flashinfer lacks the XQA decode entry
point vLLM 0.28 calls, and the engine dies on the first forward (``VLLM_ATTENTION_BACKEND`` is no longer read; the setting is
``attention_config.backend``). Both runs use the same attention backend, so the comparison isolates the MoE layers.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

MODEL = "nvidia/Qwen3-30B-A3B-NVFP4"
NEW_TOKENS = 64
BATCH = 16


def prompts(seed: int = 20261002, n_items: int = 300, n_free: int = 50) -> list[dict]:
    rng = random.Random(seed)
    out = []
    names = ["Alder", "Birch", "Cedar", "Dogwood", "Elm", "Fir", "Ginkgo", "Hazel", "Juniper", "Larch", "Maple", "Oak",
             "Pine", "Rowan", "Spruce", "Willow"]
    for i in range(n_items):
        k = rng.randint(6, 10)
        picked = rng.sample(names, k)
        vals = {p: rng.randint(100, 999) for p in picked}
        target = rng.choice(picked)
        lines = "\n".join(f"{p}: {v}" for p, v in vals.items())
        out.append({"id": f"item-{i:03d}", "kind": "retrieval", "answer": vals[target],
                    "prompt": f"Records:\n{lines}\n\nQuestion: What is the value recorded for {target}?\nAnswer:"})
    topics = ["the mechanism of a four-stroke engine", "how a hash table handles collisions", "why the sky is blue",
              "the difference between TCP and UDP", "how vaccines train the immune system", "what a mortgage amortization schedule is",
              "how a transistor switches", "the rules of chess castling", "how photosynthesis stores energy", "what a GPU warp is"]
    forms = ["Explain {} in three sentences.", "Write a short paragraph about {}.", "List four facts about {}.",
             "Describe {} to a ten-year-old.", "Summarise {} in one sentence."]
    for i in range(n_free):
        out.append({"id": f"free-{i:03d}", "kind": "free", "answer": None,
                    "prompt": rng.choice(forms).format(rng.choice(topics))})
    return out


def run(a) -> int:
    if not a.no_wsl_defaults:
        os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    import torch
    from vllm import LLM, SamplingParams
    mode = "sm120" if os.environ.get("SM120FP4_MOE") == "1" else "stock"
    reg = None
    try:
        from vllm.model_executor.layers.quantization import get_quantization_config
        reg = get_quantization_config("modelopt_fp4").__name__
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal
        reg = f"unavailable: {type(exc).__name__}"
    t0 = time.time()
    llm = LLM(model=a.model, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem, max_num_seqs=BATCH,
              enforce_eager=a.enforce_eager, seed=0,
              attention_config={"backend": a.attention_backend} if a.attention_backend else None)
    load_s = time.time() - t0
    ps = prompts()
    sp = SamplingParams(temperature=0.0, max_tokens=NEW_TOKENS, seed=0)
    t1 = time.time()
    outs = llm.generate([p["prompt"] for p in ps], sp, use_tqdm=False)
    gen_s = time.time() - t1
    rows = []
    for p, o in zip(ps, outs):
        c = o.outputs[0]
        text = c.text
        pred = None
        if p["kind"] == "retrieval":
            digits = "".join(ch if ch.isdigit() else " " for ch in text).split()
            pred = int(digits[0]) if digits else None
        rows.append({"id": p["id"], "kind": p["kind"], "token_ids": list(c.token_ids), "text": text,
                     "correct": (pred == p["answer"]) if p["kind"] == "retrieval" else None})
    n_ret = sum(r["kind"] == "retrieval" for r in rows)
    report = {"model": a.model, "mode": mode, "modelopt_fp4_resolves_to": reg, "device": torch.cuda.get_device_name(0),
              "vllm": __import__("vllm").__version__, "torch": torch.__version__, "new_tokens": NEW_TOKENS, "batch": BATCH,
              "max_model_len": a.max_model_len, "enforce_eager": a.enforce_eager, "attention_backend": a.attention_backend, "load_s": load_s, "generate_s": gen_s,
              "prompts": len(rows), "retrieval_correct": sum(bool(r["correct"]) for r in rows), "retrieval_items": n_ret,
              "rows": rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        print(f"refusing to overwrite {a.out}", file=sys.stderr)
        return 2
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"{mode}: modelopt_fp4 -> {reg}; retrieval {report['retrieval_correct']}/{n_ret}; load {load_s:.0f}s generate {gen_s:.0f}s -> {a.out}")
    return 0


def compare(a) -> int:
    x = json.loads(Path(a.a).read_text(encoding="utf-8"))
    y = json.loads(Path(a.b).read_text(encoding="utf-8"))
    rx = {r["id"]: r for r in x["rows"]}
    ry = {r["id"]: r for r in y["rows"]}
    ids = sorted(set(rx) & set(ry))
    same, first_diff = 0, []
    for i in ids:
        tx, ty = rx[i]["token_ids"], ry[i]["token_ids"]
        if tx == ty:
            same += 1
            continue
        pos = next((j for j, (u, v) in enumerate(zip(tx, ty)) if u != v), min(len(tx), len(ty)))
        first_diff.append(pos)
    res = {"a": {"path": a.a, "mode": x["mode"], "retrieval_correct": x["retrieval_correct"]},
           "b": {"path": a.b, "mode": y["mode"], "retrieval_correct": y["retrieval_correct"]},
           "prompts": len(ids), "identical": same, "identical_fraction": same / len(ids) if ids else None,
           "first_differing_position": {"median": sorted(first_diff)[len(first_diff) // 2] if first_diff else None,
                                        "min": min(first_diff) if first_diff else None, "values": first_diff},
           "retrieval_items_differing": sum(bool(rx[i]["correct"]) != bool(ry[i]["correct"]) for i in ids if rx[i]["kind"] == "retrieval")}
    print(json.dumps(res, indent=1))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--model", default=MODEL)
    r.add_argument("--max-model-len", type=int, default=1024)
    r.add_argument("--gpu-mem", type=float, default=0.85)
    r.add_argument("--enforce-eager", action="store_true", help="no CUDA graphs (the plugin's kernels set PDL per call)")
    r.add_argument("--attention-backend", default="TRITON_ATTN",
                   help="vLLM attention backend (attention_config.backend); FLASHINFER is chosen by default on this host but its XQA "
                        "decode entry point is missing from the installed flashinfer, so the engine fails on the first forward")
    r.add_argument("--no-wsl-defaults", action="store_true")
    r.set_defaults(fn=run)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--out", type=Path)
    c.set_defaults(fn=compare)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
