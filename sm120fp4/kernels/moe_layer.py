"""Stage 2, the whole decode MoE layer with each GEMM's kernel chosen by batch size: GPU router, FC1, FC2, one graph, PDL.

The rule is fixed from the separate kernel measurements (reports/fc1-mma-*.json, reports/fc2-mma-pf-*.json), not picked per
row from this run:

- FC1: the CUDA-core kernel (scripts/fc1_w4a16.py) up to 8 tokens, the tensor-core kernel (scripts/fc1_mma.py) at 9 to 16.
- FC2: the CUDA-core kernel (scripts/fc2_w4a16.py) at 1 token, the prefetch kernel (scripts/fc2_mma_pf.py, one group) at 2 to
  4, prefetch with two groups per column tile at 5 to 16.

Every launch after the router uses programmatic dependent launch. Each row also times the all-CUDA-core layer (the stage-2
layer before the tensor-core kernels) in the same session, so the two compositions share the machine state.

    PYTHONPATH=. python scripts/moe_layer.py --out reports/moe-layer-<device>-<date>.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_here = Path(__file__).resolve().parent
for _name in ("bench_moe_baseline", "micro_floor", "fc1_w4a16", "fc2_w4a16", "fc1_mma", "fc2_mma", "fc2_mma_pf", "moe_w4a16"):
    _spec = importlib.util.spec_from_file_location(_name, _here / f"{_name}.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)
bench = sys.modules["bench_moe_baseline"]
floor = sys.modules["micro_floor"]
fc1 = sys.modules["fc1_w4a16"]
fc2 = sys.modules["fc2_w4a16"]
fc1m = sys.modules["fc1_mma"]
fc2p = sys.modules["fc2_mma_pf"]
moe = sys.modules["moe_w4a16"]


SWEPT_SHAPES = {(2048, 768): "qwen", (2048, 1024): "qwen", (2816, 704): "gemma"}


def choice(m: int, hidden: int = 2048, inter: int = 768) -> tuple[str, str]:
    """(FC1 kernel, FC2 kernel) for a batch of m tokens at a shape.

    Two shapes have been swept. At 2048 x 768 (the Qwen3-30B-A3B layer; 1024 shares the rule) the tensor-core kernels
    take over from 2 tokens, as below. At 2816 x 704 (the Gemma-4-26B-A4B layer; reports/moe-layer-gemma-shape-fc1-*-fc2-*-
    rtx5090-20261009.json) the CUDA-core pair leads by 4 to 5 us through 4 tokens on random routing and the tensor-core pair
    from 8 (2 us at 8, 37 us at 16), so that shape switches at 8. Any other shape takes the Qwen rule, and the report says the
    shape was not swept (shape_swept false).

    FC2 runs the prefetch kernel with one group per column tile up to 16 tokens: in the layer on real weights, with the
    activations FC1 leaves in L2, one group is 6 us faster than two at 8 and 16 random tokens and equal on 8 experts
    (reports/real-ckpt-layer0-fc2groups*-rtx5090-2026-10-01.json). The FC2 kernels take at most 16 tokens (MAXM).

    FC1 runs the tensor-core kernel from 2 tokens up: on random routing the two FC1 kernels time the same in the layer
    at 1 to 8 tokens, and on routing that puts every token on the same 8 experts the CUDA-core kernel takes 12 us longer
    at 4 tokens and 30 us longer at 8 (reports/real-ckpt-layer0-fc1sweep-rtx5090-2026-10-01.json)."""
    if not 1 <= m <= 16:
        raise ValueError(f"the layer's kernels take 1 to 16 tokens, not {m}")
    cutoff = 8 if SWEPT_SHAPES.get((hidden, inter)) == "gemma" else 2
    f1 = "cuda_core" if m < cutoff else "tensor_core"
    f2 = "cuda_core" if m < cutoff else "prefetch"
    return f1, f2



def use_pdl(m: int) -> bool:
    """Whether the layer's kernels launch with programmatic dependent launch for a batch of m tokens: on below 16 tokens,
    where it saves 0.9 to 2.1 us at 1 and 2 tokens and changes nothing at 4 and 8, off at 16, where it costs 1.8 to 2.0 us
    on both routings in two sessions (reports/real-ckpt-layer0-fc1sweep-{rule,pdl-rep2,nopdl,nopdl-rep2}-...json)."""
    return m < 16


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="JSON report, written into the repository")
    ap.add_argument("--baseline", type=Path, default=Path("reports/moe-baseline-rtx5090-2026-09-29.json"))
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=2048, help="hidden size; 2816 for the Gemma-4-26B-A4B shape (with --inter 704 and its baseline report)")
    ap.add_argument("--inter", type=int, default=768, help="expert intermediate size; 704 for the Gemma-4-26B-A4B shape")
    ap.add_argument("--fc1", default="rule", choices=("rule", "cuda_core", "tensor_core"), help="override the FC1 kernel for every batch size (default: the rule, choice(m))")
    ap.add_argument("--fc2", default="rule", choices=("rule", "cuda_core", "prefetch", "prefetch_split2"), help="override the FC2 kernel for every batch size (default: the rule)")
    a = ap.parse_args(argv)
    mr, m1, m2, m1m, m2p = moe.build(), fc1.build(), fc2.build(), fc1m.build(), fc2p.build()
    dev = torch.device("cuda")
    e, k, h, i = a.experts, a.topk, a.hidden, a.inter
    print(f"shape: {e} experts, top-{k}, hidden {h}, intermediate {i}")
    w = bench.build(e, h, i, dev)
    q1, s1, q2, s2 = (w[n].contiguous() for n in ("q1", "s1", "q2", "s2"))
    alpha = torch.ones(e, device=dev)
    scratch = torch.zeros(4 * 16 * h, device=dev)
    counters = torch.zeros(h // 16, dtype=torch.int32, device=dev)
    base = json.loads(a.baseline.read_text(encoding="utf-8"))
    best: dict[int, tuple[str, float]] = {}
    for r in base["rows"]:
        if "graph_cold_median_us" in r:
            cur = best.get(r["tokens"])
            if cur is None or r["graph_cold_median_us"] < cur[1]:
                best[r["tokens"]] = (r["backend"], r["graph_cold_median_us"])
    for fn in (m1.fc1_set_pdl, m2.fc2_set_pdl, m1m.fc1_mma_set_pdl, m2p.fc2_pf_set_pdl):
        fn(True)
    rows = []
    print("routing | tokens | FC1 | FC2 | normwise vs fp32 | bit-identical x50 | layer us | all-CUDA-core layer us | best existing")

    def case(label, m, ids, wts, x):
        P = m * k
        umax = min(e, P)
        experts = torch.empty(umax, dtype=torch.int32, device=dev)
        offsets = torch.empty(umax + 1, dtype=torch.int32, device=dev)
        pairs = torch.empty(P, dtype=torch.int32, device=dev)
        act = torch.empty(P, i, device=dev, dtype=torch.bfloat16)
        out = torch.empty(m, h, device=dev, dtype=torch.bfloat16)
        out_cc = torch.empty(m, h, device=dev, dtype=torch.bfloat16)
        wflat = wts.reshape(-1).contiguous()
        f1, f2 = choice(m, h, i)
        if a.fc1 != "rule":
            f1 = a.fc1
        if a.fc2 != "rule":
            f2 = a.fc2
        for fn in (m1.fc1_set_pdl, m2.fc2_set_pdl, m1m.fc1_mma_set_pdl, m2p.fc2_pf_set_pdl):
            fn(use_pdl(m))                        # dependent launch per batch size, the layer's rule

        def run_fc1():
            if f1 == "cuda_core":
                m1.fc1_w4a16(q1, s1, x, experts, offsets, pairs, alpha, act, i, k)
            else:
                m1m.fc1_mma(q1, s1, x, experts, offsets, pairs, alpha, act, i, k)

        def run_fc2(o):
            if f2 == "cuda_core":
                m2.fc2_w4a16(q2, s2, act, experts, offsets, pairs, wflat, alpha, o, k)
            else:
                m2p.fc2_pf(q2, s2, act, experts, offsets, pairs, wflat, alpha, o, scratch, counters, k,
                           1 if f2 == "prefetch" else 2)

        def layer():
            mr.route(ids, e, experts, offsets, pairs)
            run_fc1()
            run_fc2(out)

        def layer_cc():
            mr.route(ids, e, experts, offsets, pairs)
            m1.fc1_w4a16(q1, s1, x, experts, offsets, pairs, alpha, act, i, k)
            m2.fc2_w4a16(q2, s2, act, experts, offsets, pairs, wflat, alpha, out_cc, k)

        layer()
        torch.cuda.synchronize()
        ref = bench.reference(x, w, ids, wts, i, act_quant=False)
        rel = float((out.float() - ref).norm() / ref.norm())
        first = out.clone()
        stable = True
        for _ in range(50):
            layer()
            stable = stable and bool(torch.equal(out, first))
        t = floor.graph_time(layer)
        layer_cc()
        torch.cuda.synchronize()
        rel_cc = float((out_cc.float() - ref).norm() / ref.norm())
        t_cc = floor.graph_time(layer_cc)
        bname, bus = best.get(m, (None, None)) if label == "random" else (None, None)
        row = {"routing": label, "tokens": m, "fc1": f1, "fc2": f2, "rel_err_vs_fp32_moe": rel, "bit_identical_50": stable,
               "layer_us": t, "all_cuda_core_layer_us": t_cc, "all_cuda_core_rel_err": rel_cc,
               "best_existing": bname, "best_existing_us": bus}
        rows.append(row)
        be = "-" if bus is None else f"{bname} {bus:.1f} ({bus / t:.2f}x)"
        print(f"{label:7s} | {m:6d} | {f1:11s} | {f2:15s} | {rel:16.5f} | {str(stable):17s} | {t:8.1f} | {t_cc:23.1f} | {be}")

    for m in (1, 2, 4, 8, 16):
        g = torch.Generator().manual_seed(1000 + m)
        x = torch.randn(m, h, generator=g).to(device=dev, dtype=torch.bfloat16)
        wts, ids = torch.topk(F.softmax(torch.randn(m, e, generator=g), dim=-1), k, dim=-1)
        wts = (wts / wts.sum(-1, keepdim=True)).float().to(dev).contiguous()
        case("random", m, ids.to(torch.int32).to(dev).contiguous(), wts, x)
    fixed = torch.arange(8, dtype=torch.int32, device=dev) * 16
    for m in (1, 4, 8, 16):
        x = torch.randn(m, h, generator=torch.Generator().manual_seed(7)).to(device=dev, dtype=torch.bfloat16)
        wts = torch.full((m, k), 1.0 / k, device=dev)
        case("fixed8", m, fixed.repeat(m, 1).contiguous(), wts, x)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"device": torch.cuda.get_device_name(0), "baseline": str(a.baseline), "shape": {"experts": e, "top_k": k, "hidden": h, "inter": i},
                                 "rule": {str(m): choice(m, h, i) for m in (1, 2, 4, 8, 16)}, "shape_swept": (h, i) in SWEPT_SHAPES, "override": {"fc1": a.fc1, "fc2": a.fc2}, "rows": rows}, indent=1) + "\n",
                     encoding="utf-8")
    print(f"written {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
