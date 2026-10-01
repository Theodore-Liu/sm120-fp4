"""The distribution, not the median, of the FC2 prefetch kernel's time over many graph replays.

micro_floor.graph_time reports the median of 50 replays, which hides sporadic outliers; Nsight Compute saw the kernel
take 79.7 and 1659.7 us on two of ten repeats at 16 random tokens where the other eight took 55 to 59
(reports/ncu-jitter-16-random-clock-none-rtx5090-2026-10-01.csv). This replays the kernel N times with the activations
warm, each replay behind an L2 flush as the benches do, and records every replay's time, so the question "does the
kernel stall outside the profiler too" has a number: the max, the p99 and how many replays exceed twice the median.
FC1 (tensor core) and the stream read are timed the same way beside it as controls.

    PYTHONPATH=. python scripts/fc2_replay_dist.py --tokens 16 --routing random --replays 400 --out reports/fc2-replay-dist-<device>-<date>.json
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
mods = {}
for _name in ("bench_moe_baseline", "micro_floor", "fc1_w4a16", "fc1_mma", "fc2_mma_pf", "moe_w4a16", "moe_layer"):
    _spec = importlib.util.spec_from_file_location(_name, _here / f"{_name}.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)
    mods[_name] = _mod


def replay_times(fn, n: int, after_flush, flush) -> list[float]:
    """Every replay's time in us, each replay after an L2 flush and the warm-up read, as micro_floor.graph_time does
    for its median; the graph is captured once."""
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(n):
        flush()
        if after_flush is not None:
            after_flush()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b) * 1000)
    return out


def summary(ts: list[float]) -> dict:
    s = sorted(ts)
    med = s[len(s) // 2]
    return {"n": len(s), "min": s[0], "median": med, "p90": s[int(0.9 * len(s))], "p99": s[int(0.99 * len(s))],
            "max": s[-1], "over_2x_median": sum(1 for t in s if t > 2 * med), "over_1p5x_median": sum(1 for t in s if t > 1.5 * med)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--routing", choices=("random", "fixed8"), default="random")
    ap.add_argument("--replays", type=int, default=400)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; a report is never overwritten")
    bench, floor, fc1, fc1m, pf, moe, layer_mod = (mods[n] for n in ("bench_moe_baseline", "micro_floor", "fc1_w4a16", "fc1_mma",
                                                                    "fc2_mma_pf", "moe_w4a16", "moe_layer"))
    dev = torch.device("cuda")
    e_n, k, h, i, m = 128, 8, 2048, 768, a.tokens
    w = bench.build(e_n, h, i, dev)
    q1, s1, q2, s2 = (w[n].contiguous() for n in ("q1", "s1", "q2", "s2"))
    alpha = torch.ones(e_n, device=dev)
    mr, m1, m1m, m2p, fl = moe.build(), fc1.build(), fc1m.build(), pf.build(), floor.build()
    for fn in (m1.fc1_set_pdl, m1m.fc1_mma_set_pdl, m2p.fc2_pf_set_pdl):
        fn(False)                                     # one kernel per graph here; dependent launch has nothing to overlap
    g = torch.Generator().manual_seed(1000 + m)
    x = torch.randn(m, h, generator=g).to(device=dev, dtype=torch.bfloat16)
    if a.routing == "random":
        wts, ids = torch.topk(F.softmax(torch.randn(m, e_n, generator=g), dim=-1), k, dim=-1)
        wts = (wts / wts.sum(-1, keepdim=True)).float().to(dev).contiguous()
        ids = ids.to(torch.int32).to(dev).contiguous()
    else:
        ids = (torch.arange(k, dtype=torch.int32, device=dev) * (e_n // k)).repeat(m, 1).contiguous()
        wts = torch.full((m, k), 1.0 / k, device=dev)
    P, umax = m * k, min(e_n, m * k)
    experts = torch.empty(umax, dtype=torch.int32, device=dev)
    offsets = torch.empty(umax + 1, dtype=torch.int32, device=dev)
    pairs = torch.empty(P, dtype=torch.int32, device=dev)
    act = torch.empty(P, i, device=dev, dtype=torch.bfloat16)
    out = torch.empty(m, h, device=dev, dtype=torch.bfloat16)
    scratch = torch.zeros(4 * 16 * h, device=dev)
    counters = torch.zeros(h // 16, dtype=torch.int32, device=dev)
    wflat = wts.reshape(-1).contiguous()
    sink = torch.zeros(4, dtype=torch.int32, device=dev)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    big = torch.empty(256 * 1024 * 1024 // 4, device=dev)
    mr.route(ids, e_n, experts, offsets, pairs)
    m1m.fc1_mma(q1, s1, x, experts, offsets, pairs, alpha, act, i, k)
    torch.cuda.synchronize()
    warm_sink = torch.zeros(1, device=dev)

    def flush():
        big.zero_()

    def warm():
        warm_sink.add_(act.float().sum() * 0 + offsets.float().sum() * 0 + pairs.float().sum() * 0)
    touched = [int(v) for v in experts.tolist() if v >= 0]
    ptrs = torch.tensor([q2.data_ptr() + t * h * i // 2 for t in touched], dtype=torch.int64, device=dev)
    fc2 = lambda: m2p.fc2_pf(q2, s2, act, experts, offsets, pairs, wflat, alpha, out, scratch, counters, k, 1)  # noqa: E731
    fc1_fn = lambda: m1m.fc1_mma(q1, s1, x, experts, offsets, pairs, alpha, act, i, k)                      # noqa: E731
    read = lambda: fl.stream_read(ptrs, h * i // 2, 0, h * i // 2, sms * 4, 256, sink)                       # noqa: E731
    rep = {"device": torch.cuda.get_device_name(0), "tokens": m, "routing": a.routing, "replays": a.replays,
           "fc2_prefetch_1group_warm": None, "fc1_mma_cold": None, "stream_read_cold": None}
    t = replay_times(fc2, a.replays, warm, flush)
    rep["fc2_prefetch_1group_warm"] = {**summary(t), "all_us": [round(v, 2) for v in t]}
    t = replay_times(fc1_fn, a.replays, None, flush)
    rep["fc1_mma_cold"] = {**summary(t), "all_us": [round(v, 2) for v in t]}
    t = replay_times(read, a.replays, None, flush)
    rep["stream_read_cold"] = {**summary(t), "all_us": [round(v, 2) for v in t]}
    for name in ("fc2_prefetch_1group_warm", "fc1_mma_cold", "stream_read_cold"):
        print(name, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in rep[name].items() if k != "all_us"})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(f"written {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
