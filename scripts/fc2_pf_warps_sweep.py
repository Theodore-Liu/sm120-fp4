"""FC2 prefetch kernel: warps per block at a shape, timed by one method in one session.

fc2_mma_pf.py builds with 8 warps per block (one expert in flight per warp, 8 experts per column tile). At 2816 x 704 the tile count
is 176 against the RTX 5090's 170 SMs and the kernel runs at 2.2 to 3.7x a read of its bytes (BACKLOG item 10). This times the
kernel at 4, 8 and 16 warps per block, one and two groups, on random routing at 1 to 16 tokens and on eight fixed experts, with
micro_floor.graph_time (graph replay, L2 flushed), and checks every variant's output against the 8-warp one-group output.
--groups widens the split (1 to 4 blocks per column tile): at 2816 x 704 one group is 176 blocks, a hair over one wave of 170 SMs,
so three and four groups (528, 704 blocks) are the untested half of the grid question. --check-only runs the correctness and
determinism checks without timing.

    PYTHONPATH=. python scripts/fc2_pf_warps_sweep.py --hidden 2816 --inter 704 --out reports/fc2-pf-warps-<shape>-<device>-<date>.json
    PYTHONPATH=. python scripts/fc2_pf_warps_sweep.py --hidden 2816 --inter 704 --warps 4 8 --groups 1 2 3 4 --out reports/fc2-pf-groups-<shape>-<device>-<date>.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_here = Path(__file__).resolve().parent.parent / "sm120fp4" / "kernels"
for _name in ("bench_moe_baseline", "micro_floor", "fc1_w4a16", "fc2_w4a16", "fc1_mma", "fc2_mma", "fc2_mma_pf"):
    _spec = importlib.util.spec_from_file_location(_name, _here / f"{_name}.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)
bench = sys.modules["bench_moe_baseline"]
floor = sys.modules["micro_floor"]
fc1 = sys.modules["fc1_w4a16"]
pfm = sys.modules["fc2_mma_pf"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--inter", type=int, default=768)
    ap.add_argument("--warps", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--groups", type=int, nargs="+", default=[1, 2], help="blocks per column tile (the kernel's split, 1 to 4)")
    ap.add_argument("--check-only", action="store_true", help="correctness and determinism only, no timing (us and the read floor are null)")
    a = ap.parse_args(argv)
    assert all(1 <= g <= 4 for g in a.groups), a.groups
    mods = {w: pfm.build(warps=w) for w in sorted(set(a.warps) | {8})}
    m1, fl = fc1.build(), floor.build()
    dev = torch.device("cuda")
    e, k, h, i = 128, 8, a.hidden, a.inter
    print(f"shape: {e} experts, top-{k}, hidden {h}, intermediate {i}")
    w = bench.build(e, h, i, dev)
    q1, s1, q2, s2 = (w[n].contiguous() for n in ("q1", "s1", "q2", "s2"))
    alpha = torch.ones(e, device=dev)
    sink = torch.zeros(4, dtype=torch.int32, device=dev)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    scratch = torch.zeros(4 * 16 * h, device=dev)
    counters = torch.zeros(h // 16, dtype=torch.int32, device=dev)
    rows = []
    print("routing | tokens | warps | groups | normwise vs fp32 | bit-identical x50 | equals w8 g1 | us")

    def case(label, m, ids, wts, x):
        experts, offsets, pairs = fc1.route(ids)
        act = torch.empty(ids.numel(), i, device=dev, dtype=torch.bfloat16)
        m1.fc1_w4a16(q1, s1, x, experts, offsets, pairs, alpha, act, i, k)
        wf = wts.reshape(-1).contiguous()
        ref = bench.reference(x, w, ids, wts, i, act_quant=False)
        o_ref = torch.empty(m, h, device=dev, dtype=torch.bfloat16)
        mods[8].fc2_pf(q2, s2, act, experts, offsets, pairs, wf, alpha, o_ref, scratch, counters, k, 1)
        torch.cuda.synchronize()
        ptrs = torch.tensor([q2.data_ptr() + int(x_) * h * i // 2 for x_ in experts.tolist()], dtype=torch.int64, device=dev)
        t_read = None if a.check_only else floor.graph_time(lambda: fl.stream_read(ptrs, h * i // 2, 0, h * i // 2, sms * 4, 256, sink))
        for wp in a.warps:
            for g in a.groups:
                out = torch.empty(m, h, device=dev, dtype=torch.bfloat16)
                go = lambda: mods[wp].fc2_pf(q2, s2, act, experts, offsets, pairs, wf, alpha, out, scratch, counters, k, g)  # noqa: E731
                go()
                torch.cuda.synchronize()
                rel = float((out.float() - ref).norm() / ref.norm())
                first = out.clone()
                stable = True
                for _ in range(50):
                    go()
                    stable = stable and bool(torch.equal(out, first))
                same = bool(torch.equal(first, o_ref))
                t = None if a.check_only else floor.graph_time(go)
                rows.append({"routing": label, "tokens": m, "warps": wp, "groups": g, "experts_touched": int(experts.numel()), "rel_err_vs_fp32_moe": rel,
                             "bit_identical_50": stable, "equals_w8_g1": same, "us": t, "read_fc2_codes_us": t_read})
                print(f"{label:7s} | {m:6d} | {wp:5d} | {g:6d} | {rel:16.5f} | {str(stable):17s} | {str(same):12s} | " + ("-" if t is None else f"{t:.1f}"))
        if not a.check_only:
            print(f"{'':7s} | {'':6s} | {'':5s} | {'':6s} | {'read floor':16s} | {'':17s} | {'':12s} | {t_read:.1f}")

    for m in (1, 2, 4, 8, 16):
        g = torch.Generator().manual_seed(1000 + m)
        x = torch.randn(m, h, generator=g).to(device=dev, dtype=torch.bfloat16)
        wts, ids = torch.topk(F.softmax(torch.randn(m, e, generator=g), dim=-1), k, dim=-1)
        wts = (wts / wts.sum(-1, keepdim=True)).float().to(dev).contiguous()
        case("random", m, ids.to(torch.int32).to(dev).contiguous(), wts, x)
    fixed = torch.arange(8, dtype=torch.int32, device=dev) * 16
    for m in (1, 4, 8, 16):
        x = torch.randn(m, h, generator=torch.Generator().manual_seed(7)).to(device=dev, dtype=torch.bfloat16)
        ids = fixed.unsqueeze(0).repeat(m, 1).contiguous()
        wts = torch.full((m, k), 1.0 / k, device=dev)
        case("fixed8", m, ids, wts, x)
    a.out.write_text(json.dumps({"device": torch.cuda.get_device_name(0), "shape": {"experts": e, "top_k": k, "hidden": h, "inter": i}, "rows": rows}, indent=1) + "\n", encoding="utf-8")
    print("written", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
