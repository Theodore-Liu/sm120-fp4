"""Are b12x W4A4's run-to-run outliers reordered bf16 additions, or lost contributions?

diag_b12x_nondeterminism.py shows the W4A4 output varies only when an output element receives more than one atomic
bf16 add (several intermediate slices, or several experts). Reordering bf16 additions moves an element by about one
bf16 ulp of the partial sum. A lost update (one contribution never lands, or is overwritten) moves it by a whole
contribution. This script uses top-1 routing, so each output element is the sum of the per-slice FC2 partials of one
expert, computes those partials in fp32 for candidate slice widths, and asks whether each outlier's deviation from the
median run equals minus one partial.

    PYTHONPATH=. python scripts/diag_b12x_lost_update.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_here = Path(__file__).resolve().parent
for _name in ("bench_moe_baseline", "diag_b12x_actquant"):
    _spec = importlib.util.spec_from_file_location(_name, _here / f"{_name}.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)
bench = sys.modules["bench_moe_baseline"]
diag = sys.modules["diag_b12x_actquant"]

from sm120fp4 import to_128x4  # noqa: E402


def main() -> int:
    from flashinfer import fused_moe as fm
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    dev = torch.device("cuda")
    e, k, h, i, m, runs = 128, 1, 2048, 768, 4, 100
    w = bench.build(e, h, i, dev)
    s1 = convert_sf_to_mma_layout(to_128x4(w["s1"]), m=2 * i, k=h, num_groups=e, sf_vec_size=16)
    s2 = convert_sf_to_mma_layout(to_128x4(w["s2"]), m=h, k=i, num_groups=e, sf_vec_size=16)
    g = torch.Generator().manual_seed(77)
    x = torch.randn(m, h, generator=g).to(device=dev, dtype=torch.bfloat16)
    ids = torch.randint(0, e, (m, k), generator=g).to(torch.int32).to(dev)
    wts = torch.ones(m, k, device=dev)
    ones = torch.ones(e, device=dev)
    one = torch.ones(1, device=dev)
    outs = torch.stack([fm.b12x_fused_moe(x, w["q1"], s1, w["q2"], s2, ids, wts, e, k, w1_alpha=ones, w2_alpha=ones,
                                          fc2_input_scale=one, quant_mode="nvfp4").float() for _ in range(runs)])
    torch.cuda.synchronize()
    med = outs.median(dim=0).values
    dev_ = outs - med                                   # [runs, m, h]
    # reference per-slice partials of FC2 for each token's single expert (SwiGLU output rounded to bf16, then NVFP4)
    xq = diag.q_ref(x.float())
    partials = {}
    for width in (64, 128, 256, 384):
        if i % width:
            continue
        p = torch.zeros(i // width, m, h, device=dev)
        for t in range(m):
            ex = int(ids[t, 0])
            hmid = xq[t:t + 1] @ w["w1d"][ex].T
            a = diag.q_bf16_scaled(F.silu(hmid[:, i:]) * hmid[:, :i])
            for sl in range(i // width):
                cols = slice(sl * width, (sl + 1) * width)
                p[sl, t] = (a[:, cols] @ w["w2d"][ex][:, cols].T)[0]
        partials[width] = p
    # A reordering of bf16 additions moves an element by about one bf16 ulp of the largest partial sum it passes
    # through; a lost contribution moves it by a whole partial. Compare the largest deviations with the partials.
    absdev = dev_.abs()
    p128 = partials.get(128)
    ulp_scale = torch.exp2(torch.floor(torch.log2(torch.maximum(med.abs(), p128.abs().amax(0)).clamp(min=1e-6))) - 7)
    in_ulps = absdev / ulp_scale
    print(f"{runs} runs, top-1, inter {i}: distinct outputs "
          f"{len({o.to(torch.bfloat16).view(torch.int16).cpu().numpy().tobytes() for o in outs})}; "
          f"elements off the median run: {float((dev_ != 0).float().mean()) * 100:.1f}%; "
          f"max |deviation| {float(absdev.max()):.4f}; mean |partial| (w128) {float(p128.abs().mean()):.3f}; "
          f"max deviation in bf16 ulps of the largest partial/output: {float(in_ulps.max()):.2f}; "
          f"elements > 2 such ulps: {int((in_ulps > 2).sum())} of {in_ulps.numel()}")
    bad = in_ulps > 8                                     # far beyond any reordering of bf16 additions
    groups = bad.view(runs, m, h // 8, 8).any(-1)         # 8-column aligned groups (one v4 bf16x2 store)
    within = bad.view(runs, m, h // 8, 8)
    print(f"  > 8 ulps: {int(bad.sum())} elements; runs affected {int(bad.flatten(1).any(1).sum())} of {runs}; "
          f"8-column groups affected {int(groups.sum())} (per affected run: "
          f"{[int(v) for v in groups.flatten(1).sum(1).tolist() if v]}); "
          f"groups whose bad elements all fall inside one aligned 8-column group: by construction; "
          f"bad elements per affected group: mean {float(within.sum(-1)[groups].float().mean()):.1f}")
    top = absdev.flatten().topk(8).indices
    for n in top.tolist():
        r_, rem = divmod(n, m * h)
        t, c = divmod(rem, h)
        d = float(dev_[r_, t, c])
        line = (f"  run {r_:3d} token {t} col {c:4d}: deviation {d:+.4f} = {float(in_ulps[r_, t, c]):.2f} ulps "
                f"(median {float(med[t, c]):+.3f})")
        for width, p_ in partials.items():
            line += f" | w{width} partials {[round(float(v), 2) for v in p_[:, t, c]]}"
        print(line[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
