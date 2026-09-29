"""Where does b12x's W4A4 MoE depart from the NVFP4 reference arithmetic?

bench_moe_baseline.py finds b12x_fused_moe(quant_mode="nvfp4") 2 to 4% (normwise) from a reference that quantizes both
GEMM inputs with the reference quantizer, while cutlass_fused_moe on the same bytes is within 0.24% (bf16 rounding) and
b12x's own W4A16 path is within 0.4%. The weights are shared, so the difference is in how b12x quantizes activations.
This script compares b12x against references that quantize only the FC1 input, only the FC2 input, or FC2's input under
alternative rules, to localise it. One MoE layer, Qwen3-30B-A3B shape, 16 tokens.

    PYTHONPATH=. python scripts/diag_b12x_actquant.py
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from sm120fp4 import dequantize_nvfp4, quantize_nvfp4, to_128x4
from sm120fp4.reference import E2M1_GRID  # noqa: F401  (kept for readers following the rules below)

import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("bench", Path(__file__).with_name("bench_moe_baseline.py"))
bench = importlib.util.module_from_spec(spec)
sys.modules["bench"] = bench
spec.loader.exec_module(bench)


def q_ref(t):
    one = torch.ones(1, device=t.device)
    p, s, _ = quantize_nvfp4(t.float(), one)
    return dequantize_nvfp4(p, s, one)


def q_scale_up(t):
    """Block scale rounded up to the next E4M3 value instead of to nearest, so no element exceeds 6 after scaling."""
    tf = t.float()
    m, k = tf.shape
    b = tf.view(m, k // 16, 16)
    amax = b.abs().amax(-1, keepdim=True)
    s_near = (amax / 6.0).to(torch.float8_e4m3fn).float()
    bump = s_near < amax / 6.0
    # step the E4M3 bit pattern to the next representable value where nearest rounding went down
    bits = s_near.to(torch.float8_e4m3fn).view(torch.uint8).to(torch.int16)
    bits = torch.where(bump, bits + 1, bits).clamp(max=0x7E).to(torch.uint8)
    s_up = bits.view(torch.float8_e4m3fn).float()
    safe = torch.where(s_up == 0, torch.ones_like(s_up), s_up)
    from sm120fp4.reference import e2m1_decode, e2m1_encode
    codes = e2m1_encode(b / safe)
    return (e2m1_decode(codes) * s_up).view(m, k)


def q_bf16_scaled(t):
    """Reference rule applied to the bf16-rounded tensor (a kernel that holds the activation in bf16 first)."""
    return q_ref(t.to(torch.bfloat16).float())


def moe(x, w, ids, wts, i, q_in, q_mid):
    out = torch.zeros(x.shape[0], w["w2d"].shape[1], device=x.device)
    xq = q_in(x.float())
    for t in range(x.shape[0]):
        for j in range(ids.shape[1]):
            ex = int(ids[t, j])
            h = xq[t:t + 1] @ w["w1d"][ex].T
            a = F.silu(h[:, i:]) * h[:, :i]
            out[t] += float(wts[t, j]) * (q_mid(a) @ w["w2d"][ex].T)[0]
    return out


def main() -> int:
    from flashinfer import fused_moe as fm
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    dev = torch.device("cuda")
    e, k, h, i, m = 128, 8, 2048, 768, 16
    w = bench.build(e, h, i, dev)
    s1 = convert_sf_to_mma_layout(to_128x4(w["s1"]), m=2 * i, k=h, num_groups=e, sf_vec_size=16)
    s2 = convert_sf_to_mma_layout(to_128x4(w["s2"]), m=h, k=i, num_groups=e, sf_vec_size=16)
    g = torch.Generator().manual_seed(1000 + m)
    x = torch.randn(m, h, generator=g).to(device=dev, dtype=torch.bfloat16)
    wts, ids = torch.topk(F.softmax(torch.randn(m, e, generator=g), dim=-1), k, dim=-1)
    wts = (wts / wts.sum(-1, keepdim=True)).float().to(dev)
    ids = ids.to(torch.int32).to(dev)
    ones = torch.ones(e, device=dev)
    one = torch.ones(1, device=dev)
    o = fm.b12x_fused_moe(x, w["q1"], s1, w["q2"], s2, ids, wts, e, k, w1_alpha=ones, w2_alpha=ones,
                          fc2_input_scale=one, quant_mode="nvfp4").float()
    ident = lambda t: t.float()  # noqa: E731
    variants = {
        "both quantized (reference rule)": (q_ref, q_ref),
        "FC1 input only": (q_ref, ident),
        "FC2 input only": (ident, q_ref),
        "neither": (ident, ident),
        "FC1 ref, FC2 from bf16": (q_ref, q_bf16_scaled),
        "FC1 ref, FC2 scale rounded up": (q_ref, q_scale_up),
        "both scale rounded up": (q_scale_up, q_scale_up),
    }
    for name, (qi, qm) in variants.items():
        r = moe(x, w, ids, wts, i, qi, qm)
        print(f"{name:34s} ||b12x - ref|| / ||ref|| = {float((o - r).norm() / r.norm()):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
