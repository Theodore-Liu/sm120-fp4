"""An oracle for a fused W4A16 MoE layer: bf16 activations, NVFP4 expert weights, gated SiLU, top-k routing.

A fused MoE is not one GEMM, so there is no single exact value for a kernel to approximate: between the two GEMMs the kernel
rounds the gate and up projections, evaluates SiLU with an approximate exponential, rounds the product to bf16, and after the
second GEMM multiplies by a routing weight and sums the top-k experts. The oracle therefore follows the rounding points of
the kernel it judges (the SM120 b12x W4A16 path, `moe_w4a16_kernel.py`, read on 2026-10-11) and, at each of them, computes
the set of values a correct implementation of that rounding can produce. The only freedom a correct kernel has is the order
of its fp32 accumulations and the last bits of its approximate exponential and division; the oracle turns each into an
interval of the exact value and rounds the interval's ends. Almost everywhere the two ends round to the same bf16 value, so
the interval has width zero and the final bound is the fp32-accumulation bound of the second GEMM plus one output rounding;
near a rounding tie the interval is one bf16 ulp wide and the bound widens by that ulp times the weight column.

The rounding points modelled (one line each in the kernel):
  FC1     fp32 accumulation of bf16 x NVFP4 (exact products), written as bf16                 `fc1_bf16_flat`
  gate    bf16 -> fp32; silu = gate / (1 + exp(-gate)) in fp32 with a fast exponential; -> bf16  `_cast_elem(silu)`
  up      bf16 (the same FC1 write)                                                             `_cast_elem(up_term)`
  h       bf16(silu) * bf16(up) in fp32 (exact), -> bf16                                        `_cast_elem(... * ...)`
  FC2     fp32 accumulation of bf16 h x NVFP4 W2 (exact products)
  weight  topk_weight * alpha in fp32, packed as a bf16 pair before the multiply                 `_broadcast_f32_to_elem2`
  combine fp32 sum over the top-k experts (`fc2_c_tmp`, float32), one bf16 rounding of the output

A fault moves an output by far more than this bound: a dropped column group or a mis-addressed scale by the size of an
expert's contribution, the wrong gate half by the whole output. The floor is higher than a dense GEMM's, and it is set by the
first GEMM, not the second: the fp32-accumulation bound of FC1 (K * 2^-24 of the sum of absolute products, worst case over
accumulation orders) straddles a bf16 rounding boundary on a large share of the gate, up and product elements once K is in
the thousands (about 80 percent at K = 1024 on random operands), and each such element carries a one-ulp-wide interval into
every output through |W2|. Measured on the selftest's shape (K = 256): a contribution off by 2^-8 is caught on about a tenth
of its elements, off by 2^-5 on about two thirds (the misses are the elements where that expert's contribution is small
against the bound built from both experts' absolute products), off by 2^-3 on about nine tenths; on the probe's shape (K = 1024)
the mean bound is about 6 percent of an expert's contribution. Faults smaller than that are below this oracle's floor; faults of the kinds the census has found are not.

Conventions follow `sm120fp4.reference.quantize_nvfp4`: packed E2M1 codes [rows, K/2], E4M3 scale bits [rows, K/16] row-major
as uint8, global scale fp32 with dequant = q * sf / gs. w13 is [E, 2I, H/2], w2 is [E, H, I/2]. Which half of w13 is the gate
is a parameter (`gate_first`); the SM120 W4A16 path reads the second half as the gate (see the probe).
"""
from __future__ import annotations

import math

import torch

from .oracle import BLOCK, U32, U64, _ulp, dequant_integer_parts, exact_gemm_bf16_nvfp4

SILU_MIN_X = -1.2784645427610738                      # argmin of x * sigmoid(x)
SILU_MIN = SILU_MIN_X / (1.0 + math.exp(-SILU_MIN_X))  # about -0.27846
FAST_EXP_REL = 2.0 ** -20                              # ex2.approx and the fp32 division, generously


def _silu64(x: torch.Tensor) -> torch.Tensor:
    return x / (1.0 + torch.exp(-x))


def _hull_silu(lo: torch.Tensor, hi: torch.Tensor):
    a, b = _silu64(lo), _silu64(hi)
    smin, smax = torch.minimum(a, b), torch.maximum(a, b)
    straddles = (lo <= SILU_MIN_X) & (hi >= SILU_MIN_X)
    smin = torch.where(straddles, torch.full_like(smin, SILU_MIN), smin)
    return smin, smax


def _hull_prod(alo, ahi, blo, bhi):
    c = torch.stack([alo * blo, alo * bhi, ahi * blo, ahi * bhi])
    return c.min(dim=0).values, c.max(dim=0).values


def _round_bf16_interval(lo: torch.Tensor, hi: torch.Tensor):
    """The bf16 values a correct fp32 -> bf16 rounding of any value in [lo, hi] can produce, as (lowest, highest) in float64.
    The fp32 rounding on the way is covered by widening the interval by one fp32 ulp first."""
    w = _ulp(torch.maximum(lo.abs(), hi.abs()), torch.float32)
    lo_b = (lo - w).to(torch.float32).to(torch.bfloat16).double()
    hi_b = (hi + w).to(torch.float32).to(torch.bfloat16).double()
    return torch.minimum(lo_b, hi_b), torch.maximum(lo_b, hi_b)


def dequant_expert_w2(w2_packed_e: torch.Tensor, w2_sf_e: torch.Tensor, gs: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """W2[e] as exact float64 values [H, I] (q * sf / gs; the division is the one float64 rounding, counted in the bound)."""
    twoq, sf = dequant_integer_parts(w2_packed_e, w2_sf_e, block)
    h, i = twoq.shape
    return (twoq.view(h, i // block, block) * sf[:, :, None]).view(h, i) / (2.0 * torch.as_tensor(gs, dtype=torch.float64).reshape(()))


def fused_moe_w4a16_oracle(x_bf16: torch.Tensor, w13_packed: torch.Tensor, w13_sf: torch.Tensor, w13_gs: torch.Tensor,
                           w2_packed: torch.Tensor, w2_sf: torch.Tensor, w2_gs: torch.Tensor,
                           topk_ids: torch.Tensor, topk_wts: torch.Tensor, block: int = BLOCK, gate_first: bool = False,
                           weight_rounding: str = "bf16") -> dict:
    """Returns {"nominal": float64 [M, H], "bound": [M, H], "abs_contrib": [M, H], "tie_elements": int}.
    A kernel output y is conformant when |y - nominal| <= bound everywhere. `weight_rounding` is "bf16" (the SM120 path packs
    topk_weight * alpha as bf16) or "fp32"."""
    x = x_bf16.detach().cpu()
    if x.dtype != torch.bfloat16:
        raise ValueError("x must be bf16; the kernel takes bf16 activations")
    m, hdim = x.shape
    e_count, i2, _ = w13_packed.shape
    inter = i2 // 2
    ids = topk_ids.detach().cpu().long()
    wts = topk_wts.detach().cpu().float()
    if ids.shape != wts.shape or ids.shape[0] != m:
        raise ValueError(f"routing shapes {tuple(ids.shape)} vs {tuple(wts.shape)} for {m} tokens")
    topk = ids.shape[1]
    gs13 = torch.as_tensor(w13_gs, dtype=torch.float64).reshape(-1)
    gs2 = torch.as_tensor(w2_gs, dtype=torch.float64).reshape(-1)
    out = torch.zeros(m, hdim, dtype=torch.float64)
    bound = torch.zeros(m, hdim, dtype=torch.float64)
    abs_contrib = torch.zeros(m, hdim, dtype=torch.float64)
    ties = 0
    for e in range(e_count):
        tok, slot = (ids == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        g13 = gs13[e] if gs13.numel() == e_count else gs13[0]
        g2 = gs2[e] if gs2.numel() == e_count else gs2[0]
        ex13, bd13 = exact_gemm_bf16_nvfp4(x[tok], w13_packed[e], w13_sf[e], g13.reshape(1), torch.float32, block)
        # bd13 is the fp32-accumulation bound with half an fp32 ulp of output rounding, i.e. the fp32 accumulator's uncertainty
        if gate_first:
            gate, bg, up, bu = ex13[:, :inter], bd13[:, :inter], ex13[:, inter:], bd13[:, inter:]
        else:
            up, bu, gate, bg = ex13[:, :inter], bd13[:, :inter], ex13[:, inter:], bd13[:, inter:]
        glo, ghi = _round_bf16_interval(gate - bg, gate + bg)          # the bf16 gate the kernel reads back
        ulo, uhi = _round_bf16_interval(up - bu, up + bu)
        slo, shi = _hull_silu(glo, ghi)                                 # silu in fp32 of the bf16 gate ...
        smag = torch.maximum(slo.abs(), shi.abs())
        gmag = torch.maximum(glo.abs(), ghi.abs())
        slack = FAST_EXP_REL * (smag + gmag)                            # ... with a fast exponential and an fp32 division
        slo, shi = _round_bf16_interval(slo - slack, shi + slack)      # ... rounded to bf16
        plo, phi = _hull_prod(slo, shi, ulo, uhi)                       # bf16 * bf16 is exact in fp32
        hlo, hhi = _round_bf16_interval(plo, phi)                       # the bf16 intermediate buffer
        ties += int((hhi > hlo).sum())
        hmid, hw = (hlo + hhi) / 2.0, (hhi - hlo) / 2.0
        w2 = dequant_expert_w2(w2_packed[e], w2_sf[e], g2, block)                        # [H, I]
        c = hmid @ w2.t()                                                                # [rows, H]
        cabs = hmid.abs() @ w2.abs().t()
        cb = hw @ w2.abs().t() + inter * U32 * cabs + (2.0 + inter) * U64 * cabs
        alpha = 1.0 / g2
        wa = wts[tok, slot].double() * alpha                                             # topk_weight * alpha in fp32
        if weight_rounding == "bf16":
            w_eff = wa.float().to(torch.bfloat16).double() / alpha
            w_unc = 0.0                                                                  # the rounding is deterministic
        elif weight_rounding == "fp32":
            w_eff = wa / alpha
            w_unc = 2.0 * U32
        else:
            raise ValueError(weight_rounding)
        w_eff = w_eff[:, None]
        contrib = w_eff * c
        out.index_add_(0, tok, contrib)
        abs_contrib.index_add_(0, tok, contrib.abs())
        bound.index_add_(0, tok, w_eff.abs() * cb + (w_unc + 2.0 * U32) * contrib.abs())  # the fp32 multiply by the weight
    bound += topk * U32 * abs_contrib + 0.5 * _ulp(out, torch.bfloat16) + 2.0 * U64 * abs_contrib
    return {"nominal": out, "bound": bound, "abs_contrib": abs_contrib, "tie_elements": ties}


def emulate_kernel_w4a16(x_bf16, w13_packed, w13_sf, w13_gs, w2_packed, w2_sf, w2_gs, topk_ids, topk_wts, block: int = BLOCK,
                         gate_first: bool = False, weight_rounding: str = "bf16", fault=None) -> torch.Tensor:
    """A faithful emulation of the SM120 W4A16 path's rounding points, for the selftest. `fault(contrib, e)` may corrupt an
    expert's contribution."""
    from .reference import dequantize_nvfp4
    x = x_bf16.detach().cpu().float()
    m, hdim = x.shape
    e_count, i2, _ = w13_packed.shape
    inter = i2 // 2
    ids = topk_ids.cpu().long()
    wts = topk_wts.cpu().float()
    gs13 = torch.as_tensor(w13_gs, dtype=torch.float32).reshape(-1)
    gs2 = torch.as_tensor(w2_gs, dtype=torch.float32).reshape(-1)
    acc = torch.zeros(m, hdim, dtype=torch.float32)
    for e in range(e_count):
        tok, slot = (ids == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        g13 = gs13[e] if gs13.numel() == e_count else gs13[0]
        g2 = gs2[e] if gs2.numel() == e_count else gs2[0]
        w13 = dequantize_nvfp4(w13_packed[e], w13_sf[e], g13, block)
        w2q = dequantize_nvfp4(w2_packed[e], w2_sf[e], g2, block) * g2            # q * sf, as the kernel accumulates it
        fc1 = (x[tok].double() @ w13.double().t()).float().to(torch.bfloat16).float()
        gate, up = (fc1[:, :inter], fc1[:, inter:]) if gate_first else (fc1[:, inter:], fc1[:, :inter])
        silu = (gate / (1.0 + torch.exp(-gate))).to(torch.bfloat16).float()
        h = (silu * up).to(torch.bfloat16).float()
        c = (h.double() @ w2q.double().t()).float()
        wa = wts[tok, slot] * (1.0 / g2)
        if weight_rounding == "bf16":
            wa = wa.to(torch.bfloat16).float()
        contrib = wa[:, None] * c
        if fault is not None:
            contrib = fault(contrib, e)
        acc.index_add_(0, tok, contrib)
    return acc.to(torch.bfloat16)


def selftest() -> int:
    from .reference import quantize_nvfp4
    torch.manual_seed(20261011)
    e_count, hdim, inter, m, topk = 4, 256, 128, 6, 2
    x = torch.randn(m, hdim).to(torch.bfloat16)
    w13 = [quantize_nvfp4(torch.randn(2 * inter, hdim) * 0.05) for _ in range(e_count)]
    w2 = [quantize_nvfp4(torch.randn(hdim, inter) * 0.05) for _ in range(e_count)]
    w13_p, w13_s, w13_g = torch.stack([q[0] for q in w13]), torch.stack([q[1] for q in w13]), torch.cat([q[2] for q in w13])
    w2_p, w2_s, w2_g = torch.stack([q[0] for q in w2]), torch.stack([q[1] for q in w2]), torch.cat([q[2] for q in w2])
    ids = torch.stack([torch.randperm(e_count)[:topk] for _ in range(m)]).to(torch.int32)
    wts = torch.rand(m, topk)
    wts = (wts / wts.sum(-1, keepdim=True)).float()
    args = (x, w13_p, w13_s, w13_g, w2_p, w2_s, w2_g, ids, wts)
    ok = True
    for gf in (False, True):
        o = fused_moe_w4a16_oracle(*args, gate_first=gf)
        nominal, b = o["nominal"], o["bound"]
        y = emulate_kernel_w4a16(*args, gate_first=gf).double()
        inside = bool(((y - nominal).abs() <= b).all())
        ok &= inside
        print(f"gate_first={gf}: faithful emulation inside the bound: {inside}; max err/bound {float(((y - nominal).abs() / b).max()):.3f}; "
              f"tie elements {o['tie_elements']}; mean bound / |contrib| {float((b / o['abs_contrib'].clamp(min=1e-30)).mean()):.2e}")
    o = fused_moe_w4a16_oracle(*args)
    nominal, b = o["nominal"], o["bound"]
    y_other = emulate_kernel_w4a16(*args, gate_first=True).double()           # the other half as the gate: a wrong contract
    frac = float(((y_other - nominal).abs() > b).double().mean())
    ok &= frac > 0.9
    print(f"the swapped gate order is outside the bound on {frac:.3f} of the elements")
    y_fp32w = emulate_kernel_w4a16(*args, weight_rounding="fp32").double()   # the weight in fp32: a design difference of 2^-9
    frac = float(((y_fp32w - nominal).abs() > b).double().mean())            # relative, below the floor; recorded, not asserted
    print(f"an fp32 routing weight (not the kernel's bf16 pack) is outside the bound on {frac:.3f} of the elements (below the floor)")
    o32 = fused_moe_w4a16_oracle(*args, weight_rounding="fp32")
    inside = bool(((y_fp32w - o32["nominal"]).abs() <= o32["bound"]).all())
    ok &= inside
    print("... and inside the bound of the oracle told weight_rounding='fp32':", inside)

    def drop_group(contrib, e):  # a lost 8-column group of expert 1's contribution
        c = contrib.clone()
        if e == 1:
            c[:, 64:72] = 0.0
        return c
    y3 = emulate_kernel_w4a16(*args, fault=drop_group).double()
    affected = (ids == 1).any(dim=1)
    caught = bool(((y3 - nominal).abs() > b)[affected][:, 64:72].all())
    ok &= caught
    print("a dropped 8-column group is outside the bound on every affected element:", caught)

    def off_by(rel):
        def fault(contrib, e):
            return contrib * (1.0 + rel) if e == 2 else contrib
        return fault
    affected = (ids == 2).any(dim=1)
    y4 = emulate_kernel_w4a16(*args, fault=off_by(2.0 ** -8)).double()
    frac8 = float(((y4 - nominal).abs() > b)[affected].double().mean())
    y5 = emulate_kernel_w4a16(*args, fault=off_by(2.0 ** -5)).double()
    frac5 = float(((y5 - nominal).abs() > b)[affected].double().mean())
    y6 = emulate_kernel_w4a16(*args, fault=off_by(2.0 ** -3)).double()
    frac3 = float(((y6 - nominal).abs() > b)[affected].double().mean())
    ok &= frac5 > 0.5 and frac3 > 0.8
    print(f"the floor: a contribution off by 2^-8 is outside the bound on {frac8:.3f} of the affected elements, "
          f"off by 2^-5 on {frac5:.3f}, off by 2^-3 on {frac3:.3f}")
    ok &= bool(torch.isfinite(nominal).all() and torch.isfinite(b).all() and (b > 0).all())
    print("selftest", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(selftest())
