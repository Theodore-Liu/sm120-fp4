"""An oracle for a fused W4A4 MoE layer: NVFP4 activations quantised inside the kernel, NVFP4 expert weights, gated SiLU.

The W4A16 oracle (`moe_oracle.py`) follows a kernel's rounding points and keeps, at each, the set of values a correct
rounding of an fp32-accumulated quantity can produce. The W4A4 path (FlashInfer's b12x `quant_mode="nvfp4"`, read in
`docs/b12x-w4a4-oracle-notes.md`) adds two block quantisers whose outputs are also set-valued, and two combine designs:

  in     x (bf16, exact) -> block quantiser: scale = E4M3(min(max|x| * gs_in / 6, 448)), codes = E2M1(x * gs_in / scale)
  FC1    exact products of the quantised x and W13 (float64), fp32-accumulation bound
  act    silu(gate) * up in fp32 (exp and an approximate reciprocal), written as bf16
  mid    block quantiser again on the bf16 activations with gs_mid (`fc2_input_scale`)
  FC2    exact products of the quantised activations and W2, fp32-accumulation bound
  static FC2 in K-chunks of 256: each partial rounded to bf16, multiplied in bf16x2 by combined_scale = bf16(topk_weight *
         alpha2), every weighted partial of every route summed in fp32 by the finalize, one bf16 output rounding (found by
         fitting identity-FC1 probes to the kernel bit for bit, 2026-10-11; the chunk is 256 at K = 256, 512 and 1024)
  micro  FC2 in one fp32 accumulation; the route result times alpha2 held as bf16, multiplied by the fp32 weight, rounded
         to bf16 and added into the output with a bf16 atomic (`red.add.noftz.bf16x2`), every partial sum rounded to bf16
         in an order the hardware chooses (at top-2 this is one rounding and matched the kernel bit for bit)

Both quantisers are the fast path (`quantize_block_fp4_fast`, the dispatcher's default): scale = gs * (max * rcp(6)) in
fp32 then `cvt.rn` to E4M3; codes = value * (rcp(scale) * gs) in fp32 then `cvt.rn.satfinite` to E2M1. On the RTX 5090
`rcp.approx.ftz.f32` equals the fp32 round-to-nearest reciprocal for 6 and for every E4M3 mantissa and scales exactly with
powers of two (`scripts/measure_rcp_approx.py`, `reports/rcp-approx-rtx5090-20261011.json`), so fp32 arithmetic reproduces
the quantiser bit for bit. That matters because bf16 inputs produce exact ties often (a bf16 block maximum over 6 lands on an
E4M3 midpoint for about one block in nine): the input quantiser is therefore emulated exactly on the exact bf16 x, and the
mid quantiser is evaluated at both ends of each activation's interval and under both candidate scales (from the block
maximum's two ends), the hull being the element's interval. A block whose two scale candidates differ is counted as
"ambiguous". The combine model is a parameter: `"static"`, `"micro"`, or `"dynamic"` (the gated dynamic kernel: the static
design's bf16 partials and bf16x2 weighting, but each weighted partial added into the output by a bf16 atomic
`red.global.add.noftz.v4.bf16x2`, one rounding per partial in an order the hardware chooses; fitted on 2026-10-11 with identity
probes at 2,048 routed rows: chunk 256, every run inside the set of orderings, the fp32-sum alternative excluded).

The floor is as for the W4A16 oracle: set by the first GEMM's worst-case accumulation bound at the bf16 write before the
mid quantiser, and for the micro combine by the k bf16 roundings of the partial sums. Faults of the kinds the census has
found (a dropped column group, a mis-addressed scale, a wrong half of w13) are far above it.

Conventions as in `sm120fp4.reference`: packed E2M1 codes [rows, K/2] (even element in the low nibble), E4M3 scale bits
[rows, K/16] row-major uint8, dequant = q * sf / gs. w13 is [E, 2I, H/2] with the b12x order (second half the gate unless
`gate_first`), w2 is [E, H, I/2].
"""
from __future__ import annotations

import torch

from .moe_oracle import _hull_prod, _hull_silu, _round_bf16_interval, dequant_expert_w2
from .oracle import BLOCK, U32, U64, _ulp, dequant_integer_parts
from .reference import E2M1_MAX, E4M3_MAX, e2m1_decode, e2m1_encode

RCP_REL = 2.0 ** -21     # the activation stage's fixed part: the approximate reciprocal in silu and the exponential's own ulps
F32_INV6 = (1.0 / torch.tensor(6.0, dtype=torch.float32))   # rcp.approx(6) on the RTX 5090 equals the fp32 RN reciprocal


def _f32(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32)


def fast_block_scale(max_abs_f32: torch.Tensor, gs: float) -> torch.Tensor:
    """The fast quantiser's E4M3 block scale from an exact fp32 block maximum: gs * (max * rcp(6)) in fp32, clamped to 448,
    `cvt.rn` to E4M3 (round to nearest even). `rcp.approx` was measured equal to the fp32 RN reciprocal on this device for 6
    and for every E4M3 mantissa (`reports/rcp-approx-rtx5090-20261011.json`), so fp32 arithmetic reproduces it bit for bit."""
    s = _f32(torch.tensor(gs)) * (max_abs_f32 * F32_INV6)
    return s.clamp(max=E4M3_MAX).to(torch.float8_e4m3fn).float()


def fast_block_codes(values_f32: torch.Tensor, scale_f32: torch.Tensor, gs: float) -> torch.Tensor:
    """The fast quantiser's E2M1 codes as values: value * (rcp(scale) * gs) in fp32, `cvt.rn.satfinite` to E2M1 (RNE, 6 max)."""
    safe = torch.where(scale_f32 == 0, torch.ones_like(scale_f32), scale_f32)
    mult = (1.0 / safe) * _f32(torch.tensor(gs))
    q = values_f32 * mult
    codes = e2m1_decode(e2m1_encode(q))
    return torch.where(scale_f32 == 0, torch.zeros_like(codes), codes)


def quantize_block_fast_exact(v_f32: torch.Tensor, gs: float, block: int = BLOCK):
    """Exact emulation of the fast quantiser on exact fp32 inputs: dequantised values (float64 [rows, K]) and scales."""
    rows, k = v_f32.shape
    b = v_f32.view(rows, k // block, block)
    scale = fast_block_scale(b.abs().amax(dim=-1, keepdim=True), gs)
    codes = fast_block_codes(b, scale, gs)
    return (codes.double() * scale.double() / gs).view(rows, k), scale


def _e4m3_grid() -> torch.Tensor:
    """Every non-negative E4M3 value, ascending (float32), the scale candidates a block can take."""
    vals = torch.arange(0, 127, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).float()   # 0x00..0x7E
    vals = vals[torch.isfinite(vals)]
    return torch.unique(vals)


E4M3_GRID = _e4m3_grid()


def block_quantize_interval(lo: torch.Tensor, hi: torch.Tensor, gs: float, block: int = BLOCK):
    """Set-valued fast quantisation of values in [lo, hi] (float64 [rows, K]; the ends are fp32-representable when they come
    from a bf16 rounding). The block maximum lies in [max_lo, max_hi]; the scale is monotone in the maximum, so its candidates
    are every E4M3 value between the scale at max_lo and the scale at max_hi (inclusive); for each candidate the codes are
    evaluated at both ends of every element (monotone in the value), and the hull over candidates and ends is the element's
    interval. Returns (vlo, vhi, ambiguous_blocks) with ambiguous_blocks the number of blocks with more than one candidate."""
    rows, k = lo.shape
    blo, bhi = _f32(lo).view(rows, k // block, block), _f32(hi).view(rows, k // block, block)
    mag_hi = torch.maximum(blo.abs(), bhi.abs())
    mag_lo = torch.where((blo <= 0) & (bhi >= 0), torch.zeros_like(blo), torch.minimum(blo.abs(), bhi.abs()))
    s_a = fast_block_scale(mag_lo.amax(dim=-1, keepdim=True), gs)
    s_b = fast_block_scale(mag_hi.amax(dim=-1, keepdim=True), gs)
    ambiguous = int((s_a != s_b).sum())
    vlo = torch.full_like(blo, float("inf"), dtype=torch.float64)
    vhi = torch.full_like(blo, float("-inf"), dtype=torch.float64)
    for g in E4M3_GRID.tolist():
        use = (s_a <= g) & (g <= s_b)                                   # [rows, blocks, 1]
        if not bool(use.any()):
            continue
        s = torch.full_like(s_a, g)
        for v in (blo, bhi):
            c = fast_block_codes(v, s, gs).double() * float(g) / gs
            vlo = torch.where(use, torch.minimum(vlo, c), vlo)
            vhi = torch.where(use, torch.maximum(vhi, c), vhi)
    return vlo.reshape(rows, k), vhi.reshape(rows, k), ambiguous


def _dequant_w(packed_e: torch.Tensor, sf_e: torch.Tensor, gs, block: int) -> torch.Tensor:
    twoq, sf = dequant_integer_parts(packed_e, sf_e, block)
    n, k = twoq.shape
    return (twoq.view(n, k // block, block) * sf[:, :, None]).view(n, k) / (2.0 * torch.as_tensor(gs, dtype=torch.float64).reshape(()))


def _interval_gemm(alo: torch.Tensor, ahi: torch.Tensor, w: torch.Tensor):
    """A in [alo, ahi] (float64 [rows, K]) against exact W [N, K]: (mid product, half-width through |W| plus the fp32
    accumulation bound and the float64 slack)."""
    mid, hw = (alo + ahi) / 2.0, (ahi - alo) / 2.0
    k = w.shape[1]
    c = mid @ w.t()
    cabs = mid.abs() @ w.abs().t()
    bound = hw @ w.abs().t() + k * U32 * cabs + (2.0 + k) * U64 * cabs
    return c, bound, cabs


def fused_moe_w4a4_oracle(x_bf16: torch.Tensor, w13_packed, w13_sf, w13_gs, w2_packed, w2_sf, w2_gs,
                          topk_ids: torch.Tensor, topk_wts: torch.Tensor, gs_in: float = 1.0, gs_mid: float = 1.0,
                          block: int = BLOCK, gate_first: bool = False, combine: str = "static", fc2_chunk: int = 256) -> dict:
    """Returns {"nominal", "bound", "abs_contrib", "ambiguous_blocks_in", "ambiguous_blocks_mid", "tie_elements"}.
    Conformant when |y - nominal| <= bound everywhere."""
    x = x_bf16.detach().cpu()
    if x.dtype != torch.bfloat16:
        raise ValueError("x must be bf16")
    m, hdim = x.shape
    e_count, i2, _ = w13_packed.shape
    inter = i2 // 2
    ids = topk_ids.detach().cpu().long()
    wts = topk_wts.detach().cpu().float()
    topk = ids.shape[1]
    gs13 = torch.as_tensor(w13_gs, dtype=torch.float64).reshape(-1)
    gs2 = torch.as_tensor(w2_gs, dtype=torch.float64).reshape(-1)
    xd = x.double()
    xq, _ = quantize_block_fast_exact(x.float(), gs_in, block)                 # x is exact, so the fast quantiser is too
    xq_lo, xq_hi, amb_in = xq, xq, 0
    out = torch.zeros(m, hdim, dtype=torch.float64)
    bound = torch.zeros(m, hdim, dtype=torch.float64)
    abs_contrib = torch.zeros(m, hdim, dtype=torch.float64)
    amb_mid = 0
    ties = 0
    for e in range(e_count):
        tok, slot = (ids == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        g13 = gs13[e] if gs13.numel() == e_count else gs13[0]
        g2 = gs2[e] if gs2.numel() == e_count else gs2[0]
        w13 = _dequant_w(w13_packed[e], w13_sf[e], g13, block)                     # [2I, H]
        f, fb, _ = _interval_gemm(xq_lo[tok], xq_hi[tok], w13)                      # FC1 in fp32
        if gate_first:
            gate, bg, up, bu = f[:, :inter], fb[:, :inter], f[:, inter:], fb[:, inter:]
        else:
            up, bu, gate, bg = f[:, :inter], fb[:, :inter], f[:, inter:], fb[:, inter:]
        glo, ghi = gate - bg, gate + bg
        slo, shi = _hull_silu(glo, ghi)
        smag = torch.maximum(slo.abs(), shi.abs()) + torch.maximum(glo.abs(), ghi.abs())
        # the fast exponential: ex2.approx is within 2 ulp and its argument scaling adds |g| * 2^-24 relative, so the slack
        # grows with |g|; the approximate reciprocal adds one more ulp
        gabs = torch.maximum(glo.abs(), ghi.abs())
        act_slack = (RCP_REL + gabs * (2.0 ** -22)) * smag
        slo, shi = slo - act_slack, shi + act_slack
        plo, phi = _hull_prod(slo, shi, up - bu, up + bu)
        hlo, hhi = _round_bf16_interval(plo, phi)                                   # written as bf16
        ties += int((hhi > hlo).sum())
        qlo, qhi, amb = block_quantize_interval(hlo, hhi, gs_mid, block)            # the mid quantiser
        amb_mid += amb
        w2 = dequant_expert_w2(w2_packed[e], w2_sf[e], g2, block)                   # [H, I], includes 1/gs2
        alpha = 1.0 / g2
        w = wts[tok, slot].double()
        if combine in ("static", "dynamic"):
            # FC2 is computed in K-chunks of `fc2_chunk` columns; each partial is rounded to bf16 (in the kernel's units, times
            # gs2), multiplied in bf16 by combined_scale = bf16(w * alpha) (`mul.rn.bf16x2`), and every weighted partial of every
            # route is summed in fp32 by the finalize, which rounds once (bit-exact against the kernel on 2026-10-11)
            wa = (w * alpha).float().to(torch.bfloat16).double()[:, None]
            lo_sum = torch.zeros(tok.numel(), hdim, dtype=torch.float64)
            hi_sum = torch.zeros_like(lo_sum)
            for c0 in range(0, inter, fc2_chunk):
                c, cb, _ = _interval_gemm(qlo[:, c0:c0 + fc2_chunk], qhi[:, c0:c0 + fc2_chunk], w2[:, c0:c0 + fc2_chunk])
                r_lo, r_hi = _round_bf16_interval((c - cb) * g2, (c + cb) * g2)
                p_lo, p_hi = _hull_prod(r_lo, r_hi, wa, wa)
                k_lo, k_hi = _round_bf16_interval(p_lo, p_hi)
                lo_sum, hi_sum = lo_sum + k_lo, hi_sum + k_hi
            contrib, cbound = (lo_sum + hi_sum) / 2.0, (hi_sum - lo_sum) / 2.0
            cbound = cbound + (inter // fc2_chunk) * U32 * torch.maximum(lo_sum.abs(), hi_sum.abs())
        elif combine == "micro":
            # FC2 in one fp32 accumulation; the route result (times alpha) sits in shared memory as bf16, is multiplied by the
            # fp32 weight, rounded to bf16 by cvt.rn and added into the output by a bf16 atomic
            c, cb, cabs = _interval_gemm(qlo, qhi, w2)
            r_lo, r_hi = _round_bf16_interval((c - cb) * g2 * alpha, (c + cb) * g2 * alpha)
            wf = w[:, None]
            p_lo, p_hi = _hull_prod(r_lo, r_hi, wf, wf)
            slack = 2.0 * U32 * torch.maximum(p_lo.abs(), p_hi.abs())
            k_lo, k_hi = _round_bf16_interval(p_lo - slack, p_hi + slack)
            contrib, cbound = (k_lo + k_hi) / 2.0, (k_hi - k_lo) / 2.0
        else:
            raise ValueError(combine)
        out.index_add_(0, tok, contrib)
        abs_contrib.index_add_(0, tok, contrib.abs())
        bound.index_add_(0, tok, cbound)
    # the output rounding's half ulp is taken at the top of the interval, not at its midpoint: the sum may sit in the binade above
    pre = bound + topk * U32 * abs_contrib + 2.0 * U64 * abs_contrib
    if combine == "static":
        bound = pre + 0.5 * _ulp(out.abs() + pre, torch.bfloat16)
    elif combine == "dynamic":
        # every weighted bf16 partial of every route lands in the output through a bf16 atomic: one rounding per partial
        n_partials = topk * max(1, inter // fc2_chunk)
        bound = pre + n_partials * 0.5 * _ulp(abs_contrib + pre, torch.bfloat16)
    else:
        bound = pre + topk * 0.5 * _ulp(abs_contrib + pre, torch.bfloat16)   # k bf16 roundings of partial sums
    return {"nominal": out, "bound": bound, "abs_contrib": abs_contrib, "ambiguous_blocks_in": amb_in,
            "ambiguous_blocks_mid": amb_mid, "tie_elements": ties}


def _quantize_block_faithful(v: torch.Tensor, gs: float, block: int = BLOCK) -> torch.Tensor:
    """The kernel's fast quantiser on fp32 values, as fp32 dequantised values (for the selftest's emulation)."""
    return quantize_block_fast_exact(v, gs, block)[0].float()


def emulate_kernel_w4a4(x_bf16, w13_packed, w13_sf, w13_gs, w2_packed, w2_sf, w2_gs, topk_ids, topk_wts, gs_in=1.0, gs_mid=1.0,
                        block: int = BLOCK, gate_first: bool = False, combine: str = "static", fault=None,
                        atomic_order: int = 0, fc2_chunk: int = 256) -> torch.Tensor:
    """A faithful emulation of the rounding points of either combine design, for the selftest."""
    from .reference import dequantize_nvfp4
    x = x_bf16.detach().cpu().float()
    m, hdim = x.shape
    e_count, i2, _ = w13_packed.shape
    inter = i2 // 2
    ids = topk_ids.cpu().long()
    wts = topk_wts.cpu().float()
    gs13 = torch.as_tensor(w13_gs, dtype=torch.float32).reshape(-1)
    gs2 = torch.as_tensor(w2_gs, dtype=torch.float32).reshape(-1)
    xq = _quantize_block_faithful(x, gs_in, block)
    acc32 = torch.zeros(m, hdim, dtype=torch.float32)
    acc_bf = torch.zeros(m, hdim, dtype=torch.float32)
    order = list(range(e_count))
    if atomic_order:
        order = order[::-1]
    for e in order:
        tok, slot = (ids == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        g13 = gs13[e] if gs13.numel() == e_count else gs13[0]
        g2 = gs2[e] if gs2.numel() == e_count else gs2[0]
        w13 = dequantize_nvfp4(w13_packed[e], w13_sf[e], g13, block)
        w2q = dequantize_nvfp4(w2_packed[e], w2_sf[e], g2, block) * g2
        fc1 = (xq[tok].double() @ w13.double().t()).float()
        gate, up = (fc1[:, :inter], fc1[:, inter:]) if gate_first else (fc1[:, inter:], fc1[:, :inter])
        h = (gate / (1.0 + torch.exp(-gate)) * up).to(torch.bfloat16).float()
        hq = _quantize_block_faithful(h, gs_mid, block)
        w = wts[tok, slot] * (1.0 / g2)
        if combine == "static":
            wb = w.to(torch.bfloat16).float()[:, None]
            contrib = torch.zeros(tok.numel(), hdim, dtype=torch.float32)
            for c0 in range(0, inter, fc2_chunk):
                part = (hq[:, c0:c0 + fc2_chunk].double() @ w2q[:, c0:c0 + fc2_chunk].double().t()).float().to(torch.bfloat16).float()
                contrib = contrib + (part * wb).to(torch.bfloat16).float()
        else:
            r = (hq.double() @ w2q.double().t()).float()
            rb = r.to(torch.bfloat16).float()
            contrib = (rb * w[:, None]).to(torch.bfloat16).float()
        if fault is not None:
            contrib = fault(contrib, e)
        if combine == "static":
            acc32.index_add_(0, tok, contrib)
        else:
            acc_bf[tok] = (acc_bf[tok] + contrib).to(torch.bfloat16).float()
    return (acc32.to(torch.bfloat16) if combine == "static" else acc_bf.to(torch.bfloat16))


def selftest() -> int:
    from .reference import quantize_nvfp4
    torch.manual_seed(20261011)
    e_count, hdim, inter, m, topk = 4, 256, 128, 6, 2
    one = torch.ones(1)
    x = torch.randn(m, hdim).to(torch.bfloat16)
    w13 = [quantize_nvfp4(torch.randn(2 * inter, hdim) * 0.05, one) for _ in range(e_count)]
    w2 = [quantize_nvfp4(torch.randn(hdim, inter) * 0.05, one) for _ in range(e_count)]
    w13_p, w13_s, w13_g = torch.stack([q[0] for q in w13]), torch.stack([q[1] for q in w13]), torch.cat([q[2] for q in w13])
    w2_p, w2_s, w2_g = torch.stack([q[0] for q in w2]), torch.stack([q[1] for q in w2]), torch.cat([q[2] for q in w2])
    ids = torch.stack([torch.randperm(e_count)[:topk] for _ in range(m)]).to(torch.int32)
    wts = torch.rand(m, topk)
    wts = (wts / wts.sum(-1, keepdim=True)).float()
    args = (x, w13_p, w13_s, w13_g, w2_p, w2_s, w2_g, ids, wts)
    ok = True
    for combine in ("static", "micro"):
        o = fused_moe_w4a4_oracle(*args, combine=combine)
        nominal, b = o["nominal"], o["bound"]
        for order in (0, 1):
            y = emulate_kernel_w4a4(*args, combine=combine, atomic_order=order).double()
            inside = bool(((y - nominal).abs() <= b).all())
            ok &= inside
            print(f"combine={combine} order={order}: faithful emulation inside the bound: {inside}; max err/bound "
                  f"{float(((y - nominal).abs() / b).max()):.3f}; ambiguous blocks in/mid {o['ambiguous_blocks_in']}/{o['ambiguous_blocks_mid']}; "
                  f"ties {o['tie_elements']}; mean bound/|contrib| {float((b / o['abs_contrib'].clamp(min=1e-30)).mean()):.2e}")
    o = fused_moe_w4a4_oracle(*args)
    nominal, b = o["nominal"], o["bound"]
    y_other = emulate_kernel_w4a4(*args, gate_first=True).double()
    frac = float(((y_other - nominal).abs() > b).double().mean())
    ok &= frac > 0.9
    print(f"the swapped gate order is outside the bound on {frac:.3f} of the elements")

    def drop_group(contrib, e):
        c = contrib.clone()
        if e == 1:
            c[:, 64:72] = 0.0
        return c
    y3 = emulate_kernel_w4a4(*args, fault=drop_group).double()
    affected = (ids == 1).any(dim=1)
    caught = bool(((y3 - nominal).abs() > b)[affected][:, 64:72].all())
    ok &= caught
    print("a dropped 8-column group is outside the bound on every affected element:", caught)

    def off_by(rel):
        def fault(contrib, e):
            return contrib * (1.0 + rel) if e == 2 else contrib
        return fault
    affected = (ids == 2).any(dim=1)
    fr = {}
    for p in (8, 5, 3):
        y = emulate_kernel_w4a4(*args, fault=off_by(2.0 ** -p)).double()
        fr[p] = float(((y - nominal).abs() > b)[affected].double().mean())
    ok &= fr[3] > 0.8
    print(f"the floor: a contribution off by 2^-8 is outside on {fr[8]:.3f}, 2^-5 on {fr[5]:.3f}, 2^-3 on {fr[3]:.3f} of the affected elements")
    ok &= bool(torch.isfinite(nominal).all() and torch.isfinite(b).all() and (b > 0).all())
    print("selftest", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(selftest())
