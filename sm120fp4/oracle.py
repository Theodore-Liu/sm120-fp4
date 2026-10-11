"""Exact oracle for NVFP4 GEMMs and quantizers: conformance without a hand-chosen tolerance.

`reference.py` compares a kernel against an fp32 GEMM over dequantized operands and a relative-error threshold. That
catches a kernel that is wrong by a lot and says nothing precise about one that is wrong by a little, which is the class
the 2026 conformance literature shows tolerance checks miss (arXiv 2609.00363: four of five injected epilogue faults
undetected by any tolerance check). This module computes the value the format defines, exactly, and a bound on how far a
correct kernel may sit from it; a result outside the bound is a defect by definition, one inside it is conformant.

Why it can be exact. An NVFP4 operand element is q * sf / gs with q on the E2M1 grid (a half-integer in [-6, 6], so 2q is
an integer with |2q| <= 12), sf an E4M3 value (an integer mantissa of at most 4 bits times a power of two) and gs the
per-tensor fp32 scale. Pull the per-tensor scales out: for row i of A and row j of B,

    C[i, j] = (1 / (gs_a * gs_b)) * sum_k (qa[i, k] * sfa[i, k // 16]) * (qb[j, k] * sfb[j, k // 16]).

Each product (2qa * 2qb) * (sfa * sfb) is an integer times a power of two with at most 4 + 4 + 8 = 16 significant bits;
a sum of K <= 2^20 of them has at most 36 significant bits, below float64's 53, so the sum is computed exactly in
float64 when the powers of two are aligned (they are: every E4M3 value is a multiple of 2^-9, so every product is an
integer multiple of 2^-20 after the factor 1/4 from the two half-integers). The division by gs_a * gs_b is one float64
rounding (relative 2^-53), and the cast to the output dtype is the final rounding: those two are the only inexact
steps, and both are accounted for in the bound.

The bound for a kernel that accumulates in fp32 (every shipped SM120 path does) and writes bf16 or fp16:

    |kernel - exact| <= K * u32 * sum_k |a_k * b_k|   (fp32 accumulation, any order; u32 = 2^-24)
                      + 0.5 ulp_out(|exact|)          (the output rounding)
                      + 2^-52 * |exact|               (the oracle's own division)

K * u32 is the classical worst case for K additions in any order (Higham, gamma_K to first order), so it over-covers a
tree reduction; a kernel outside it is wrong, not unlucky. The bound is per element, from the operands, so it is a
function of the test, not of the device.

The detectability floor that follows. A fault of size d at one output element is visible only where d exceeds the
bound, and the bound's fp32 term grows with K: a single wrong operand code changes one of K products, so at K = 4096 it
is smaller than the worst case of fp32 accumulation and no fp32-accumulating conformance check, this one included, can
see it; at K <= 1024 on random operands it is outside the bound (the selftest records both). Faults that drop or
duplicate a whole 16-wide block partial (the b12x W4A4 class in docs/stage2-baselines.md) are above the floor at every
K tested. A census that wants single-code faults at large K splits K into chunks the kernel can be run on separately,
or compares two kernels bitwise under a forced accumulation order; the floor is stated so a "conformant" verdict is read
with it.

    from sm120fp4.oracle import exact_gemm_nvfp4, conformance
    exact, bound = exact_gemm_nvfp4(a_packed, a_sf, a_gs, b_packed, b_sf, b_gs, out_dtype=torch.bfloat16)
    verdict = conformance(kernel_out, exact, bound)
    python -m sm120fp4.oracle --selftest
"""
from __future__ import annotations

import sys

import torch

from .reference import BLOCK, E2M1_MAX, E4M3_MAX, e2m1_decode, unpack_e2m1

# The per-addition unit of the accumulation bound. Tensor cores accumulate in fp32 by aligning the addends to the largest
# exponent and truncating (round toward zero), so one addition can err by a full ulp of the running sum, not the half ulp of
# round-to-nearest (Fasi, Higham, Mikaitis and Pranesh, "Numerical behavior of NVIDIA tensor cores", 2021). The dense census
# rows read against a 2^-24 unit sat at 0.95 of that bound, and a fused-MoE row crossed it by 4 percent on one element of
# 65,536; with the truncation unit every row is inside. 2026-10-11.
U32 = 2.0 ** -23
U64 = 2.0 ** -53


def _ulp(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The spacing of `dtype` at |x| (float64 in, float64 out); subnormals take the smallest normal spacing."""
    info = torch.finfo(dtype)
    mant = {torch.bfloat16: 7, torch.float16: 10, torch.float32: 23, torch.float64: 52}[dtype]
    ax = x.abs().clamp(min=info.tiny)
    e = torch.floor(torch.log2(ax))
    return torch.pow(2.0, e - mant)


def dequant_integer_parts(packed: torch.Tensor, sf_bits: torch.Tensor, block: int = BLOCK):
    """Return (2q as float64 integers [M, K], sf as float64 [M, K/block]) so products are exact in float64."""
    codes = unpack_e2m1(packed)
    twoq = (2.0 * e2m1_decode(codes).double())            # integers in [-12, 12]
    sf = sf_bits.view(torch.float8_e4m3fn).double()        # exact: E4M3 values are float64-representable
    return twoq, sf


def exact_gemm_nvfp4(a_packed, a_sf, a_gs, b_packed, b_sf, b_gs, out_dtype=torch.bfloat16, block: int = BLOCK):
    """Exact C[M, N] = dequant(A)[M, K] @ dequant(B)[N, K]^T (float64, exact up to the two roundings named in the module
    docstring) and the per-element bound an fp32-accumulating kernel writing `out_dtype` must respect.

    Returns (exact as float64 [M, N], bound as float64 [M, N])."""
    twoqa, sfa = dequant_integer_parts(a_packed, a_sf, block)
    twoqb, sfb = dequant_integer_parts(b_packed, b_sf, block)
    m, k = twoqa.shape
    n = twoqb.shape[0]
    if k % block or twoqb.shape[1] != k:
        raise ValueError(f"K must match and be a multiple of {block}: {twoqa.shape} vs {twoqb.shape}")
    # scaled integer operands: (2q * sf) is an integer multiple of 2^-9 with <= 12 significant bits
    sa = twoqa.view(m, k // block, block) * sfa[:, :, None]
    sb = twoqb.view(n, k // block, block) * sfb[:, :, None]
    sa = sa.view(m, k)
    sb = sb.view(n, k)
    # the sum of products is exact in float64 (<= 36 significant bits for K <= 2^20); matmul order is irrelevant
    s = sa @ sb.t()
    sabs = sa.abs() @ sb.abs().t()
    inv = 1.0 / (4.0 * a_gs.double().reshape(()) * b_gs.double().reshape(()))   # the 4 undoes the two factors of 2
    exact = s * inv
    sum_abs = sabs * inv
    bound = k * U32 * sum_abs + 0.5 * _ulp(exact, out_dtype) + U64 * exact.abs() * 2
    return exact, bound


def exact_gemm_bf16_nvfp4(a_bf16, b_packed, b_sf, b_gs, out_dtype=torch.bfloat16, block: int = BLOCK):
    """W4A16: C[M, N] = A[M, K] (bf16, exact in float64) @ dequant(B)[N, K]^T with B in NVFP4. Each product of a bf16 (8 significant
    bits) and a scaled code (at most 7) is exact in float64; the sum over K is a float64 sum whose own rounding (K * 2^-53 relative)
    is added to the bound, so the result is exact to within that and the two roundings the NVFP4 oracle names. The bound for an
    fp32-accumulating kernel is as in exact_gemm_nvfp4."""
    a = a_bf16.detach().cpu().double()
    twoqb, sfb = dequant_integer_parts(b_packed, b_sf, block)
    n, k = twoqb.shape
    if a.shape[1] != k:
        raise ValueError(f"K must match: {tuple(a.shape)} vs {tuple(twoqb.shape)}")
    sb = (twoqb.view(n, k // block, block) * sfb[:, :, None]).view(n, k)       # 2 * q * sf, exact
    inv = 1.0 / (2.0 * b_gs.double().reshape(()))
    s = a @ sb.t()
    sabs = a.abs() @ sb.abs().t()
    exact = s * inv
    sum_abs = sabs * inv
    bound = k * U32 * sum_abs + 0.5 * _ulp(exact, out_dtype) + U64 * exact.abs() * 2 + k * U64 * sum_abs
    return exact, bound


def exact_grouped_gemm_nvfp4(a_packed, a_sf, a_gs, b_packed, b_sf, b_gs, m_indptr, out_dtype=torch.bfloat16, block: int = BLOCK):
    """Exact grouped GEMM: rows m_indptr[e]:m_indptr[e+1] of A against expert e's weights B[e] ([E, N, K/2] codes, [E, N, K/block]
    scales, per-expert or shared fp32 global scale), assembled expert by expert from exact_gemm_nvfp4. Returns (exact, bound) of
    shape [cum_m, N]; an expert with no rows contributes nothing. This is the oracle for grouped_mm_fp4-style kernels and for the
    expert GEMMs inside a fused MoE whose activations are already quantised."""
    m_indptr = [int(v) for v in torch.as_tensor(m_indptr).reshape(-1).tolist()]
    e_count = b_packed.shape[0]
    if len(m_indptr) != e_count + 1:
        raise ValueError(f"m_indptr has {len(m_indptr)} entries for {e_count} experts")
    n = b_packed.shape[1]
    exact = torch.zeros(m_indptr[-1], n, dtype=torch.float64)
    bound = torch.zeros(m_indptr[-1], n, dtype=torch.float64)
    b_gs_t = torch.as_tensor(b_gs, dtype=torch.float64).reshape(-1)
    for e in range(e_count):
        lo, hi = m_indptr[e], m_indptr[e + 1]
        if hi <= lo:
            continue
        gs_e = b_gs_t[e] if b_gs_t.numel() == e_count else b_gs_t[0]
        ex, bd = exact_gemm_nvfp4(a_packed[lo:hi], a_sf[lo:hi], a_gs, b_packed[e], b_sf[e], gs_e.reshape(1), out_dtype, block)
        exact[lo:hi] = ex
        bound[lo:hi] = bd
    return exact, bound


def conformance(kernel_out: torch.Tensor, exact: torch.Tensor, bound: torch.Tensor) -> dict:
    """Compare a kernel's output with the exact value under the bound. Every number a report needs, no threshold."""
    k = kernel_out.detach().double().cpu()
    e = exact.cpu()
    b = bound.cpu()
    err = (k - e).abs()
    outside = err > b
    ratio = torch.where(b > 0, err / b, torch.zeros_like(err))
    finite = torch.isfinite(k)
    return {
        "elements": int(err.numel()),
        "non_finite": int((~finite).sum()),
        "outside_bound": int(outside.sum()),
        "max_err_over_bound": float(ratio.max()) if ratio.numel() else 0.0,
        "max_abs_err": float(err.max()) if err.numel() else 0.0,
        "all_zero_output": bool((k == 0).all()) if k.numel() else False,
        "conformant": bool(finite.all() and not outside.any()),
    }


def _e4m3_tie_mask(sf_fp32: torch.Tensor) -> torch.Tensor:
    """True where an fp32 scale value sits exactly halfway between two adjacent positive E4M3 values (saturation at 448 aside),
    so that round-to-nearest-even and round-half-away-from-zero legitimately give different E4M3 codes."""
    grid = torch.arange(0, 256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    grid = grid[torch.isfinite(grid) & (grid > 0)].unique().sort().values
    v = sf_fp32.float().reshape(-1)
    idx = torch.searchsorted(grid, v)
    lo = grid[(idx - 1).clamp(min=0)]
    hi = grid[idx.clamp(max=grid.numel() - 1)]
    tie = (v == (lo + hi) / 2) & (lo < hi) & (v > 0) & (v < E4M3_MAX)
    return tie.reshape(sf_fp32.shape)


def quantizer_conformance(x: torch.Tensor, packed: torch.Tensor, sf_bits: torch.Tensor, gs: torch.Tensor, block: int = BLOCK) -> dict:
    """Check a library quantizer's codes and scales against the format's definition (reference.quantize_nvfp4 with the
    library's own global scale). Ties are reported apart, for the scales as for the codes: at an exact midpoint the
    reference's round-to-nearest-even and a kernel's round-half-away-from-zero give adjacent E4M3 codes and both are
    valid roundings of the same number (vLLM 0.28.0 and FlashInfer 0.6.16 both round such scale ties up; torch's
    float8 cast rounds them to even), and a scale that differs at a tie changes the codes of its whole block, so a
    code difference inside such a block is counted with the scale tie and not as a code defect."""
    from .reference import quantize_nvfp4, E2M1_GRID
    ref_packed, ref_sf, _ = quantize_nvfp4(x.float(), gs.float(), block)
    codes = unpack_e2m1(packed.cpu())
    ref_codes = unpack_e2m1(ref_packed.cpu())
    sf_lib = sf_bits.cpu().view(torch.uint8).reshape(x.shape[0], -1)
    sf_ref = ref_sf.cpu().view(torch.uint8).reshape(x.shape[0], -1)
    sf_diff = sf_lib != sf_ref
    xm = x.float().cpu().view(x.shape[0], -1, block)
    gsf = gs.float().cpu().reshape(())
    sf_f32 = (xm.abs().amax(dim=-1) / E2M1_MAX * gsf)
    sf_tie = _e4m3_tie_mask(sf_f32)
    # a tie excuses a difference only when both sides chose one of the two neighbours of the midpoint
    lib_val = sf_lib.view(torch.float8_e4m3fn).float()
    ref_val = sf_ref.view(torch.float8_e4m3fn).float()
    grid = torch.arange(0, 256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    grid = grid[torch.isfinite(grid) & (grid > 0)].unique().sort().values
    idx = torch.searchsorted(grid, sf_f32.reshape(-1))
    lo = grid[(idx - 1).clamp(min=0)].reshape(sf_f32.shape)
    hi = grid[idx.clamp(max=grid.numel() - 1)].reshape(sf_f32.shape)
    neighbours = ((lib_val == lo) | (lib_val == hi)) & ((ref_val == lo) | (ref_val == hi))
    sf_diff_tie = sf_diff & sf_tie & neighbours
    sf_diff_other = sf_diff & ~sf_diff_tie
    # codes: ties on the element grid, with the library's own scale, excluding blocks whose scale differs at a tie
    sf_val = sf_lib.view(torch.float8_e4m3fn).float()
    scale = torch.where(sf_val > 0, gsf / sf_val, torch.zeros_like(sf_val))
    y = (xm * scale[..., None]).abs().reshape(x.shape[0], -1)
    # a code tie: |x * gs / sf| within two fp32 ulps of an E2M1 midpoint. The exact midpoint is the round-to-even
    # against round-half-away case; the two-ulp band covers the order in which a kernel multiplies and divides
    # (x * (gs / sf) against (x * gs) / sf), which moves a value sitting on a midpoint by one ulp either way.
    mids = (E2M1_GRID[1:] + E2M1_GRID[:-1]) / 2
    ulp_y = torch.finfo(torch.float32).eps * y.clamp(min=torch.finfo(torch.float32).tiny)
    at_tie = ((y[..., None] - mids).abs() <= 2 * ulp_y[..., None]).any(dim=-1)
    in_tie_block = sf_diff_tie.repeat_interleave(block, dim=1)
    diff = codes != ref_codes
    return {
        "elements": int(codes.numel()),
        "blocks": int(sf_diff.numel()),
        "scales_differ": int(sf_diff.sum()),
        "scales_differ_at_tie": int(sf_diff_tie.sum()),
        "scales_differ_not_at_tie": int(sf_diff_other.sum()),
        "scales_identical": bool(not sf_diff.any()),
        "codes_differ": int(diff.sum()),
        "codes_differ_at_tie": int((diff & at_tie & ~in_tie_block).sum()),
        "codes_differ_in_scale_tie_blocks": int((diff & in_tie_block).sum()),
        "codes_differ_not_at_tie": int((diff & ~at_tie & ~in_tie_block).sum()),
        "conformant": int(sf_diff_other.sum()) == 0 and int((diff & ~at_tie & ~in_tie_block).sum()) == 0,
    }


def selftest() -> int:
    from .reference import quantize_nvfp4, reference_gemm_nvfp4
    torch.manual_seed(0)
    ok = True
    for (m, n, k) in ((4, 8, 64), (16, 32, 1024), (3, 5, 4096)):
        a = torch.randn(m, k) * 3.0
        b = torch.randn(n, k)
        ap, asf, ags = quantize_nvfp4(a)
        bp, bsf, bgs = quantize_nvfp4(b)
        exact, bound = exact_gemm_nvfp4(ap, asf, ags, bp, bsf, bgs, out_dtype=torch.bfloat16)
        # 1. the fp32 reference GEMM (a correct fp32-accumulating implementation) sits inside the bound after bf16 rounding
        ref = reference_gemm_nvfp4(ap, asf, ags, bp, bsf, bgs, out_dtype=torch.bfloat16)
        v = conformance(ref, exact, bound)
        ok &= v["conformant"]
        print(f"  {m}x{n}x{k}: fp32 reference inside the bound: {v['conformant']} (max err/bound {v['max_err_over_bound']:.3f})")
        # 2. the exact value agrees with an independent integer computation on the first element
        twoqa, sfa = dequant_integer_parts(ap, asf)
        twoqb, sfb = dequant_integer_parts(bp, bsf)
        s0 = 0.0
        for kk in range(k):
            s0 += (twoqa[0, kk] * sfa[0, kk // BLOCK]).item() * (twoqb[0, kk] * sfb[0, kk // BLOCK]).item()
        e0 = s0 / (4.0 * ags.double().item() * bgs.double().item())
        ok &= abs(e0 - exact[0, 0].item()) <= 4 * U64 * abs(e0) + 1e-300
        # 3. a flipped nibble in one operand code (a wrong element) is outside the bound somewhere in its row
        bad = ap.clone()
        bad[0, 0] ^= 0x07
        wrong, _ = exact_gemm_nvfp4(bad, asf, ags, bp, bsf, bgs, out_dtype=torch.bfloat16)
        v3 = conformance(wrong.to(torch.bfloat16), exact, bound)
        # The detectability floor: a fault of size d at one element is visible only where d exceeds the bound, and the
        # fp32 term of the bound grows with K. One wrong code is one term of K, so at K = 4096 it sits inside the
        # worst case of fp32 accumulation and no fp32-accumulating conformance check can see it; at K <= 1024 on these
        # operands it is outside. The selftest asserts the small-K case and records the large-K count as the floor.
        if k <= 1024:
            ok &= v3["outside_bound"] >= 1 and not v3["conformant"]
        print(f"  flipped code -> outside the bound: {v3['outside_bound']} element(s)"
              + ("" if k <= 1024 else "  (K = 4096: a one-code fault is below the fp32 floor, as the module documents)"))
        # 4. a lost 16-wide partial (one block's contribution dropped) is caught
        lost = exact.clone()
        blk = (twoqa[1].view(-1, BLOCK)[0] * sfa[1, 0])[None, :] * (twoqb.view(n, -1, BLOCK)[:, 0, :] * sfb[:, 0:1])
        lost[1] -= blk.sum(dim=1) / (4.0 * ags.double().item() * bgs.double().item())
        v4 = conformance(lost.to(torch.bfloat16), exact, bound)
        ok &= v4["outside_bound"] >= 1
        print(f"  dropped block partial -> outside the bound: {v4['outside_bound']} element(s)")
        # 5. the quantizer check accepts the reference quantizer's own output and rejects a changed code
        q = quantizer_conformance(a, ap, asf, ags)
        ok &= q["conformant"] and q["codes_differ"] == 0 and q["scales_identical"]
        # a scale moved up at an exact E4M3 midpoint (the kernels' rounding) is a tie, not a defect; moved elsewhere it is one
        tie_x = torch.full((1, 16), 2.265625); tie_gs = torch.tensor([307.2])      # sf_fp32 = 2.265625 / 6 * 307.2 = 116, midway 112 | 120
        tp, tsf, _ = quantize_nvfp4(tie_x, tie_gs)
        up = tsf.clone(); up[0, 0] = torch.tensor(120.0).to(torch.float8_e4m3fn).view(torch.uint8)
        from .reference import e2m1_encode
        codes_up = e2m1_encode(tie_x * (tie_gs / 120.0)).view(1, 16)
        from .reference import pack_e2m1
        qt = quantizer_conformance(tie_x, pack_e2m1(codes_up), up, tie_gs)
        ok &= bool(qt["scales_differ_at_tie"] == 1 and qt["scales_differ_not_at_tie"] == 0 and qt["conformant"])
        print(f"  scale tie rounded up -> tie, conformant: {qt['conformant']}")
        off = tsf.clone(); off[0, 0] = torch.tensor(128.0).to(torch.float8_e4m3fn).view(torch.uint8)
        qo = quantizer_conformance(tie_x, tp, off, tie_gs)
        ok &= bool(qo["scales_differ_not_at_tie"] == 1 and not qo["conformant"])
        print(f"  scale moved off a midpoint -> defect: {not qo['conformant']}")
        q2 = quantizer_conformance(a, bad, asf, ags)
        ok &= not q2["conformant"] and q2["codes_differ_not_at_tie"] >= 1
    # 7. the grouped oracle equals the dense oracle expert by expert, and a row routed to the wrong expert is outside the bound
    E, n, k = 3, 8, 256
    a = torch.randn(10, k); ap, asf, ags = quantize_nvfp4(a)
    bs = [quantize_nvfp4(torch.randn(n, k)) for _ in range(E)]
    bp = torch.stack([b[0] for b in bs]); bsf = torch.stack([b[1] for b in bs]); bgs = torch.cat([b[2] for b in bs])
    indptr = torch.tensor([0, 4, 4, 10])
    ge, gb = exact_grouped_gemm_nvfp4(ap, asf, ags, bp, bsf, bgs, indptr)
    d0, _ = exact_gemm_nvfp4(ap[0:4], asf[0:4], ags, bp[0], bsf[0], bgs[0:1])
    d2, _ = exact_gemm_nvfp4(ap[4:10], asf[4:10], ags, bp[2], bsf[2], bgs[2:3])
    ok &= bool(torch.equal(ge[0:4], d0) and torch.equal(ge[4:10], d2))
    swapped, _ = exact_gemm_nvfp4(ap[0:4], asf[0:4], ags, bp[1], bsf[1], bgs[1:2])
    ok &= not conformance(torch.cat([swapped, d2]).to(torch.bfloat16), ge, gb)["conformant"]
    print("  grouped oracle: equals the dense one per expert and catches a mis-routed expert")
    # 8. the W4A16 oracle agrees with the NVFP4 one when the activation is itself an exactly representable dequantised NVFP4 row set
    a = torch.randn(6, 256); ap, asf, ags = quantize_nvfp4(a)
    from .reference import dequantize_nvfp4
    a_deq = dequantize_nvfp4(ap, asf, ags)                       # fp32 values; representable in bf16 only when small codes: use fp32->bf16 round trip check
    b = torch.randn(5, 256); bp, bsf, bgs = quantize_nvfp4(b)
    e_w, _ = exact_gemm_bf16_nvfp4(a_deq.to(torch.bfloat16), bp, bsf, bgs)
    twoqb, sfb = dequant_integer_parts(bp, bsf)                  # the float64 dequantisation, not reference.py's fp32 one
    b_deq64 = (twoqb.view(5, -1, BLOCK) * sfb[:, :, None]).view(5, -1) / (2.0 * bgs.double())
    ref = a_deq.to(torch.bfloat16).double() @ b_deq64.t()
    ok &= bool((e_w - ref).abs().max() <= 1e-12 * ref.abs().max() + 1e-300)
    print("  W4A16 oracle equals the float64 product of the dequantised operands")
    # 6. a zero bound never divides by zero and an all-zero output is named
    z = conformance(torch.zeros(2, 2), torch.zeros(2, 2, dtype=torch.float64), torch.zeros(2, 2, dtype=torch.float64))
    ok &= z["conformant"] and z["all_zero_output"]
    print("selftest", "ok" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    print(__doc__)
