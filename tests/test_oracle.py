"""The exact oracle (sm120fp4.oracle): CPU-only tests, no device needed.

Failure classes it is built to catch: a kernel wrong by less than a relative-error threshold would notice (the class
tolerance checks miss, arXiv 2609.00363), a dropped or duplicated partial (the b12x W4A4 8-column group of
docs/stage2-baselines.md), an all-zero output (FlashInfer #2577), and a quantizer that rounds off the format's grid.
"""
import pytest
import torch

from sm120fp4 import oracle
from sm120fp4.reference import quantize_nvfp4, reference_gemm_nvfp4


@pytest.mark.parametrize("m,n,k", [(4, 8, 64), (16, 32, 1024), (3, 5, 4096), (1, 1, 16)])
def test_fp32_reference_is_inside_the_bound(m, n, k):
    torch.manual_seed(1)
    ap, asf, ags = quantize_nvfp4(torch.randn(m, k) * 3)
    bp, bsf, bgs = quantize_nvfp4(torch.randn(n, k))
    exact, bound = oracle.exact_gemm_nvfp4(ap, asf, ags, bp, bsf, bgs, out_dtype=torch.bfloat16)
    ref = reference_gemm_nvfp4(ap, asf, ags, bp, bsf, bgs, out_dtype=torch.bfloat16)
    v = oracle.conformance(ref, exact, bound)
    assert v["conformant"], v


def test_exact_matches_scalar_integer_sum():
    torch.manual_seed(2)
    ap, asf, ags = quantize_nvfp4(torch.randn(2, 128))
    bp, bsf, bgs = quantize_nvfp4(torch.randn(3, 128))
    exact, _ = oracle.exact_gemm_nvfp4(ap, asf, ags, bp, bsf, bgs)
    twoqa, sfa = oracle.dequant_integer_parts(ap, asf)
    twoqb, sfb = oracle.dequant_integer_parts(bp, bsf)
    s = sum((twoqa[1, kk] * sfa[1, kk // 16]).item() * (twoqb[2, kk] * sfb[2, kk // 16]).item() for kk in range(128))
    e = s / (4.0 * ags.double().item() * bgs.double().item())
    assert abs(e - exact[1, 2].item()) <= 4 * oracle.U64 * abs(e) + 1e-300


def test_flipped_code_and_dropped_partial_are_outside_the_bound():
    torch.manual_seed(3)
    ap, asf, ags = quantize_nvfp4(torch.randn(8, 512) * 2)
    bp, bsf, bgs = quantize_nvfp4(torch.randn(8, 512))
    exact, bound = oracle.exact_gemm_nvfp4(ap, asf, ags, bp, bsf, bgs, out_dtype=torch.bfloat16)
    bad = ap.clone()
    bad[0, 0] ^= 0x07
    wrong, _ = oracle.exact_gemm_nvfp4(bad, asf, ags, bp, bsf, bgs)
    assert not oracle.conformance(wrong.to(torch.bfloat16), exact, bound)["conformant"]
    lost = exact.clone()
    twoqa, sfa = oracle.dequant_integer_parts(ap, asf)
    twoqb, sfb = oracle.dequant_integer_parts(bp, bsf)
    blk = (twoqa[1].view(-1, 16)[0] * sfa[1, 0])[None, :] * (twoqb.view(8, -1, 16)[:, 0, :] * sfb[:, 0:1])
    lost[1] -= blk.sum(dim=1) / (4.0 * ags.double().item() * bgs.double().item())
    v = oracle.conformance(lost.to(torch.bfloat16), exact, bound)
    assert v["outside_bound"] >= 1


def test_all_zero_output_is_named_and_non_conformant_when_exact_is_not_zero():
    torch.manual_seed(4)
    ap, asf, ags = quantize_nvfp4(torch.randn(4, 64))
    bp, bsf, bgs = quantize_nvfp4(torch.randn(4, 64))
    exact, bound = oracle.exact_gemm_nvfp4(ap, asf, ags, bp, bsf, bgs)
    v = oracle.conformance(torch.zeros(4, 4), exact, bound)
    assert v["all_zero_output"] and not v["conformant"]


def test_quantizer_conformance_accepts_reference_and_rejects_a_changed_code():
    torch.manual_seed(5)
    x = torch.randn(6, 256)
    ap, asf, ags = quantize_nvfp4(x)
    assert oracle.quantizer_conformance(x, ap, asf, ags)["conformant"]
    bad = ap.clone()
    bad[2, 3] ^= 0x10
    q = oracle.quantizer_conformance(x, bad, asf, ags)
    assert not q["conformant"] and q["codes_differ_not_at_tie"] == 1


def test_selftest():
    assert oracle.selftest() == 0
