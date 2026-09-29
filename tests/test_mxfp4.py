"""MXFP4 conformance: FlashInfer's fp4_quantize(sf_vec_size=32, sf_use_ue8m0=True) against the reference MXFP4 quantizer.

Failure class caught: a checkpoint or kernel that assumes a different E8M0 rounding rule for the block exponent (floor, round
or ceil of log2 of the block maximum, with or without the division by 6) - each yields a valid-looking scale tensor and a
different set of codes.
"""
import torch

from sm120fp4.reference import dequantize_mxfp4, quantize_mxfp4, unpack_e2m1


def test_reference_round_trip_error_is_bounded():
    torch.manual_seed(0)
    x = torch.randn(64, 512) * 3
    q, sf = quantize_mxfp4(x)
    y = dequantize_mxfp4(q, sf)
    unit = torch.exp2(sf.float() - 127).view(64, 16, 1)      # one grid unit in x's units
    err = (y.view(64, 16, 32) - x.view(64, 16, 32)).abs()
    assert (err <= unit + 1e-5).all()


def test_matches_flashinfer_scales_and_codes(device, fi):
    torch.manual_seed(0)
    total = mismatched = 0
    for m, k in [(64, 512), (128, 2048)]:
        x = (torch.randn(m, k, device=device) * 3).to(torch.bfloat16)
        q_fi, sf_fi = fi.fp4_quantize(x, None, 32, True, False)
        sf_fi = sf_fi.view(-1)[: m * (k // 32)].view(m, k // 32)
        q_us, sf_us = quantize_mxfp4(x.float())
        assert torch.equal(sf_fi, sf_us), "E8M0 block exponents differ: the rule is ceil(log2(amax / 6))"
        d = unpack_e2m1(q_fi) != unpack_e2m1(q_us)
        total += d.numel()
        mismatched += int(d.sum())
    assert mismatched / total < 0.005, f"{mismatched} of {total} codes differ"
