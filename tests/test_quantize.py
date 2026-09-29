"""Value conformance: FlashInfer's fp4_quantize against the reference quantizer, element by element.

Failure class caught: a quantizer whose rounding, scale convention or packing differs from what the GEMM assumes.
Both sides here are checked against FlashInfer's own dequantizer (`e2m1_and_ufp8sf_scale_to_float`) too, so a
disagreement is localised to the quantizer, the packing or the scale.
"""
import torch

from sm120fp4 import dequantize_nvfp4, quantize_nvfp4
from sm120fp4.reference import e2m1_decode, unpack_e2m1


def _codes(packed):
    return unpack_e2m1(packed)


def test_reference_round_trip_error_is_bounded_by_one_grid_unit_per_block():
    torch.manual_seed(0)
    x = torch.randn(64, 256) * 3
    q, sf, gs = quantize_nvfp4(x)
    y = dequantize_nvfp4(q, sf, gs)
    # the E2M1 grid's widest gap is 2 (4 -> 6), so round-to-nearest lands within one grid unit of the scaled value;
    # saturation above 6 adds at most the excess, which the block scale (amax / 6, rounded to E4M3) keeps small
    unit = (sf.view(torch.float8_e4m3fn).float() / gs.reshape(())).view(64, 16, 1)      # one grid unit, in x's units
    err = (y.view(64, 16, 16) - x.view(64, 16, 16)).abs()
    assert (err <= unit * 1.0 + 1e-5).all(), f"max error {err.max():.4f} against unit {unit.max():.4f}"
    assert err.mean() < unit.mean() * 0.35  # rounding error averages well under half a unit


def test_matches_flashinfer_values(device, fi):
    torch.manual_seed(1)
    total, mismatched, off_by_more = 0, 0, 0
    for m, k in [(128, 512), (256, 2048), (64, 4096)]:
        x = (torch.randn(m, k, device=device) * 2).to(torch.bfloat16)
        gs = (448.0 * 6.0 / x.abs().max().float()).reshape(1)
        q_fi, sf_fi = fi.fp4_quantize(x, gs, 16, False, False)
        sf_fi = sf_fi.view(-1)[: m * (k // 16)].view(m, k // 16)
        q_us, sf_us, _ = quantize_nvfp4(x.float(), gs)
        assert torch.equal(sf_fi, sf_us), f"block scales differ at {m}x{k}: {(sf_fi != sf_us).sum().item()} of {sf_us.numel()}"
        c_fi, c_us = _codes(q_fi), _codes(q_us)
        diff = c_fi != c_us
        total += diff.numel()
        mismatched += int(diff.sum())
        # a mismatch may only be a rounding-tie difference of one grid step
        v_fi, v_us = e2m1_decode(c_fi), e2m1_decode(c_us)
        grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=device)
        i_fi = torch.searchsorted(grid, v_fi.abs().reshape(-1)).view(v_fi.shape)
        i_us = torch.searchsorted(grid, v_us.abs().reshape(-1)).view(v_us.shape)
        off_by_more += int(((i_fi - i_us).abs() > 1).sum())
    assert off_by_more == 0, "codes differ by more than one grid step"
    assert mismatched / total < 0.005, f"{mismatched} of {total} codes differ (ties or fast-math reciprocal); above 0.5%"


def test_flashinfer_dequant_agrees_with_reference(device, fi):
    torch.manual_seed(2)
    x = torch.randn(128, 1024, device=device).to(torch.bfloat16)
    gs = (448.0 * 6.0 / x.abs().max().float()).reshape(1)
    q, sf = fi.fp4_quantize(x, gs, 16, False, False)
    sf = sf.view(-1)[: 128 * 64].view(128, 64)
    ours = dequantize_nvfp4(q, sf, gs).cpu()
    # FlashInfer's dequantiser takes the per-tensor scale in the checkpoint direction (weight_scale_2 = amax / (448 * 6),
    # multiplied in), which is the reciprocal of the global_scale its quantizer takes (divided out). Measured on this
    # device: passing the quantizer's global_scale straight through scales every output by global_scale ** 2.
    theirs = fi.e2m1_and_ufp8sf_scale_to_float(q.cpu(), sf.cpu(), (1.0 / gs).cpu(), 16, 1, False).float()
    assert torch.allclose(theirs, ours, rtol=1e-5, atol=1e-6), "dequantisation differs even with the reciprocal per-tensor scale"
    wrong = fi.e2m1_and_ufp8sf_scale_to_float(q.cpu(), sf.cpu(), gs.cpu(), 16, 1, False).float()
    nz = ours != 0
    ratio = (wrong[nz] / ours[nz])
    assert torch.allclose(ratio, (gs * gs).cpu().expand_as(ratio), rtol=1e-4), "the direction trap is not gs**2 any more; re-document"
