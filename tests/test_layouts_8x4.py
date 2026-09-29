"""The 8x4 scale layout (FlashInfer `is_sf_8x4_layout`): derived by one-hot probing, then checked as a converter.

Failure class caught: a scale tensor prepared in the 128x4 layout handed to a kernel that expects 8x4 (or the reverse):
same dtype, same byte count for many shapes, different order, no error.
"""
import pytest
import torch

from sm120fp4 import from_8x4, linear_to_8x4_index, padded_sf_shape_8x4, to_8x4


@pytest.mark.parametrize("rows,cols", [(8, 4), (16, 8), (9, 5), (130, 12)])
def test_round_trip_and_padding(rows, cols):
    sf = torch.randint(0, 256, (rows, cols), dtype=torch.uint8)
    sw = to_8x4(sf)
    assert sw.shape == padded_sf_shape_8x4(rows, cols)
    assert torch.equal(from_8x4(sw, rows, cols), sf)
    idx = linear_to_8x4_index(rows, cols)
    assert idx.unique().numel() == rows * cols


def test_8x4_matches_flashinfer_one_hot(device, fi):
    """Each scale position, probed alone through fp4_quantize(..., swizzled=True, 8x4=True), lands where the formula says."""
    for rows, cols in [(16, 8), (9, 5)]:
        k = cols * 16
        idx = linear_to_8x4_index(rows, cols)
        gs = torch.tensor([448.0 * 6.0], device=device)
        for r in range(rows):
            for c in range(cols):
                y = torch.zeros(rows, k, device=device)
                y[r, c * 16] = 1.0
                _, s = fi.fp4_quantize(y.to(torch.bfloat16), gs, 16, False, True, True)
                assert tuple(s.shape) == padded_sf_shape_8x4(rows, cols)
                nz = (s.view(-1) != 0).nonzero().flatten().tolist()
                assert nz == [int(idx[r, c])], f"({r},{c}) at {nz}, formula says {int(idx[r, c])}"


def test_8x4_flag_without_swizzle_is_row_major(device, fi):
    rows, cols = 16, 8
    gs = torch.tensor([448.0 * 6.0], device=device)
    for (r, c) in [(0, 4), (1, 0), (15, 7)]:
        y = torch.zeros(rows, cols * 16, device=device)
        y[r, c * 16] = 1.0
        _, s = fi.fp4_quantize(y.to(torch.bfloat16), gs, 16, False, False, True)
        nz = (s.view(-1) != 0).nonzero().flatten().tolist()
        assert nz == [r * cols + c], "with is_sf_swizzled_layout=False the 8x4 flag is expected to be ignored"


def test_8x4_bulk_matches_flashinfer(device, fi):
    torch.manual_seed(5)
    for m, k in [(128, 512), (200, 1024)]:
        x = torch.randn(m, k, device=device).to(torch.bfloat16)
        gs = (448.0 * 6.0 / x.abs().max().float()).reshape(1)
        _, s_lin = fi.fp4_quantize(x, gs, 16, False, False, False)
        _, s_8 = fi.fp4_quantize(x, gs, 16, False, True, True)
        s_lin = s_lin.view(-1)[: m * (k // 16)].view(m, k // 16)
        assert torch.equal(to_8x4(s_lin).reshape(-1), s_8.reshape(-1))
