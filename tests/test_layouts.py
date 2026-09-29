"""Layout conformance: our 128x4 converter against the libraries that own the layout.

Failure class caught: a scale tensor handed to a kernel in the wrong layout, which produces wrong output with no error
(FlashInfer #2577 reported `mm_fp4` on SM120 "silently returns all zeros" on the CUTLASS backend; the layout is the first
thing such a report has to rule out).
"""
import pytest
import torch

from sm120fp4 import from_128x4, linear_to_128x4_index, padded_sf_shape, to_128x4


@pytest.mark.parametrize("rows,cols", [(128, 4), (256, 32), (100, 5), (1, 1), (129, 9), (1024, 256)])
def test_round_trip_and_padding(rows, cols):
    sf = torch.randint(0, 256, (rows, cols), dtype=torch.uint8)
    sw = to_128x4(sf)
    assert sw.shape == padded_sf_shape(rows, cols)
    assert torch.equal(from_128x4(sw, rows, cols), sf)
    # padding is zero and the index map is a bijection onto the used entries
    idx = linear_to_128x4_index(rows, cols)
    assert idx.unique().numel() == rows * cols
    used = torch.zeros(sw.numel(), dtype=torch.bool)
    used[idx.reshape(-1)] = True
    assert (sw.reshape(-1)[~used] == 0).all()


def test_worked_example_matches_cudnn_formula():
    # cuDNN frontend: offset = (outer % 32) * 16 + (outer / 32) * 4 + inner, for one 128x4 tile
    idx = linear_to_128x4_index(128, 4)
    for outer in (0, 1, 31, 32, 33, 96, 127):
        for inner in range(4):
            assert idx[outer, inner].item() == (outer % 32) * 16 + (outer // 32) * 4 + inner


def test_matches_flashinfer_block_scale_interleave(device, fi):
    """FlashInfer's own swizzler is the oracle for the whole buffer, including multi-tile ordering."""
    for rows, cols in [(128, 4), (256, 32), (384, 8), (130, 6)]:
        sf = torch.randint(1, 255, (rows, cols), dtype=torch.uint8, device=device)
        theirs = fi.nvfp4_block_scale_interleave(sf)
        ours = to_128x4(sf)
        assert theirs.numel() == ours.numel(), (rows, cols, theirs.shape, ours.shape)
        assert torch.equal(theirs.reshape(-1), ours.reshape(-1)), f"128x4 layout differs from FlashInfer at {rows}x{cols}"


def test_matches_flashinfer_quantize_swizzled_vs_linear(device, fi):
    """fp4_quantize with is_sf_swizzled_layout=True must be to_128x4 of the same call with False."""
    torch.manual_seed(0)
    for m, k in [(256, 512), (128, 1024), (200, 2048)]:
        x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        gs = (448.0 * 6.0 / x.abs().max().float()).reshape(1)
        q_sw, sf_sw = fi.fp4_quantize(x, gs, 16, False, True)
        q_li, sf_li = fi.fp4_quantize(x, gs, 16, False, False)
        assert torch.equal(q_sw, q_li)
        sf_li = sf_li.view(-1)[: m * (k // 16)].view(m, k // 16)
        assert torch.equal(to_128x4(sf_li).reshape(-1), sf_sw.reshape(-1)), f"swizzled scale differs at {m}x{k}"
