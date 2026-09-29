"""Scale-factor layouts for block-scaled FP4/FP8 GEMMs, as pure PyTorch index arithmetic.

The layout every Blackwell tensor-core path consumes is the "128x4 tiled" (also "swizzled", "blocked",
`to_blocked`) layout. One tile holds the scale factors of 128 consecutive rows and 4 consecutive scale
columns (4 x 16 = 64 elements of K for NVFP4; 4 x 32 = 128 for MXFP4), stored as 512 contiguous bytes.
Inside the tile, memory order is [outerM (32), innerM (4), innerK (4)] with

    offset_in_tile = (row % 32) * 16 + ((row % 128) // 32) * 4 + (col % 4)

Tiles are laid out row-major over [ceil(rows/128), ceil(cols/4)], rows padded to a multiple of 128 and
scale columns to a multiple of 4. This is the same formula in three places:

* TensorRT-LLM / FlashInfer, `get_sf_out_offset_128x4` in `cpp/tensorrt_llm/kernels/quantization.cuh`:
  SF layout [numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4 (kTile)]; strides 1, 4, 16, 512, numKTiles*512.
* cuDNN frontend, "The 128x4 tiled layout for block scaling factors":
  offset = (outer % 32) * 16 + (outer / 32) * 4 + inner.
* CUTLASS, `include/cutlass/detail/sm100_blockscaled_layout.hpp` (used by the SM120 collectives too):
  SF atom Layout<Shape<Shape<_32,_4>, Shape<SFVecSize,_4>>, Stride<Stride<_16,_4>, Stride<_0,_1>>>,
  Blk_MN = 128, Blk_SF = 4, tiled over (M, K) with Step<_2,_1>.

The equivalence is tested, not assumed (tests/test_layouts.py).
"""
from __future__ import annotations

import torch

SF_BLOCK_ROWS = 128
SF_BLOCK_COLS = 4
_TILE = SF_BLOCK_ROWS * SF_BLOCK_COLS  # 512 scale factors per tile


def padded_sf_shape(rows: int, sf_cols: int) -> tuple[int, int]:
    """Rows padded to a multiple of 128 and scale columns to a multiple of 4."""
    pr = -(-rows // SF_BLOCK_ROWS) * SF_BLOCK_ROWS
    pc = -(-sf_cols // SF_BLOCK_COLS) * SF_BLOCK_COLS
    return pr, pc


def linear_to_128x4_index(rows: int, sf_cols: int, device=None) -> torch.Tensor:
    """For every (row, col) of a row-major [rows, sf_cols] scale tensor, the flat offset into the 128x4 buffer.

    Returns an int64 tensor of shape [rows, sf_cols]. The buffer it indexes has padded_sf_shape(rows, sf_cols)
    elements in total (padded rows x padded cols)."""
    pr, pc = padded_sf_shape(rows, sf_cols)
    num_k_tiles = pc // SF_BLOCK_COLS
    r = torch.arange(rows, device=device, dtype=torch.int64)[:, None]
    c = torch.arange(sf_cols, device=device, dtype=torch.int64)[None, :]
    m_tile = r // SF_BLOCK_ROWS
    k_tile = c // SF_BLOCK_COLS
    outer_m = r % 32
    inner_m = (r % SF_BLOCK_ROWS) // 32
    inner_k = c % SF_BLOCK_COLS
    return ((m_tile * num_k_tiles + k_tile) * _TILE + outer_m * 16 + inner_m * 4 + inner_k)


def to_128x4(sf_linear: torch.Tensor) -> torch.Tensor:
    """Row-major [rows, sf_cols] scale factors -> 128x4 swizzled buffer, returned as [padded_rows, padded_cols].

    The returned tensor's flat memory is the swizzled buffer; its 2-D shape is only the padded extent, the way
    FlashInfer returns it. Padding entries are zero."""
    if sf_linear.dim() != 2:
        raise ValueError(f"expected a 2-D [rows, sf_cols] tensor, got {tuple(sf_linear.shape)}")
    rows, cols = sf_linear.shape
    pr, pc = padded_sf_shape(rows, cols)
    out = torch.zeros(pr * pc, dtype=sf_linear.dtype, device=sf_linear.device)
    idx = linear_to_128x4_index(rows, cols, device=sf_linear.device)
    out[idx.reshape(-1)] = sf_linear.reshape(-1)
    return out.view(pr, pc)


def from_128x4(sf_swizzled: torch.Tensor, rows: int, sf_cols: int) -> torch.Tensor:
    """Inverse of to_128x4: a swizzled buffer (any 2-D view of padded_rows x padded_cols elements) -> row-major [rows, sf_cols]."""
    pr, pc = padded_sf_shape(rows, sf_cols)
    flat = sf_swizzled.reshape(-1)
    if flat.numel() != pr * pc:
        raise ValueError(f"swizzled buffer has {flat.numel()} elements; expected {pr}*{pc}={pr * pc} for rows={rows}, sf_cols={sf_cols}")
    idx = linear_to_128x4_index(rows, sf_cols, device=sf_swizzled.device)
    return flat[idx.reshape(-1)].view(rows, sf_cols)


# ----------------------------------------------------------------------------- the 8x4 layout
# FlashInfer's fp4_quantize(..., is_sf_swizzled_layout=True, is_sf_8x4_layout=True) and mm_fp4(..., use_8x4_sf_layout=True):
# a tile of 8 rows x 4 scale columns (32 bytes), in-tile offset (row % 8) * 4 + (col % 4), tiles row-major over
# [ceil(rows/8), ceil(cols/4)], rows padded to a multiple of 8 and columns to a multiple of 4. Derived by one-hot probing of
# fp4_quantize on an RTX 5090 (tests/test_layouts.py::test_8x4_matches_flashinfer_one_hot). With is_sf_swizzled_layout=False
# the 8x4 flag is ignored and the output is row-major.
SF8_BLOCK_ROWS = 8
_TILE8 = SF8_BLOCK_ROWS * SF_BLOCK_COLS  # 32


def padded_sf_shape_8x4(rows: int, sf_cols: int) -> tuple[int, int]:
    pr = -(-rows // SF8_BLOCK_ROWS) * SF8_BLOCK_ROWS
    pc = -(-sf_cols // SF_BLOCK_COLS) * SF_BLOCK_COLS
    return pr, pc


def linear_to_8x4_index(rows: int, sf_cols: int, device=None) -> torch.Tensor:
    pr, pc = padded_sf_shape_8x4(rows, sf_cols)
    num_k_tiles = pc // SF_BLOCK_COLS
    r = torch.arange(rows, device=device, dtype=torch.int64)[:, None]
    c = torch.arange(sf_cols, device=device, dtype=torch.int64)[None, :]
    return ((r // SF8_BLOCK_ROWS) * num_k_tiles + c // SF_BLOCK_COLS) * _TILE8 + (r % SF8_BLOCK_ROWS) * SF_BLOCK_COLS + (c % SF_BLOCK_COLS)


def to_8x4(sf_linear: torch.Tensor) -> torch.Tensor:
    rows, cols = sf_linear.shape
    pr, pc = padded_sf_shape_8x4(rows, cols)
    out = torch.zeros(pr * pc, dtype=sf_linear.dtype, device=sf_linear.device)
    out[linear_to_8x4_index(rows, cols, device=sf_linear.device).reshape(-1)] = sf_linear.reshape(-1)
    return out.view(pr, pc)


def from_8x4(sf_8x4: torch.Tensor, rows: int, sf_cols: int) -> torch.Tensor:
    pr, pc = padded_sf_shape_8x4(rows, sf_cols)
    flat = sf_8x4.reshape(-1)
    if flat.numel() != pr * pc:
        raise ValueError(f"8x4 buffer has {flat.numel()} elements; expected {pr}*{pc}={pr * pc}")
    return flat[linear_to_8x4_index(rows, sf_cols, device=sf_8x4.device).reshape(-1)].view(rows, sf_cols)


def describe() -> dict:
    """The layout constants, for reports."""
    return {"name": "128x4", "tile_rows": SF_BLOCK_ROWS, "tile_cols": SF_BLOCK_COLS, "tile_bytes_e4m3": _TILE,
            "offset_in_tile": "(row % 32) * 16 + ((row % 128) // 32) * 4 + (col % 4)",
            "tile_order": "row-major over [ceil(rows/128), ceil(cols/4)]"}
