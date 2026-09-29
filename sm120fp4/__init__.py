"""sm120fp4: FP4 scale-factor layouts, reference kernels and a conformance suite for consumer Blackwell (SM120/SM121)."""

from .layouts import (  # noqa: F401
    SF_BLOCK_ROWS,
    SF_BLOCK_COLS,
    from_128x4,
    linear_to_128x4_index,
    padded_sf_shape,
    to_128x4,
)
from .reference import (  # noqa: F401
    E2M1_GRID,
    dequantize_nvfp4,
    quantize_nvfp4,
    reference_gemm_nvfp4,
)

__version__ = "0.0.1"
