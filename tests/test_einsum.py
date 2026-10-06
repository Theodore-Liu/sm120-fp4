"""FP8 einsum `bhr,hdr->bhd` (scripts/fp8_einsum_sm120.py): every kernel version against torch.einsum on the dequantised operands, and v1 and v2
bit for bit against v0.

Failure classes caught: a tile or chunk edge that drops or double-counts b rows (B not a multiple of 16 or 64), a d tile that reads the wrong
per-block scale (D a multiple of 64 but not of 128 would mis-index; the shapes keep D a multiple of 128 as the kernels require), and a
reordering of the accumulation that changes the bf16 output.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "fp8_einsum_sm120.py"


@pytest.fixture(scope="module")
def einsum(device):
    spec = importlib.util.spec_from_file_location("fp8_einsum_sm120", _PATH)
    mod = importlib.util.module_from_spec(spec)
    saved = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = saved
    return mod, mod.build()


@pytest.mark.parametrize("kernel", ("v0", "v1", "v2"))
@pytest.mark.parametrize("B,H,D,R,seed", [(1, 4, 128, 256, 1), (5, 8, 256, 512, 2), (33, 8, 512, 2048, 5), (8, 16, 1024, 4096, 6), (200, 4, 256, 1024, 7)])
def test_einsum_matches_reference_and_v0(device, einsum, kernel, B, H, D, R, seed):
    mod, built = einsum
    r, _ = mod.run_case(built, B, H, D, R, seed, device, kernel)
    assert r["max_abs_err"] <= 2 * r["bf16_half_ulp_at_max"] + 1e-6, r
    assert r["rel_fro_err"] < 4e-3, r
    if kernel != "v0":
        assert r["elements_differing_from_v0"] == 0, r
