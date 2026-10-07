"""Every stage-3 kernel module's own selftest, run by pytest so the nightly CI covers it (the CI runs `pytest tests/` only).

Each module's `main(["--selftest"])` builds the extension and checks its kernels against the reference and, where versions exist, against each
other bit for bit (the paged indexers against the flat ones through random page permutations); it returns non-zero on any failure. The
einsum has its own parametrised test (tests/test_einsum.py).
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULES = {
    "ue8m0_reference": ROOT / "sm120fp4" / "kernels" / "ue8m0_reference.py",
    "fp8_fp4_gemm": ROOT / "scripts" / "fp8_fp4_gemm_sm120.py",
    "fp8_fp4_mqa_logits_v0_v3": ROOT / "sm120fp4" / "kernels" / "fp8_fp4_mqa_logits_sm120.py",
    "fp8_fp4_mqa_logits_v4": ROOT / "sm120fp4" / "kernels" / "fp8_fp4_mqa_logits_v4_sm120.py",
    "fp4_fp4_mqa_logits_v6": ROOT / "sm120fp4" / "kernels" / "fp4_fp4_mqa_logits_v6_sm120.py",
    "fp8_mqa_logits_v5": ROOT / "sm120fp4" / "kernels" / "fp8_mqa_logits_v5_sm120.py",
}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(f"selftest_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    saved = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = saved
    return mod


@pytest.mark.parametrize("name", sorted(MODULES))
def test_module_selftest(device, name):
    mod = _load(name, MODULES[name])
    rc = mod.main(["--selftest"])
    assert rc == 0, f"{name}: selftest returned {rc}"
