"""The fused W4A16 MoE oracle's selftest, as a test (CPU only)."""
from sm120fp4 import moe_oracle


def test_moe_oracle_selftest():
    assert moe_oracle.selftest() == 0
