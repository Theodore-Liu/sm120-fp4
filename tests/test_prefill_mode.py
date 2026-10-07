"""The prefill switch: the CUTLASS hand-off by default, the slices on SM120FP4_PREFILL=slices, and an explicit cutlass setting recognised."""
from sm120fp4 import vllm_backend as vb


def test_default_is_cutlass(monkeypatch):
    monkeypatch.delenv(vb.PREFILL_ENV, raising=False)
    assert vb.prefill_mode() == "cutlass"
    assert not vb.prefill_explicit()


def test_slices_fallback(monkeypatch):
    monkeypatch.setenv(vb.PREFILL_ENV, "slices")
    assert vb.prefill_mode() == "slices"
    assert not vb.prefill_explicit()


def test_explicit_cutlass(monkeypatch):
    monkeypatch.setenv(vb.PREFILL_ENV, "cutlass")
    assert vb.prefill_mode() == "cutlass"
    assert vb.prefill_explicit()


def test_unknown_value_is_the_default(monkeypatch):
    monkeypatch.setenv(vb.PREFILL_ENV, "something-else")
    assert vb.prefill_mode() == "cutlass"
    assert not vb.prefill_explicit()
