"""Thin shim: this kernel module lives in the package since 2026-10-04 (sm120fp4/kernels/fp8_fp4_mqa_logits_sm120.py), so that a pip install
carries it. Loading this file by path (as the bench and diagnostic scripts beside it do) executes the package copy here."""
import importlib.util as _ilu
from pathlib import Path as _P

_target = _P(__file__).resolve().parents[1] / "sm120fp4" / "kernels" / "fp8_fp4_mqa_logits_sm120.py"
_spec = _ilu.spec_from_file_location(__name__, _target)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
globals().update({k: v for k, v in vars(_mod).items() if k not in ("__name__", "__file__", "__spec__", "__loader__", "__package__", "__builtins__", "__cached__")})
