"""Bit-identity of the paged fp8 v5s against the flat v5 at H = 24 and 32, where v5s restages its quarters per head block (the path the module
selftest, H <= 16, does not reach), with bf16 and fp32 weights. Prints one line per case and exits non-zero on any mismatch.

    ~/mlsys-5090-runtime/vllm028/.venv/bin/python scripts/check_v5s_h32.py
"""
import importlib.util
import sys
from pathlib import Path

import torch

path = Path(__file__).resolve().parents[1] / "sm120fp4" / "kernels" / "fp8_mqa_logits_v5_sm120.py"
spec = importlib.util.spec_from_file_location("v5mod", path)
v5 = importlib.util.module_from_spec(spec)
sys.argv = [sys.argv[0]]
spec.loader.exec_module(v5)
dev = torch.device("cuda")
v5.base.SM_COUNT = v5.sm_count()
mod = v5.build()
ok = True
for (S, N, H, seed, f32) in ((8, 512, 32, 61, False), (16, 2048, 32, 62, False), (5, 700, 24, 63, False), (9, 1500, 32, 64, True)):
    r, _ = v5.run_paged_case(mod, S, N, H, seed, dev, weights_fp32=f32, quarter=True)
    ok &= r["pass"]
    print(f"S={S} N={N} H={H} fp32 weights {f32}: v5s bit-identical to flat v5 {r['bit_identical_to_flat_v5']}, rel max err {r['rel_max_err']:.2e} -> {'ok' if r['pass'] else 'FAIL'}")
print("v5s H > 16 check:", "ok" if ok else "FAIL")
sys.exit(0 if ok else 1)
