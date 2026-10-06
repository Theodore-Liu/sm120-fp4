"""Bit-identity of the paged v6i, v6j, v6k, v6l and v6m against the paged v6e at H = 24 and 32, where both restage their quarters per head block (and v6j adds into
the logits after the first) (the path the module selftest, H <= 16, does
not reach). Prints one line per case and exits non-zero on any mismatch.

    ~/mlsys-5090-runtime/vllm028/.venv/bin/python scripts/check_v6i_h32.py
"""
import importlib.util
import sys
from pathlib import Path

import torch

path = Path(__file__).resolve().parents[1] / "sm120fp4" / "kernels" / "fp4_fp4_mqa_logits_v6_sm120.py"
spec = importlib.util.spec_from_file_location("v6mod", path)
v6 = importlib.util.module_from_spec(spec)
sys.argv = [sys.argv[0]]
spec.loader.exec_module(v6)
dev = torch.device("cuda")
v6.base.SM_COUNT = v6.v4.sm_count()
mod = v6.build()
ok = True
for (S, N, H, seed) in ((8, 512, 32, 51), (16, 2048, 32, 52), (5, 700, 24, 53)):
    r, _ = v6.run_paged_case(mod, S, N, H, seed, dev)
    good = r["paged_v6i_bit_identical"] and r["paged_v6j_bit_identical"] and r["paged_v6k_bit_identical"] and r["paged_v6l_bit_identical"] and r["paged_v6m_bit_identical"] and r["paged_v6e_bit_identical"]
    ok &= good
    print(f"S={S} N={N} H={H}: v6i bit-identical to v6e {r['paged_v6i_bit_identical']}, v6j {r['paged_v6j_bit_identical']}, v6k {r['paged_v6k_bit_identical']}, v6l {r['paged_v6l_bit_identical']}, v6m {r['paged_v6m_bit_identical']}, v6e to flat {r['bit_identical_to_flat_v6']} -> {'ok' if good else 'FAIL'}")
print("v6i to v6m H > 16 check:", "ok" if ok else "FAIL")
sys.exit(0 if ok else 1)
