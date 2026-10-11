#!/usr/bin/env python3
"""Run FlashInfer's bmm_fp8 on one backend at one shape, writing a marker before and after the call to a file on the Windows
side, so that if the call takes the WSL virtual machine down the marker says which backend did it.

    PYTHONPATH=. python scripts/bmm_fp8_isolate.py --backend cudnn --log /mnt/c/.../bmm-isolate.log
"""
import argparse, time, torch

ap = argparse.ArgumentParser()
ap.add_argument("--backend", required=True)
ap.add_argument("--log", required=True)
ap.add_argument("--m", type=int, default=128); ap.add_argument("--n", type=int, default=4096); ap.add_argument("--k", type=int, default=1024)
a = ap.parse_args()
def mark(s):
    with open(a.log, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {a.backend} {s}\n"); f.flush()
mark("start")
import flashinfer
dev = torch.device("cuda")
gen = torch.Generator().manual_seed(1)
x = (torch.randn(a.m, a.k, generator=gen) * 2).to(torch.float8_e4m3fn).to(dev)
w = torch.randn(a.n, a.k, generator=gen).to(torch.float8_e4m3fn).to(dev)
sa = torch.tensor([0.01], device=dev); sw = torch.tensor([0.02], device=dev)
mark("calling")
try:
    y = flashinfer.bmm_fp8(x[None], w.t()[None], sa, sw, torch.bfloat16, backend=a.backend)
    torch.cuda.synchronize()
    mark(f"ok shape={tuple(y.shape)} finite={bool(torch.isfinite(y).all())}")
except Exception as e:
    mark(f"error {type(e).__name__}: {str(e)[:200]}")
