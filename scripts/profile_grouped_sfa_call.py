#!/usr/bin/env python3
"""Where does a patched grouped_mm_fp4 call spend its time? Python-level cProfile of 20 calls plus CUDA-event timing of the call,
on one MoE-like shape, with whichever FlashInfer is installed (patched or not).

    PYTHONPATH=. python scripts/profile_grouped_sfa_call.py
"""
import cProfile
import io
import pstats

import torch

import flashinfer
from flashinfer import SfLayout
from flashinfer.fp4_quantization import nvfp4_quantize

dev = torch.device("cuda")
e_count, tpe, n, k = 8, 128, 4096, 4096
cum_m = e_count * tpe
a_bf = torch.randn(cum_m, k, device=dev, dtype=torch.bfloat16)
b_bf = torch.randn(e_count * n, k, device=dev, dtype=torch.bfloat16)
gsa = (448 * 6) / a_bf.float().abs().max(); gsb = (448 * 6) / b_bf.float().abs().max()
a_fp4, a_sf = nvfp4_quantize(a_bf, gsa, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
b_fp4, b_sf = nvfp4_quantize(b_bf, gsb, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
a_sf = a_sf.view(torch.float8_e4m3fn).reshape(-1, k // 16)
b_fp4 = b_fp4.reshape(e_count, n, k // 2); b_sf = b_sf.view(torch.float8_e4m3fn).reshape(e_count, -1, k // 16)
m_indptr = (torch.arange(e_count + 1, device=dev) * tpe).to(torch.int32)
alpha = torch.tensor([1.0 / float(gsa * gsb)], device=dev)


def call():
    return flashinfer.grouped_mm_fp4(a_fp4, b_fp4, a_sf, b_sf, m_indptr, alpha=alpha, out_dtype=torch.bfloat16, block_size=16)


for _ in range(5):
    call()
torch.cuda.synchronize()
pr = cProfile.Profile()
pr.enable()
for _ in range(20):
    call()
torch.cuda.synchronize()
pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(14)
print(s.getvalue()[:4000])
