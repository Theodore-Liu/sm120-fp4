"""Median CUDA-event time of grouped_mm_fp4 at the census's MoE-like shapes (expert segments 128-aligned so both the patched and
unpatched code are correct), with whichever FlashInfer is installed; prints one JSON line."""
import json
import torch
import flashinfer
from flashinfer import SfLayout
from flashinfer.fp4_quantization import nvfp4_quantize
from scripts.bench_grouped_sfa_relayout2 import median_ms

dev = torch.device("cuda")
out = {}
for e_count, tpe, n, k in ((8, 128, 4096, 4096), (8, 1024, 4096, 4096), (64, 128, 2048, 7168), (128, 64, 2048, 7168), (128, 512, 1536, 4096)):
    g = torch.Generator(device=dev).manual_seed(5)
    cum_m = e_count * tpe
    a_bf = torch.randn(cum_m, k, generator=g, device=dev, dtype=torch.bfloat16)
    b_bf = torch.randn(e_count * n, k, generator=g, device=dev, dtype=torch.bfloat16)
    gsa = (448 * 6) / a_bf.float().abs().max(); gsb = (448 * 6) / b_bf.float().abs().max()
    a_fp4, a_sf = nvfp4_quantize(a_bf, gsa, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    b_fp4, b_sf = nvfp4_quantize(b_bf, gsb, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    a_sf = a_sf.view(torch.float8_e4m3fn).reshape(-1, k // 16)
    b_fp4 = b_fp4.reshape(e_count, n, k // 2); b_sf = b_sf.view(torch.float8_e4m3fn).reshape(e_count, -1, k // 16)
    m_indptr = (torch.arange(e_count + 1, device=dev) * tpe).to(torch.int32)
    alpha = torch.tensor([1.0 / float(gsa * gsb)], device=dev)
    f = lambda: flashinfer.grouped_mm_fp4(a_fp4, b_fp4, a_sf, b_sf, m_indptr, alpha=alpha, out_dtype=torch.bfloat16, block_size=16)  # noqa: E731
    out[f"{e_count}x{tpe} n{n} k{k}"] = round(median_ms(f, warm=20, reps=100), 4)
print(json.dumps(out))
