"""Driver for Nsight Compute, and a warm/flushed timer, for vLLM's CUTLASS FP8 GEMM at 2048 x 14336 x 4096.

    python scripts/ncu_vllm_fp8_driver.py time        # median of 50, warm and with a 256 MB L2 flush between iterations
    ncu --launch-skip 13 --launch-count 1 --kernel-name regex:"(?i)gemm|cutlass|kernel" --metrics ... python scripts/ncu_vllm_fp8_driver.py
"""
import statistics
import sys

import torch
from vllm import _custom_ops as ops

M, N, K = 2048, 14336, 4096
gen = torch.Generator().manual_seed(1)
a8 = torch.randn((M, K), generator=gen).to(torch.float8_e4m3fn).cuda()
w8 = torch.randn((N, K), generator=gen).to(torch.float8_e4m3fn).cuda().t()
sa = torch.ones(M, 1, device="cuda"); sb = torch.ones(N, 1, device="cuda")
flush = torch.empty(64 * 1024 * 1024, dtype=torch.float32, device="cuda")


def call():
    return ops.cutlass_scaled_mm(a8, w8, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)


for _ in range(10):
    call()
torch.cuda.synchronize()


def timeit(fb):
    ts = []
    for _ in range(50):
        if fb is not None:
            fb.fill_(1.0); torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); call(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) * 1000)
    return statistics.median(ts)


if len(sys.argv) > 1 and sys.argv[1] == "time":
    for cond, fb in (("warm", None), ("flushed", flush)):
        t = timeit(fb); print(cond, round(t, 1), "us", round(2 * M * N * K / (t * 1e-6) / 1e12, 1), "TFLOPS")
else:
    for _ in range(4):
        call()
    torch.cuda.synchronize(); print("done")
