"""Empirical probe of `mma.sync.m16n8k32.kind::f8f6f4.f32.e4m3.e2m1.f32` on SM120: fragment layouts and the e2m1 container.

The kernel runs one MMA per warp on fragments handed in from Python (a 32 x 6 uint32 tensor: a0..a3, b0, b1 per lane)
and returns the 32 x 4 fp32 C fragment. Python builds A (16 x 32, e4m3) and B (32 x 8, e2m1) from a hypothesis about
the layout and the nibble container, runs the MMA, and compares with the fp32 product of the dequantized matrices. The
hypothesis that matches on random inputs is the layout the kernel in scripts/fp8_fp4_gemm_sm120.py must use.

    PYTHONPATH=. python scripts/probe_f8f6f4.py
"""
from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ue8m0_reference", _here / "ue8m0_reference.py")
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)

CPP = "#include <torch/extension.h>\nvoid probe(torch::Tensor frags, torch::Tensor out);\n"
CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
__global__ void k_probe(const uint32_t* __restrict__ f, float* __restrict__ out) {
  const int lane = threadIdx.x;
  uint32_t a[4] = {f[lane * 6 + 0], f[lane * 6 + 1], f[lane * 6 + 2], f[lane * 6 + 3]};
  uint32_t b[2] = {f[lane * 6 + 4], f[lane * 6 + 5]};
  float c[4] = {0.f, 0.f, 0.f, 0.f};
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e2m1.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  for (int i = 0; i < 4; ++i) out[lane * 4 + i] = c[i];
}
void probe(torch::Tensor frags, torch::Tensor out) {
  TORCH_CHECK(frags.scalar_type() == torch::kInt && frags.numel() == 32 * 6 && out.numel() == 32 * 4);
  k_probe<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const uint32_t*>(frags.data_ptr<int>()), out.data_ptr<float>());
}
"""


def build():
    return load_inline(name="sm120fp4_probe_f8f6f4", cpp_sources=CPP, cuda_sources=CUDA, functions=["probe"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a"])


def e4m3_bytes(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float8_e4m3fn).view(torch.uint8)


def pack4(bytes4: torch.Tensor) -> int:
    b = [int(v) & 0xFF for v in bytes4]
    return b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)


def main() -> int:
    dev = torch.device("cuda")
    mod = build()
    torch.manual_seed(0)
    # A: 16 x 32 values on the e4m3 grid; B: 32 x 8 values on the e2m1 grid (codes 0..15)
    a = (torch.randn(16, 32) * 2).to(torch.float8_e4m3fn).float()
    b_codes = torch.randint(0, 16, (32, 8), dtype=torch.int8)
    b = ref.dequantize_from_fp4_e2m1(b_codes)
    want = a @ b  # 16 x 8 fp32 exact (all products representable)
    a8 = e4m3_bytes(a)
    results = []
    # hypotheses: A k-chunk order (a0..a3 = (row g, k0..), (row g+8, k0..), (row g, k16..), (row g+8, k16..)) vs the
    # k-major alternative ((g,k0),(g,k16),(g+8,k0),(g+8,k16)); B container: code in low nibble, high nibble, or
    # e2m1 bits spread as sign<<7 | exp<<5?... (the FP8-style placement: sign in bit 7, exponent in bits 6:5, mantissa in bit 4)
    def place(code: int, how: str) -> int:
        s, mag = (code >> 3) & 1, code & 7
        if how == "low":
            return code
        if how == "high":
            return code << 4
        if how == "fp8pos":  # sign at bit 7, e2m1's two exponent bits and one mantissa bit left-aligned below it
            return (s << 7) | (mag << 4)
        if how == "shift2":  # the 6-bit field convention the one-hot probe found: e2m1 code in bits 5:2
            return code << 2
        raise ValueError(how)
    for a_order, b_how in itertools.product(("rows-then-k", "k-then-rows"), ("low", "high", "fp8pos", "shift2")):
        frags = torch.zeros(32, 6, dtype=torch.int32)
        for lane in range(32):
            g, t = lane >> 2, lane & 3
            chunks = {"g,k0": a8[g, 4 * t:4 * t + 4], "g8,k0": a8[g + 8, 4 * t:4 * t + 4],
                      "g,k16": a8[g, 16 + 4 * t:16 + 4 * t + 4], "g8,k16": a8[g + 8, 16 + 4 * t:16 + 4 * t + 4]}
            order = ("g,k0", "g8,k0", "g,k16", "g8,k16") if a_order == "rows-then-k" else ("g,k0", "g,k16", "g8,k0", "g8,k16")
            for i, key in enumerate(order):
                frags[lane, i] = pack4(chunks[key]) - (1 << 32 if pack4(chunks[key]) >= 1 << 31 else 0)
            for j, k0 in enumerate((4 * t, 16 + 4 * t)):
                bytes4 = [place(int(b_codes[k0 + i, g]), b_how) for i in range(4)]
                v = bytes4[0] | (bytes4[1] << 8) | (bytes4[2] << 16) | (bytes4[3] << 24)
                frags[lane, 4 + j] = v - (1 << 32 if v >= 1 << 31 else 0)
        out = torch.zeros(32, 4, device=dev)
        mod.probe(frags.to(dev), out)
        torch.cuda.synchronize()
        got = torch.zeros(16, 8)
        for lane in range(32):
            g, t = lane >> 2, lane & 3
            got[g, 2 * t], got[g, 2 * t + 1], got[g + 8, 2 * t], got[g + 8, 2 * t + 1] = out[lane].cpu()
        err = float((got - want).abs().max())
        results.append((a_order, b_how, err))
        print(f"A {a_order:12s} B {b_how:7s}: max|err| {err:.4g}")
    best = min(results, key=lambda r: r[2])
    print("match:", best if best[2] == 0.0 else ("none exact; closest", best))
    return 0 if best[2] == 0.0 else 1


if __name__ == "__main__":
    sys.exit(main())
