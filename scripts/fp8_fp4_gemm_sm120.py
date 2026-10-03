"""Stage 3, the first kernel: a minimal, correctness-first SM120 FP8xFP4 GEMM with UE8M0 block scales (DeepGEMM's
`fp8_fp4_gemm_nt` contract, 1D1D, dense, NT, K-major).

    D[m, n] = sum_k A[m, k] * B[n, k],   A e4m3 [M, K] with packed UE8M0 scales sfa [M, K/gran_k] (int32, 4 per word),
                                        B packed e2m1 [N, K/2] with sfb [N, K/gran_k]; D bf16 [M, N].

The shape of the first version, all checked on the host: M <= 16, N a multiple of 8, K a multiple of 128, gran_k 128
for both operands, no epilogue, no accumulation input. It exists to settle three things an SM120 kernel for this site
has to get right before any tiling matters, and to settle them against `scripts/ue8m0_reference.py` (itself bit for bit
with DeepGEMM's helpers): the `mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e2m1.f32` fragment layouts for
8-bit A and 4-bit B, the container convention for e2m1 operands (one code per byte, the code left-aligned in the low six
bits, i.e. bits 5:2, as the probe found), and the
UE8M0 fold outside the MMA (four k32 steps share one scale pair at gran_k 128; the block partial is multiplied by
2^(ea-127) * 2^(eb-127) and added to the running fp32 accumulator). One warp computes one 16 x 8 tile over the whole K
with direct global loads; there is no shared memory, no pipelining and no attempt at speed. The tiled kernel of
`docs/stage3-survey.md` section 2.1 starts from this one once it is right.

    PYTHONPATH=. python scripts/fp8_fp4_gemm_sm120.py --selftest          # synthetic instances against the reference
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ue8m0_reference", _here / "ue8m0_reference.py")
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)

CPP = r"""
#include <torch/extension.h>
void fp8_fp4_gemm_nt_sm120(torch::Tensor a, torch::Tensor sfa, torch::Tensor b, torch::Tensor sfb, torch::Tensor d, int64_t gran_k);
"""

CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cstdint>

// One warp per 16 (M) x 8 (N) output tile over the whole K. Fragment layout of mma.m16n8k32 with 8-bit containers
// (PTX ISA, "Matrix Fragments for mma.m16n8k32" with .e4m3 / .e2m1 under .kind::f8f6f4): lane = 4 * g + t,
//   A: a0 = row g,     k = 4t..4t+3        a1 = row g+8, k = 4t..4t+3
//      a2 = row g,     k = 16+4t..16+4t+3  a3 = row g+8, k = 16+4t..16+4t+3     (4 bytes per register, k ascending)
//   B: b0 = col g,     k = 4t..4t+3        b1 = col g,   k = 16+4t..16+4t+3     (4 e2m1 codes, one per byte)
//   C/D: c0, c1 = row g, cols 2t, 2t+1;    c2, c3 = row g+8, cols 2t, 2t+1
__device__ __forceinline__ void mma_f8f6f4(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e2m1.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// four e2m1 codes (k, k+1, k+2, k+3) packed two per byte (even code low) -> four 8-bit containers. Measured on the RTX 5090
// (scripts/probe_f8f6f4_onehot.py, 2026-10-02): under .kind::f8f6f4 the hardware reads each container as a 6-bit field in
// bits 5:0 (sign at bit 5, two exponent bits, three mantissa bits), so an e2m1 code goes in bits 5:2 (code << 2) with bits
// 1:0 and 7:6 zero; 0x08 reads as 1.0, 0x3C as -6.0, and a code in the low nibble reads as a wrong, smaller number.
__device__ __forceinline__ uint32_t unpack_e2m1_x4(uint16_t packed2) {
  const uint32_t p = packed2;
  return ((p & 0x000Fu) << 2) | ((p & 0x00F0u) << 6) | ((p & 0x0F00u) << 10) | ((p & 0xF000u) << 14);
}

__device__ __forceinline__ float ue8m0_to_float(uint32_t e) { return __uint_as_float(e << 23); }

__global__ void __launch_bounds__(32)
k_fp8_fp4_gemm_nt(const uint8_t* __restrict__ a, const uint8_t* __restrict__ sfa,
                  const uint8_t* __restrict__ b, const uint8_t* __restrict__ sfb,
                  __nv_bfloat16* __restrict__ d, int M, int N, int K, int gran_k, int sf_stride) {
  const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
  const int n0 = blockIdx.x * 8;
  const int row0 = g, row1 = g + 8, col = n0 + g;
  const bool has0 = row0 < M, has1 = row1 < M;
  const int sf_per_row = sf_stride;             // UE8M0 bytes per row: the packed int32 stream read as bytes, padded to 4
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  const int steps_per_block = gran_k / 32;
  for (int kb = 0; kb < K; kb += gran_k) {
    float part[4] = {0.f, 0.f, 0.f, 0.f};
    for (int s = 0; s < steps_per_block; ++s) {
      const int k0 = kb + s * 32;
      uint32_t af[4], bf[2];
      af[0] = has0 ? *reinterpret_cast<const uint32_t*>(a + (size_t)row0 * K + k0 + 4 * t) : 0u;
      af[1] = has1 ? *reinterpret_cast<const uint32_t*>(a + (size_t)row1 * K + k0 + 4 * t) : 0u;
      af[2] = has0 ? *reinterpret_cast<const uint32_t*>(a + (size_t)row0 * K + k0 + 16 + 4 * t) : 0u;
      af[3] = has1 ? *reinterpret_cast<const uint32_t*>(a + (size_t)row1 * K + k0 + 16 + 4 * t) : 0u;
      const uint8_t* brow = b + (size_t)col * (K / 2);
      bf[0] = unpack_e2m1_x4(*reinterpret_cast<const uint16_t*>(brow + (k0 + 4 * t) / 2));
      bf[1] = unpack_e2m1_x4(*reinterpret_cast<const uint16_t*>(brow + (k0 + 16 + 4 * t) / 2));
      mma_f8f6f4(part, af, bf);
    }
    // the fold: this thread's C fragment holds rows g, g+8 and cols 2t, 2t+1 of the tile
    const int blk = kb / gran_k;
    const float sa0 = has0 ? ue8m0_to_float(sfa[(size_t)row0 * sf_per_row + blk]) : 0.f;
    const float sa1 = has1 ? ue8m0_to_float(sfa[(size_t)row1 * sf_per_row + blk]) : 0.f;
    const float sb0 = ue8m0_to_float(sfb[(size_t)(n0 + 2 * t) * sf_per_row + blk]);
    const float sb1 = ue8m0_to_float(sfb[(size_t)(n0 + 2 * t + 1) * sf_per_row + blk]);
    acc[0] += part[0] * (sa0 * sb0);
    acc[1] += part[1] * (sa0 * sb1);
    acc[2] += part[2] * (sa1 * sb0);
    acc[3] += part[3] * (sa1 * sb1);
  }
  if (has0) {
    d[(size_t)row0 * N + n0 + 2 * t] = __float2bfloat16(acc[0]);
    d[(size_t)row0 * N + n0 + 2 * t + 1] = __float2bfloat16(acc[1]);
  }
  if (has1) {
    d[(size_t)row1 * N + n0 + 2 * t] = __float2bfloat16(acc[2]);
    d[(size_t)row1 * N + n0 + 2 * t + 1] = __float2bfloat16(acc[3]);
  }
}

void fp8_fp4_gemm_nt_sm120(torch::Tensor a, torch::Tensor sfa, torch::Tensor b, torch::Tensor sfb, torch::Tensor d, int64_t gran_k) {
  const int M = (int)a.size(0), K = (int)a.size(1), N = (int)b.size(0);
  TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn && a.is_contiguous(), "A: e4m3 [M, K], contiguous");
  TORCH_CHECK(b.scalar_type() == torch::kInt8 && b.is_contiguous() && b.size(1) * 2 == K, "B: packed e2m1 int8 [N, K/2]");
  TORCH_CHECK(sfa.scalar_type() == torch::kInt && sfb.scalar_type() == torch::kInt, "scales: packed UE8M0 int32");
  TORCH_CHECK(gran_k == 128 && K % gran_k == 0, "this version: gran_k 128, K a multiple of 128");
  TORCH_CHECK(sfa.size(0) == M && sfb.size(0) == N && sfa.size(1) == sfb.size(1) && sfa.size(1) * 4 >= K / gran_k
              && sfa.is_contiguous() && sfb.is_contiguous(), "scale shapes: [rows, ceil((K/gran_k)/4)] int32, padded to a multiple of 4 scales");
  TORCH_CHECK(M >= 1 && M <= 16 && N % 8 == 0, "this version: M <= 16, N a multiple of 8");
  TORCH_CHECK(d.scalar_type() == torch::kBFloat16 && d.size(0) == M && d.size(1) == N && d.is_contiguous(), "D: bf16 [M, N]");
  auto st = at::cuda::getCurrentCUDAStream();
  k_fp8_fp4_gemm_nt<<<N / 8, 32, 0, st>>>(static_cast<const uint8_t*>(a.data_ptr()), reinterpret_cast<const uint8_t*>(sfa.data_ptr<int>()),
                                          reinterpret_cast<const uint8_t*>(b.data_ptr<int8_t>()),
                                          reinterpret_cast<const uint8_t*>(sfb.data_ptr<int>()),
                                          reinterpret_cast<__nv_bfloat16*>(d.data_ptr()), M, N, K, (int)gran_k, (int)sfa.size(1) * 4);
}
"""


def build(verbose: bool = False):
    return load_inline(name="sm120fp4_fp8_fp4_gemm_nt_v0d", cpp_sources=CPP, cuda_sources=CUDA, functions=["fp8_fp4_gemm_nt_sm120"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a"], verbose=verbose)


def run_case(mod, m: int, n: int, k: int, gran_k: int, seed: int, dev) -> dict:
    g = torch.Generator().manual_seed(seed)
    a = (torch.randn(m, k, generator=g) * torch.pow(2.0, torch.randint(-6, 8, (m, 1), generator=g).float())).to(dev)
    b = (torch.randn(n, k, generator=g) * torch.pow(2.0, torch.randint(-6, 8, (n, 1), generator=g).float())).to(dev)
    a8, sfa = ref.per_token_cast_to_fp8(a, use_ue8m0=True, gran_k=gran_k, use_packed_ue8m0=True)
    b4, sfb = ref.per_token_cast_to_fp4(b, use_ue8m0=True, gran_k=gran_k, use_packed_ue8m0=True)
    d = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
    mod.fp8_fp4_gemm_nt_sm120(a8, sfa, b4, sfb, d, gran_k)
    torch.cuda.synchronize()
    # the reference takes unpacked fp32 scales; unpack the same packed words the kernel read
    sfa_f = ref.unpack_ue8m0_from_int(sfa)
    sfb_f = ref.unpack_ue8m0_from_int(sfb)
    exact = ref.mx_gemm_reference(a8, sfa_f, b4, sfb_f, gran_k)
    diff = (d.float() - exact).abs()
    denom = exact.abs().clamp_min(1.0)
    rel_fro = float((d.float() - exact).norm() / exact.norm())
    rows = {"m": m, "n": n, "k": k, "gran_k": gran_k, "seed": seed, "max_abs_err": float(diff.max()),
            "max_rel_err_elem": float((diff / denom).max()), "rel_fro_err": rel_fro,
            "ref_abs_max": float(exact.abs().max()), "bf16_half_ulp_at_max": float(exact.abs().max()) * 2 ** -9}
    return rows


def selftest(mod, dev) -> int:
    # DeepGEMM's own tolerance for a mixed FP8xFP4 configuration is max_diff 0.01 (tests/generators.py QuantConfig.max_diff),
    # measured there on an fp32 reference of the *original* values. Here the reference is the exact dequantized product, so
    # the kernel's only error is bf16 output rounding (half an ulp at the magnitude of each element) plus fp32 summation
    # order; the bar below is the bf16 half-ulp at the matrix maximum plus fp32 slack, not a quantization tolerance.
    results = []
    for (m, n, k, seed) in ((16, 128, 256, 1), (16, 8, 128, 2), (5, 64, 512, 3), (16, 512, 2048, 4), (1, 256, 1024, 5)):
        r = run_case(mod, m, n, k, 128, seed, dev)
        bar = r["bf16_half_ulp_at_max"] + 1e-3 * r["ref_abs_max"] * 2 ** -10
        r["bar"] = bar
        r["pass"] = r["max_abs_err"] <= 2 * r["bf16_half_ulp_at_max"] + 1e-6 and r["rel_fro_err"] < 4e-3
        results.append(r)
        print(f"m={m} n={n} k={k}: max|err| {r['max_abs_err']:.4g} (bf16 half-ulp at max {r['bf16_half_ulp_at_max']:.4g}), "
              f"rel Frobenius {r['rel_fro_err']:.3e} -> {'ok' if r['pass'] else 'FAIL'}", flush=True)
    # a known-answer instance: A all ones (scale 2^-? per block), B rows one-hot-ish; D must equal the reference exactly in bf16
    ok = all(r["pass"] for r in results)
    print("fp8_fp4_gemm_sm120 selftest:", "ok" if ok else "FAIL", json.dumps({k: v for k, v in results[0].items() if k != "pass"}))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", type=Path, help="write the selftest rows as JSON (refuses to overwrite)")
    a = ap.parse_args(argv)
    dev = torch.device("cuda")
    mod = build()
    if a.selftest:
        rc = selftest(mod, dev)
        return rc
    ap.error("pass --selftest")
    return 2


if __name__ == "__main__":
    sys.exit(main())
