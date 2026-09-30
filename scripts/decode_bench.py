"""Stage 2: two ways to decode FP4 x E4M3 into bf16 pairs, checked for equality on every input and timed.

A (the kernels' way today): cvt.rn.f16x2.e2m1x2, the pair to floats, times the scale, packed to bf16x2.
C (lookup): the bf16 bits of the eight E2M1 magnitudes held in two byte tables, selected with prmt by each nibble's
low three bits, the sign put in by bit operations, then one bf16x2 multiply by the scale. An E2M1 value has at most
two significant bits and an E4M3 scale at most four, so the product is exact in bf16 and C must equal A bit for bit.

The equality check is exhaustive: every code byte (two values) against every E4M3 scale byte, 65,536 pairs. The timing
decodes a large array of codes, each value eight times under different scales so the decode rather than the read is
timed, with a fold of the results written out so nothing is eliminated.

    PYTHONPATH=. python scripts/decode_bench.py --out reports/decode-bench-<device>-<date>.json
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
_spec = importlib.util.spec_from_file_location("micro_floor", _here / "micro_floor.py")
floor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(floor)

CPP = r"""
#include <torch/extension.h>
void decode_all(torch::Tensor out_a, torch::Tensor out_c);
void decode_timed(torch::Tensor codes, torch::Tensor scales, torch::Tensor sink, int64_t which);
"""

CUDA = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ float e4m3(unsigned b) { __nv_fp8_e4m3 v; v.__x = (unsigned char)b; return float(v); }

__device__ __forceinline__ unsigned decode_a(unsigned byte, float sc) {
  unsigned o;
  unsigned short in = (unsigned short)byte;
  asm("{ .reg .b8 lo, hi;\n mov.b16 {lo, hi}, %1;\n cvt.rn.f16x2.e2m1x2 %0, lo; }\n" : "=r"(o) : "h"(in));
  const float2 f = __half22float2(*reinterpret_cast<__half2*>(&o));
  __nv_bfloat162 v = __floats2bfloat162_rn(f.x * sc, f.y * sc);
  return *reinterpret_cast<unsigned*>(&v);
}

// bf16 bits of the E2M1 magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6: low bytes and high bytes, eight each
constexpr unsigned LO0 = 0xC0800000u, LO1 = 0xC0804000u, HI0 = 0x3F3F3F00u, HI1 = 0x40404040u;

// four values (two code bytes) of w's low 16 bits -> two bf16x2 (low nibble in the low half)
__device__ __forceinline__ void decode_c4(unsigned w, __nv_bfloat162 s2, unsigned& p0, unsigned& p1) {
  const unsigned sel = w & 0x7777u;
  const unsigned lo = __byte_perm(LO0, LO1, sel), hi = __byte_perm(HI0, HI1, sel);
  unsigned a = __byte_perm(lo, hi, 0x5140), b = __byte_perm(lo, hi, 0x7362);
  a |= ((w << 12) & 0x8000u) | ((w << 24) & 0x80000000u);
  b |= ((w << 4) & 0x8000u) | ((w << 16) & 0x80000000u);
  __nv_bfloat162 ra = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), s2);
  __nv_bfloat162 rb = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&b), s2);
  p0 = *reinterpret_cast<unsigned*>(&ra);
  p1 = *reinterpret_cast<unsigned*>(&rb);
}

constexpr int REPS = 8;

__global__ void k_all(unsigned* out_a, unsigned* out_c) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;   // code byte (low 8 bits) x scale byte (high 8 bits)
  if (idx >= 65536) return;
  const unsigned code = idx & 0xff, sbyte = idx >> 8;
  const float sc = e4m3(sbyte);
  out_a[idx] = decode_a(code, sc);
  unsigned p0, p1;
  decode_c4(code, __float2bfloat162_rn(sc), p0, p1);
  out_c[idx] = p0;
}

template <int WHICH>
__global__ void k_timed(const uint4* __restrict__ codes, const unsigned char* __restrict__ scales, unsigned* sink, long long n) {
  unsigned acc = 0;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
    const uint4 q = codes[i];
    const float sc0 = e4m3(scales[i]);
    const unsigned w[4] = {q.x, q.y, q.z, q.w};
    // each loaded value is decoded REPS times under a different scale, so the timing is the decode, not the read
#pragma unroll 1
    for (int r = 0; r < REPS; ++r) {
    const float sc = sc0 * (float)(r + 1);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      if (WHICH == 0) {
#pragma unroll
        for (int b = 0; b < 4; ++b) acc ^= decode_a((w[j] >> (8 * b)) & 0xff, sc);
      } else {
        const __nv_bfloat162 s2 = __float2bfloat162_rn(sc);
        unsigned p0, p1, p2, p3;
        decode_c4(w[j], s2, p0, p1);
        decode_c4(w[j] >> 16, s2, p2, p3);
        acc ^= p0 ^ p1 ^ p2 ^ p3;
      }
    }
    }
  }
  if (acc == 0x9e3779b9u) sink[0] = acc;
}

void decode_all(torch::Tensor out_a, torch::Tensor out_c) {
  k_all<<<256, 256, 0, at::cuda::getCurrentCUDAStream()>>>((unsigned*)out_a.data_ptr<int>(), (unsigned*)out_c.data_ptr<int>());
}

void decode_timed(torch::Tensor codes, torch::Tensor scales, torch::Tensor sink, int64_t which) {
  const long long n = codes.numel() / 16;
  auto st = at::cuda::getCurrentCUDAStream();
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  if (which == 0) k_timed<0><<<sms * 8, 256, 0, st>>>((const uint4*)codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), (unsigned*)sink.data_ptr<int>(), n);
  else k_timed<1><<<sms * 8, 256, 0, st>>>((const uint4*)codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), (unsigned*)sink.data_ptr<int>(), n);
}
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    m = load_inline(name="sm120fp4_decode_bench", cpp_sources=CPP, cuda_sources=CUDA, functions=["decode_all", "decode_timed"],
                    extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a"], verbose=False)
    dev = torch.device("cuda")
    out_a = torch.empty(65536, dtype=torch.int32, device=dev)
    out_c = torch.empty(65536, dtype=torch.int32, device=dev)
    m.decode_all(out_a, out_c)
    torch.cuda.synchronize()
    idx = torch.arange(65536, device=dev)
    finite = ~((idx >> 8) & 0x7f).eq(0x7f)          # E4M3 0x7f / 0xff is NaN: excluded from the equality
    diff = (out_a != out_c) & finite
    n_diff = int(diff.sum())
    print("exhaustive: pairs compared", int(finite.sum()), "differ", n_diff)
    fl_ = floor.build()
    n_bytes = 64 << 20                               # 64 MiB of codes: 128 Mi values
    g = torch.Generator(device="cpu").manual_seed(0)
    codes = torch.randint(0, 256, (n_bytes,), dtype=torch.uint8, generator=g).to(dev)
    scales = torch.randint(0, 0x7e, (n_bytes // 16,), dtype=torch.uint8, generator=g).to(dev)
    sink = torch.zeros(4, dtype=torch.int32, device=dev)
    t_a = floor.graph_time(lambda: m.decode_timed(codes, scales, sink, 0))
    t_c = floor.graph_time(lambda: m.decode_timed(codes, scales, sink, 1))
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    ptr = torch.tensor([codes.data_ptr()], dtype=torch.int64, device=dev)
    t_r = floor.graph_time(lambda: fl_.stream_read(ptr, n_bytes, 0, n_bytes, sms * 4, 256, sink))
    rep = {"device": torch.cuda.get_device_name(0), "exhaustive_pairs": int(finite.sum()), "exhaustive_differ": n_diff,
           "timed_code_bytes": n_bytes, "decodes_per_value": 8, "decode_a_us": t_a, "decode_c_us": t_c, "read_codes_us": t_r}
    print(json.dumps(rep))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
