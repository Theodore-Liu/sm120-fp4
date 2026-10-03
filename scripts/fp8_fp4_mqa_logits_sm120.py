#!/usr/bin/env python3
"""FP8 x FP4 MQA logits on SM120 (RTX 5090), v0: the indexer kernel DeepGEMM ships for SM100 only (`fp8_fp4_mqa_logits`),
written against the `mma.sync.m16n8k32.kind::f8f6f4` path this repository measured for the GEMM (scripts/fp8_fp4_gemm_sm120.py).

What it computes (DeepGEMM tests/test_attention.py's reference, non-paged form): for query row i with heads h and kv rows j,
    score[h, i, j] = sum_d q[i, h, d] * kv[j, d];   logits[i, j] = sum_h relu(score[h, i, j]) * w[i, h]
for j in [ks[i], ke[i]), stored compressed as logits[i, j - ks[i]] in fp32 of shape [seq_len, max_seqlen_k]; columns past the
row's span are left as written by the caller (the test fills them with -inf).

Operands: q is e4m3 [seq_len, heads, 128] with one UE8M0 scale per (row, head) (head_dim 128 is one 128-wide scale block); kv is
packed e2m1 [seq_len_kv, 64] with one UE8M0 scale per kv row; weights bf16 [seq_len, heads]. The fold of the two UE8M0 scales
happens outside the MMA per (head, kv row), as in the GEMM.

v0 is for correctness: one warp per query row, heads as the MMA's 16 rows (padded with zero rows above `heads`; more than 16
heads loop over head tiles), kv rows as the MMA's n8 columns, four k32 steps over head_dim 128, then relu and the head-weighted
sum done in the C-fragment layout with a shuffle over the eight lanes that share a column pair.

    ~/mlsys-5090-runtime/vllm028/.venv/bin/python scripts/fp8_fp4_mqa_logits_sm120.py --selftest
    ~/mlsys-5090-runtime/vllm028/.venv/bin/python scripts/fp8_fp4_mqa_logits_sm120.py --bench --out reports/fp8-fp4-mqa-logits-v0-rtx5090-20261003.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ue8m0_reference as ref  # noqa: E402

HEAD_DIM = 128

CPP = r"""
#include <torch/extension.h>
void fp8_fp4_mqa_logits_sm120_v0(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv, torch::Tensor sfkv, torch::Tensor w,
                                 torch::Tensor ks, torch::Tensor ke, torch::Tensor logits);
"""

CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cstdint>

// The same fragment layout, container convention and fold as scripts/fp8_fp4_gemm_sm120.py (measured there, 2026-10-02).
__device__ __forceinline__ void mma_f8f6f4(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e2m1.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t unpack_e2m1_x4(uint32_t packed2) {
  const uint32_t p = packed2 & 0xFFFFu;
  return ((p & 0x000Fu) << 2) | ((p & 0x00F0u) << 6) | ((p & 0x0F00u) << 10) | ((p & 0xF000u) << 14);
}
__device__ __forceinline__ float ue8m0_to_float(uint32_t e) { return __uint_as_float(e << 23); }

// v0: one warp per query row. A = q[row] as [heads(16 padded), 128] e4m3; B = kv rows as columns; per n8 tile of kv rows the
// four k32 steps accumulate the exact-integer-scaled dot products, the two UE8M0 scales fold per (head, kv row), relu, weight,
// and the eight lanes with equal t (heads g and g+8 each) sum over heads with xor shuffles over lane bits 2..4.
__global__ void __launch_bounds__(32)
k_mqa_logits_v0(const uint8_t* __restrict__ q, const uint8_t* __restrict__ sfq, const uint8_t* __restrict__ kv,
                const uint8_t* __restrict__ sfkv, const __nv_bfloat16* __restrict__ w, const int* __restrict__ ks,
                const int* __restrict__ ke, float* __restrict__ logits, int S, int H, int N, int max_k) {
  const int i = blockIdx.x;
  const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
  const int k_start = ks[i], k_end = ke[i];
  const uint8_t* qrow = q + (size_t)i * H * 128;
  float* out = logits + (size_t)i * max_k;
  for (int h0 = 0; h0 < H; h0 += 16) {
    const int ha = h0 + g, hb = h0 + g + 8;
    const bool has_a = ha < H, has_b = hb < H;
    const float sa = has_a ? ue8m0_to_float(sfq[(size_t)i * H + ha]) : 0.f;
    const float sb = has_b ? ue8m0_to_float(sfq[(size_t)i * H + hb]) : 0.f;
    const float wa = has_a ? __bfloat162float(w[(size_t)i * H + ha]) : 0.f;
    const float wb = has_b ? __bfloat162float(w[(size_t)i * H + hb]) : 0.f;
    // A fragments for the four k32 steps of this head tile: the q row's bytes are reused for every kv tile
    uint32_t af[4][4];
    for (int s = 0; s < 4; ++s) {
      const int k0 = s * 32;
      af[s][0] = has_a ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)ha * 128 + k0 + 4 * t) : 0u;
      af[s][1] = has_b ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)hb * 128 + k0 + 4 * t) : 0u;
      af[s][2] = has_a ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)ha * 128 + k0 + 16 + 4 * t) : 0u;
      af[s][3] = has_b ? *reinterpret_cast<const uint32_t*>(qrow + (size_t)hb * 128 + k0 + 16 + 4 * t) : 0u;
    }
    for (int n0 = k_start; n0 < k_end; n0 += 8) {
      const int col = n0 + g;                       // this lane's B column (kv row) for the loads
      const bool has_col = col < k_end && col < N;
      float part[4] = {0.f, 0.f, 0.f, 0.f};
      const uint8_t* brow = kv + (size_t)(has_col ? col : 0) * 64;
      for (int s = 0; s < 4; ++s) {
        const int k0 = s * 32;
        uint32_t bf[2];
        bf[0] = has_col ? unpack_e2m1_x4(*reinterpret_cast<const uint16_t*>(brow + (k0 + 4 * t) / 2)) : 0u;
        bf[1] = has_col ? unpack_e2m1_x4(*reinterpret_cast<const uint16_t*>(brow + (k0 + 16 + 4 * t) / 2)) : 0u;
        mma_f8f6f4(part, af[s], bf);
      }
      // C: part[0], part[1] = head ha, kv cols n0+2t, n0+2t+1; part[2], part[3] = head hb, same cols
      const int c0 = n0 + 2 * t, c1 = n0 + 2 * t + 1;
      const float sk0 = (c0 < k_end && c0 < N) ? ue8m0_to_float(sfkv[c0]) : 0.f;
      const float sk1 = (c1 < k_end && c1 < N) ? ue8m0_to_float(sfkv[c1]) : 0.f;
      float v0 = fmaxf(part[0] * (sa * sk0), 0.f) * wa + fmaxf(part[2] * (sb * sk0), 0.f) * wb;
      float v1 = fmaxf(part[1] * (sa * sk1), 0.f) * wa + fmaxf(part[3] * (sb * sk1), 0.f) * wb;
      // sum over the eight lanes that share t (lane bits 2..4 = g)
      for (int m = 4; m < 32; m <<= 1) {
        v0 += __shfl_xor_sync(0xffffffffu, v0, m);
        v1 += __shfl_xor_sync(0xffffffffu, v1, m);
      }
      if (g == 0) {
        if (c0 < k_end) { if (h0 == 0) out[c0 - k_start] = v0; else out[c0 - k_start] += v0; }
        if (c1 < k_end) { if (h0 == 0) out[c1 - k_start] = v1; else out[c1 - k_start] += v1; }
      }
    }
  }
}

void fp8_fp4_mqa_logits_sm120_v0(torch::Tensor q, torch::Tensor sfq, torch::Tensor kv, torch::Tensor sfkv, torch::Tensor w,
                                 torch::Tensor ks, torch::Tensor ke, torch::Tensor logits) {
  const int S = (int)q.size(0), H = (int)q.size(1), N = (int)kv.size(0), max_k = (int)logits.size(1);
  TORCH_CHECK(q.scalar_type() == torch::kFloat8_e4m3fn && q.is_contiguous() && q.size(2) == 128, "q: e4m3 [S, H, 128]");
  TORCH_CHECK(sfq.scalar_type() == torch::kUInt8 && sfq.numel() == (int64_t)S * H && sfq.is_contiguous(), "sfq: uint8 UE8M0 [S, H]");
  TORCH_CHECK(kv.scalar_type() == torch::kInt8 && kv.is_contiguous() && kv.size(1) == 64, "kv: packed e2m1 int8 [N, 64]");
  TORCH_CHECK(sfkv.scalar_type() == torch::kUInt8 && sfkv.numel() == N && sfkv.is_contiguous(), "sfkv: uint8 UE8M0 [N]");
  TORCH_CHECK(w.scalar_type() == torch::kBFloat16 && w.size(0) == S && w.size(1) == H && w.is_contiguous(), "weights: bf16 [S, H]");
  TORCH_CHECK(ks.scalar_type() == torch::kInt && ke.scalar_type() == torch::kInt && ks.numel() == S && ke.numel() == S, "ks, ke: int32 [S]");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat && logits.size(0) == S && logits.is_contiguous(), "logits: fp32 [S, max_seqlen_k]");
  auto st = at::cuda::getCurrentCUDAStream();
  k_mqa_logits_v0<<<S, 32, 0, st>>>(static_cast<const uint8_t*>(q.data_ptr()), sfq.data_ptr<uint8_t>(),
                                    reinterpret_cast<const uint8_t*>(kv.data_ptr<int8_t>()), sfkv.data_ptr<uint8_t>(),
                                    reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), ks.data_ptr<int>(), ke.data_ptr<int>(),
                                    logits.data_ptr<float>(), S, H, N, max_k);
}
"""


def build(verbose: bool = False):
    return load_inline(name="sm120fp4_mqa_logits_v0a", cpp_sources=CPP, cuda_sources=CUDA, functions=["fp8_fp4_mqa_logits_sm120_v0"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a"], verbose=verbose)


def quantize_inputs(q: torch.Tensor, kv: torch.Tensor):
    """q [S, H, 128] -> e4m3 + UE8M0 per (row, head); kv [N, 128] -> packed e2m1 + UE8M0 per row (one 128-wide scale block)."""
    S, H, D = q.shape
    q8, sfq_packed = ref.per_token_cast_to_fp8(q.reshape(S * H, D), use_ue8m0=True, gran_k=D, use_packed_ue8m0=True)
    kv4, sfkv_packed = ref.per_token_cast_to_fp4(kv, use_ue8m0=True, gran_k=D, use_packed_ue8m0=True)
    # one scale per row: unpack the int32 packing to the exponent byte
    sfq = ref.unpack_ue8m0_from_int(sfq_packed)[:, :1]
    sfkv = ref.unpack_ue8m0_from_int(sfkv_packed)[:, :1]
    # UE8M0 byte = biased exponent: value 2^(e - 127); the scales are exact powers of two (ceil_to_ue8m0)
    to_u8 = lambda s: (torch.round(torch.log2(s.float())) + 127).clamp(0, 255).to(torch.uint8)
    return q8.reshape(S, H, D), to_u8(sfq).reshape(S, H).contiguous(), kv4, to_u8(sfkv).reshape(N_of(kv)).contiguous(), sfq, sfkv


def N_of(kv):
    return kv.shape[0]


def dequant_fp8(x8: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    return ref.cast_back_from_fp8(x8, sf.float(), gran_k=HEAD_DIM)


def dequant_fp4(x4: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    return ref.cast_back_from_fp4(x4, sf.float(), gran_k=HEAD_DIM)


def reference(q8, sfq_f, kv4, sfkv_f, w, ks, ke, max_k):
    """DeepGEMM's test reference on the dequantised operands, in fp32 (so the only differences are summation order)."""
    S, H, D = q8.shape
    qf = dequant_fp8(q8.reshape(S * H, D), sfq_f).reshape(S, H, D)
    kf = dequant_fp4(kv4, sfkv_f)
    score = torch.einsum("mhd,nd->hmn", qf, kf)
    logits_full = torch.einsum("hmn,mh->mn", score.relu(), w.float())   # w is [S, H]; the test file's reference indexes it per row
    out = torch.full((S, max_k), float("-inf"), device=q8.device, dtype=torch.float32)
    for i in range(S):
        a, b = int(ks[i]), int(ke[i])
        out[i, : b - a] = logits_full[i, a:b]
    return out


def make_case(S, N, H, seed, dev, full_span=False):
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(S, H, HEAD_DIM, generator=g) * torch.pow(2.0, torch.randint(-4, 5, (S, H, 1), generator=g).float())).to(dev)
    kv = (torch.randn(N, HEAD_DIM, generator=g) * torch.pow(2.0, torch.randint(-4, 5, (N, 1), generator=g).float())).to(dev)
    w = torch.randn(S, H, generator=g).abs().to(dev).to(torch.bfloat16)
    if full_span:
        ks = torch.zeros(S, dtype=torch.int32); ke = torch.full((S,), N, dtype=torch.int32)
    else:
        ks = torch.randint(0, max(1, N // 4), (S,), generator=g).to(torch.int32)
        ke = torch.minimum(ks + torch.randint(1, N, (S,), generator=g).to(torch.int32), torch.full((S,), N, dtype=torch.int32))
    return q, kv, w, ks.to(dev), ke.to(dev)


def run_case(mod, S, N, H, seed, dev, full_span=False):
    q, kv, w, ks, ke = make_case(S, N, H, seed, dev, full_span)
    q8, sfq_u8, kv4, sfkv_u8, sfq_f, sfkv_f = quantize_inputs(q, kv)
    max_k = int((ke - ks).max())
    out = torch.full((S, max_k), float("-inf"), device=dev, dtype=torch.float32)
    mod.fp8_fp4_mqa_logits_sm120_v0(q8, sfq_u8, kv4, sfkv_u8, w, ks, ke, out)
    torch.cuda.synchronize()
    exact = reference(q8, sfq_f, kv4, sfkv_f, w, ks, ke, max_k)
    valid = torch.isfinite(exact)
    diff = (out[valid] - exact[valid]).abs()
    scale = exact[valid].abs().max().clamp_min(1e-30)
    res = {"S": S, "N": N, "H": H, "max_k": max_k, "max_abs_err": float(diff.max()), "ref_abs_max": float(scale),
           "rel_max_err": float(diff.max() / scale), "untouched_outside_span": bool(torch.equal(torch.isfinite(out), valid))}
    res["pass"] = res["rel_max_err"] < 1e-5 and res["untouched_outside_span"]
    return res, (q8, sfq_u8, kv4, sfkv_u8, w, ks, ke, out)


def selftest(mod, dev) -> int:
    ok = True
    print("v0 against the reference (fp32 on the dequantised operands)")
    for (S, N, H, seed, full) in ((8, 256, 8, 1, False), (8, 256, 16, 2, True), (32, 1024, 16, 3, False), (32, 1024, 8, 4, True), (5, 300, 12, 5, False), (16, 2048, 32, 6, False)):
        r, _ = run_case(mod, S, N, H, seed, dev, full)
        ok &= r["pass"]
        print(f"  S={S} N={N} H={H} {'full span' if full else 'random spans'}: rel max err {r['rel_max_err']:.2e} (ref max {r['ref_abs_max']:.3g}), "
              f"outside-span untouched {r['untouched_outside_span']} -> {'ok' if r['pass'] else 'FAIL'}", flush=True)
    print("fp8_fp4_mqa_logits_sm120 v0 selftest:", "ok" if ok else "FAIL")
    return 0 if ok else 1


def bench(mod, dev, out: Path | None) -> int:
    if out is not None and out.exists():
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 2
    props = torch.cuda.get_device_properties(0)
    flush_buf = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    rows = []
    for (S, N, H) in ((32, 1024, 16), (32, 8192, 16), (8, 65536, 16), (128, 8192, 16)):
        r, args = run_case(mod, S, N, H, 100 + S, dev, full_span=True)
        q8, sfq_u8, kv4, sfkv_u8, w, ks, ke, o = args
        times = []
        for _ in range(10):
            flush_buf.fill_(1)
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); mod.fp8_fp4_mqa_logits_sm120_v0(q8, sfq_u8, kv4, sfkv_u8, w, ks, ke, o); e1.record()
            torch.cuda.synchronize()
            times.append(e0.elapsed_time(e1) * 1000)
        nbytes = q8.numel() + kv4.numel() + sfq_u8.numel() + sfkv_u8.numel() + w.numel() * 2 + o.numel() * 4
        flops = 2.0 * S * H * N * HEAD_DIM
        r.update({"us_median": statistics.median(times), "us_min": min(times), "bytes": nbytes, "GBps": nbytes / statistics.median(times) / 1e3,
                  "TFLOPs": flops / statistics.median(times) / 1e6, "floor_us_at_1792": nbytes / 1792.0 / 1e3})
        rows.append(r)
        print(f"S={S} N={N} H={H}: {r['us_median']:.1f} us ({r['GBps']:.0f} GB/s, {r['TFLOPs']:.2f} TFLOP/s; floor {r['floor_us_at_1792']:.1f} us); rel max err {r['rel_max_err']:.1e}", flush=True)
    report = {"kernel": "fp8_fp4_mqa_logits_sm120_v0 (one warp per query row, correctness version)", "device": props.name,
              "note": "cold L2 (256 MB fill before each launch); median of 10; full spans; bytes = q e4m3 + kv packed e2m1 + scales + bf16 weights + fp32 logits; floor at 1792 GB/s; the logits output dominates the bytes at these shapes", "rows": rows}
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print("->", out)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    dev = torch.device("cuda")
    mod = build()
    rc = 0
    if a.selftest:
        rc = selftest(mod, dev)
    if a.bench:
        rc = rc or bench(mod, dev, a.out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
