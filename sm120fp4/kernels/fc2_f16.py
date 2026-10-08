"""Stage 2, FC2 with the FP4 codes fed to an f16 MMA and the block scale applied after it ("f16": the decode without the decode).

Nsight Compute on the 32-column kernel (reports/ncu-fc2-cols32-opcodes-16-random-rtx5090-20261008.txt) put 79 percent of its instructions in the
decode of the codes: for every pair of FP4 values `decode_pairs` converts to f16x2, widens to two floats, multiplies each by the E4M3 block scale and
packs the pair to bf16x2, and the MMAs are 5 percent. This kernel keeps the 32-column kernel's shape (two 16-row tiles per warp, G expert groups per
column block, the same expert order and the same fixed summation order) and changes the arithmetic:

- the codes go into the MMA as they come out of `cvt.rn.f16x2.e2m1x2`: e2m1 values are exact in f16, so the A fragment needs no widening, no
  multiply and no pack;
- the MMA is `m16n8k16 f32.f16.f16.f32`, so the activations are read as f16 (the caller converts FC1's bf16 output once; in a layer FC1 would write
  f16); a bf16 value in [2^-14, 65504] converts to f16 exactly, so the activation operand is unchanged in value on this layer's ranges;
- the block scale is constant over each MMA's 16 k values once the codes are laid out so that one MMA covers one scale block: the codes are
  repacked once at load (`repack_codes`: within each 128-wide k chunk, lane tig holds k = 16 b + 4 tig .. 4 tig + 3 for the eight blocks b, where the
  storage layout gives it k = 32 tig .. 32 tig + 31), every lane reads the chunk's eight scales, and each MMA's fragment is added to the running total
  with the row's scale for that block in fp32: four FFMA per lane per MMA in place of the decode's 32 FMUL and 32 conversions per 32 codes. The
  activations are read in the matching 4-element pieces (eight 8-byte loads per lane per chunk in place of four 16-byte ones). A first form that
  applied one lane's scale after an MMA whose 16 k values came from four lanes with four different scale blocks read a 28 percent error against the
  reference (the lanes' scales differ inside one MMA); this form is the correction.

The products are no longer rounded to bf16 before the MMA (the decode rounded code x scale to bf16), so the output is not bit-identical to the
32-column kernel; its error against the fp32 reference is reported beside the 32-column kernel's and must not be larger.

    PYTHONPATH=. python scripts/fc2_f16.py --check-only
    PYTHONPATH=. python scripts/fc2_f16.py --out reports/fc2-f16-<device>-<date>.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_here = Path(__file__).resolve().parent
for _name in ("bench_moe_baseline", "micro_floor", "fc1_w4a16", "fc2_w4a16", "fc1_mma", "fc2_mma", "fc2_mma_pf"):
    _spec = importlib.util.spec_from_file_location(_name, _here / f"{_name}.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)
_c32_spec = importlib.util.spec_from_file_location("fc2_cols32", _here.parent.parent / "scripts" / "fc2_cols32.py")
_c32 = importlib.util.module_from_spec(_c32_spec)
sys.modules["fc2_cols32"] = _c32
_c32_spec.loader.exec_module(_c32)
bench = sys.modules["bench_moe_baseline"]
floor = sys.modules["micro_floor"]
fc1 = sys.modules["fc1_w4a16"]
fc2m = sys.modules["fc2_mma"]
pfm = sys.modules["fc2_mma_pf"]

CPP = r"""
#include <torch/extension.h>
void fc2_f16(torch::Tensor q2, torch::Tensor s2, torch::Tensor act, torch::Tensor experts, torch::Tensor offsets,
             torch::Tensor pairs, torch::Tensor weights, torch::Tensor alpha, torch::Tensor out, torch::Tensor scratch,
             torch::Tensor counters, int64_t top_k, int64_t groups);
void fc2_f16_set_pdl(bool on);
"""

# the prefetch kernel's helpers and load_expert, verbatim (e4m3, fp4x2, pack_bf16, decode_pairs, mma are unused here but harmless)
_PF = pfm.CUDA
_HELPERS = _PF[: _PF.index("// PF_CHAIN: everything an expert's routing needs")]

CUDA = _HELPERS + r"""
constexpr int TILES = 2;                // 16-row tiles per warp
constexpr int COLS2 = 16 * TILES;       // output columns per block

// the 32 codes of one lane segment as 16 f16x2 words, no scale: e2m1 is exact in f16
__device__ __forceinline__ void decode_codes(uint4 q, unsigned* dst) {
  const unsigned w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const unsigned short in = (unsigned short)((w[j >> 2] >> (8 * (j & 3))) & 0xff);
    unsigned o;
    asm("{ .reg .b8 lo, hi;\n mov.b16 {lo, hi}, %1;\n cvt.rn.f16x2.e2m1x2 %0, lo; }\n" : "=r"(o) : "h"(in));
    dst[j] = o;
  }
}

__device__ __forceinline__ void mma_f16(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int CH>
__device__ __forceinline__ void load_expert_f16(const unsigned char* __restrict__ q2p, const unsigned char* __restrict__ s2,
                                                long long r0, int I, int tig, uint4 (&wq)[CH][2], uint2 (&ws)[CH][2]) {
  // wq: the lane's 16 bytes of repacked codes per chunk (the same addresses as the storage layout); ws: the chunk's eight scales for both rows
  const long long r1 = r0 + 8;
#pragma unroll
  for (int ch = 0; ch < CH; ++ch) {
    const int k = ch * 128 + tig * 32;
    wq[ch][0] = __ldcs(reinterpret_cast<const uint4*>(q2p + r0 * (I / 2) + k / 2));
    wq[ch][1] = __ldcs(reinterpret_cast<const uint4*>(q2p + r1 * (I / 2) + k / 2));
    ws[ch][0] = *reinterpret_cast<const uint2*>(s2 + r0 * (I / 16) + ch * 8);
    ws[ch][1] = *reinterpret_cast<const uint2*>(s2 + r1 * (I / 16) + ch * 8);
  }
}

__device__ __forceinline__ float scale_at(uint2 w, int b) {
  const unsigned word = b < 4 ? w.x : w.y;
  return e4m3((word >> (8 * (b & 3))) & 0xff);
}

template <int NT, int CH>
__global__ void __launch_bounds__(WARPS * 32)
k_fc2_f16(const unsigned char* __restrict__ q2, const unsigned char* __restrict__ s2, const __half* __restrict__ act,
          const int* __restrict__ experts, const int* __restrict__ offsets, const int* __restrict__ pairs,
          const float* __restrict__ weights, const float* __restrict__ alpha, __nv_bfloat16* __restrict__ out,
          float* __restrict__ scratch, int* __restrict__ counters, int U, int M, int H, int I, int top_k) {
  __shared__ float part[WARPS][COLS2][MAXM];
  __shared__ int last;
  cudaGridDependencySynchronize();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
  const int n0 = blockIdx.x * COLS2, g = blockIdx.y, G = gridDim.y;
  for (int z = threadIdx.x; z < WARPS * COLS2 * MAXM; z += blockDim.x) (&part[0][0][0])[z] = 0.f;
  __syncthreads();

  for (int u = g * WARPS + warp; u < U; u += WARPS * G) {
    const int e = experts[u];
    if (e >= 0) {                                   // a negative id is a padding slot from the GPU router
      uint4 wq[TILES][CH][2];
      uint2 ws[TILES][CH][2];
#pragma unroll
      for (int j = 0; j < TILES; ++j) load_expert_f16<CH>(q2, s2, (long long)e * H + n0 + 16 * j + gid, I, tig, wq[j], ws[j]);
      const int p0 = offsets[u];
      const int cnt = min(offsets[u + 1] - p0, 8 * NT);
      int pr[NT];
      bool valid[NT];
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        const int slot = t * 8 + gid;
        valid[t] = slot < cnt;
        pr[t] = valid[t] ? pairs[p0 + slot] : 0;
      }
      float c[TILES][NT][4];
#pragma unroll
      for (int j = 0; j < TILES; ++j)
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
          for (int i = 0; i < 4; ++i) c[j][t][i] = 0.f;
#pragma unroll
      for (int ch = 0; ch < CH; ++ch) {
        unsigned xb[NT][16];                        // the activations, f16: for MMA s, k = ch * 128 + 16 s + 4 tig .. + 3 (words 2 s, 2 s + 1)
#pragma unroll
        for (int t = 0; t < NT; ++t) {
          const __half* ar = act + (long long)pr[t] * I + ch * 128 + 4 * tig;
#pragma unroll
          for (int sblk = 0; sblk < 8; ++sblk) {
            const uint2 q = valid[t] ? __ldg(reinterpret_cast<const uint2*>(ar + 16 * sblk)) : make_uint2(0, 0);
            xb[t][2 * sblk] = q.x; xb[t][2 * sblk + 1] = q.y;
          }
        }
#pragma unroll
        for (int j = 0; j < TILES; ++j) {
          unsigned a[2][16];
          decode_codes(wq[j][ch][0], a[0]);
          decode_codes(wq[j][ch][1], a[1]);
          // after the repack, MMA s covers scale block s of this chunk for every lane: words 2 s and 2 s + 1 hold k = 16 s + 4 tig .. + 3
#pragma unroll
          for (int s = 0; s < 8; ++s) {
            const float scr0 = scale_at(ws[j][ch][0], s), scr1 = scale_at(ws[j][ch][1], s);
#pragma unroll
            for (int t = 0; t < NT; ++t) {
              float acc[4] = {0.f, 0.f, 0.f, 0.f};
              mma_f16(acc, a[0][2 * s], a[1][2 * s], a[0][2 * s + 1], a[1][2 * s + 1], xb[t][2 * s], xb[t][2 * s + 1]);
              c[j][t][0] = fmaf(scr0, acc[0], c[j][t][0]);
              c[j][t][1] = fmaf(scr0, acc[1], c[j][t][1]);
              c[j][t][2] = fmaf(scr1, acc[2], c[j][t][2]);
              c[j][t][3] = fmaf(scr1, acc[3], c[j][t][3]);
            }
          }
        }
      }
      const float al = alpha[e];
#pragma unroll
      for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int slot = t * 8 + 2 * tig + h;
          if (slot < cnt) {
            const int p = pairs[p0 + slot];
            const float wt = weights[p] * al;
            const int tok = p / top_k;
#pragma unroll
            for (int j = 0; j < TILES; ++j) {
              part[warp][16 * j + gid][tok] += wt * c[j][t][h];
              part[warp][16 * j + gid + 8][tok] += wt * c[j][t][2 + h];
            }
          }
        }
    }
    __syncwarp();                                   // the next expert may put a token in another lane
  }
  __syncthreads();
  if (G == 1) {
    for (int idx = threadIdx.x; idx < COLS2 * M; idx += blockDim.x) {
      const int r = idx / M, t = idx % M;
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < WARPS; ++w) v += part[w][r][t];   // fixed warp order
      out[(long long)t * H + n0 + r] = __float2bfloat16(v);
    }
    return;
  }
  for (int idx = threadIdx.x; idx < COLS2 * M; idx += blockDim.x) {
    const int r = idx / M, t = idx % M;
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) v += part[w][r][t];
    scratch[((long long)g * MAXM + t) * H + n0 + r] = v;
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = (atomicAdd(&counters[blockIdx.x], 1) == G - 1);
  __syncthreads();
  if (!last) return;
  __threadfence();
  for (int idx = threadIdx.x; idx < COLS2 * M; idx += blockDim.x) {
    const int r = idx / M, t = idx % M;
    float v = 0.f;
    for (int gg = 0; gg < G; ++gg) v += __ldcg(&scratch[((long long)gg * MAXM + t) * H + n0 + r]);   // fixed group order
    out[(long long)t * H + n0 + r] = __float2bfloat16(v);
  }
  if (threadIdx.x == 0) counters[blockIdx.x] = 0;   // ready for the next launch
}

static bool g_pdl = false;
template <typename... KArgs, typename... Args>
static void pdl_launch(void (*kernel)(KArgs...), dim3 grid, dim3 block, cudaStream_t st, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = g_pdl ? 1 : 0;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, kernel, args...) == cudaSuccess, "launch");
}
void fc2_f16_set_pdl(bool on) { g_pdl = on; }

template <int NT, int CH>
static void launch(dim3 grid, cudaStream_t st, const unsigned char* q2, const unsigned char* s2, const __half* act,
                   const int* experts, const int* offsets, const int* pairs, const float* weights, const float* alpha,
                   __nv_bfloat16* out, float* scratch, int* counters, int U, int M, int H, int I, int top_k) {
  pdl_launch(k_fc2_f16<NT, CH>, grid, dim3(WARPS * 32), st, q2, s2, act, experts, offsets, pairs, weights, alpha, out,
             scratch, counters, U, M, H, I, top_k);
}

void fc2_f16(torch::Tensor q2, torch::Tensor s2, torch::Tensor act, torch::Tensor experts, torch::Tensor offsets,
             torch::Tensor pairs, torch::Tensor weights, torch::Tensor alpha, torch::Tensor out, torch::Tensor scratch,
             torch::Tensor counters, int64_t top_k, int64_t groups) {
  const int M = (int)out.size(0), H = (int)out.size(1), I = (int)act.size(1), U = (int)experts.numel();
  TORCH_CHECK(act.scalar_type() == at::kHalf, "the activations are read as f16");
  TORCH_CHECK(M <= MAXM && H % COLS2 == 0 && (I == 768 || I == 1024), "shape");
  TORCH_CHECK(groups >= 1 && groups <= 4 && scratch.numel() >= groups * MAXM * H && counters.numel() >= H / COLS2, "scratch");
  auto st = at::cuda::getCurrentCUDAStream();
  dim3 grid(H / COLS2, (unsigned)groups);
#define F16_ARGS grid, st, q2.data_ptr<uint8_t>(), s2.data_ptr<uint8_t>(), reinterpret_cast<const __half*>(act.data_ptr()), \
    experts.data_ptr<int>(), offsets.data_ptr<int>(), pairs.data_ptr<int>(), weights.data_ptr<float>(), alpha.data_ptr<float>(), \
    reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), scratch.data_ptr<float>(), counters.data_ptr<int>(), U, M, H, I, (int)top_k
  if (I == 768) { if (M <= 8) launch<1, 6>(F16_ARGS); else launch<2, 6>(F16_ARGS); }
  else          { if (M <= 8) launch<1, 8>(F16_ARGS); else launch<2, 8>(F16_ARGS); }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "launch");
}
"""


def repack_codes(q2: torch.Tensor) -> torch.Tensor:
    """The storage layout gives lane tig the 32 codes k = 32 tig .. 32 tig + 31 of each 128-wide chunk (16 bytes); this layout gives it, for each
    scale block b, the 4 codes k = 16 b + 4 tig .. + 3 (2 bytes), so that one MMA covers one scale block. q2: [rows, I / 2] uint8, byte = 2 codes."""
    nbytes = q2.shape[-1]
    assert nbytes % 64 == 0
    c = q2.reshape(-1, nbytes // 64, 8, 4, 2)           # row, chunk, block b, lane tig, 2 bytes: byte offset 8 b + 2 tig + i
    return c.permute(0, 1, 3, 2, 4).reshape(q2.shape).contiguous()   # lane tig, block b, 2 bytes: byte offset 16 tig + 2 b + i


def build(verbose: bool = False):
    return load_inline(name="sm120fp4_fc2_f16", cpp_sources=CPP, cuda_sources=CUDA, functions=["fc2_f16", "fc2_f16_set_pdl"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a"] + (["-Xptxas=-v"] if verbose else []),
                       verbose=verbose)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, help="JSON report, written into the repository")
    ap.add_argument("--check-only", action="store_true", help="correctness and determinism only, no timing")
    ap.add_argument("--ptxas", action="store_true", help="print register use and spills")
    a = ap.parse_args(argv)
    if not a.check_only and a.out is None:
        ap.error("--out is required unless --check-only")
    f16, c32, pf, m1, fl = build(a.ptxas), _c32.build(), pfm.build(), fc1.build(), floor.build()
    dev = torch.device("cuda")
    e, k, h, i = 128, 8, 2048, 768
    w = bench.build(e, h, i, dev)
    q1, s1, q2, s2 = (w[n].contiguous() for n in ("q1", "s1", "q2", "s2"))
    alpha = torch.ones(e, device=dev)
    sink = torch.zeros(4, dtype=torch.int32, device=dev)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    scratch = torch.zeros(4 * 16 * h, device=dev)
    counters = torch.zeros(h // 16, dtype=torch.int32, device=dev)
    rows = []
    print("routing | tokens | variant | normwise vs fp32 MoE | bit-identical x50 | equals cols32 same G | us")

    def case(label, m, ids, wts, x):
        experts, offsets, pairs = fc1.route(ids)
        act = torch.empty(ids.numel(), i, device=dev, dtype=torch.bfloat16)
        m1.fc1_w4a16(q1, s1, x, experts, offsets, pairs, alpha, act, i, k)
        act16 = act.half()                            # what FC1 would write in a layer built for this kernel; not timed
        q2p = repack_codes(q2)                        # once per weight, at load; not timed
        inexact = int((act16.float() != act.float()).sum())   # bf16 values outside f16's range (below 2^-24, above 65504) round or clamp
        wf = wts.reshape(-1).contiguous()
        ref = bench.reference(x, w, ids, wts, i, act_quant=False)
        row = {"routing": label, "tokens": m, "activations_not_exact_in_f16": inexact, "activations": int(act.numel())}
        variants = {"cols16_prefetch_g2": lambda o: pf.fc2_pf(q2, s2, act, experts, offsets, pairs, wf, alpha, o, scratch, counters, k, 2)}
        for G in (2, 4):
            variants[f"cols32_g{G}"] = (lambda o, G=G: c32.fc2_c32(q2, s2, act, experts, offsets, pairs, wf, alpha, o, scratch, counters, k, G))
        for G in (2, 4):
            variants[f"f16_g{G}"] = (lambda o, G=G: f16.fc2_f16(q2p, s2, act16, experts, offsets, pairs, wf, alpha, o, scratch, counters, k, G))
        outs = {}
        for name, fn in variants.items():
            out = torch.empty(m, h, device=dev, dtype=torch.bfloat16)
            go = lambda: fn(out)  # noqa: E731
            go()
            torch.cuda.synchronize()
            rel = float((out.float() - ref).norm() / ref.norm())
            first = out.clone()
            stable = True
            for _ in range(50):
                go()
                stable = stable and bool(torch.equal(out, first))
            outs[name] = first
            twin = name.replace("f16", "cols32")
            same = bool(torch.equal(first, outs[twin])) if name.startswith("f16") and twin in outs else None
            t = None if a.check_only else floor.graph_time(go)
            row[name] = {"rel_err_vs_fp32_moe": rel, "bit_identical_50": stable, "equals_cols32_same_groups": same, "us": t}
            print(f"{label:7s} | {m:6d} | {name:19s} | {rel:20.5f} | {str(stable):17s} | {str(same):20s} | "
                  + ("-" if t is None else f"{t:.1f}"))
        if not a.check_only:
            ptrs = torch.tensor([q2.data_ptr() + int(x_) * h * i // 2 for x_ in experts.tolist()], dtype=torch.int64, device=dev)
            row["read_fc2_codes_us"] = floor.graph_time(lambda: fl.stream_read(ptrs, h * i // 2, 0, h * i // 2, sms * 4, 256, sink))
            print(f"{'':7s} | {'':6s} | {'read floor':19s} | {'':20s} | {'':17s} | {'':20s} | {row['read_fc2_codes_us']:.1f}")
        rows.append(row)

    for m in (1, 2, 4, 8, 16):
        g = torch.Generator().manual_seed(1000 + m)
        x = torch.randn(m, h, generator=g).to(device=dev, dtype=torch.bfloat16)
        wts, ids = torch.topk(F.softmax(torch.randn(m, e, generator=g), dim=-1), k, dim=-1)
        wts = (wts / wts.sum(-1, keepdim=True)).float().to(dev).contiguous()
        case("random", m, ids.to(torch.int32).to(dev).contiguous(), wts, x)
    fixed = torch.arange(8, dtype=torch.int32, device=dev) * 16
    for m in (1, 16):
        x = torch.randn(m, h, generator=torch.Generator().manual_seed(7)).to(device=dev, dtype=torch.bfloat16)
        wts = torch.full((m, k), 1.0 / k, device=dev)
        case("fixed8", m, fixed.repeat(m, 1).contiguous(), wts, x)
    worse = [(r["routing"], r["tokens"]) for r in rows if r["f16_g4"]["rel_err_vs_fp32_moe"] > 1.05 * r["cols32_g4"]["rel_err_vs_fp32_moe"] + 1e-5]
    if a.out is not None:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"device": torch.cuda.get_device_name(0), "check_only": a.check_only, "rows": rows, "error_worse_than_cols32": worse},
                                    indent=1) + "\n", encoding="utf-8")
        print(f"written {a.out}")
    if worse:
        print("ERROR LARGER THAN THE 32-COLUMN KERNEL'S:", worse)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
