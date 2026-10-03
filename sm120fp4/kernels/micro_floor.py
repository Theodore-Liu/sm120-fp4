"""The two measurements docs/stage2-design.md asks for before a fused MoE kernel is written.

1. Achievable weight-read floor. A kernel that does nothing but read the FP4 weight and scale bytes of the experts a
   decode batch touches (the same chunk sizes and the same random routing as scripts/bench_moe_baseline.py), with L2
   flushed before every replay of a CUDA graph. Swept over blocks per SM; the best configuration is the floor a real
   kernel can hope for, as opposed to the datasheet's 1792 GB/s.
2. The price of the two-phase structure. The same bytes read in two phases, FC1's share (gate and up, two thirds of each
   expert) then FC2's share (one third), separated by (a) a cooperative-launch grid barrier inside one kernel, or (b) two
   kernels chained with programmatic dependent launch (PDL). Plus an empty kernel, an empty grid barrier and an empty
   PDL pair, for the fixed costs.

    PYTHONPATH=. python scripts/micro_floor.py --out reports/micro-floor-<device>-<date>.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

CPP = r"""
#include <torch/extension.h>
void stream_read(torch::Tensor ptrs, int64_t chunk_bytes, int64_t begin, int64_t end, int64_t blocks, int64_t threads, torch::Tensor sink);
void two_phase_coop(torch::Tensor ptrs, int64_t chunk_bytes, int64_t split, int64_t blocks, int64_t threads, torch::Tensor sink);
void two_phase_pdl(torch::Tensor ptrs, int64_t chunk_bytes, int64_t split, int64_t blocks, int64_t threads, torch::Tensor sink);
void empty_kernel(int64_t blocks, int64_t threads, torch::Tensor sink);
void empty_coop_sync(int64_t blocks, int64_t threads, torch::Tensor sink);
void empty_pdl_pair(int64_t blocks, int64_t threads, torch::Tensor sink);
int64_t max_coop_blocks_per_sm(int64_t threads);
"""

CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <ATen/cuda/CUDAContext.h>
namespace cg = cooperative_groups;

// Read bytes [begin, end) of every chunk (16 bytes per load, streaming hint), fold them so the loads cannot be removed.
__device__ __forceinline__ unsigned read_range(const long long* __restrict__ ptrs, int nchunks, long long chunk_vec,
                                               long long begin_vec, long long end_vec) {
  const long long span = end_vec - begin_vec;
  const long long total = span * nchunks;
  unsigned acc = 0;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
    const long long c = i / span;
    const long long off = begin_vec + (i - c * span);
    const uint4* base = reinterpret_cast<const uint4*>(ptrs[c]);
    uint4 v = __ldcs(base + off);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  return acc;
}

__global__ void k_stream(const long long* ptrs, int nchunks, long long chunk_vec, long long b, long long e, unsigned* sink) {
  unsigned acc = read_range(ptrs, nchunks, chunk_vec, b, e);
  if (acc == 0x9e3779b9u) sink[0] = acc;   // practically never true; keeps the loads live
}

__global__ void k_coop(const long long* ptrs, int nchunks, long long chunk_vec, long long split, unsigned* sink) {
  unsigned acc = read_range(ptrs, nchunks, chunk_vec, 0, split);
  cg::this_grid().sync();
  acc ^= read_range(ptrs, nchunks, chunk_vec, split, chunk_vec);
  if (acc == 0x9e3779b9u) sink[0] = acc;
}

__global__ void k_phase_a(const long long* ptrs, int nchunks, long long chunk_vec, long long split, unsigned* sink) {
  unsigned acc = read_range(ptrs, nchunks, chunk_vec, 0, split);
  if (acc == 0x9e3779b9u) sink[0] = acc;
  cudaTriggerProgrammaticLaunchCompletion();
}

__global__ void k_phase_b(const long long* ptrs, int nchunks, long long chunk_vec, long long split, unsigned* sink) {
  cudaGridDependencySynchronize();
  unsigned acc = read_range(ptrs, nchunks, chunk_vec, split, chunk_vec);
  if (acc == 0x9e3779b9u) sink[0] = acc;
}

__global__ void k_empty(unsigned* sink) { if (threadIdx.x == 1023 && blockIdx.x == 1 << 30) sink[0] = 1; }
__global__ void k_empty_sync(unsigned* sink) {
  cg::this_grid().sync();
  if (threadIdx.x == 1023 && blockIdx.x == 1 << 30) sink[0] = 1;
}
__global__ void k_empty_a(unsigned* sink) {
  if (threadIdx.x == 1023 && blockIdx.x == 1 << 30) sink[0] = 1;
  cudaTriggerProgrammaticLaunchCompletion();
}
__global__ void k_empty_b(unsigned* sink) {
  cudaGridDependencySynchronize();
  if (threadIdx.x == 1023 && blockIdx.x == 1 << 30) sink[0] = 1;
}

static void check(cudaError_t e) { TORCH_CHECK(e == cudaSuccess, cudaGetErrorString(e)); }

static void launch_pdl(void (*kern)(const long long*, int, long long, long long, unsigned*), int blocks, int threads,
                       const long long* p, int n, long long cv, long long s, unsigned* sink, cudaStream_t st) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(blocks); cfg.blockDim = dim3(threads); cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr; cfg.numAttrs = 1;
  check(cudaLaunchKernelEx(&cfg, kern, p, n, cv, s, sink));
}

void stream_read(torch::Tensor ptrs, int64_t chunk_bytes, int64_t begin, int64_t end, int64_t blocks, int64_t threads, torch::Tensor sink) {
  auto st = at::cuda::getCurrentCUDAStream();
  k_stream<<<blocks, threads, 0, st>>>((const long long*)ptrs.data_ptr<int64_t>(), (int)ptrs.numel(), chunk_bytes / 16,
                                        begin / 16, end / 16, (unsigned*)sink.data_ptr<int32_t>());
  check(cudaGetLastError());
}

void two_phase_coop(torch::Tensor ptrs, int64_t chunk_bytes, int64_t split, int64_t blocks, int64_t threads, torch::Tensor sink) {
  auto st = at::cuda::getCurrentCUDAStream();
  const long long* p = (const long long*)ptrs.data_ptr<int64_t>();
  int n = (int)ptrs.numel();
  long long cv = chunk_bytes / 16, sv = split / 16;
  unsigned* s = (unsigned*)sink.data_ptr<int32_t>();
  void* args[] = {&p, &n, &cv, &sv, &s};
  check(cudaLaunchCooperativeKernel((void*)k_coop, dim3(blocks), dim3(threads), args, 0, st));
}

void two_phase_pdl(torch::Tensor ptrs, int64_t chunk_bytes, int64_t split, int64_t blocks, int64_t threads, torch::Tensor sink) {
  auto st = at::cuda::getCurrentCUDAStream();
  const long long* p = (const long long*)ptrs.data_ptr<int64_t>();
  int n = (int)ptrs.numel();
  unsigned* s = (unsigned*)sink.data_ptr<int32_t>();
  k_phase_a<<<blocks, threads, 0, st>>>(p, n, chunk_bytes / 16, split / 16, s);
  check(cudaGetLastError());
  launch_pdl(k_phase_b, (int)blocks, (int)threads, p, n, chunk_bytes / 16, split / 16, s, st);
}

void empty_kernel(int64_t blocks, int64_t threads, torch::Tensor sink) {
  auto st = at::cuda::getCurrentCUDAStream();
  k_empty<<<blocks, threads, 0, st>>>((unsigned*)sink.data_ptr<int32_t>());
  check(cudaGetLastError());
}

void empty_coop_sync(int64_t blocks, int64_t threads, torch::Tensor sink) {
  auto st = at::cuda::getCurrentCUDAStream();
  unsigned* s = (unsigned*)sink.data_ptr<int32_t>();
  void* args[] = {&s};
  check(cudaLaunchCooperativeKernel((void*)k_empty_sync, dim3(blocks), dim3(threads), args, 0, st));
}

void empty_pdl_pair(int64_t blocks, int64_t threads, torch::Tensor sink) {
  auto st = at::cuda::getCurrentCUDAStream();
  unsigned* s = (unsigned*)sink.data_ptr<int32_t>();
  k_empty_a<<<blocks, threads, 0, st>>>(s);
  check(cudaGetLastError());
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(blocks); cfg.blockDim = dim3(threads); cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr; cfg.numAttrs = 1;
  check(cudaLaunchKernelEx(&cfg, k_empty_b, s));
}

int64_t max_coop_blocks_per_sm(int64_t threads) {
  int n = 0;
  check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, k_coop, (int)threads, 0));
  return n;
}
"""


def build():
    return load_inline(name="sm120fp4_micro_floor", cpp_sources=CPP, cuda_sources=CUDA,
                       functions=["stream_read", "two_phase_coop", "two_phase_pdl", "empty_kernel", "empty_coop_sync",
                                  "empty_pdl_pair", "max_coop_blocks_per_sm"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a"], verbose=False)


_FLUSH = None


def flush_l2():
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.empty(256 * 2**20, dtype=torch.uint8, device="cuda")
    _FLUSH.fill_(1)


def graph_time(fn, iters=50, warmup=5, cold=True, after_flush=None):
    """Median time of one replay of a CUDA graph of fn, in us, L2 flushed before each replay when cold.

    The flush also keeps the GPU busy while the host enqueues the start event and the graph, so the host's submission
    latency does not land between the two events. Timed on an idle GPU, an empty kernel reads about 7 us, which is that
    latency, not the kernel; every timing here is therefore taken behind the flush, fixed costs included.

    after_flush: called after each flush and before the start event, untimed - e.g. a read of the buffers a kernel's
    predecessor would have left in L2, so a kernel timed alone sees the cache it would see inside a sequence."""
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        if cold:
            flush_l2()
        if after_flush is not None:
            after_flush()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports") / f"micro-floor-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--inter", type=int, default=768)
    ap.add_argument("--tokens", default="1,2,4,8,16")
    ap.add_argument("--dram-gbps", type=float, default=1792.0)
    a = ap.parse_args(argv)
    mod = build()
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    e, k, h, i = a.experts, a.topk, a.hidden, a.inter
    # one expert: FC1 codes 2i*h/2 + FC2 codes h*i/2, plus one E4M3 scale per 16 values; FC1's share is two thirds
    chunk = 3 * i * h // 2 + 3 * i * h // 16
    chunk = (chunk + 15) // 16 * 16
    split = (2 * chunk // 3) // 16 * 16
    bank = torch.randint(0, 255, (e * chunk,), dtype=torch.uint8, device=dev)
    sink = torch.zeros(4, dtype=torch.int32, device=dev)
    threads = 256
    coop_max = int(mod.max_coop_blocks_per_sm(threads))
    report = {"device": torch.cuda.get_device_name(0), "sms": sms, "chunk_bytes": chunk, "split_bytes": split,
              "threads": threads, "coop_max_blocks_per_sm": coop_max, "rows": [], "fixed_costs_us": {}}

    for bps in (1, 2, 4, 8):
        blocks = sms * bps
        report["fixed_costs_us"][f"empty kernel, {bps}/SM"] = graph_time(lambda: mod.empty_kernel(blocks, threads, sink))
        if bps <= coop_max:
            report["fixed_costs_us"][f"empty cooperative grid sync, {bps}/SM"] = graph_time(lambda: mod.empty_coop_sync(blocks, threads, sink))
        report["fixed_costs_us"][f"empty PDL pair, {bps}/SM"] = graph_time(lambda: mod.empty_pdl_pair(blocks, threads, sink))
    for name, v in report["fixed_costs_us"].items():
        print(f"{name:40s} {v:7.2f} us")

    for m in [int(t) for t in a.tokens.split(",")]:
        g = torch.Generator().manual_seed(1000 + m)   # the routing of bench_moe_baseline.py for this token count
        torch.randn(m, h, generator=g)
        _, ids = torch.topk(F.softmax(torch.randn(m, e, generator=g), dim=-1), k, dim=-1)
        touched = torch.unique(ids).tolist()
        ptrs = torch.tensor([bank.data_ptr() + x * chunk for x in touched], dtype=torch.int64, device=dev)
        nbytes = len(touched) * chunk
        floor_us = nbytes / (a.dram_gbps * 1e3)
        best = None
        sweep = {}
        for bps in (1, 2, 4, 8, 16):
            blocks = sms * bps
            t = graph_time(lambda: mod.stream_read(ptrs, chunk, 0, chunk, blocks, threads, sink))
            sweep[bps] = t
            if best is None or t < best[1]:
                best = (bps, t)
        coop_bps = min(best[0], coop_max)
        t_coop = graph_time(lambda: mod.two_phase_coop(ptrs, chunk, split, sms * coop_bps, threads, sink))
        t_pdl = graph_time(lambda: mod.two_phase_pdl(ptrs, chunk, split, sms * best[0], threads, sink))
        row = {"tokens": m, "experts": len(touched), "bytes": nbytes, "datasheet_floor_us": floor_us,
               "stream_us_by_blocks_per_sm": sweep, "best_blocks_per_sm": best[0], "stream_us": best[1],
               "stream_gbps": nbytes / best[1] / 1e3, "two_phase_coop_us": t_coop, "coop_blocks_per_sm": coop_bps,
               "two_phase_pdl_us": t_pdl}
        report["rows"].append(row)
        print(f"M={m:2d} experts {len(touched):3d} {nbytes / 2**20:6.1f} MiB: datasheet {floor_us:6.1f} us | stream {best[1]:6.1f} us "
              f"({row['stream_gbps']:6.0f} GB/s, {best[0]}/SM) | two-phase coop {t_coop:6.1f} us ({coop_bps}/SM) | "
              f"two-phase PDL {t_pdl:6.1f} us | sweep {{{', '.join(f'{b}: {v:.1f}' for b, v in sweep.items())}}}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"written {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
