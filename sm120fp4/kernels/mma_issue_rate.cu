// Issue rate of the SM120 mma.sync tensor instructions, measured on registers with no memory traffic.
//
// Three instruction forms, each run as NCHAIN independent accumulator chains per warp for ITERS iterations:
//   fp8   : mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32            (8,192 FLOP per instruction)
//   fp4c  : mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32            (e2m1 in 8-bit containers; 8,192 FLOP)
//   nvfp4 : mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3  (16,384 FLOP)
// The last is the instruction the SM120 block-scaled CUTLASS kernels use (their names carry "s16864").
//
// Each block records clock64() around its loop; the host reports cycles per instruction per SM sub-partition and the
// implied FLOP per clock per SM, independent of the clock frequency, plus wall-clock TFLOPS from CUDA events.
//
//   nvcc -O3 -arch=sm_120a -o mma_issue_rate mma_issue_rate.cu && ./mma_issue_rate
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#ifndef NCHAIN
#define NCHAIN 8
#endif
#ifndef ITERS
#define ITERS 4096
#endif

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

struct Fp8 {
    static constexpr double flop = 2.0 * 16 * 8 * 32;
    static constexpr const char* name = "fp8 m16n8k32 kind::f8f6f4 e4m3xe4m3 f32";
    __device__ static inline void run(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
};
struct Fp8H {   // fp16 accumulate: the sheet's "FP8 Tensor with FP16 accumulate" row (838 TFLOPS dense on the 5090)
    static constexpr double flop = 2.0 * 16 * 8 * 32;
    static constexpr const char* name = "fp8 m16n8k32 kind::f8f6f4 e4m3xe4m3 f16 accumulate";
    __device__ static inline void run(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
        unsigned* h = reinterpret_cast<unsigned*>(d);   // two b32 registers hold four f16 accumulators
        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f16.e4m3.e4m3.f16 "
                     "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                     : "+r"(h[0]), "+r"(h[1])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
};
struct Fp4c {
    static constexpr double flop = 2.0 * 16 * 8 * 32;
    static constexpr const char* name = "fp4 m16n8k32 kind::f8f6f4 e2m1xe2m1 f32 (8-bit containers)";
    __device__ static inline void run(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
};
struct Nvfp4 {
    static constexpr double flop = 2.0 * 16 * 8 * 64;
    static constexpr const char* name = "nvfp4 m16n8k64 kind::mxf4nvf4 block_scale 4X e2m1xe2m1 f32 ue4m3";
    __device__ static inline void run(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
        const unsigned sa = 0x38383838u, sb = 0x38383838u;   // ue4m3 1.0 in every byte
        asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb));
    }
};

template <class Op>
__global__ void bench(float* out, long long* cycles, unsigned seed) {
    unsigned a[4] = {seed ^ 0x3c3c3c3cu, seed ^ 0x24242424u, seed ^ 0x18181818u, seed ^ 0x0c0c0c0cu};
    unsigned b[2] = {seed ^ 0x38383838u, seed ^ 0x28282828u};
    float d[NCHAIN][4];
    for (int c = 0; c < NCHAIN; ++c) for (int i = 0; i < 4; ++i) d[c][i] = 0.f;
    __syncthreads();
    long long t0 = clock64();
    #pragma unroll 1
    for (int it = 0; it < ITERS; ++it) {
        #pragma unroll
        for (int c = 0; c < NCHAIN; ++c) Op::run(d[c], a, b);
    }
    long long t1 = clock64();
    float s = 0.f;
    for (int c = 0; c < NCHAIN; ++c) for (int i = 0; i < 4; ++i) s += d[c][i];
    if (threadIdx.x == 0) cycles[blockIdx.x] = t1 - t0;
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;   // keep the chains alive
}

template <class Op>
void run_one(int sms, int blocks_per_sm, int warps_per_block) {
    int blocks = sms * blocks_per_sm, threads = warps_per_block * 32;
    float* out; long long* cyc;
    CK(cudaMalloc(&out, sizeof(float) * blocks * threads));
    CK(cudaMalloc(&cyc, sizeof(long long) * blocks));
    bench<Op><<<blocks, threads>>>(out, cyc, 1u);   // warm-up
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0));
    bench<Op><<<blocks, threads>>>(out, cyc, 2u);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms = 0.f; CK(cudaEventElapsedTime(&ms, e0, e1));
    long long* h = (long long*)malloc(sizeof(long long) * blocks);
    CK(cudaMemcpy(h, cyc, sizeof(long long) * blocks, cudaMemcpyDeviceToHost));
    double sum = 0; long long mx = 0; for (int i = 0; i < blocks; ++i) { sum += h[i]; if (h[i] > mx) mx = h[i]; }
    double mean_cycles = sum / blocks;
    double instr_per_block = (double)warps_per_block * NCHAIN * ITERS;
    double total_instr = instr_per_block * blocks;
    double flop = total_instr * Op::flop;
    // per SM: blocks_per_sm blocks run concurrently (if they fit), so the SM executes blocks_per_sm * instr_per_block in ~mean_cycles
    double instr_per_clk_per_sm = blocks_per_sm * instr_per_block / mean_cycles;
    double flop_per_clk_per_sm = instr_per_clk_per_sm * Op::flop;
    printf("{\"op\": \"%s\", \"blocks\": %d, \"warps_per_block\": %d, \"nchain\": %d, \"iters\": %d, "
           "\"mean_block_cycles\": %.0f, \"max_block_cycles\": %lld, \"wall_ms\": %.3f, \"tflops_wall\": %.1f, "
           "\"instr_per_clock_per_sm\": %.4f, \"clocks_per_instr_per_sm\": %.2f, \"flop_per_clock_per_sm\": %.0f, \"implied_tflops_at_2407mhz\": %.0f}\n",
           Op::name, blocks, warps_per_block, NCHAIN, ITERS, mean_cycles, mx, ms, flop / (ms * 1e-3) / 1e12,
           instr_per_clk_per_sm, 1.0 / instr_per_clk_per_sm, flop_per_clk_per_sm, flop_per_clk_per_sm * sms * 2.407e9 / 1e12);
    free(h); CK(cudaFree(out)); CK(cudaFree(cyc));
}

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("{\"device\": \"%s\", \"sms\": %d, \"cc\": \"%d.%d\"}\n", p.name, p.multiProcessorCount, p.major, p.minor);
    int sms = p.multiProcessorCount;
    for (int wpb : {4, 8}) for (int bps : {1, 2, 4}) {
        run_one<Fp8>(sms, bps, wpb);
        run_one<Fp8H>(sms, bps, wpb);
        run_one<Fp4c>(sms, bps, wpb);
        run_one<Nvfp4>(sms, bps, wpb);
    }
    return 0;
}
