"""Empirical probe of the FP4 x FP4 MMA forms on SM120, the q side of vLLM's MXFP4 indexer path (wiring doc, step 2).

Arm 1: `mma.sync.m16n8k32.kind::f8f6f4.f32.e2m1.e2m1.f32` (one e2m1 value per byte, both operands).
Arm 2: `mma.sync.m16n8k64.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0` (two e2m1 values per byte), with
every UE8M0 scale byte 127 (2^0) so only the packing is tested.

Each arm hands fragments built from a layout hypothesis to one warp, runs one MMA and compares the 16 x 8 fp32 C with the
fp32 product of the dequantised matrices (exact: every product and every partial sum of e2m1 values is representable).
Exit 0 only if every arm has an exact hypothesis.

    PYTHONPATH=. python scripts/probe_f4f4.py [--out reports/probe-f4f4-<gpu>-<date>.json]
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

CPP = "#include <torch/extension.h>\nvoid probe(torch::Tensor frags, torch::Tensor out, int64_t arm);\n"
CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
__global__ void k_probe(const uint32_t* __restrict__ f, float* __restrict__ out, int arm) {
  const int lane = threadIdx.x;
  uint32_t a[4] = {f[lane * 8 + 0], f[lane * 8 + 1], f[lane * 8 + 2], f[lane * 8 + 3]};
  uint32_t b[2] = {f[lane * 8 + 4], f[lane * 8 + 5]};
  float c[4] = {0.f, 0.f, 0.f, 0.f};
  if (arm == 1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  } else {
    const uint32_t sa = f[lane * 8 + 6], sb = f[lane * 8 + 7];
    const uint16_t z = 0;
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%12, %12}, %11, {%12, %12};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb), "h"(z));
  }
  for (int i = 0; i < 4; ++i) out[lane * 4 + i] = c[i];
}
void probe(torch::Tensor frags, torch::Tensor out, int64_t arm) {
  TORCH_CHECK(frags.scalar_type() == torch::kInt && frags.numel() == 32 * 8 && out.numel() == 32 * 4);
  k_probe<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const uint32_t*>(frags.data_ptr<int>()), out.data_ptr<float>(), (int)arm);
}
"""


def build():
    return load_inline(name="sm120fp4_probe_f4f4", cpp_sources=CPP, cuda_sources=CUDA, functions=["probe"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a"])


def s32(v: int) -> int:
    return v - (1 << 32) if v >= 1 << 31 else v


def place(code: int, how: str) -> int:
    return code << 2 if how == "shift2" else code


def unpack_c(out: torch.Tensor) -> torch.Tensor:
    got = torch.zeros(16, 8)
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        got[g, 2 * t], got[g, 2 * t + 1], got[g + 8, 2 * t], got[g + 8, 2 * t + 1] = out[lane].cpu()
    return got


def run(mod, frags, arm, dev):
    out = torch.zeros(32, 4, device=dev)
    mod.probe(frags.to(dev), out, arm)
    torch.cuda.synchronize()
    return unpack_c(out)


def arm1(mod, dev) -> bool:
    """Byte containers, k = 32: lane (g, t) holds four consecutive k per register, the e4m3 x e2m1 probe's layout."""
    ac = torch.randint(0, 16, (16, 32), dtype=torch.int8)
    bc = torch.randint(0, 16, (32, 8), dtype=torch.int8)
    want = ref.dequantize_from_fp4_e2m1(ac) @ ref.dequantize_from_fp4_e2m1(bc)
    best = None
    for a_how, b_how in itertools.product(("low", "shift2"), ("low", "shift2")):
        frags = torch.zeros(32, 8, dtype=torch.int32)
        for lane in range(32):
            g, t = lane >> 2, lane & 3
            for i, (r, k0) in enumerate(((g, 4 * t), (g + 8, 4 * t), (g, 16 + 4 * t), (g + 8, 16 + 4 * t))):
                frags[lane, i] = s32(sum(place(int(ac[r, k0 + j]), a_how) << (8 * j) for j in range(4)))
            for i, k0 in enumerate((4 * t, 16 + 4 * t)):
                frags[lane, 4 + i] = s32(sum(place(int(bc[k0 + j, g]), b_how) << (8 * j) for j in range(4)))
        err = float((run(mod, frags, 1, dev) - want).abs().max())
        print(f"  arm 1 A {a_how:6s} B {b_how:6s}: max|err| {err:.4g}")
        best = min(best or (a_how, b_how, err), (a_how, b_how, err), key=lambda x: x[2])
    print("  arm 1 match:", best if best[2] == 0.0 else ("none exact; closest", best))
    return best[2] == 0.0


def arm2(mod, dev) -> bool:
    """Packed nibbles, k = 64: lane (g, t) holds eight consecutive k per register; nibble order low-first or high-first."""
    ac = torch.randint(0, 16, (16, 64), dtype=torch.int8)
    bc = torch.randint(0, 16, (64, 8), dtype=torch.int8)
    want = ref.dequantize_from_fp4_e2m1(ac) @ ref.dequantize_from_fp4_e2m1(bc)
    best = None
    for order in ("low-first", "high-first"):
        def pack8(vals):
            v = 0
            for j, c in enumerate(vals):
                byte, hi = j // 2, j % 2
                sh = 8 * byte + (4 * hi if order == "low-first" else 4 * (1 - hi))
                v |= int(c) << sh
            return s32(v)
        frags = torch.zeros(32, 8, dtype=torch.int32)
        for lane in range(32):
            g, t = lane >> 2, lane & 3
            for i, (r, k0) in enumerate(((g, 8 * t), (g + 8, 8 * t), (g, 32 + 8 * t), (g + 8, 32 + 8 * t))):
                frags[lane, i] = pack8([ac[r, k0 + j] for j in range(8)])
            for i, k0 in enumerate((8 * t, 32 + 8 * t)):
                frags[lane, 4 + i] = pack8([bc[k0 + j, g] for j in range(8)])
        frags[:, 6] = frags[:, 7] = s32(0x7F7F7F7F)
        err = float((run(mod, frags, 2, dev) - want).abs().max())
        print(f"  arm 2 nibbles {order:10s}: max|err| {err:.4g}")
        best = min(best or (order, err), (order, err), key=lambda x: x[1])
    # both nibble orders permute k identically on A and B, which no dot product can see; only a scale block could
    print("  arm 2: packing exact" if best[1] == 0.0 else ("  arm 2: none exact; closest", best),
          "(nibble order within a byte is invisible when both operands share it)")
    return best[1] == 0.0


def arm3(mod, dev) -> bool:
    """Scale mapping: A nonzero only in k block h (k 32h .. 32h+31 under arm 2's packing), all scale bytes 2^0 except one byte of one
    lane's scale-a or scale-b register set to 2^1; record which C rows (scale-a) or columns (scale-b) double. A byte that doubles a
    whole row in exactly one block, and every (row, block) reached by exactly one byte, is the mapping; a partial change means the
    scale block does not coincide with the packed k block."""
    ac = torch.randint(1, 8, (16, 64), dtype=torch.int8)   # positive codes, so no row sums to zero
    bc = torch.randint(1, 8, (64, 8), dtype=torch.int8)
    def pack8(vals):
        v = 0
        for j, c in enumerate(vals):
            v |= int(c) << (4 * j)
        return s32(v)
    base = {}
    hits = {"a": {}, "b": {}}
    partial = 0
    for h in (0, 1):
        a_h = ac.clone(); a_h[:, 32 * (1 - h):32 * (1 - h) + 32] = 0
        frags = torch.zeros(32, 8, dtype=torch.int32)
        for lane in range(32):
            g, t = lane >> 2, lane & 3
            for i, (r, k0) in enumerate(((g, 8 * t), (g + 8, 8 * t), (g, 32 + 8 * t), (g + 8, 32 + 8 * t))):
                frags[lane, i] = pack8([a_h[r, k0 + j] for j in range(8)])
            for i, k0 in enumerate((8 * t, 32 + 8 * t)):
                frags[lane, 4 + i] = pack8([bc[k0 + j, g] for j in range(8)])
        frags[:, 6] = frags[:, 7] = s32(0x7F7F7F7F)
        c0 = run(mod, frags, 2, dev)
        want = ref.dequantize_from_fp4_e2m1(a_h) @ ref.dequantize_from_fp4_e2m1(bc)
        assert float((c0 - want).abs().max()) == 0.0
        for side, col in (("a", 6), ("b", 7)):
            for lane in range(32):
                for byte in range(4):
                    f2 = frags.clone()
                    f2[lane, col] = s32(0x7F7F7F7F + (1 << (8 * byte)))
                    c1 = run(mod, f2, 2, dev)
                    ratio = c1 / c0
                    changed = (c1 != c0)
                    if not changed.any():
                        continue
                    doubled = (ratio == 2.0) & changed
                    if (changed & ~doubled).any():
                        partial += 1
                    if side == "a":
                        rows = tuple(r for r in range(16) if doubled[r].all())
                        if len(rows) != int(changed.any(dim=1).sum()):
                            partial += 1
                        hits["a"][(lane, byte, h)] = rows
                    else:
                        cols = tuple(n for n in range(8) if doubled[:, n].all())
                        if len(cols) != int(changed.any(dim=0).sum()):
                            partial += 1
                        hits["b"][(lane, byte, h)] = cols
    for side, n_idx in (("a", 16), ("b", 8)):
        reach = {}
        for (lane, byte, h), idx in hits[side].items():
            for i in idx:
                reach.setdefault((i, h), []).append((lane, byte))
        complete = all(len(reach.get((i, h), [])) == 1 for i in range(n_idx) for h in (0, 1))
        print(f"  arm 3 scale-{side}: {len(hits[side])} (lane, byte, block) settings change C; every ({'row' if side == 'a' else 'col'}, block) reached by exactly one: {complete}")
        for (i, h), who in sorted(reach.items())[:6]:
            print(f"    {'row' if side == 'a' else 'col'} {i} block {h} <- lane, byte {who}")
        if not complete:
            partial += 1
        hits[side + "_reach"] = reach
    print(f"  arm 3 partial or ambiguous changes: {partial}")
    arm3.hits = hits
    return partial == 0


def main() -> int:
    dev = torch.device("cuda")
    mod = build()
    torch.manual_seed(0)
    ok1 = arm1(mod, dev)
    ok2 = arm2(mod, dev)
    ok3 = arm3(mod, dev)
    if "--out" in sys.argv:
        import json
        h = arm3.hits
        rep = {"device": torch.cuda.get_device_name(), "arm1_container": "e2m1 code in bits 5:2 of its byte on both operands" if ok1 else None,
               "arm2_packing_exact": ok2, "arm3_exact_mapping": ok3,
               "scale_a": {f"row{i}_block{b}": [list(x) for x in who] for (i, b), who in sorted(h["a_reach"].items())},
               "scale_b": {f"col{i}_block{b}": [list(x) for x in who] for (i, b), who in sorted(h["b_reach"].items())},
               "selectors": "byte-id 0 and thread-id 0 for both operands"}
        out = sys.argv[sys.argv.index("--out") + 1]
        open(out, "w").write(json.dumps(rep, indent=1) + "\n")
    print("fp4 x fp4 probe:", "ok" if ok1 and ok2 and ok3 else "incomplete")
    return 0 if ok1 and ok2 and ok3 else 1


if __name__ == "__main__":
    sys.exit(main())
