#!/usr/bin/env python3
"""The container convention of every `kind::f8f6f4` operand format on SM120, measured one-hot (the R4 letter's table).

The one-hot probe of 2026-10-02 (probe_f8f6f4_onehot.py) measured one cell: with A = e4m3 and B = e2m1, the 5090 reads
an e2m1 B container as a 6-bit field, code in bits 5:2 of the byte. This script measures the whole table: for each of
the formats e4m3, e3m2, e2m1 on each operand side, which bit position of the byte holds the value's code.

Method. One MMA `mma.sync.m16n8k32.kind::f8f6f4.f32.<A>.<B>.f32` per warp with fragments handed in from Python
(32 x 6 uint32: a0..a3, b0, b1 per lane). To read side X's convention, the other side holds the value 1.0 in a format
whose byte is known (e4m3: 0x38 in bit 7:0, measured and documented), placed at lane 0 register 0 byte 0; side X holds
one candidate byte in every container. The candidate is the format's code for 1.0 shifted by s bits (s = 0..4); a
single product 1.0 x decode(candidate) appears once in the C fragment. The shift that returns exactly 1.0 is the
container convention; shifts that return other values tell what the hardware decodes there. e2m1 on both sides is
resolved by first fixing B through an e4m3 A, then A through an e4m3 B, then checking the pair.

Codes for 1.0: e4m3 (bias 7) 0_0111_000 = 0x38; e3m2 (bias 3) 0_011_00 = 0x0C; e2m1 (bias 1) 0_01_0 = 0x02.

    PYTHONPATH=. python scripts/probe_f8f6f4_formats.py [--out reports/probe-f8f6f4-formats-<gpu>-<date>.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

ONE = {"e4m3": 0x38, "e3m2": 0x0C, "e2m1": 0x02}
WIDTH = {"e4m3": 8, "e3m2": 6, "e2m1": 4}
CPP = "#include <torch/extension.h>" + chr(10) + "void probe(torch::Tensor frags, torch::Tensor out);" + chr(10)


def cuda_source(atype: str, btype: str) -> str:
    instr = f"mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.{atype}.{btype}.f32 "
    lines = [
        "#include <torch/extension.h>",
        "#include <ATen/cuda/CUDAContext.h>",
        "#include <cstdint>",
        "__global__ void k_probe(const uint32_t* __restrict__ f, float* __restrict__ out) {",
        "  const int lane = threadIdx.x;",
        "  uint32_t a[4] = {f[lane * 6 + 0], f[lane * 6 + 1], f[lane * 6 + 2], f[lane * 6 + 3]};",
        "  uint32_t b[2] = {f[lane * 6 + 4], f[lane * 6 + 5]};",
        "  float c[4] = {0.f, 0.f, 0.f, 0.f};",
        '  asm volatile("' + instr + '"',
        '               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"',
        '               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])',
        '               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));',
        "  for (int i = 0; i < 4; ++i) out[lane * 4 + i] = c[i];",
        "}",
        "void probe(torch::Tensor frags, torch::Tensor out) {",
        "  k_probe<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(",
        "      reinterpret_cast<const uint32_t*>(frags.data_ptr<int32_t>()), out.data_ptr<float>());",
        "}",
    ]
    return chr(10).join(lines) + chr(10)


def build(atype: str, btype: str):
    return load_inline(name=f"sm120fp4_probe_f8f6f4_{atype}_{btype}", cpp_sources=CPP, cuda_sources=cuda_source(atype, btype),
                       functions=["probe"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a"], verbose=False)


def to_i32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= 1 << 31 else v


def run(mod, frags: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(32, 4, device="cuda")
    mod.probe(frags.to("cuda"), out)
    torch.cuda.synchronize()
    return out.cpu()


def one_hot(mod, side: str, candidate: int, known_byte: int) -> dict:
    """side 'B': A holds known_byte at lane 0 reg 0 byte 0, every B container = candidate. side 'A': the mirror."""
    frags = torch.zeros(32, 6, dtype=torch.int32)
    word = to_i32(candidate | (candidate << 8) | (candidate << 16) | (candidate << 24))
    if side == "B":
        frags[0, 0] = known_byte
        frags[:, 4] = word
        frags[:, 5] = word
    else:
        frags[0, 4] = known_byte
        frags[:, 0] = word
        frags[:, 1] = word
        frags[:, 2] = word
        frags[:, 3] = word
    out = run(mod, frags)
    nz = (out != 0).nonzero().tolist()
    vals = sorted(set(float(v) for v in out[out != 0]))
    return {"candidate": f"{candidate:#04x}", "nonzero_positions": len(nz), "values": vals[:8]}


def measure_side(mod, side: str, fmt: str, known_byte: int) -> dict:
    """Try the format's 1.0 code at every shift that fits in the byte; report the shift that decodes to exactly 1.0."""
    trials = []
    hit = None
    for s in range(0, 9 - WIDTH[fmt]):
        cand = ONE[fmt] << s
        r = one_hot(mod, side, cand, known_byte)
        r["shift"] = s
        trials.append(r)
        if r["values"] == [1.0]:
            hit = s
    # the all-ones control: every bit set must not decode to 1.0 in any convention (sign bit set) -> negative or NaN
    ctrl = one_hot(mod, side, 0xFF, known_byte)
    return {"side": side, "format": fmt, "code_for_one": f"{ONE[fmt]:#04x}", "shift_that_reads_one": hit,
            "code_bits_in_byte": (f"{hit + WIDTH[fmt] - 1}:{hit}" if hit is not None else None), "trials": trials,
            "all_ones_control": ctrl}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": name, "capability": list(cap),
              "torch": torch.__version__, "cuda": torch.version.cuda, "instruction": "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.<A>.<B>.f32",
              "known_reference": "e4m3 1.0 = 0x38 in bits 7:0 (the e4m3 byte is the format's own byte)", "sides": [], "pairs": []}
    e4 = ONE["e4m3"]
    # Step 0: the control cell, A e4m3 x B e4m3, one-hot both ways: the known byte must read 1.0 on both sides.
    mod = build("e4m3", "e4m3")
    for side in ("B", "A"):
        report["sides"].append(measure_side(mod, side, "e4m3", e4))
    # Step 1: B conventions through an e4m3 A.
    for fmt in ("e3m2", "e2m1"):
        mod = build("e4m3", fmt)
        report["sides"].append(measure_side(mod, "B", fmt, e4))
    # Step 2: A conventions through an e4m3 B.
    for fmt in ("e3m2", "e2m1"):
        mod = build(fmt, "e4m3")
        report["sides"].append(measure_side(mod, "A", fmt, e4))
    # Step 3: the sub-byte pairs, with each side's measured byte for 1.0, one product -> must be exactly 1.0.
    conv = {(r["side"], r["format"]): r["shift_that_reads_one"] for r in report["sides"]}
    for at, bt in (("e3m2", "e3m2"), ("e2m1", "e2m1"), ("e3m2", "e2m1"), ("e2m1", "e3m2")):
        sa, sb = conv.get(("A", at)), conv.get(("B", bt))
        if sa is None or sb is None:
            report["pairs"].append({"A": at, "B": bt, "skipped": "a side had no shift that read 1.0"})
            continue
        mod = build(at, bt)
        frags = torch.zeros(32, 6, dtype=torch.int32)
        frags[0, 0] = ONE[at] << sa
        frags[0, 4] = ONE[bt] << sb
        out = run(mod, frags)
        nz = (out != 0).nonzero().tolist()
        report["pairs"].append({"A": at, "B": bt, "a_byte": f"{ONE[at] << sa:#04x}", "b_byte": f"{ONE[bt] << sb:#04x}",
                                "nonzero_positions": nz, "values": sorted(set(float(v) for v in out[out != 0])), "exact_one": nz == [[0, 0]] and float(out[0, 0]) == 1.0})
    for r in report["sides"]:
        print(f"{r['side']} {r['format']}: code {r['code_for_one']} reads 1.0 at shift {r['shift_that_reads_one']} -> code bits {r['code_bits_in_byte']}; "
              + "; ".join(f"s{t['shift']}={t['candidate']}->{t['values']}" for t in r["trials"]) + f"; 0xff->{r['all_ones_control']['values']}")
    for p in report["pairs"]:
        print("pair", p)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
