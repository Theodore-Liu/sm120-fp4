"""One-hot probes of `mma.sync.m16n8k32.kind::f8f6f4.f32.e4m3.e2m1.f32` on SM120 (uses the probe kernel of probe_f8f6f4.py).

Experiment 1 (container and A-row mapping): B registers filled with one byte value `bb` in every container, A zero
except one byte (e4m3 1.0) at lane L, register r, byte j. The output tells (i) which C row/col lights up and (ii) the
value of a single product 1.0 * decode(bb) summed once, which identifies how the e2m1 container is read.
Experiment 2 (k mapping): A lane 0 register 0 byte 0 = 1.0 (one (row, k) position), B one byte at lane L, register r,
byte j = container of 1.0; the (L, r, j) that produce a nonzero output share A's k and give the B col mapping.

    PYTHONPATH=. python scripts/probe_f8f6f4_onehot.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("probe_f8f6f4", _here / "probe_f8f6f4.py")
pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf)


def run(mod, frags: torch.Tensor):
    out = torch.zeros(32, 4, device="cuda")
    mod.probe(frags.to("cuda"), out)
    torch.cuda.synchronize()
    return out.cpu()


def to_i32(v: int) -> int:
    return v - (1 << 32) if v >= 1 << 31 else v


def main() -> int:
    mod = pf.build()
    one8 = int(torch.tensor(1.0).to(torch.float8_e4m3fn).view(torch.uint8))  # 0x38
    print("e4m3 byte for 1.0:", hex(one8))
    # Experiment 1: which container byte value reads as e2m1 1.0? Try candidates in every B container, A = 1.0 at lane 0 reg 0 byte 0.
    for bb in (0x02, 0x20, 0x08, 0x80, 0x40, 0x04, 0x10, 0x01, 0x3C, 0x30):
        frags = torch.zeros(32, 6, dtype=torch.int32)
        frags[0, 0] = one8
        bword = to_i32(bb | (bb << 8) | (bb << 16) | (bb << 24))
        frags[:, 4] = bword
        frags[:, 5] = bword
        out = run(mod, frags)
        nz = (out != 0).nonzero().tolist()
        vals = sorted(set(round(float(v), 4) for v in out[out != 0]))
        print(f"B byte {bb:#04x}: nonzero at (lane, c) {nz[:6]}{'...' if len(nz) > 6 else ''}, values {vals[:6]}")
    # Experiment 2: A row mapping: A = 1.0 at (lane L, reg r, byte 0), B all = 0x02 (candidate for 1.0 in the low nibble)
    print("--- A placement -> C position (B all containers 0x02)")
    for L, r in ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (4, 0), (5, 2)):
        frags = torch.zeros(32, 6, dtype=torch.int32)
        frags[L, r] = one8
        frags[:, 4] = to_i32(0x02020202)
        frags[:, 5] = to_i32(0x02020202)
        out = run(mod, frags)
        nz = (out != 0).nonzero().tolist()
        print(f"A lane {L} reg {r} byte 0 -> nonzero (lane, c) {nz}, values {sorted(set(round(float(v), 4) for v in out[out != 0]))}")
    # Experiment 3: k pairing: A = 1.0 at lane 0 reg 0 byte j; B = 0x02 at lane L reg r byte jb only
    print("--- k pairing (A lane 0 reg 0 byte j vs B lane L reg r byte jb)")
    for j in (0, 1):
        for (L, r, jb) in ((0, 4, 0), (0, 4, 1), (0, 5, 0), (1, 4, 0), (4, 4, 0)):
            frags = torch.zeros(32, 6, dtype=torch.int32)
            frags[0, 0] = one8 << (8 * j)
            frags[L, r] = to_i32(0x02 << (8 * jb))
            out = run(mod, frags)
            nz = (out != 0).nonzero().tolist()
            print(f"A byte {j} x B lane {L} reg {r - 4} byte {jb} -> {nz} {sorted(set(round(float(v), 4) for v in out[out != 0]))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
