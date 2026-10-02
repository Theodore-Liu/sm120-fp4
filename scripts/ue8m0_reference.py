"""Stage 3, step 2: the UE8M0 (MX) block-scale path as a reference that mirrors DeepGEMM's own, with a self-test.

DeepGEMM's SM100 FP8xFP4 GEMM (`sm100_fp8_fp4_gemm_1d1d`) takes each operand as codes plus scaling factors in "packed
UE8M0": four 8-bit exponents per int32, one scale per `gran_k` elements along K (the default recipe on SM100 is
`(1, 1, 128)`, so `gran_k` 128; 32 is also accepted), applied inside the `kind::mxf8f6f4` MMA. SM120's `mma.sync` has
no scale operand, so an SM120 kernel applies them outside the MMA; this file fixes, before any kernel exists, exactly what
the codes and scales mean, by mirroring `deep_gemm/utils/math.py` at commit 057ca5964aae (2026-09-30):

- scale of a block = `ceil_to_ue8m0(max|x|.clamp_min(1e-4) / fmax)` with fmax 448 (e4m3) or 6 (e2m1): the smallest
  power of two that brings the block's largest magnitude inside the format; a zero block gets 2^-15 (the clamp), not 1;
- e2m1 codes are nibbles `sign << 3 | index` into {0, 0.5, 1, 1.5, 2, 3, 4, 6}, rounded to nearest with ties to the even
  index (0.25 down, 0.75 up, 1.25 down, 1.75 up, 2.5 down, 3.5 up, 5.0 down), two per byte with the even element in the
  low nibble; negative zero is +0;
- e4m3 codes are torch's `float8_e4m3fn` cast of the scaled value;
- packed UE8M0 = exponent bytes of the fp32 scales, four per int32, element 0 in the low byte (`pack_ue8m0_to_int`);
- dequantize = code value times the block's scale; the GEMM reference is the fp32 matmul of the two dequantized operands.

The self-test has two layers. The synthetic layer needs only torch: known-answer blocks (powers of two that round-trip
exactly, the tie points of the e2m1 grid, a zero block, an overflow step), packing round trips, and the GEMM against a
plain fp32 matmul. The conformance layer runs when a DeepGEMM checkout is present (`DEEPGEMM=/path` or
`~/oss/DeepGEMM`): it loads `deep_gemm/utils/math.py` by file path and checks, bit for bit, that this file's
`per_token_cast_to_fp4` / `per_token_cast_to_fp8` / `cast_back_from_fp4` / packing agree with DeepGEMM's on random
inputs at gran_k 32 and 128, with and without UE8M0 rounding.

    python scripts/ue8m0_reference.py --selftest
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import torch

E4M3_MAX = 448.0
E2M1_MAX = 6.0
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_BOUNDARIES = ((0.25, False), (0.75, True), (1.25, False), (1.75, True), (2.5, False), (3.5, True), (5.0, False))
UE8M0_BIAS = 127
AMAX_FLOOR = 1e-4


def align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    """Round each positive fp32 up to the next power of two (a power of two stays); exponent clamped to 1..254."""
    bits = x.abs().float().view(torch.int)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float)


def pack_ue8m0_to_int(sf: torch.Tensor) -> torch.Tensor:
    """fp32 power-of-two scales [..., 4j] -> int32 [..., j], element 0 in the low byte."""
    assert sf.dtype == torch.float and sf.size(-1) % 4 == 0
    x_int = sf.contiguous().view(torch.int)
    assert (x_int >= 0).all() and (x_int & 0x7FFFFF == 0).all(), "scales must be positive powers of two"
    return (x_int >> 23).to(torch.uint8).view(torch.int)


def unpack_ue8m0_from_int(packed: torch.Tensor) -> torch.Tensor:
    return (packed.view(torch.uint8).to(torch.int) << 23).view(torch.float)


def quantize_to_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> e2m1 nibbles (int8 holding 0..15): sign << 3 | index, ties to the even index; -0 is +0."""
    ax = x.abs()
    code = torch.zeros_like(x, dtype=torch.uint8)
    for boundary, round_up in E2M1_BOUNDARIES:
        code += (ax >= boundary if round_up else ax > boundary).to(torch.uint8)
    sign = (x < 0) & (code != 0)
    return (code | (sign.to(torch.uint8) << 3)).view(torch.int8)


def dequantize_from_fp4_e2m1(nib: torch.Tensor) -> torch.Tensor:
    vals = torch.tensor(E2M1_VALUES, device=nib.device, dtype=torch.float)
    sign, idx = (nib & 0x08) != 0, (nib & 0x07).to(torch.int)
    v = vals[idx]
    return torch.where(sign & (idx != 0), -v, v)


def per_token_cast_to_fp4(x: torch.Tensor, use_ue8m0: bool, gran_k: int = 128, use_packed_ue8m0: bool = False):
    """x [m, n] -> (packed e2m1 codes int8 [m, n // 2], scales fp32 [m, ceil(n / gran_k)] or packed int32)."""
    m, n = x.shape
    assert n % 2 == 0 and (not use_packed_ue8m0 or use_ue8m0)
    padded_n = align(n, gran_k)
    xp = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
    xp[:, :n] = x
    xv = xp.view(m, -1, gran_k)
    sf = xv.abs().float().amax(dim=2).clamp_min(AMAX_FLOOR) / E2M1_MAX
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    codes = quantize_to_fp4_e2m1(xv * (1.0 / sf.unsqueeze(2))).view(m, padded_n)
    c2 = codes.view(m, padded_n // 2, 2)
    packed = (c2[:, :, 0] & 0x0F) | ((c2[:, :, 1] & 0x0F) << 4)
    if use_packed_ue8m0:
        num_sf = sf.size(-1)
        if num_sf % 4 != 0:
            sf = torch.nn.functional.pad(sf, (0, align(num_sf, 4) - num_sf), value=1.0)
        return packed[:, :n // 2].contiguous(), pack_ue8m0_to_int(sf)
    return packed[:, :n // 2].contiguous(), sf


def per_token_cast_to_fp8(x: torch.Tensor, use_ue8m0: bool, gran_k: int = 128, use_packed_ue8m0: bool = False):
    m, n = x.shape
    assert not use_packed_ue8m0 or use_ue8m0
    padded_n = align(n, gran_k)
    xp = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
    xp[:, :n] = x
    xv = xp.view(m, padded_n // gran_k, gran_k)
    sf = xv.abs().float().amax(dim=2).clamp(AMAX_FLOOR) / E4M3_MAX
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x8 = (xv * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, padded_n)[:, :n].contiguous()
    if use_packed_ue8m0:
        num_sf = sf.size(-1)
        if num_sf % 4 != 0:
            sf = torch.nn.functional.pad(sf, (0, align(num_sf, 4) - num_sf), value=1.0)
        return x8, pack_ue8m0_to_int(sf)
    return x8, sf


def cast_back_from_fp4(packed: torch.Tensor, sf: torch.Tensor, gran_k: int = 128, use_packed_ue8m0: bool = False):
    m, n2 = packed.shape
    n = n2 * 2
    if use_packed_ue8m0:
        sf = unpack_ue8m0_from_int(sf)
    nib = torch.zeros((m, n), dtype=torch.int8, device=packed.device)
    nib[:, ::2] = packed & 0x0F
    nib[:, 1::2] = (packed >> 4) & 0x0F
    group = torch.arange(n, device=packed.device) // gran_k
    return dequantize_from_fp4_e2m1(nib) * sf[:, group]


def cast_back_from_fp8(x8: torch.Tensor, sf: torch.Tensor, gran_k: int = 128, use_packed_ue8m0: bool = False):
    m, n = x8.shape
    if use_packed_ue8m0:
        sf = unpack_ue8m0_from_int(sf)
    group = torch.arange(n, device=x8.device) // gran_k
    return x8.float() * sf[:, group]


def mx_gemm_reference(a8, a_sf, b4, b_sf, gran_k: int) -> torch.Tensor:
    """D[m, n] = sum_k A[m, k] B[n, k], A in e4m3 and B in packed e2m1, each dequantized block by block; fp32 accumulation."""
    return cast_back_from_fp8(a8, a_sf, gran_k) @ cast_back_from_fp4(b4, b_sf, gran_k).T


def _deepgemm_math():
    root = os.environ.get("DEEPGEMM") or os.path.expanduser("~/oss/DeepGEMM")
    f = Path(root) / "deep_gemm/utils/math.py"
    if not f.is_file():
        return None
    spec = importlib.util.spec_from_file_location("deep_gemm_utils_math", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def selftest() -> int:
    torch.manual_seed(0)
    # 1. ceil_to_ue8m0: powers of two stay, anything above rounds to the next, over the whole exponent range
    p = torch.pow(2.0, torch.arange(-120, 120, dtype=torch.float))
    assert torch.equal(ceil_to_ue8m0(p), p)
    assert torch.equal(ceil_to_ue8m0(p * 1.0001), p * 2)
    # 2. packing round trip and byte order
    sf = torch.pow(2.0, torch.tensor([[0.0, 1.0, -3.0, -20.0]]))
    w = pack_ue8m0_to_int(sf)
    assert int(w[0, 0]) == 127 | (128 << 8) | (124 << 16) | (107 << 24)  # exponents 127, 128, 124, 107; top byte below 128 keeps the int32 positive
    assert torch.equal(unpack_ue8m0_from_int(w), sf)
    # 3. e2m1 rounding at the grid points and the tie points, and the sign nibble
    y = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0, -2.5, -0.1, 0.5, 6.0])
    q = dequantize_from_fp4_e2m1(quantize_to_fp4_e2m1(y))
    assert torch.equal(q, torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, -2.0, 0.0, 0.5, 6.0])), q
    # 4. known-answer blocks: max 48 -> scale 8 (exponent 130); a block of ones -> scale 0.25 (6 * 0.25 >= 1); zeros -> 2^-15
    x = torch.zeros(1, 96)
    x[0, :4] = torch.tensor([48.0, -48.0, 8.0, 16.0])
    x[0, 32:64] = 1.0
    codes, s = per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=32)
    assert torch.equal(s[0], torch.tensor([8.0, 0.25, 2.0 ** -15])), s
    back = cast_back_from_fp4(codes, s, 32)
    assert torch.equal(back[0, :4], x[0, :4]) and torch.all(back[0, 32:64] == 1.0) and torch.all(back[0, 64:] == 0)
    assert torch.isfinite(back).all()
    # 5. without UE8M0 the scale is amax / 6 exactly
    _, s_raw = per_token_cast_to_fp4(x, use_ue8m0=False, gran_k=32)
    assert torch.equal(s_raw[0, :2], torch.tensor([8.0, 1.0 / 6.0]))
    # 6. e4m3 side: a block whose max is 448 * 4 gets scale 4 and round-trips
    z = torch.zeros(1, 128)
    z[0, :3] = torch.tensor([1792.0, -4.0, 224.0])
    c8, s8 = per_token_cast_to_fp8(z, use_ue8m0=True, gran_k=128)
    assert float(s8[0, 0]) == 4.0 and torch.equal(cast_back_from_fp8(c8, s8, 128)[0, :3], z[0, :3])
    # 7. the GEMM reference equals the matmul of the dequantized operands and is in range of the fp32 answer
    for gran_k in (32, 128):
        m, n, k = 16, 48, 256
        a = torch.randn(m, k) * 3
        b = torch.randn(n, k)
        a8, asf = per_token_cast_to_fp8(a, True, gran_k)
        b4, bsf = per_token_cast_to_fp4(b, True, gran_k)
        d = mx_gemm_reference(a8, asf, b4, bsf, gran_k)
        ref = cast_back_from_fp8(a8, asf, gran_k) @ cast_back_from_fp4(b4, bsf, gran_k).T
        assert torch.isfinite(d).all() and torch.equal(d, ref)
        rel = float((d - a @ b.T).norm() / (a @ b.T).norm())
        assert rel < 0.25, rel
    # 8. conformance with DeepGEMM's own helpers, bit for bit, when a checkout is present
    dg = _deepgemm_math()
    if dg is None:
        print("ue8m0_reference selftest: synthetic layer ok; DeepGEMM checkout not found, conformance layer skipped")
        return 0
    checked = 0
    for gran_k in (32, 128):
        for use_ue8m0 in (False, True):
            for packed in ((False, True) if use_ue8m0 else (False,)):
                x = torch.randn(8, 512) * torch.pow(2.0, torch.randint(-8, 12, (8, 1)).float())
                x[2, :gran_k] = 0.0
                ours = per_token_cast_to_fp4(x, use_ue8m0, gran_k, packed)
                theirs = dg.per_token_cast_to_fp4(x, use_ue8m0=use_ue8m0, gran_k=gran_k, use_packed_ue8m0=packed)
                assert torch.equal(ours[0], theirs[0]) and torch.equal(ours[1], theirs[1]), (gran_k, use_ue8m0, packed, "fp4")
                assert torch.equal(cast_back_from_fp4(*ours, gran_k, packed), dg.cast_back_from_fp4(*theirs, gran_k=gran_k, use_packed_ue8m0=packed))
                ours8 = per_token_cast_to_fp8(x, use_ue8m0, gran_k, packed)
                theirs8 = dg.per_token_cast_to_fp8(x, use_ue8m0=use_ue8m0, gran_k=gran_k, use_packed_ue8m0=packed)
                assert torch.equal(ours8[0].view(torch.uint8), theirs8[0].view(torch.uint8)) and torch.equal(ours8[1], theirs8[1]), (gran_k, use_ue8m0, packed, "fp8")
                checked += 1
    print(f"ue8m0_reference selftest: synthetic layer ok; conformance with DeepGEMM's math.py bit for bit on {checked} configurations")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    ap.error("nothing to do; pass --selftest")
    return 2


if __name__ == "__main__":
    sys.exit(main())
