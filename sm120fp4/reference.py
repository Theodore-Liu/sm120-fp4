"""Reference NVFP4 quantizer, dequantizer and GEMM, in plain PyTorch (fp32 arithmetic, CPU or GPU).

NVFP4 as the Blackwell tensor cores and the shipped checkpoints define it:

* elements: FP4 E2M1, the grid {0, 0.5, 1, 1.5, 2, 3, 4, 6} with a sign bit; encoding (sign<<3)|(exp<<1)|mantissa,
  two elements per byte, the even element in the low nibble (TensorRT-LLM / FlashInfer packing; checked in tests);
* block scale: one FP8 E4M3 (unsigned use, "UE4M3") per 16 consecutive elements along K;
* per-tensor scale: one fp32. TensorRT-LLM/FlashInfer take it as `global_scale = (448 * 6) / amax(|x|)` and store the
  block scale as `sf = e4m3(block_amax / 6 * global_scale)`; the element is quantised as `q = e2m1(x * global_scale / sf)`
  and dequantised as `x ~ q * sf / global_scale`. Checkpoints (ModelOpt, compressed-tensors, ComfyUI) store the same
  two tensors under names like `weight_scale` (E4M3) and `weight_scale_2` / `weight_global_scale` (fp32, the inverse
  of `global_scale` in some conventions - the tests pin which).

Rounding: E2M1 conversion is round-to-nearest with ties to the even encoding, which is what `cvt.rn.satfinite.e2m1x2`
does; values beyond 6 saturate to 6. E4M3 conversion uses PyTorch's float8_e4m3fn cast (round to nearest even,
saturating is applied explicitly at 448).
"""
from __future__ import annotations

import torch

E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E4M3_MAX = 448.0
E2M1_MAX = 6.0
BLOCK = 16


def e2m1_encode(x: torch.Tensor) -> torch.Tensor:
    """fp32 values -> E2M1 codes (uint8 0..15), round to nearest, ties to the even code, saturating at 6."""
    grid = E2M1_GRID.to(x.device)
    mag = x.abs().clamp(max=E2M1_MAX)
    # nearest grid point; ties resolved to the even code (0, 2, 4, 6 -> values 0, 1, 2, 4)
    lo = torch.searchsorted(grid, mag.reshape(-1), right=True).clamp(1, 7).view(mag.shape) - 1  # grid[lo] <= mag
    hi = (lo + 1).clamp(max=7)
    dlo = (mag - grid[lo]).abs()
    dhi = (grid[hi] - mag).abs()
    pick_hi = dhi < dlo
    tie = dhi == dlo
    even_hi = (hi % 2) == 0
    code = torch.where(pick_hi | (tie & even_hi), hi, lo)
    sign = (x < 0) & (code != 0)
    return (code | (sign.to(code.dtype) << 3)).to(torch.uint8)


def e2m1_decode(codes: torch.Tensor) -> torch.Tensor:
    grid = E2M1_GRID.to(codes.device)
    mag = grid[(codes & 0x7).long()]
    return torch.where((codes & 0x8) != 0, -mag, mag)


def pack_e2m1(codes: torch.Tensor) -> torch.Tensor:
    """[..., K] codes -> [..., K/2] bytes, even element in the low nibble."""
    if codes.shape[-1] % 2:
        raise ValueError("K must be even")
    lo = codes[..., 0::2].to(torch.uint8)
    hi = codes[..., 1::2].to(torch.uint8)
    return (lo | (hi << 4)).to(torch.uint8)


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    return torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def global_scale_for(x: torch.Tensor) -> torch.Tensor:
    """TensorRT-LLM's convention: global_scale = 448 * 6 / amax(|x|) (fp32, shape [1])."""
    amax = x.abs().max().float().clamp(min=1e-12)
    return (E4M3_MAX * E2M1_MAX / amax).reshape(1)


def quantize_nvfp4(x: torch.Tensor, global_scale: torch.Tensor | None = None, block: int = BLOCK):
    """x [M, K] (any float dtype) -> (packed uint8 [M, K/2], sf uint8 [M, K/block] as E4M3 bits, row-major, global_scale fp32 [1])."""
    if x.dim() != 2 or x.shape[1] % block:
        raise ValueError(f"expected [M, K] with K a multiple of {block}, got {tuple(x.shape)}")
    xf = x.float()
    if global_scale is None:
        global_scale = global_scale_for(xf)
    gs = global_scale.float().reshape(())
    m, k = xf.shape
    blocks = xf.view(m, k // block, block)
    block_amax = blocks.abs().amax(dim=-1)                                  # [M, K/16]
    sf_f = (block_amax / E2M1_MAX * gs).clamp(max=E4M3_MAX)
    sf_e4m3 = sf_f.to(torch.float8_e4m3fn)
    sf_val = sf_e4m3.float()
    scale = torch.where(sf_val > 0, gs / sf_val, torch.zeros_like(sf_val))   # per element multiplier
    q = e2m1_encode(blocks * scale[..., None]).view(m, k)
    return pack_e2m1(q), sf_e4m3.view(torch.uint8), global_scale.float().reshape(1)


def dequantize_nvfp4(packed: torch.Tensor, sf_e4m3_bits: torch.Tensor, global_scale: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """Inverse of quantize_nvfp4 (sf row-major [M, K/block] as uint8 E4M3 bits) -> fp32 [M, K]."""
    codes = unpack_e2m1(packed)
    m, k = codes.shape
    vals = e2m1_decode(codes).view(m, k // block, block)
    sf = sf_e4m3_bits.view(torch.float8_e4m3fn).float().view(m, k // block, 1)
    gs = global_scale.float().reshape(())
    return (vals * sf / gs).view(m, k)


def reference_gemm_nvfp4(a_packed, a_sf, a_gs, b_packed, b_sf, b_gs, out_dtype=torch.float32) -> torch.Tensor:
    """C[M, N] = dequant(A)[M, K] @ dequant(B)[N, K]^T in fp32, the reference every FP4 GEMM is compared with."""
    a = dequantize_nvfp4(a_packed, a_sf, a_gs)
    b = dequantize_nvfp4(b_packed, b_sf, b_gs)
    return (a @ b.t()).to(out_dtype)
