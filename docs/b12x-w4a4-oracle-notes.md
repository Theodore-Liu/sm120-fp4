# Reading notes for a W4A4 fused-MoE oracle (b12x static path, FlashInfer 0.7.1, read 2026-10-11)

The W4A16 oracle (`sm120fp4/moe_oracle.py`) follows the dispatched kernel's rounding points. For the W4A4 (NVFP4 activations)
path the points differ, and the activation quantiser adds set-valued steps. Read from `fused_moe/cute_dsl/blackwell_sm12x/`:

- **FC1 and activation** (`moe_static_kernel.py` around lines 2242 to 2258): the fp32 FC1 accumulator goes through
  `gated_activation_f32` (SiLU: `exp(-g, fastmath=False)` then `rcp_approx` of `1 + exp`, times up) in fp32, and the activated
  values are written to shared memory as bf16 (`load_shared_bf16x8_to_f32x8` reads them back).
- **Block quantiser** (`quantize_block_fp4` / `_fast` in `moe_w4a16_fp4_helpers.py`, called from the static kernel's
  "quantize one retained FC1 slice" step): `block_max` over the 16 bf16 values; scale = `cvt.rn.satfinite.e4m3x2.f32` of
  `min(max * gs / 6, 448)` (fast path: `gs * (max * rcp_approx(6))`); codes = `cvt.rn` E2M1 of `value * gs / qs`
  (fast path: `value * rcp_approx(qs) * gs`, `cvt_e2m1x8_f32`); the plain path's threshold ladder rounds ties to the even code
  (0.25 to 0, 0.75 to 2, 1.25 to 2, 1.75 to 4, 2.5 to 4, 3.5 to 6, 5 to 6). The global scale is a multiplier here; when
  `input_scales_are_reciprocal` it is first inverted with `rcp_approx_ftz` (fast path) or an exact division.
  `fast_math` defaults to True in `moe_dispatch.py` (line about 900) and is forced off for non-gated activations.
- **FC2 and the route weight** (`moe_static_kernel.py` lines 150 to 180): each route's FC2 result is held as bf16 pairs;
  `combined_scale = route_weight * down_alpha` in fp32 is rounded to bf16 (`cvt.rn.bf16.f32`) and the multiply is done in
  **bf16x2 arithmetic** (`mul.rn.bf16x2`), then stored to a private per-route buffer ("without a global REDG").
- **Combine** (lines 190 to 262): a finalize loads the route-major bf16x8 contributions as fp32, sums them in fp32 and rounds
  once (`store_global_f32x8_as_bf16`). So the 0.7.1 static path is deterministic by construction; the non-determinism the
  census found at M = 1 and 4 belongs to the micro (decode) backend, which must be read separately
  (`moe_micro_kernel.py`, `moe_direct_micro_kernel.py`).

## Oracle design that follows

1. FC1 interval as in W4A16 (fp32 accumulation bound), SiLU hull in fp32 with the fast-reciprocal slack, one bf16 rounding
   (set-valued at straddles).
2. Block scale: the interval of `max * gs / 6` over the block's bf16 candidates, widened by the reciprocal slack (about 2^-22
   relative), rounded to E4M3 at both ends. One value almost always (E4M3 has 3 mantissa bits, so a straddle is rare); if two,
   mark the block ambiguous and widen that block's contribution bound by its whole magnitude (sound, coarse, counted).
3. Codes: each element's `value * gs / qs` interval rounded to E2M1 at both ends (ties to even); one or two codes; h interval =
   codes times qs / gs (exact in float64).
4. FC2: exact products of the h interval midpoint with the dequantised W2, half-width through |W2|, fp32 accumulation bound,
   then one bf16 rounding of the contribution (set-valued at straddles).
5. Route weight: `bf16(w * alpha2)` exactly as the kernel computes it; the multiply in bf16: the product of two bf16 values
   is exact in fp32 and then rounded to bf16 (`mul.rn.bf16x2`), so one more bf16 rounding of the contribution.
6. Combine: fp32 sum over the top-k, one bf16 output rounding.

The micro backend (small M) needs its own reading before any verdict on it; the known non-determinism there means the oracle
should report per-run conformance and distinct-output counts side by side, as the W4A16 probe does.

## As measured (2026-10-11, after the oracle was fitted to the kernel)

The design above was corrected in three places by fitting against the kernel (`docs/flashinfer-b12x-w4a4-moe-rtx5090.md`):
the static backend's FC2 is computed in K-chunks of 256 with each partial rounded to bf16 and weighted per partial, not as one
fp32 accumulation; the micro backend's FC2 is one fp32 accumulation with the route result held as bf16 and the weight applied
in fp32; and the block quantisers are emulated exactly rather than as set-valued with a slack, since `rcp.approx` equals the
fp32 RN reciprocal on this device. `w1_alpha` is the input-quant global scale in this mode, not an FC1 multiplier.
