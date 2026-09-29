# Scale-factor layouts for block-scaled FP4 on Blackwell

Every number and formula here is quoted from the source that owns it, and every equivalence claimed between two sources
is backed by a test in `tests/` that runs both against each other on an SM120 GPU. Where a claim is not yet tested it
says so.

## 1. The formats

| format | element | block | block scale | per-tensor scale |
|---|---|---|---|---|
| NVFP4 | FP4 E2M1: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}, code `(sign<<3)|(exp<<1)|mantissa` | 16 along K | FP8 E4M3, used unsigned ("UE4M3"), max 448 | fp32, one per tensor |
| MXFP4 | same E2M1 grid | 32 along K | E8M0 (a power of two), no per-tensor scale | none |

Two elements pack into one byte; the even element takes the low nibble (TensorRT-LLM/FlashInfer `fp4_quantize`,
ComfyUI `comfy-quants` "HIGH=even index, LOW=odd" describes the same byte from the other end; tested against FlashInfer
in `tests/test_quantize.py`).

Scale conventions (TensorRT-LLM / FlashInfer; checked against FlashInfer's own dequantiser
`e2m1_and_ufp8sf_scale_to_float` in `tests/test_quantize.py`):

```
global_scale = 448 * 6 / amax(|x|)                       # fp32, shape [1]
sf[block]    = e4m3( amax(|x_block|) / 6 * global_scale ) # saturating at 448
q[i]         = e2m1( x[i] * global_scale / sf[block] )    # round to nearest, ties to even code, saturate at 6
x[i]        ~= q[i] * sf[block] / global_scale
```

Checkpoint conventions store the same two tensors under other names: `weight_scale` (E4M3, block) and
`weight_scale_2` or `weight_global_scale` (fp32). ComfyUI's `comfy-quants` documents its per-tensor scale as
`amax / (448 * 6)`, the reciprocal of `global_scale` above, and its dequantisation as `e2m1 * weight_scale * weight_scale_2`.
Which direction a given checkpoint family stores is the first thing a loader has to get right; a converter that
assumes the wrong direction produces outputs scaled by `global_scale^2`, which is silent. (Not yet tested here against
real checkpoints; on the list.)

## 2. The 128x4 layout, and the three names it goes by

The tensor cores' block-scaled MMA wants the scale factors of 128 consecutive rows and 4 consecutive scale columns
(4 x 16 = 64 elements of K for NVFP4) as one 512-byte tile. Inside a tile the order is `[outerM 32][innerM 4][innerK 4]`:

```
offset_in_tile(row, col) = (row % 32) * 16 + ((row % 128) // 32) * 4 + (col % 4)
tile(row, col)           = (row // 128) * ceil(cols / 4) + (col // 4)          # tiles row-major
offset                   = tile * 512 + offset_in_tile
rows padded to a multiple of 128, scale columns to a multiple of 4, padding zero-filled
```

The same layout under three names:

* **TensorRT-LLM / FlashInfer**, `get_sf_out_offset_128x4` in `cpp/tensorrt_llm/kernels/quantization.cuh`:
  "SF layout [numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]", strides innerK 1, innerM 4, outerM 16,
  kTile 512, mTile numKTiles*512; "Each SF block has 128 rows so pad rows to the multiple of 128"; "Round the number of
  scale-factor columns up to a multiple of four". FlashInfer exposes it as `fp4_quantize(..., is_sf_swizzled_layout=True)`
  and as the converter `nvfp4_block_scale_interleave`.
* **cuDNN frontend**, "The 128x4 tiled layout for block scaling factors": `offset = (outer % 32) * 16 + (outer / 32) * 4 + inner`,
  inverse `outer = ((offset % 16) / 4) * 32 + (offset / 16)`, `inner = offset % 4`; "one NVFP4 tile covers 128x64 data
  elements".
* **CUTLASS**, `include/cutlass/detail/sm100_blockscaled_layout.hpp` (the SM120 collectives `sm120_blockscaled_mma_tma.hpp`
  and `sm120_blockscaled_mma_array_tma.hpp` consume it): SF atom
  `Layout<Shape<Shape<_32,_4>, Shape<Int<SFVecSize>,_4>>, Stride<Stride<_16,_4>, Stride<_0,_1>>>`, `Blk_MN = 128`,
  `Blk_SF = 4`, tiled over (M, K) with `tile_to_shape(SfAtom{}, make_shape(M,K), Step<_2,_1>{})`. Reading the atom:
  row stride 16 within 32 rows, row-group stride 4, scale-column stride 1, the K-vector stride 0 because 16 elements
  share one scale - the same three strides as above.

**Tested** (`tests/test_layouts.py`): `to_128x4` reproduces `nvfp4_block_scale_interleave` byte for byte on 128x4, 256x32,
384x8 and 130x6 scale tensors (the last two exercise multi-tile ordering and padding), and `fp4_quantize`'s swizzled
output equals `to_128x4` of its linear output at 256x512, 128x1024 and 200x2048. The cuDNN formula is checked on a hand
table of 7 rows x 4 columns. The CUTLASS atom is the same arithmetic; a byte-level test against a CUTLASS SM120 kernel's
expected scale tensor is the next item.

## 3. Other layouts in circulation

* **Row-major ("linear")**: `sf[row, col]` at `row * cols + col`, no padding. What most checkpoints store and what
  FlashInfer emits with `is_sf_swizzled_layout=False`. `to_128x4` / `from_128x4` convert.
* **8x4** (`fp4_quantize(..., is_sf_8x4_layout=True)`, `mm_fp4(..., use_8x4_sf_layout=True)`): a smaller tile FlashInfer
  offers for some paths. Formula not yet extracted; not yet tested.
* **trtllm-gen weight shuffles** (`shuffle_matrix_a`, `shuffle_matrix_sf_a` with an epilogue tile M): a permutation of
  the weight and its scales for the TensorRT-LLM generated kernels; SM100-only kernels today, so out of scope for
  SM120 until they are not.
* **DeepGEMM** packs UE8M0 (MX) scales into int32 words, "TMA-aligned and MN-major", per recipe `(1, 128)` / `(128, 128)`
  / `(1, gran_k)` (`csrc/apis/layout.hpp`, `transform_sf_into_required_layout`); its FP4 scale transformation is gated to
  SM100 (vllm-project/vllm#41063 lists the gates). Not an NVFP4 128x4 variant; a different family. Not yet tested here.
* **Marlin (W4A16)**: a repacked weight order for its dequantise-then-BF16-MMA kernels; not a tensor-core block-scale
  layout at all, but it is what beats native FP4 at batch 1 on SM120 today (flashinfer-ai/flashinfer#2723), so a
  converter to it belongs in the stage-2 small-batch path.

## 4. Why this document exists

Reports of "garbage output" from FP4 kernels on SM120 (flashinfer-ai/flashinfer#2723, NVIDIA/cutlass#3096) and of a GEMM
that "silently returns all zeros" (flashinfer-ai/flashinfer#2577) share a property: nothing raised. A scale tensor in the
wrong layout, a scale in the wrong direction, or a tactic that cannot run on this architecture all produce numbers. The
converters here are the reference the conformance suite compares kernels against, and the tests are the record of which
libraries agree.

## Sources

- TensorRT-LLM `quantization.cuh`: https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/quantization.cuh
- cuDNN frontend, 128x4 tiled layout: https://nvidia.github.io/cudnn-frontend/mxfp8-scale-factor-128x4-layout/
- CUTLASS `sm100_blockscaled_layout.hpp`: https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/detail/sm100_blockscaled_layout.hpp
- CUTLASS SM120 collectives: https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective
- Colfax, NVFP4 blockscaled GEMM on SM12x: https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/
- comfy-quants NVFP4 format: https://github.com/Comfy-Org/comfy-quants/blob/main/docs/formats/nvfp4.md
- DeepGEMM `layout.hpp`: https://github.com/deepseek-ai/DeepGEMM/blob/main/csrc/apis/layout.hpp
