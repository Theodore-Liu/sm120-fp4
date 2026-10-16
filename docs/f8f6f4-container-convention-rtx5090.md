# How the RTX 5090 reads every `kind::f8f6f4` operand byte: one 6-bit field, bits 5:0, for both sub-byte formats

Measured 2026-10-16 on the RTX 5090 (compute capability 12.0), torch 2.13.0+cu130, CUDA 13.0, one
`mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.<A>.<B>.f32` per warp with fragments handed in from Python.
Script `scripts/probe_f8f6f4_formats.py`; data `reports/probe-f8f6f4-formats-rtx5090-20261016.json`. This completes the
one cell measured on 2026-10-02 (`probe_f8f6f4_onehot.py`: e2m1 on the B side only) to the whole table: three formats,
both operand sides, and the four sub-byte pairs.

## Method

One-hot: the other operand holds 1.0 in e4m3 (byte 0x38, the format's own byte, confirmed on both sides first), at one
position; the operand under test holds one candidate byte in every container. The single product 1.0 x decode(candidate)
lands once in the C fragment and reads the hardware's decode of that byte directly. Candidates are the format's code
for 1.0 shifted by 0 to 4 bits, plus the all-ones byte 0xFF.

## The table

| side | format | code for 1.0 | byte that reads 1.0 | code bits in the byte | what the other shifts decode to | 0xFF decodes to |
|---|---|---|---|---|---|---|
| A and B | e4m3 | 0x38 | 0x38 | 7:0 | (full byte) | NaN |
| A and B | e3m2 | 0b001100 | 0x0C | 5:0 | 0x18 = 8.0, 0x30 = -2.0 | -28.0 |
| A and B | e2m1 | 0b0010 | 0x08 | 5:2 | 0x02 = 0.25, 0x04 = 0.5, 0x10 = 2.0, 0x20 = -0.0 | -7.5 |

Both sides agree, format by format. The four sub-byte pairs (e3m2 x e3m2 with 0x0C and 0x0C, e2m1 x e2m1 with 0x08 and
0x08, and the two mixed pairs) each return exactly 1.0 at C[0, 0] and nothing elsewhere.

## Reading

- **Every sub-byte operand is decoded as a 6-bit field in bits 5:0 of the byte.** Bits 7:6 are ignored: 0xFF decodes
  to a finite value for both sub-byte formats (-28.0 for e3m2 is sign, exponent 111 = 2^4, mantissa 1.11; -7.5 for
  e2m1 is sign, exponent 11 = 2^2, mantissa 1.111), where an 8-bit reading would give NaN or a different magnitude.
- **The e2m1 code sits in bits 5:2, and bits 1:0 are not ignored: they are read as two further mantissa bits.** The byte
  0x02 decodes to 0.25 and 0x04 to 0.5, which are the subnormals 0.01 and 0.10 of a 6-bit e2m3 field (bias 1). The
  datapath is a 6-bit field with sign, two exponent bits and three mantissa bits; e2m1 is the subset whose low two
  mantissa bits are zero. An e2m1 packer that leaves bits 1:0 of a container non-zero changes the value the tensor core
  sees (by up to 0.75 of an e2m1 ulp); a packer must zero them (ours does; what other packers do is read from their source, not assumed).
- **e3m2 occupies the field exactly**, so for e3m2 the "6-bit field" and "the format's own 6 bits" coincide and nothing
  of this is visible; it is visible only for e2m1, which is why the 2026-10-02 probe saw a 4-bit code "in bits 5:2".
- No e3m2 or e2m1 code decodes to NaN or infinity on this part, consistent with the formats having no such encodings.

## What it settles

The open queue question of 2026-10-02 (`f8f6f4-container-convention-across-formats`): yes, the same convention holds for
e3m2, for e2m1 on the A side, and for the mixed and the both-FP4 pairs; `kind::f8f6f4` takes e2m1 on both operands at
k32 (the one-value-per-byte form; the two-per-byte k64 form is `kind::mxf4` with block scales, measured 2026-10-04 by
`probe_f4f4.py`). The PTX ISA's container text is read by this hardware as: the value occupies the low 6 bits, left-aligned
within them for e2m1. This is the measured table the IEEE CAL letter (R4 in the mlsys repository's result-first
directions) is built around.
