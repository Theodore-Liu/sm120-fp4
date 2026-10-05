# The paged v5's row width: what the remaining gap to the paged v4 is made of, and the two ways to close it (a design, 2026-10-04)

Status: a design page, no kernel. The numbers are the saved reports' (`reports/fp8-paged-mqa-logits-v5h-rerun-rtx5090-20261004.json`,
`reports/fp8-fp4-paged-mqa-logits-v4-rtx5090-20261003.json`, `reports/ncu-paged-v5-rtx5090-20261004.txt`, `reports/ncu-paged-v5d-v5h-rtx5090-20261004.txt`).

## Where the time is

| shape (S, N, H) | paged v4 (e2m1 rows, 64 B), us | paged v5h (e4m3 rows, 128 B + 4), us | v5h / v4 | bytes per kv row read, v5h / v4 |
|---|---:|---:|---:|---|
| 64, 8192, 8 | 23.3 | 33.5 | 1.44 | 132 / 68 |
| 128, 16384, 8 | 72.4 | 103.2 | 1.43 | 132 / 68 |
| 32, 65536, 16 | 74.5 | 113.4 | 1.52 | 132 / 68 |

The e4m3 row is 1.94 times the e2m1 row's bytes (132 against 68 with the scales), and the kernel runs at 1.43 to 1.52 times the time, so
per byte it is already level with or ahead of v4; the gap is the bytes. The profile says the rows come from L2 (hit 85 to 94 percent)
and that neither DRAM nor the SM is near its ceiling; v5h at two blocks per SM (33 percent occupancy, 128 registers per thread) is where the
shared-memory lever ends: the register double buffer (v5d) lost because it cost the second block (200 registers).

## Lever 1, withdrawn: two q heads per MMA tile is already how the kernel works

The first version of this page proposed putting two heads of a query row into one MMA tile so that each kv row's B fragment is read once per
head pair. Reading the kernel again (`k_paged_mqa_logits_v5h`, the head loop), the A fragment already holds sixteen heads of one query row:
rows `g` and `g + 8` of the m16 tile are heads `h0 + g` and `h0 + g + 8`, so each 8-column slice of the staged page is read from shared memory
once per sixteen heads, not once per head. The lever is in place; the page's premise was wrong, and nothing is gained by building it.

What the reading does show: with H = 8 (the first two bench shapes) only eight of the tile's sixteen rows carry a head, so half of every MMA
is padding; with H = 16 (the third shape) the tile is full. Filling the H = 8 tile would take two query rows per tile, which only works when the
two rows read the same pages (two draft positions of one request under speculative decoding, `next_n = 2`, share a block table; two
requests do not). The H = 16 shape's gap to v4 (1.52) is no smaller than the H = 8 shapes' (1.43, 1.44), so the padding is not where the gap
is; the gap is the bytes, as the table above says.

## Lever 2: read the 128-byte row as two 64-byte halves on the v4 path

The paged v4 reads 64-byte e2m1 rows and runs at 1.44 to 1.52 times less time. An e4m3 row is two 64-byte halves; the v4 kernel's
staging and fragment loads are written for 64-byte rows with one scale per row. Reading the e4m3 row as two half-rows would need the k32
steps to take their B fragment from the right half (steps 0 and 1 from the first 64 bytes, 2 and 3 from the second) and the per-row fp32
scale instead of v4's UE8M0: a new kernel with v4's staging shape and v5's arithmetic, not a parameter of either. The bytes do not change,
so it would recover only what v4's staging shape (64-byte rows, 16-byte aligned, no 132-byte scatter) saves over v5h's 132-byte scatter;
the profile does not separate that cost, so the expected gain is unknown and would be measured, not predicted.

## Lever 2's first form: raw-row staging (v5r, 2026-10-04, untimed)

The cheap form of lever 2 does not change the cache bytes (they are vLLM's 132-byte rows) but removes the staging scatter: `fp8_paged_mqa_logits_sm120_v5r` copies each 32-row half as 264 straight 16-byte chunks into a 4224-byte shared buffer and reads the B fragments and the scales at a 132-byte (33-word) row stride, so no per-word divide and no split of codes from scales; a 33-word stride puts the eight `g` rows of a fragment read on distinct banks. Shared memory per block is unchanged (8 x 4224 bytes). Bit-identical to the flat v5 on the five half-page selftest shapes. It has not been timed: the GPU was running another job when it was written, and the timing is taken idle against v5h's 33.5 / 103.2 / 113.4 us before anything is claimed.

## What this page does not decide

Lever 2 is the one left, and it changes the staging; the `next_n = 2` row pairing is a narrower variant for speculative decoding only. Both
are kernel changes queued behind adoption item 2's step 3 (the two-GPU run), which does not depend on them: the adapter serves the engine
with v5h today, and the indexer is a small share of a decode step on these models until the context is long.
