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

## Lever 1: two q heads per MMA tile

Today one MMA tile is 16 query rows x 8 kv columns x 32 k, and the kernel runs the H heads of one query row as separate tiles with the same
B fragment (the k row) re-read from shared memory for each head. For the indexer's shapes H is 8 or 16, so the same kv row is read 8 or 16
times from shared memory per query row; the shared-memory traffic, not the global bytes, sets the pipe at 49 to 55 percent. Putting two
heads of the same query row into the 16-row A fragment (8 rows of head a, 8 of head b, as the fragment already does for `ha` and `hb` in
pairs) halves the B re-reads per head pair; a second step, four heads per tile with the k32 steps split across them, would need the
accumulator laid out per head and costs registers (the budget is 128 to keep two blocks). Expected: a reduction of the shared-memory
reads by up to half with no change in global bytes; whether it moves the time depends on whether the pipe at 55 percent is the limiter or
the latency chain is, which the warp-state section of the profile (7.5 to 8 cycles per issued instruction at 8 warps per SM) suggests
it partly is. Cheap to try (a fragment rearrangement, no layout change), and bit-exact by construction.

## Lever 2: read the 128-byte row as two 64-byte halves on the v4 path

The paged v4 reads 64-byte e2m1 rows and runs at 1.44 to 1.52 times less time. An e4m3 row is two 64-byte halves; the v4 kernel's
staging and fragment loads are written for 64-byte rows with one scale per row. Reading the e4m3 row as two half-rows would need the k32
steps to take their B fragment from the right half (steps 0 and 1 from the first 64 bytes, 2 and 3 from the second) and the per-row fp32
scale instead of v4's UE8M0: a new kernel with v4's staging shape and v5's arithmetic, not a parameter of either. The bytes do not change,
so it would recover only what v4's staging shape (64-byte rows, 16-byte aligned, no 132-byte scatter) saves over v5h's 132-byte scatter;
the profile does not separate that cost, so the expected gain is unknown and would be measured, not predicted.

## What this page does not decide

Which lever to try first: lever 1 is cheaper and bit-exact; lever 2 changes the staging and is the one that could reach v4's shape. Both
are kernel changes queued behind adoption item 2's step 3 (the two-GPU run), which does not depend on them: the adapter serves the engine
with v5h today, and the indexer is a small share of a decode step on these models until the context is long.
