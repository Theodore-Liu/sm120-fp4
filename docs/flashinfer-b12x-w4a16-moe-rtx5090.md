# FlashInfer's b12x fused MoE, W4A16 mode, against the fused-MoE oracle (RTX 5090, 2026-10-11)

Script: `scripts/flashinfer_b12x_w4a16_moe_probe.py`. Oracle: `sm120fp4/moe_oracle.py` (selftest in `tests/test_moe_oracle.py`).
Report: `reports/flashinfer-b12x-w4a16-moe-rtx5090-fi071-20261011.json`. Stack: FlashInfer 0.7.1, PyTorch 2.9.1+cu128, driver
610.47, WSL2; `b12x_fused_moe(..., activation_precision="bf16")`, which dispatches to `moe_w4a16_kernel.run_w4a16_moe`.

## The oracle

A fused MoE has no single exact value: the kernel rounds at several points between its two GEMMs. The oracle follows the
rounding points of the dispatched kernel, read from its source (FC1 written as bf16; SiLU in fp32 from the bf16 gate with a
fast exponential, rounded to bf16; the product with the bf16 up rounded to bf16; FC2 in fp32; the routing weight times alpha
packed as bf16; the top-k sum in fp32; one bf16 output rounding), and at each point computes the set of values a correct
rounding of the fp32-accumulated quantity can produce, given only the freedom of accumulation order and of the last bits of
the approximate exponential. Where the accumulation bound does not straddle a bf16 rounding boundary the set is one value;
where it does, two adjacent values, and the width propagates through |W2| into the bound. The result is a nominal output and
a per-element bound every correct implementation of those rounding points must respect.

The floor is set by FC1's worst-case accumulation bound (K * 2^-24 of the absolute products): at K = 1024 it straddles a bf16
boundary on about 80 percent of the intermediate elements, and the mean bound is about 6 percent of an expert's
contribution. That catches the kinds of fault the census has found (a dropped column group, a mis-addressed scale, the wrong
gate half) and does not catch a fault below a few percent of a contribution. The selftest shows both: the faithful emulation
inside the bound, the swapped gate order outside on 99.8 percent of elements, a dropped 8-column group outside on every
affected element, a contribution off by 2^-3 outside on about nine tenths of its elements, off by 2^-5 on about two thirds, off by 2^-8 on
about a tenth.

## The operand contract, found by sweep

The API docstring gives shapes but not the scale layout, the alpha convention or which half of w13 is the gate. A sweep of
the eight combinations against a float reference (M = 4, scratch, not kept) read:

| scales | alpha | reference's gate half | relative error |
|---|---|---|---|
| 128x4 swizzled | 1 / gs | second | **0.0047** |
| 128x4 swizzled | 1 / gs | first | 0.92 |
| 128x4 swizzled | gs | either | NaN |
| row-major | 1 / gs | second | 0.57 |
| row-major | 1 / gs | first | 1.09 |
| row-major | gs | either | NaN |

So: scales in the 128x4 swizzled layout (as the preparation's docstring says), alpha the multiplicative global scale (1 / gs in
the reference quantiser's convention), and the second half of w13 is the gate. The last is not in the docstring; the W4A4
path's reference in `scripts/diag_b12x_actquant.py` reads the halves the same way, so it is the b12x convention across both
modes, recorded here as a convention, not a defect.

## Results

8 experts, hidden 1024, intermediate 512, top-2, random routing, 5 runs per token count.

| tokens | elements | outside the bound (each of 5 runs) | max err / bound | distinct outputs in 5 runs | rel. error vs float reference |
|---|---|---|---|---|---|
| 1 | 1,024 | 0 | 0.377 | 1 | 0.0053 |
| 4 | 4,096 | 0 | 0.362 | 1 | 0.0052 |
| 16 | 16,384 | 0 | 0.571 | 1 | 0.0049 |
| 64 | 65,536 | 0 | 0.550 | 1 | 0.0048 |

Conformant on every element of every run, and deterministic at every token count, where the W4A4 path on the same stack
was non-deterministic at M = 1 and 4 (`docs/stage2-baselines.md`). The relative error against an fp64 float reference of the
same quantised weights, about 0.5 percent, is the bf16 intermediate and the bf16-packed routing weight, both design choices
the oracle models rather than judges.

## What this row does not say

One shape family and random operands; the W4A16 path's micro / static / dynamic backend selection by routed row count was
exercised only as far as M = 64 with top-2 (128 routed rows). The W4A4 path has not yet been read under this oracle (it needs
the activation quantiser's rounding modelled as well); it is the next item.
