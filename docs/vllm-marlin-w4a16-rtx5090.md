# vLLM 0.28.0's Marlin W4A16 GEMM on the RTX 5090 outside its test enumeration (2026-10-15)

Read under the registration `mlsys-5090-research/docs/d1-vllm-marlin-w4a16-registration-2026-10-15.md` with
`scripts/vllm_marlin_w4a16_probe.py`; the readings outside the probe's one-rounding oracle were fitted with
`scripts/vllm_marlin_w4a16_diag.py` (rounding models) and localised with `scripts/vllm_marlin_w4a16_diag2.py` (seeds,
options, neighbouring shapes). vLLM 0.28.0, torch 2.13.0+cu130, the vllm028 environment under WSL2. Reports:
`reports/vllm-marlin-w4a16-rtx5090/`.

## The sweep

437 calls: dtypes bf16 and fp16; GPTQ (`uint4b8`), GPTQ with act_order, AWQ (`uint4` with zero points); group sizes -1,
32, 128; m in {1, 7, 129} (plus 4097), n in {64, 320, 4160}, k in {128, 1152, 4224}; operands from vLLM's own quantisers.
Judged: 417 (47,032,128 elements). Refused with a message: 20 (act_order with `group_size == k`, vLLM's own quantiser
assertion `groupsize must be less than size_k`). The instrument check on one of the test's own shapes (26 x 2368 x 1664,
GPTQ bf16 group 128) read 0 outside, and both the kernel and the oracle within 0.003 of the test's reference under the
test's own 0.04 judge.

## Readings and what they are

| set | cases | outside the one-rounding oracle | what the kernel does |
|---|---|---|---|
| grouped (group 32, 128 with k > group), fp32 reduce, no atomics: GPTQ, GPTQ act_order, AWQ, both dtypes | 273 | **0** (max 0.99 of bound), deterministic | one output rounding after fp32 accumulation, as modelled; also deterministic and 0 outside over six seeds, m in {64, 128, 129, 130, 256, 257, 1024}, n in {64, 256, 320, 4160}, k in {1152, 2304, 4224}, groups 32 and 64 (`diag2`) |
| channelwise (group -1) and single-group (k == group_size) | 144 | 90 cases, errors of 1 to 3 output ulps (max 52 x the bound at k = 128, where the bound is smallest) | **a precision design, fitted bit for bit** (`diag`): the integer-weighted sum `a @ (q - 8)` is accumulated in fp32, rounded to the output dtype, then multiplied by the channel scale and rounded again (model M1, bit-exact fraction 1.000 on every channelwise case in bf16 and fp16, m 1 to 129, k 128 to 4224); the one-rounding model fits 57 to 58 percent. Deterministic |
| `use_fp32_reduce=False` | 1 (`diag2`) | 4,635 of 41,280, max 5.5 x the bound, deterministic | the option the source documents: "global reductions in fp16 precision (instead of fp32)" (`marlin_utils.py:38-40`) |
| `use_atomic_add=True` | 1 in the sweep, 1 in `diag2` | 7,111 and 6,979 to 7,134 of 41,280 per run, max 5.6 x the bound, **10 distinct outputs in 10 runs** | bf16 atomics on partial sums, order-dependent by construction; the same shape without the option is deterministic and clean |

So: no reading that the kernel's stated or fitted numerics do not account for. Under the census rule none is a defect:
the channelwise path's second rounding is a design (documented here, not in vLLM's docstrings: the layer code only says
`use_fp32_reduce` chooses the reduction precision); the two options are documented choices. The grouped path with the
defaults vLLM uses (`use_fp32_reduce=True`, no atomic add) is conformant on every served shape, including m = 1 and 7,
n = 320 and 4160, k = 4224 with group 32 and act_order, and m = 4097.

## For a user of 4-bit checkpoints on this part

- GPTQ/AWQ checkpoints with group scales (the usual 128) run on the one-rounding path: outputs within one output ulp of
  the exact product of the dequantised weights.
- A channelwise (`group_size = -1`) checkpoint, or a layer whose k equals its group size, gets one extra rounding of
  the accumulator before the scale: up to two or three output ulps, deterministic.
- `use_atomic_add` trades determinism and up to a few ulps for speed; `use_fp32_reduce=False` trades a few ulps.

## Instrument notes

The one-rounding oracle is right for the grouped path and wrong for the channelwise one; a reading outside the bound on a
whole class of cases (90 of 144) with errors of a few ulps is the signature of a rounding design, and the model fit
(bit-exact or not) is the test that separates it from a defect. The `use_atomic_add` case was in the registered plan as a
single option probe and showed up as the one non-deterministic grouped case; `diag2` reproduced it in isolation.
