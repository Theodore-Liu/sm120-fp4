# CUTLASS's SM120 GeForce Blackwell example programs at sizes outside their defaults (RTX PRO 6000, 2026-10-15)

Read under the registration `mlsys-5090-research/docs/d1-cutlass-sm120-examples-registration-2026-10-15.md` with
`mlsys-5090-research/scripts/pod/cutlass-sm120-examples-pro6000.sh` on a rented RTX PRO 6000 Blackwell Server Edition
(US-NC-1, driver 595.91.07, CUDA 13.2 toolkit from NVIDIA's apt repository, CUTLASS 78e4622, `CUTLASS_NVCC_ARCHS=120a`,
only the five example targets built: 45 s). Reports: `reports/cutlass-sm120-examples-pro6000/` (the 583-run sweep as
JSON lines, the pod log, build and cmake logs). About 35 minutes of pod time.

## What the sweep measures

Each example takes `--m --n --k` (`--groups` for the grouped ones), fills its operands and scale factors from a seed,
runs the kernel once and compares the output with CUTLASS's own host block-scaling reference (`reference::host::Gemm3x`)
under exact equality (`TensorEquals`), printing `Disposition: Passed` or `Failed`; unsupported sizes are refused by
`gemm.can_implement` through `CUTLASS_CHECK`. So the first signal is the library's own reference, not the census oracle;
a `Failed` is a reading to localise.

## Results, 583 runs

| example | kernel | runs | Passed | Failed | refused | reading |
|---|---|---|---|---|---|---|
| 79a | NVFP4 x BF16 dense, bf16 output | 169 | 165 | 4 | 0 | every size accepted passes except m in {1, 2} at k = 32: an all-zero problem, see below |
| 79b | NVFP4 x NVFP4 dense, NVFP4 output with SFD | 169 | 1 | 0 | 168 | every n in {8, 24, 136, 1000} refused; n in {64, 128, 192, 256, 2048} pass at m 12 and 1024, k 64 to 1088: the rule is `N % 16 == 0` (the SF vector size of the NVFP4 output, `sm120_visitor_store_tma_warpspecialized.hpp:110`) |
| 79d | NVFP4 x NVFP4 grouped, NVFP4 output with SFD | 38 | 2 | 36 | 0 | every explicit size with n in {24, 136, 1000} fails; n in {64, 128, 192, 256, 512, 1024} and the random-size mode pass: the same `N % 16` rule, unchecked here, see below |
| 87a | FP8 blockwise (128 x 128 scales) dense, bf16 output | 169 | 1 | 0 | 168 | refused unless `M % 128 == 0`, `N % 128 == 0` and `K % 128 == 0` (the scale granularity and the K tile, `sm120_mma_tma_blockwise_scaling.hpp:345-349`); m in {128, 256, 1024} with n in {128, 512} and k in {128, 384, 1024} pass |
| 87c | FP8 groupwise grouped | 38 | 2 | 0 | 36 | the same rules; the random-size mode passes at 10 and 64 groups |

Every size the examples accept and that respects the stated scale rules passes the exact-equality check. Deterministic
in the sense the examples test (one run each); the second runs of the grouped example (its Pingpong schedule) agree.

## The two Failed readings, localised on the pod

- **79a at m in {1, 2}, k = 32** (n in {8, 24, 136, 1000}; m = 3 and k = 64 pass): a diff patch on the example's `verify`
  (`scripts/pod-side/cutlass_diffstat_patch.py`) shows 0 differing elements and a reference whose largest absolute value
  is 0. The example's `verify` also requires `TensorNorm(reference) > 0` and `TensorNorm(D) > 0`, so an all-zero problem
  reads Failed. The kernel equals the reference; the artefact is the example's random fill of a one- or two-row problem
  at a single K block. Not a kernel reading.
- **79d at n % 16 != 0** (24, 136, 1000; at m 8, 12, 65, 128, 256, 300, 1024 and every k): the NVFP4-output epilogue's
  fusion states `N % SFVecSize == 0` in its `can_implement`, and the dense 79b refuses exactly these n through it. The
  grouped (array) epilogue runs that check per group only when host problem shapes are available
  (`sm90_epilogue_array_tma_warpspecialized.hpp:400-417`), and 79d calls `run<Gemm>(options, false
  /*host_problem_shapes_available*/)` for both schedules, so the rule is never checked and the kernel runs a size outside
  its stated contract; its output then differs from the host reference. A contract violation the example does not guard,
  not a kernel defect under the census rule (the contract is stated and checked where the library can check it).
  Note for a caller: the grouped block-scaled-output path cannot validate device-side problem shapes, so a caller that
  passes only device shapes has to keep every group's N a multiple of the SF vector size itself.

Instrument note: 79d's D is `float_e2m1_t`, packed two values a byte, so a raw `host_data()` element loop (the first
version of the diff patch) compares bytes, not elements, and reports differences on passing cases; the examples'
`TensorEquals` on the typed view stays the judge for 4-bit outputs.

## Verdict for the ledger

Conformant where accepted, under the examples' own exact reference: 79a (165 sizes, m 1 to 300, n 8 to 1000, k 32 to
4128), 79b and 87a at the sizes their scale rules admit, 79d and 87c at aligned explicit sizes and in the random-size mode.
Refusals by stated rules: 79b `N % 16`; 87a and 87c `M % 128`, `N % 128`, `K % 128`. Two example-side readings, above.
No census defect; the inside-against-outside count gains an outside row with no defect. Not read: 79c (MXFP8 x MXFP6) and
the sparse 80a/b, the next record.
