# DeepGEMM's SM120 k-grouped FP8 GEMMs outside the test enumeration (RTX 5090, 2026-10-15)

Read under the registration `mlsys-5090-research/docs/d1-next-sweep-outside-enumeration-2026-10-14.md` with
`scripts/deepgemm_kgrouped_probe.py`; localised with `scripts/deepgemm_kgrouped_diag{,2,3,4}.py`. Library:
vllm-project/DeepGEMM dev 15c02c8 (deep_gemm 2.8.1), torch 2.13.0+cu130, CUDA 13.2 JIT, the vllm028 environment under
WSL2. Reports: `reports/deepgemm-kgrouped-rtx5090/`.

## The sweep

422 calls, both entry points. Judged: 360 (271,209,600 elements). Refused with a message: 62 (60 of them bf16 output
without an accumulator: the SM120 k-grouped kernel asserts `c.has_value()`, `sm120_fp8_fp4_gemm_1d1d.hpp:337`, which the
TN entry point's API does not say; 2 of them m = 1 and m = 7: the SF packing asserts `num_groups <= 128 and mn % 4 == 0`,
`smxx_layout.hpp:231`). Conformant: 356 cases, 0 elements outside, max 0.009 of the bound, deterministic. Outside the
bound: 4 cases, all `k_grouped_fp8_gemm_nt_contiguous` (K-major operands) at m = 12 with the 64-group pattern, 17,523
elements wrong by up to 32,000 times the bound; the TN entry point on the same quantised operands is conformant.

## The defect: the K-major SF orientation is guessed from its shape

`k_grouped_fp8_gemm_nt_contiguous` takes each operand's scale factors as a K-major tensor `(mn, sum_k / gran_k)`. On
SM120 the layout transform (`csrc/apis/layout.hpp:117-127`) has to turn it into the `[sf_k, mn]` orientation the common
packer consumes, and decides whether to transpose by comparing the tensor's first dimension with the expected number of
SF rows:

```cpp
sf_input = sf_contiguous.size(0) == expected_sf_k ? sf_contiguous : sf_contiguous.t().contiguous();
```

When `mn == sum_k / gran_k` the SF tensor is square, the test passes on the untransposed tensor, and the kernel reads
the scale of row i, block j as that of row j, block i: wrong output for every row whose scales differ, no error, the
same wrong output on every call in a process (the values depend only on the operands), a different set of rows across
processes only because the random operands differ.

Confirmed by prediction (`diag4`): with sum_k = 128 m the NT entry point is wrong at every m tried (4, 8, 12, 16, 24,
32, 48, 64, 128: 272 to 218,727 elements outside, DeepGEMM's own `calc_diff` against its bf16 reference 3.3e-3 to
8.9e-3, above the 1e-3 its tests allow), and with sum_k = 128 (m - 1) or 128 (m + 1) it is clean at every one of them
(0 outside, calc_diff 7.4e-5 to 7.9e-5). Passing the SF tensor contiguous instead of as a transposed view changes
nothing (the transform makes it contiguous first), so the shape, not the stride, is what decides. The grouping does
not matter: [384] x 4, [768] x 2, [128] x 12, [512] x 3 and [128, 256] x 4 (all sum 1536 at m = 12) are wrong, and
[256] x 8, [640] x 2, [896] x 2, [256] x 3 are clean. The TN entry point (MN-major SF `(sf_k, mn)`, no transpose
decision) is clean on every one of these, including the square ones.

Why the library's tests do not see it: the generic k-grouped test enumerates m in {768, 4096, 7168} with sum_k about
0.7 to 1.3 x {1024, 32768, 65536}, never equal to 128 m; the SM120 file's K-major cases use m = 128 or 256 with sum_k
384 to 8320; and `test_sm120_kgroup_zero_and_unequal_k` at m = 256 passes `sa.T.contiguous()` of shape (256, 6).

Who hits it: a caller of the K-major k-grouped FP8 GEMM (MoE weight-gradient shapes, `README: "K-axis-grouped API for
MoE weight backward"`) whose row count equals its total K in units of the scale granularity; for gran_k = 128 that is
m = sum_k / 128, e.g. 64 experts of 128 rows with m = 64. Rare in practice, silent when it happens.

Verdict under the census rule: the contract was read from the host checks and README (m = 12 and these ks are
admitted); the probe's decode is checked against the generator's reference; a second path of the same library (the TN
entry point) and the library's own metric disagree with the output; the cause is a line of host code. **Defect class 2
of the four-week test.** A local fix and a regression test come before any upstream text (`feedback-confirm-defect-and-fix-locally-before-posting`).

## The fix to try

Decide the orientation from what the caller already states, not from the shape: the NT entry point asserts
`major_a == K`, so its SF is `(mn, sf_k)` and must be transposed unconditionally; the TN entry point's is `(sf_k, mn)`
and must not. Pass the major into `transform_k_grouped_sf_into_required_layout` (or transpose at the two call sites in
`gemm.hpp`) and drop the shape comparison. Regression test: the square case, m = sum_k / gran_k, for both entry points.
