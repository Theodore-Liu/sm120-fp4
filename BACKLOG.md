# Backlog

Ranked work for the stages in `PLAN.md`. The top item is always in progress; when an item closes, the next one starts
and the list is refilled from the stage gates, so the queue is never empty. Each item names the measurement that
closes it. Dates are when an item was added, not estimates.

## In progress

3. **Stage 3 survey: the three FP4 sites DeepGEMM routes to `tcgen05`** (2026-10-02). Read the current DeepGEMM
   sources for the FP8xFP4 GEMM, the FP4 attention path and the FP4 einsum: their interfaces, tile shapes, the
   `tcgen05`/TMEM features they depend on, and what SM120 lacks (no TMEM, 99 KB shared memory, `mma.sync` only).
   Closes with `docs/stage3-survey.md` naming, per site, the SM120 design and the first kernel to write.
   State 2026-10-02: the survey is written against DeepGEMM `057ca5964aae` (the einsum site is FP8, not FP4;
   tiles, stages, recipe defaults confirmed), the first kernel and its interface are named, and the UE8M0 reference
   (`scripts/ue8m0_reference.py`) matches DeepGEMM's helpers bit for bit; next is the kernel itself.

## Ready, in order

4. **Stage 2, 8 to 16 tokens against FlashInfer's `compute_120f` grouped path by a stated margin** (2026-10-02). The
   gate clause not met: at 16 tokens the layer is level with Marlin and FlashInfer W4A16. The remaining gap is FC2
   against its own byte floor (`docs/stage2-design.md`); the candidate is a tensor-core FC2 that keeps two blocks per
   SM with the one-group tile. Closes with the 16-token rows ahead of both by the margin the report states.
5. **Prefill: a path for more than 16 tokens** (2026-10-02). Today the FC2 kernel takes at most 16 tokens, so the
   layer is a decode layer. Either a second FC2 kernel for 17 to 256 tokens or a documented hand-off to the engine's
   path, measured at 32, 64 and 128 tokens against FlashInfer and Marlin.
6. **Report the b12x W4A4 nondeterminism upstream** (2026-10-01). `reports/b12x-nondeterminism-rtx5090-2026-09-29.json`
   and `docs/stage2-baselines.md`: 20 identical calls, 20 outputs, 1.8 to 3.8% apart; the atomic scatter the source
   describes plus a lost or duplicated 8-column group in 2 of 100 calls. The issue text is drafted for the maintainer
   to review before anything is posted.
7. **Stage 2 on a second checkpoint family** (2026-10-02). A Gemma-class or Mixtral-class NVFP4 MoE checkpoint through
   the same table, to show the layer is not tuned to one expert shape.

## Closed

- Engine integration, the layer as an optional MoE backend in vLLM (closed 2026-10-02): `sm120fp4/vllm_backend.py`,
  `sm120fp4/vllm_classes.py`, `scripts/vllm_model_compare.py`, `scripts/vllm_decode_throughput.py`,
  `reports/vllm-compare-20261002.json`, `reports/vllm-decode-compare-20261002.json`, README section "Inside vLLM".
- Stage 2 gate, PRO 6000 reproduction of the real-weights table (closed 2026-10-02): `scripts/pod_real_ckpt.sh`,
  `reports/rtxpro6000-realckpt-2026-10-02/`, README section "The same table on an RTX PRO 6000".
- Stage 1 (layouts, conformance suite, RTX 5090 and RTX PRO 6000) - `docs/conformance-report-*.md`.
- Stage 2 single-kernel benches reproduced on an RTX PRO 6000 - `reports/rtxpro6000-stage2-2026-10-01/`.
