# Step 3 plan: the indexer kernel inside a running engine on two RTX PRO 6000 (adoption item 2)

Status 2026-10-04: a plan, not a run. Nothing here has been started on a cloud machine; starting one is a spend and is asked for
before it happens.

## What the run has to show

One number closes adoption item 2: with `SM120FP4_INDEXER=1`, vLLM serving DeepSeek-V4-Flash (NVFP4) on SM120 runs its sparse
indexer's MQA logits on our v5 kernel instead of the path it falls back to without DeepGEMM, answers the same, and is measured end to
end against the engine's own fallback on the same prompts, same hardware, same build. Two measurements, both against the fallback:

1. **Answers.** The 300-item retrieval set this repository's tests already use (`tests/` carries the 300-of-300 reference for the
   stage-2 MoE run), greedy, eight new tokens, first integer read; the indexer picks the same top-k when the logits are the same to
   1e-5, so the expected result is 300 of 300 agreement, and any disagreement is a bug to find, not a tolerance to widen.
2. **Throughput at long context.** Decode tokens per second at 1, 4 and 16 concurrent sequences with 64k and 256k prompts (the
   indexer's share of a decode step grows with context; the public recipe's operating point is a 500k-token request), and prefill
   tokens per second on the 256k prompt. The fallback's numbers on the same pod are the comparator; the stage-2 convention (paired
   blocks, bootstrap intervals, cold L2 where a kernel is timed alone) is kept.

## Host

Two RTX PRO 6000 Blackwell (96 GB each) at TP=2: DeepSeek-V4-Flash-0731-NVFP4 is about 160 to 168 GB of weights plus the KV cache
(wiring doc, Section 3). RunPod lists the card as `NVIDIA RTX PRO 6000 Blackwell Server Edition`; a 2-GPU pod with 200 GB of
container disk and a network volume for the weights is the shape. The pre-approved pool for cloud experiments is $25; at the card's
list price (about $1.9 per GPU-hour at the time of writing, to be read from `list-gpu-types` before starting) a 2-GPU pod costs about
$3.8 per hour, so the pool buys about six pod-hours, of which the weight download takes a large part on a first pod. The order of
work is therefore: a smoke pod with the weights pre-staged on a network volume (download once, about 170 GB at the data centre's
rate), then the measured pod attached to that volume.

## Software

- vLLM: the nightly or the first release in which PR #41834 (the SM120 DeepSeek-V4 megapatch) has landed, or 0.28 plus the
  commits the public recipe (Infatoshi/dsv4-flash-2x-rtxpro6000s) pins; which one is decided on the day from the PR's state, and the
  build is recorded in the report. State read 2026-10-04: #41834 ("Add SM12x support for DeepSeek V4 Flash with essential fixes") is
  open, last tagged 2026-08-09 (`sm120-pr-41834-stable-preview-20260809`), with a stock-deps path on released FlashInfer wheels, Triton
  sparse-MLA kernels and an indexer that selects top-k without materialising the logits; reviewers flagged that its FP4 indexer still
  needs DeepGEMM on SM120. So the pod runs that branch at its tag (not a release), and the comparison inside it is its own indexer
  path against ours behind the flag; our adapter serves the FP8 indexer cache, which is the one SM120 runs.
- Our plugin, installed from the wheel (`scripts/plugin_install_test.py --non-editable` is the check that the entry point registers on
  a stock install). Both flags are independent: `SM120FP4_INDEXER=1` alone wires the indexer; `SM120FP4_MOE=1` is stage 2's layer and
  is off for this run unless the V4 experts' format fits it (they are NVFP4 `modelopt_fp4`, so it may).
- The two readings 3f leaves to the engine: V4's `max_model_len` unit at the paged call and the compressed lengths' arrival through
  `seq_lens`. Both are logged by a one-line patch of the adapter (shape and max of each argument on the first call) before anything is
  measured.
- Known hazard: vLLM issue #53635 (the DeepGEMM paged kernel against V4's 256-row blocks on SM12x). With our adapter bound, DeepGEMM's
  kernel is not called, and the adapter's remap takes 256-row blocks (3e); if the engine refuses to start for another reason on this
  path, that is recorded and the fallback measured alone.

## Order on the pod

1. Start, attach the volume, install vLLM and the plugin, run `plugin_install_test` on the pod (the kernels compile on the PRO 6000,
   `sm_120a`).
2. Serve with the fallback, run the 300 items and a 4k-prompt throughput point: the engine works on this build (the public record says
   it does at 96 percent memory utilisation with the FP8 KV cache).
   The three engine scripts (`vllm_model_compare.py`, `vllm_prefill_latency.py`, `vllm_decode_throughput.py`) take `--tp 2` since
   2026-10-08 and record `tensor_parallel_size` in their reports; the model comparison records `+indexer` in its mode when
   `SM120FP4_INDEXER=1` is set, so the two 300-item runs are told apart by the report itself.
3. Serve with `SM120FP4_INDEXER=1`, confirm the log line that the adapter bound, run the same 300 items.
4. The throughput table, fallback and ours, paired.
5. Harvest the report into `reports/` (the pod's log, the 300-item diff, the throughput JSON), commit, terminate the pod (the convention
   is harvest, commit, terminate; stopped pods are not kept).

## What is not done here

No issue or PR text for vLLM is drafted (the author's decision: not before the project is essentially complete). The grouped FP8 x FP4
GEMM (adoption item 2, step 4) is not started until step 3 has its number.
