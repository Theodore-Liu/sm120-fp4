# Minimal CI on an SM120 runner (adoption item 3): the plan

Status 2026-10-04: a plan. Nothing is registered or running yet.

## What it has to do

Every push to `master` (or every night when there was one) runs the test suite on a real SM120 card and records the result where a
reader of the repository can see it. The kernels compile at first use through `torch.utils.cpp_extension.load_inline`, so a CI run is
a compile check as much as a correctness check; a green run means the sources still build against the pinned PyTorch and CUDA and the
numerical tests still pass on the card.

## The two shapes considered

**A. A GitHub Actions self-hosted runner on the RTX 5090 machine.** The repository is public, so the hosted runners exist but have no
SM120 GPU; a self-hosted runner registered to the repository (the `actions/runner` agent, running under WSL on the Windows machine,
labelled `sm120`) takes a workflow on every push. What it costs: a runner process that must be up when a push arrives (a scheduled
task at logon, like the other services on this machine), a GitHub token scoped to the repository held on the machine, and the risk
that a workflow from a fork runs on the author's hardware (GitHub does not run fork workflows on self-hosted runners without approval,
and `pull_request` triggers are left off; only `push` to `master` and `workflow_dispatch`). What it gives: the result as a check on
the commit, visible on the repository page, with the log.

**B. A nightly scheduled task on the machine that runs the suite and pushes a result file.** A Windows scheduled task (ASCII `.cmd`
wrapped in `run-hidden.vbs`, as every other task here) runs `pytest` in the vLLM 0.28 venv at a fixed hour, writes
`reports/ci/<date>.json` (commit, device, driver, torch, the pytest summary, per-test outcomes) and commits and pushes it on success or
failure. What it costs: nothing new on the machine. What it gives: a dated record in the repository, one commit a night, no check mark
on the commit page, and a result that lags a push by up to a day.

## Decision

Start with B, because it needs no new credential on the machine and no runner process, and it produces the artifact the README can
cite; move to A when the repository has a second contributor or a reason to gate pushes. B's task is `Sm120Ci`, 05:30 local (after
the backup chain and the other nightly jobs on this machine), with these steps:

1. `git fetch` and `git reset --hard origin/master` in a dedicated checkout (`~/ci/sm120-fp4` under WSL), so the suite runs on what is
   pushed, not on the working tree.
2. `pytest tests/ -q -p no:cacheprovider --junitxml=…` in the pinned venv, with the GPU checked idle first (`nvidia-smi`, the same
   20 percent utilisation bar the other lanes use); if the GPU is busy the run is skipped and the skip is recorded, not reported as
   green.
3. The result file written from the junit XML plus the environment, committed to `reports/ci/` on `master` and pushed.
4. The README's status line carries the latest result (date, passed/failed/skipped), written by the same step.

The suite's GPU tests take about a minute on the 5090 after the first compile; the first compile of every kernel is several minutes.

## What is not in this plan

No test runs on the RTX PRO 6000 (no standing machine); the stage-2 and stage-3 tables that cite that card are measurements, not
tests. No pull-request gating. No upstream text.
