#!/usr/bin/env python3
"""Nightly CI on the SM120 machine (adoption item 3, plan B in docs/ci-plan.md).

Runs under WSL in the pinned vLLM 0.28 venv:

1. a dedicated checkout (~/ci/sm120-fp4) is fetched and reset to origin/master, so the suite runs on what is pushed;
2. the GPU is checked idle (utilisation at most 20 percent and no other process holding more than 4 GB); a busy GPU makes the run
   `skipped`, recorded as such, never reported green;
3. `pytest tests/ -q -p no:cacheprovider --junitxml=...` in that checkout;
4. the result is written to reports/ci/<date>.json in the working copy this script lives in (commit, device, driver, torch, CUDA,
   the pytest summary, every test's outcome and time), the README's CI status line is rewritten, and with --push the two files are
   committed and pushed from the working copy.

    ~/mlsys-5090-runtime/vllm028/.venv/bin/python scripts/ci_nightly.py            # run and write the report
    ~/mlsys-5090-runtime/vllm028/.venv/bin/python scripts/ci_nightly.py --push     # the scheduled task's form
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]                    # the working copy this script lives in
CI_DIR = Path(os.environ.get("SM120_CI_DIR", str(Path.home() / "ci" / "sm120-fp4")))
REMOTE = "https://github.com/Theodore-Liu/sm120-fp4.git"
BUSY_UTIL = 20
BUSY_MEM_MB = 4096
STATUS_RE = re.compile(r"^<!-- ci-status -->.*$", re.M)


def sh(cmd: list[str], cwd: Path | None = None, check: bool = True) -> str:
    r = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} -> {r.returncode}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def gpu_state() -> dict:
    out = sh(["nvidia-smi", "--query-gpu=name,driver_version,utilization.gpu,memory.used", "--format=csv,noheader,nounits"]).strip().splitlines()[0]
    name, driver, util, mem = [x.strip() for x in out.split(",")]
    return {"device": name, "driver": driver, "utilization_pct": int(float(util)), "memory_used_mb": int(float(mem))}


def checkout() -> str:
    if not (CI_DIR / ".git").is_dir():
        CI_DIR.parent.mkdir(parents=True, exist_ok=True)
        sh(["git", "clone", "-q", REMOTE, str(CI_DIR)])
    sh(["git", "fetch", "-q", "origin"], cwd=CI_DIR)
    sh(["git", "reset", "-q", "--hard", "origin/master"], cwd=CI_DIR)
    return sh(["git", "rev-parse", "HEAD"], cwd=CI_DIR).strip()


def run_pytest(junit: Path) -> tuple[int, str]:
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "-p", "no:cacheprovider", f"--junitxml={junit}"],
                       cwd=str(CI_DIR), capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)[-4000:]


def parse_junit(junit: Path) -> dict:
    root = ET.parse(junit).getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    tests, summary = [], {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "time_s": 0.0}
    for s in suites:
        for k in ("tests", "failures", "errors", "skipped"):
            summary[k] += int(s.get(k, 0))
        summary["time_s"] += float(s.get("time", 0.0))
        for tc in s.iter("testcase"):
            outcome = "passed"
            for child in tc:
                if child.tag in ("failure", "error", "skipped"):
                    outcome = child.tag if child.tag != "error" else "error"
            tests.append({"name": f"{tc.get('classname')}::{tc.get('name')}", "outcome": outcome, "time_s": float(tc.get("time", 0.0))})
    summary["passed"] = summary["tests"] - summary["failures"] - summary["errors"] - summary["skipped"]
    return {"summary": summary, "tests": tests}


def write_status(line: str) -> None:
    readme = WORK / "README.md"
    text = readme.read_text(encoding="utf-8")
    new = f"<!-- ci-status -->{line}"
    if STATUS_RE.search(text):
        text = STATUS_RE.sub(new, text)
    else:
        anchor = "## Status\n\n"
        assert anchor in text, "README has no Status section"
        text = text.replace(anchor, anchor + new + "\n\n", 1)
    readme.write_text(text, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true", help="commit the report and the README line and push from the working copy")
    ap.add_argument("--force", action="store_true", help="run even when the GPU reads busy (a hand run)")
    a = ap.parse_args()
    import torch
    date = dt.datetime.now().strftime("%Y-%m-%d")
    rep_dir = WORK / "reports" / "ci"
    rep_dir.mkdir(parents=True, exist_ok=True)
    out = rep_dir / f"{date}.json"
    gpu = gpu_state()
    commit = checkout()
    rec = {"date": date, "commit": commit, "gpu": gpu, "torch": torch.__version__, "cuda": torch.version.cuda, "python": sys.version.split()[0],
           "checkout": str(CI_DIR)}
    busy = gpu["utilization_pct"] > BUSY_UTIL or gpu["memory_used_mb"] > BUSY_MEM_MB
    if busy and not a.force:
        rec.update({"outcome": "skipped", "why": f"GPU busy: utilisation {gpu['utilization_pct']} percent, {gpu['memory_used_mb']} MB in use"})
        line = f"CI {date}: skipped (GPU busy) on {commit[:7]}."
    else:
        junit = rep_dir / f"{date}.junit.xml"
        rc, tail = run_pytest(junit)
        parsed = parse_junit(junit) if junit.exists() else {"summary": {}, "tests": []}
        s = parsed["summary"]
        outcome = "passed" if rc == 0 else "failed"
        rec.update({"outcome": outcome, "pytest_rc": rc, "pytest_tail": tail, **parsed})
        line = (f"CI {date}: {outcome} on {commit[:7]}, {s.get('passed', 0)} passed, {s.get('failures', 0) + s.get('errors', 0)} failed, "
                f"{s.get('skipped', 0)} skipped in {s.get('time_s', 0.0):.0f} s on {gpu['device']} (driver {gpu['driver']}, torch {torch.__version__}).")
        try:
            junit.unlink()
        except OSError:
            pass
    out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    write_status(line)
    print(line)
    print("->", out)
    if a.push:
        rel = out.relative_to(WORK).as_posix()
        sh(["git", "add", rel, "README.md"], cwd=WORK)
        msg = f"CI {date}: {rec['outcome']} on {commit[:7]} ({gpu['device']})\n\nreports/ci/{date}.json written by scripts/ci_nightly.py; the README's CI status line updated."
        sh(["git", "commit", "-q", "-m", msg, "--", rel, "README.md"], cwd=WORK)
        sh(["git", "push", "-q", "origin", "master"], cwd=WORK)
        print("pushed")
    return 0 if rec["outcome"] in ("passed", "skipped") else 1


if __name__ == "__main__":
    sys.exit(main())
