#!/usr/bin/env python3
"""BACKLOG item 4: does `pip install` of this repository register the vLLM plugin on a stock vLLM 0.28, in a venv that has never
seen the checkout's own environment?

Steps, each logged, the chain stopping at the first failure:
  1. a fresh venv (uv) with stock vllm==0.28.0 from PyPI;
  2. `pip install -e <this repository>` into it (editable: the kernels live in scripts/ and compile from the checkout at first use);
  3. in that venv, with SM120FP4_MOE unset: the `vllm.general_plugins` entry point is listed, `load_general_plugins()` leaves
     vLLM's own `modelopt_fp4` config in place (the plugin is a no-op);
  4. in that venv, with SM120FP4_MOE=1: `load_general_plugins()` swaps `modelopt_fp4` to `sm120fp4.vllm_classes`.
The result is one JSON (reports/plugin-install-test-<date>.json) with each step's stdout tail and verdict.

    ~/mlsys-5090-runtime/vllm028/.venv/bin/python scripts/plugin_install_test.py --venv ~/sm120-plugin-test --out reports/plugin-install-test-20261003.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PROBE = r'''
import importlib.metadata as md, json, os, sys
out = {"env": os.environ.get("SM120FP4_MOE", ""), "python": sys.version.split()[0]}
eps = [e for e in md.entry_points(group="vllm.general_plugins")]
out["entry_points"] = [f"{e.name} = {e.value}" for e in eps]
import vllm
out["vllm"] = vllm.__version__
import sm120fp4
out["sm120fp4"] = sm120fp4.__version__
from vllm.plugins import load_general_plugins
load_general_plugins()
from vllm.model_executor.layers.quantization import get_quantization_config
cls = get_quantization_config("modelopt_fp4")
out["modelopt_fp4_class"] = f"{cls.__module__}.{cls.__name__}"
print("PROBE_JSON " + json.dumps(out))
'''


def run(cmd: list[str], env: dict | None = None, timeout: int = 3600) -> dict:
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    tail = (p.stdout + "\n" + p.stderr).strip().splitlines()[-25:]
    return {"cmd": " ".join(cmd), "rc": p.returncode, "seconds": round(time.time() - t0, 1), "tail": tail, "stdout": p.stdout}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venv", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--vllm", default="vllm==0.28.0")
    a = ap.parse_args()
    if a.out.exists():
        print(f"refusing to overwrite {a.out}", file=sys.stderr)
        return 2
    venv = Path(os.path.expanduser(a.venv))
    py = venv / "bin" / "python"
    report = {"repo_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
              "venv": str(venv), "vllm_requirement": a.vllm, "steps": []}

    def step(name, cmd, env=None, timeout=3600):
        r = run(cmd, env, timeout)
        r["step"] = name
        report["steps"].append({k: v for k, v in r.items() if k != "stdout"})
        print(f"[{name}] rc={r['rc']} {r['seconds']}s", flush=True)
        for ln in r["tail"][-5:]:
            print("   ", ln, flush=True)
        return r

    base = dict(os.environ)
    base.pop("SM120FP4_MOE", None)
    base["VLLM_LOGGING_LEVEL"] = "WARNING"
    if not py.exists():
        r = step("venv", ["uv", "venv", str(venv), "--python", "3.12", "--seed"])
        if r["rc"] != 0:
            return finish(report, a.out, "venv creation failed")
    r = step("install vllm", ["uv", "pip", "install", "--python", str(py), a.vllm])
    if r["rc"] != 0:
        return finish(report, a.out, "stock vllm install failed")
    r = step("install sm120fp4 (editable)", ["uv", "pip", "install", "--python", str(py), "-e", str(REPO)])
    if r["rc"] != 0:
        return finish(report, a.out, "editable install failed")
    probes = {}
    for label, env_value in (("plugin off", None), ("plugin on", "1")):
        env = dict(base)
        if env_value is not None:
            env["SM120FP4_MOE"] = env_value
        r = step(f"probe, {label}", [str(py), "-c", PROBE], env=env, timeout=900)
        js = [ln for ln in r["stdout"].splitlines() if ln.startswith("PROBE_JSON ")]
        probes[label] = json.loads(js[-1][len("PROBE_JSON "):]) if js else {"error": "no PROBE_JSON line"}
    report["probes"] = probes
    off, on = probes["plugin off"], probes["plugin on"]
    checks = {
        "entry point registered": any(e.startswith("sm120fp4_moe = sm120fp4.vllm_backend:register") for e in off.get("entry_points", [])),
        "stock vllm 0.28": str(off.get("vllm", "")).startswith("0.28"),
        "plugin off leaves vllm's modelopt_fp4": off.get("modelopt_fp4_class", "").startswith("vllm."),
        "plugin on installs sm120fp4's config": on.get("modelopt_fp4_class", "").startswith("sm120fp4."),
    }
    report["checks"] = checks
    verdict = "pass" if all(checks.values()) else "fail"
    return finish(report, a.out, verdict)


def finish(report, out: Path, verdict: str) -> int:
    report["verdict"] = verdict
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("VERDICT", verdict, "->", out, flush=True)
    return 0 if verdict == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
