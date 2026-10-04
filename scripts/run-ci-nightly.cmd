@echo off
REM ASCII ONLY. Nightly CI on the SM120 machine (adoption item 3, docs/ci-plan.md): the suite in a dedicated checkout, the report pushed.
wsl.exe -e bash -lc "cd /mnt/c/Users/jingz/oss/sm120-fp4 && ~/mlsys-5090-runtime/vllm028/.venv/bin/python scripts/ci_nightly.py --push >> /mnt/c/Users/jingz/oss/sm120-fp4/reports/ci/.nightly.log 2>&1; echo CI_RC=$? >> /mnt/c/Users/jingz/oss/sm120-fp4/reports/ci/.nightly.log"
