@echo off
REM ASCII ONLY. One-shot: BACKLOG item 4's clean-venv plugin install test under WSL (stock vllm 0.28 from PyPI + pip install -e of this checkout), logged.
wsl.exe -e bash -lc "export PATH=$HOME/.local/bin:$PATH; cd /mnt/c/Users/jingz/oss/sm120-fp4 && ~/mlsys-5090-runtime/vllm028/.venv/bin/python scripts/plugin_install_test.py --venv ~/sm120-plugin-test --out reports/plugin-install-test-20261003.json >> reports/.plugin-install-test-20261003.log 2>&1; echo TEST_RC=$? >> reports/.plugin-install-test-20261003.log"
