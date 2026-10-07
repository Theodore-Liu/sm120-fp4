@echo off
REM ASCII ONLY. One-shot: model-level greedy comparison with the plugin defaults (no SM120FP4_PREFILL set: the CUTLASS prefill hand-off).
wsl.exe -e bash -lc "cd /mnt/c/Users/jingz/oss/sm120-fp4 && SM120FP4_MOE=1 PYTHONPATH=. ~/mlsys-5090-runtime/vllm028/.venv/bin/python scripts/vllm_model_compare.py run --out reports/vllm-compare-sm120-default-20261007.json > reports/.vllm-compare-default-20261007.log 2>&1; echo RUN_RC $? >> reports/.vllm-compare-default-20261007.log"
wsl.exe -u root -e bash -lc "sync; echo 3 > /proc/sys/vm/drop_caches"
wsl.exe -e bash -lc "echo CHAIN_DONE >> /mnt/c/Users/jingz/oss/sm120-fp4/reports/.vllm-compare-default-20261007.log"
