@echo off
REM ASCII ONLY. One-shot: model-level greedy comparison, hand-off enabled (run 2, with its log lines) and the slices, same day.
wsl.exe -e bash -lc "cd /mnt/c/Users/jingz/oss/sm120-fp4 && SM120FP4_MOE=1 SM120FP4_PREFILL=cutlass PYTHONPATH=. ~/mlsys-5090-runtime/vllm028/.venv/bin/python scripts/vllm_model_compare.py run --out reports/vllm-compare-sm120-prefill-cutlass-20261007-run2.json > reports/.vllm-compare-prefill-20261007-run2.log 2>&1; echo RUN_RC $? >> reports/.vllm-compare-prefill-20261007-run2.log"
wsl.exe -u root -e bash -lc "sync; echo 3 > /proc/sys/vm/drop_caches"
wsl.exe -e bash -lc "cd /mnt/c/Users/jingz/oss/sm120-fp4 && SM120FP4_MOE=1 PYTHONPATH=. ~/mlsys-5090-runtime/vllm028/.venv/bin/python scripts/vllm_model_compare.py run --out reports/vllm-compare-sm120-slices-20261007.json > reports/.vllm-compare-slices-20261007.log 2>&1; echo RUN_RC $? >> reports/.vllm-compare-slices-20261007.log"
wsl.exe -u root -e bash -lc "sync; echo 3 > /proc/sys/vm/drop_caches"
wsl.exe -e bash -lc "echo CHAIN_DONE >> /mnt/c/Users/jingz/oss/sm120-fp4/reports/.vllm-compare-slices-20261007.log"
