@echo off
REM ASCII ONLY. One-shot: prefill latency inside vLLM, three runs in one session: stock, the sm120fp4 backend with the CUTLASS hand-off (default), and with the slices.
wsl.exe -e bash -lc "cd /mnt/c/Users/jingz/oss/sm120-fp4 && PYTHONPATH=. ~/mlsys-5090-runtime/vllm028/.venv/bin/python scripts/vllm_prefill_latency.py --out reports/vllm-prefill-stock-20261007.json > reports/.vllm-prefill-20261007.log 2>&1; echo RUN_RC 0 $? >> reports/.vllm-prefill-20261007.log"
wsl.exe -u root -e bash -lc "sync; echo 3 > /proc/sys/vm/drop_caches"
wsl.exe -e bash -lc "cd /mnt/c/Users/jingz/oss/sm120-fp4 && SM120FP4_MOE=1 PYTHONPATH=. ~/mlsys-5090-runtime/vllm028/.venv/bin/python scripts/vllm_prefill_latency.py --out reports/vllm-prefill-sm120-cutlass-20261007.json >> reports/.vllm-prefill-20261007.log 2>&1; echo RUN_RC 1 $? >> reports/.vllm-prefill-20261007.log"
wsl.exe -u root -e bash -lc "sync; echo 3 > /proc/sys/vm/drop_caches"
wsl.exe -e bash -lc "cd /mnt/c/Users/jingz/oss/sm120-fp4 && SM120FP4_MOE=1 SM120FP4_PREFILL=slices PYTHONPATH=. ~/mlsys-5090-runtime/vllm028/.venv/bin/python scripts/vllm_prefill_latency.py --out reports/vllm-prefill-sm120-slices-20261007.json >> reports/.vllm-prefill-20261007.log 2>&1; echo RUN_RC 2 $? >> reports/.vllm-prefill-20261007.log"
wsl.exe -u root -e bash -lc "sync; echo 3 > /proc/sys/vm/drop_caches"
wsl.exe -e bash -lc "echo CHAIN_DONE >> /mnt/c/Users/jingz/oss/sm120-fp4/reports/.vllm-prefill-20261007.log"
