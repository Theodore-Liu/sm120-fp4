@echo off
REM ASCII ONLY. Nightly CI on the SM120 machine (adoption item 3, docs/ci-plan.md): the suite in a dedicated checkout under WSL, then the
REM report and the README line committed and pushed with the Windows git (WSL's git has no identity and no push credential).
wsl.exe -e bash -lc "cd /mnt/c/Users/jingz/oss/sm120-fp4 && ~/mlsys-5090-runtime/vllm028/.venv/bin/python scripts/ci_nightly.py --stage-only >> /mnt/c/Users/jingz/oss/sm120-fp4/reports/ci/.nightly.log 2>&1; echo CI_RC=$? >> /mnt/c/Users/jingz/oss/sm120-fp4/reports/ci/.nightly.log"
if exist C:\Users\jingz\oss\sm120-fp4\reports\ci\.commit_msg (
  "C:\Program Files\Git\cmd\git.exe" -C C:\Users\jingz\oss\sm120-fp4 commit -q -F reports/ci/.commit_msg -- reports/ci README.md >> C:\Users\jingz\oss\sm120-fp4\reports\ci\.nightly.log 2>&1
  "C:\Program Files\Git\cmd\git.exe" -C C:\Users\jingz\oss\sm120-fp4 push -q origin master >> C:\Users\jingz\oss\sm120-fp4\reports\ci\.nightly.log 2>&1
  echo WIN_PUSH_RC=%ERRORLEVEL% >> C:\Users\jingz\oss\sm120-fp4\reports\ci\.nightly.log
  del C:\Users\jingz\oss\sm120-fp4\reports\ci\.commit_msg
)
