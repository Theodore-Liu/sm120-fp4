@echo off
REM ASCII ONLY. One-shot: FlashInfer main (grouped block-scale fix worktree) editable build in ~/fimain, then tests/grouped_mm, logged.
wsl.exe -e bash -lc "bash /mnt/c/Users/jingz/oss/sm120-fp4/scripts/build_flashinfer_main_venv.sh >> /mnt/c/Users/jingz/oss/sm120-fp4/reports/.fimain-build.log 2>&1; echo EXIT=$? >> /mnt/c/Users/jingz/oss/sm120-fp4/reports/.fimain-build.log"
