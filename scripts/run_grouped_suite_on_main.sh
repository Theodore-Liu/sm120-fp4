#!/usr/bin/env bash
# Owed item 2: FlashInfer's tests/grouped_mm against an editable build of FlashInfer main + the fix commit
# (~/fimain installed from the ext4 clone ~/fi-src, see build_flashinfer_main_venv.sh), first with the fix, then
# without it (the fix commit's parent version of the runner file, restored from git afterwards). One process per
# test via run_tests_isolated.sh, resumable with the same log.
set -u
S=$HOME/fi-src
R=/mnt/c/Users/jingz/oss/sm120-fp4
LOG=$R/reports/grouped-sfa-suites-main-rtx5090-20261011.log
F=flashinfer/grouped_mm/cudnn/core.py
export PATH=/usr/local/cuda-13.0/bin:$PATH CUDA_HOME=/usr/local/cuda-13.0 PY=$HOME/fimain/bin/python
# Tests run from the clone itself: it is the installed (editable) package, and main's conftest needs tests/test_helpers.
T=$S
echo "$(date +%T) source $(git -C $S log --oneline -1)" >> $LOG
if ! grep -q "main-fixed done" $LOG 2>/dev/null; then
  git -C $S diff --quiet HEAD -- $F || git -C $S checkout HEAD -- $F
  TEST_ROOT=$T bash $R/scripts/run_tests_isolated.sh $LOG main-fixed - tests/grouped_mm/test_grouped_mm_fp4.py tests/grouped_mm/test_grouped_mm_mxfp8.py
fi
git -C $S show HEAD~1:$F > $S/$F
echo "$(date +%T) runner file set to the parent commit (no fix)" >> $LOG
TEST_ROOT=$T bash $R/scripts/run_tests_isolated.sh $LOG main-unfixed - tests/grouped_mm/test_grouped_mm_fp4.py tests/grouped_mm/test_grouped_mm_mxfp8.py
git -C $S checkout HEAD -- $F && git -C $S diff --quiet HEAD -- $F && echo "$(date +%T) runner file restored to HEAD" >> $LOG
echo SUITE_RC=0 >> $LOG
