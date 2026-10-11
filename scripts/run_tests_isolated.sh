#!/usr/bin/env bash
# Run each pytest node id of the given test files in its own process, writing a marker line to a Windows-side log
# before and after each, so a test that takes the WSL2 virtual machine down is named by the last "start" line.
#   bash scripts/run_tests_isolated.sh <marker log> <label> <pythonpath or -> <test file>...
# Re-running with the same log and label resumes where a dropped run stopped.
set -u
LOG=$1; LABEL=$2; PP=$3; shift 3
PY=$HOME/fi071/.venv/bin/python
cd ${TEST_ROOT:-$HOME/gtests} || exit 2
if [ "$PP" != "-" ]; then export PYTHONPATH=$PP; fi
ids=$($PY -m pytest -q -p no:cacheprovider --collect-only "$@" 2>/dev/null | grep "::")
echo "$(date +%T) $LABEL collected $(echo "$ids" | wc -l)" >> "$LOG"
pass=0; fail=0
for id in $ids; do
  # resumable: a test already recorded as pass or FAIL under this label is not run again (WSL may drop mid-run)
  if awk -v L="$LABEL" -v I="$id" '$2==L && ($3=="pass" || $3=="FAIL") && $NF==I {f=1} END {exit !f}' "$LOG" 2>/dev/null; then continue; fi
  echo "$(date +%T) $LABEL start $id" >> "$LOG"
  if timeout 600 $PY -m pytest -q -p no:cacheprovider "$id" > /tmp/one.log 2>&1; then r=pass; pass=$((pass+1)); else r="FAIL rc=$?"; fail=$((fail+1)); fi
  echo "$(date +%T) $LABEL $r $id" >> "$LOG"
done
echo "$(date +%T) $LABEL done (this pass) pass=$pass fail=$fail; totals: pass=$(awk -v L="$LABEL" '$2==L && $3=="pass"' "$LOG" | sort -u | wc -l) fail=$(awk -v L="$LABEL" '$2==L && $3=="FAIL"' "$LOG" | wc -l)" >> "$LOG"
