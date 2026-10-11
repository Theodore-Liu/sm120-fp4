#!/usr/bin/env bash
# Run each pytest node id of the given test files in its own process, writing a marker line to a Windows-side log
# before and after each, so a test that takes the WSL2 virtual machine down is named by the last "start" line.
#   bash scripts/run_tests_isolated.sh <marker log> <label> <pythonpath or -> <test file>...
set -u
LOG=$1; LABEL=$2; PP=$3; shift 3
PY=$HOME/fi071/.venv/bin/python
cd ${TEST_ROOT:-$HOME/gtests} || exit 2
if [ "$PP" != "-" ]; then export PYTHONPATH=$PP; fi
ids=$($PY -m pytest -q -p no:cacheprovider --collect-only "$@" 2>/dev/null | grep "::")
echo "$(date +%T) $LABEL collected $(echo "$ids" | wc -l)" >> "$LOG"
pass=0; fail=0
for id in $ids; do
  echo "$(date +%T) $LABEL start $id" >> "$LOG"
  if timeout 600 $PY -m pytest -q -p no:cacheprovider "$id" > /tmp/one.log 2>&1; then r=pass; pass=$((pass+1)); else r="FAIL rc=$?"; fail=$((fail+1)); fi
  echo "$(date +%T) $LABEL $r $id" >> "$LOG"
done
echo "$(date +%T) $LABEL done pass=$pass fail=$fail" >> "$LOG"
