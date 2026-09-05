#!/bin/bash
# The three commands of one row, in order, detached; stops at the first that
# does not end in ROW-DONE. Progress goes to run-<row>.log; the last line is
# ROW1-DONE or ROW1-FAILED.
ROW="${1:-atomics_present}"
cd /data/rccl-build/pa || exit 2
L=/data/rccl-build/pa/run-$ROW.log
: > "$L"
for step in "pa_row.sh $ROW" "serve_row.sh $ROW default" "serve_row.sh $ROW triton"; do
  echo "=== $(date -u +%H:%M:%S) START $step ===" >> "$L"
  bash $step >> "$L" 2>&1; rc=$?
  echo "=== $(date -u +%H:%M:%S) END $step rc=$rc ===" >> "$L"
  [ "$rc" -eq 0 ] || { echo "ROW1-FAILED at: $step rc=$rc" >> "$L"; exit "$rc"; }
done
echo "ROW1-DONE row=$ROW" >> "$L"
