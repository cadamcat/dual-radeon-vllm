#!/bin/bash
# Both arms of vllm#54210's gsm8k evaluation, full 1 319 questions, sequential.
#
#   nohup bash /data/gsm8k-run/run_gsm8k_both.sh &
#
# Full rather than --limit N, and the reviewer's unanswered question about a
# limit does not block it: lm_eval --limit N takes the first N documents in
# order, so the first N per-document results of a full run ARE the --limit N
# score. A full run answers every limit he might name; a limited one answers
# only its own.
#
# Stock first. Its "the tree is pristine" assertion and its use_custom=False
# verdict are the two that had never been exercised before 2026-09-07 14:16.
set -u
H=/data/gsm8k-run
cd $H
echo "$(date -u +%H:%M:%S) | ===== BOTH ARMS, full gsm8k =====" >> $H/BOTH.txt
for ARM in stock widened; do
  echo "$(date -u +%H:%M:%S) | -> $ARM" >> $H/BOTH.txt
  bash run_gsm8k_arm.sh "$ARM" > /dev/null 2>&1
  rc=$?
  echo "$(date -u +%H:%M:%S) | <- $ARM exit=$rc" >> $H/BOTH.txt
  if [ $rc -ne 0 ]; then
    echo "$(date -u +%H:%M:%S) | STOPPING: $ARM failed, second arm not run" >> $H/BOTH.txt
    exit $rc
  fi
done
echo "$(date -u +%H:%M:%S) | ===== BOTH ARMS DONE =====" >> $H/BOTH.txt
