#!/bin/bash
# The reporter's Reproducer 1, verbatim commands, as a second witness that does
# not go through PyTorch at all. Their reported outcome on 2x W7800:
#   all_gather_perf -c 1  -> FAILED (#wrong ~ one rank's slot)
#   all_reduce_perf  -c 1 -> PASS
#   with HSA_DISABLE_CACHE=1 -> PASS
# -c 1 turns on rccl-tests' own correctness check.
set -u
cd /work/rccl-tests
run() {
  local label="$1"; shift
  echo "--- $label"
  echo "    \$ $*"
  out=$(env "$@" 2>&1)
  echo "$out" | grep -E "^ *[0-9]+ +[0-9]+ +(float|int)|Avg bus bandwidth|Errors|errors|Out of bounds|# Wrong|FAILED|failed" | head -12 | sed 's/^/    /'
  # rccl-tests prints "#wrong" per size; N/A means the check passed
  wrong=$(echo "$out" | awk '{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+e[+-][0-9]+$/ || $i=="N/A") last=$i} /float|int/{print last}' | sort -u | tr '\n' ' ')
  echo "    per-size error column: ${wrong:-<none parsed>}"
  if echo "$out" | grep -qiE "FAILED|Test failure"; then echo "    VERDICT: FAILED"; else echo "    VERDICT: PASS"; fi
}
run "all_gather  (the collective #6565 reports broken)" NCCL_P2P_DISABLE=1 ./build/all_gather_perf -b 8 -e 1M -f 4 -g 2 -c 1
run "reduce_scatter (also reported broken)"             NCCL_P2P_DISABLE=1 ./build/reduce_scatter_perf -b 8 -e 1M -f 4 -g 2 -c 1
run "all_reduce  (reported unaffected — control)"       NCCL_P2P_DISABLE=1 ./build/all_reduce_perf -b 8 -e 16M -f 4 -g 2 -c 1
run "reduce      (reported to HANG)"                    NCCL_P2P_DISABLE=1 ./build/reduce_perf -b 8 -e 1M -f 4 -g 2 -c 1
run "all_gather, P2P left enabled (their default path)"  ./build/all_gather_perf -b 8 -e 1M -f 4 -g 2 -c 1
run "all_gather + HSA_DISABLE_CACHE=1 (their cure)"     HSA_DISABLE_CACHE=1 NCCL_P2P_DISABLE=1 ./build/all_gather_perf -b 8 -e 1M -f 4 -g 2 -c 1
echo "STAGE2B_DONE"
