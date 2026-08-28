#!/bin/bash
# Arm runner for the cross-rank variant. Same shape as run6565b.sh so the two
# sweeps are directly comparable, with two differences that are the point:
# rccl_allgather_allranks.py exits non-zero when any rank saw a failure, so the
# exit status is read as well as the printed verdict and the two are required to
# agree; and both ranks must have printed their own local count, or the
# cross-rank property was not exercised and the run is an error rather than a
# pass. Each run is classified exactly once.
#
# The two ranks' prints often land on the same output line, so occurrences are
# counted rather than lines -- counting lines reported 6 of 20 as one-sided on
# a first attempt, and both ranks had in fact reported.
# argv: N label KEY=VAL...
set -u
N="$1"; LABEL="$2"; shift 2
# overridable so the gate check below can point it at a deliberately broken copy
SCRIPT="${ALLRANKS_SCRIPT:-/work/rccl_allgather_allranks.py}"
cd /work
echo "=== arm=$LABEL  env: $*  N=$N  script=$SCRIPT ==="
pass_n=0; fail_n=0; err_n=0
for i in $(seq 1 "$N"); do
  out=$(env "$@" torchrun --nproc-per-node=2 "$SCRIPT" 2>&1)
  rc=$?
  verdict=$(echo "$out" | grep -o 'ALL CORRECT ON EVERY RANK\|[0-9]* FAILING CASES' | tail -1)
  seen=$(echo "$out" | grep -o "failing cases locally" | wc -l | tr -d ' ')
  if [ "$verdict" = "ALL CORRECT ON EVERY RANK" ] && [ "$rc" -eq 0 ] && [ "$seen" -eq 2 ]; then
    pass_n=$((pass_n+1))
  elif [ -n "$verdict" ] && [ "$verdict" != "ALL CORRECT ON EVERY RANK" ] && [ "$rc" -ne 0 ]; then
    fail_n=$((fail_n+1)); echo "  run $i: >>> $verdict (exit $rc) <<<"; echo "$out" | sed 's/^/      /'
  else
    err_n=$((err_n+1))
    echo "  run $i: >>> verdict='$verdict' exit=$rc ranks_reporting=$seen — inconsistent <<<"
    echo "$out" | tail -25 | sed 's/^/      /'
  fi
done
echo "=== arm=$LABEL RESULT pass=$pass_n fail=$fail_n error=$err_n of $N ==="
[ "$((pass_n + fail_n + err_n))" -eq "$N" ] || { echo "  runner defect: tallies do not sum to $N"; exit 2; }
[ "$fail_n" -eq 0 ] && [ "$err_n" -eq 0 ]
