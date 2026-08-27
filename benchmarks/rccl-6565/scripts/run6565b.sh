#!/bin/bash
# Stage 2A arm runner: N cold inits of the reporter's truth test under an
# arbitrary env prefix, so the channel-count and transport axes can be swept.
# argv: N label KEY=VAL...
set -u
N="$1"; LABEL="$2"; shift 2
cd /work
echo "=== arm=$LABEL  env: $*  N=$N ==="
pass_n=0; fail_n=0; err_n=0
for i in $(seq 1 "$N"); do
  out=$(env "$@" torchrun --nproc-per-node=2 /work/rccl_allgather_truth.py 2>&1)
  verdict=$(echo "$out" | grep -o 'ALL CORRECT\|[0-9]* FAILING CASES' | tail -1)
  if [ "$verdict" = "ALL CORRECT" ]; then
    pass_n=$((pass_n+1))
  elif [ -n "$verdict" ]; then
    fail_n=$((fail_n+1)); echo "  run $i: >>> $verdict <<<"; echo "$out" | sed 's/^/      /'
  else
    err_n=$((err_n+1)); echo "  run $i: >>> NO VERDICT (init failed?) <<<"; echo "$out" | tail -20 | sed 's/^/      /'
  fi
done
echo "=== arm=$LABEL RESULT pass=$pass_n fail=$fail_n error=$err_n of $N ==="
# what did RCCL actually build for this arm?
env "$@" NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH torchrun --nproc-per-node=2 /work/rccl_allgather_truth.py 2>&1 \
  | grep -oE "Channel [0-9]+/[0-9]+|via [A-Za-z/]+|isAllDirectP2p [0-9]" | sort -u | sed 's/^/    /'
echo
