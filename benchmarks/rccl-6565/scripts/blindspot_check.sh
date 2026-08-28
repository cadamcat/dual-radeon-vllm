#!/bin/bash
# Deliberately break one thing and confirm the harness notices.
#
# Injects corruption that only rank 1 can see -- exactly the shape the
# reporter's script cannot count -- into a copy of each script, one line after
# the bad_list computation, and runs both once. The originals are untouched.
#
# Expected, and asserted:
#   rccl_allgather_truth.py  -> "ALL CORRECT", exit 0   (the blind spot)
#   rccl_allgather_allranks.py -> "1 FAILING CASES", exit 1
#
# If the two ever agree, either the injection missed or the new script does not
# do what it claims, and this exits non-zero rather than reporting a result.
set -u
cd /work
INJ='        if rank == 1 and n == 64 and dt is torch.float32: bad_list = [0]  # INJECTED'

python3 - "$INJ" <<'PY'
import sys
inj = sys.argv[1]
anchor = "bad_list = [r for r in range(world) if not (parts[r] == float(r)).all().item()]"
for src, dst in (("/work/rccl_allgather_truth.py", "/tmp/bs_truth.py"),
                 ("/work/rccl_allgather_allranks.py", "/tmp/bs_allranks.py")):
    lines = open(src).read().split("\n")
    hits = [i for i, l in enumerate(lines) if anchor in l]
    assert len(hits) == 1, f"{src}: anchor found {len(hits)} times, expected 1"
    lines.insert(hits[0] + 1, inj)
    open(dst, "w").write("\n".join(lines))
    print(f"injected into {dst} after line {hits[0]+1}")
PY
[ $? -eq 0 ] || { echo "BLINDSPOT_CHECK_FAILED injection did not apply"; exit 1; }

echo "--- reporter's script, with rank-1-only corruption injected ---"
out_t=$(NCCL_P2P_DISABLE=1 torchrun --nproc-per-node=2 /tmp/bs_truth.py 2>&1); rc_t=$?
echo "$out_t" | tail -4 | sed 's/^/    /'
echo "    exit=$rc_t"

echo "--- cross-rank variant, same injection ---"
out_a=$(NCCL_P2P_DISABLE=1 torchrun --nproc-per-node=2 /tmp/bs_allranks.py 2>&1); rc_a=$?
echo "$out_a" | tail -6 | sed 's/^/    /'
echo "    exit=$rc_a"

ok=1
echo "$out_t" | grep -q "ALL CORRECT" || { echo "  UNEXPECTED: the reporter's script did not print ALL CORRECT"; ok=0; }
[ "$rc_t" -eq 0 ] || { echo "  UNEXPECTED: the reporter's script exited $rc_t"; ok=0; }
echo "$out_a" | grep -q "FAILING CASES" || { echo "  UNEXPECTED: the cross-rank variant did not print FAILING CASES"; ok=0; }
[ "$rc_a" -ne 0 ] || { echo "  UNEXPECTED: the cross-rank variant exited 0"; ok=0; }

echo "--- and once through the arm runner, which is what the sweep actually uses ---"
out_r=$(ALLRANKS_SCRIPT=/tmp/bs_allranks.py /work/run6565_allranks.sh 1 gatecheck NCCL_P2P_DISABLE=1 2>&1); rc_r=$?
echo "$out_r" | grep -E "RESULT|>>>" | sed 's/^/    /'
echo "    runner exit=$rc_r"
echo "$out_r" | grep -q "RESULT pass=0 fail=1 error=0 of 1" || { echo "  UNEXPECTED: the arm runner did not tally this as one failure"; ok=0; }
[ "$rc_r" -ne 0 ] || { echo "  UNEXPECTED: the arm runner exited 0 on a failing run"; ok=0; }

[ "$ok" -eq 1 ] || { echo "BLINDSPOT_CHECK_FAILED"; exit 1; }
echo "BLINDSPOT_CHECK_OK  one-sided corruption: truth=ALL CORRECT/exit 0, allranks=FAILING/exit $rc_a"
