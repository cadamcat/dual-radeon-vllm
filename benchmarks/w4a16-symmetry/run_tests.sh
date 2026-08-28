#!/bin/bash
# Two-way validation: the asymmetric tests must FAIL on the stock tree, proving
# the defect is there, and must PASS after the patch, proving it is fixed. A
# run that cannot tell those apart is worth nothing, so each stage asserts its
# direction and the script exits non-zero when one comes out the wrong way.
#
# Fixed 2026-08-28: `pytest ... | tail` then `$?` reads tail, which is always 0,
# so both directions were silently unchecked. The exit code is now captured
# before anything is piped.
set -u -o pipefail
IMG=rocm/vllm:rocm7.14.0_rdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0
sudo docker run --rm --name w4a16-tests --device /dev/kfd --device /dev/dri \
  --group-add video --ipc host --shm-size 8g -v /data/50603:/work \
  --entrypoint bash "$IMG" -c '
set -u
cd /work/tests
python -c "import pytest; print(\"pytest\", pytest.__version__)" || pip install -q pytest

run_expect () {   # want(fail|pass)  label  lines  pytest args...
  want=$1; label=$2; lines=$3; shift 3
  out=$(python -m pytest "$@" -q --no-header 2>&1); rc=$?
  printf "%s\n" "$out" | tail -"$lines"
  echo "---- $label exit=$rc ----"
  if [ "$want" = fail ] && [ "$rc" -eq 0 ]; then
    echo "UNEXPECTED: $label passed on the stock tree, so the defect is not reproduced"
    exit 1
  fi
  if [ "$want" = pass ] && [ "$rc" -ne 0 ]; then
    echo "UNEXPECTED: $label failed after the patch"
    exit 1
  fi
}

echo "######## STOCK: the new asymmetric tests must FAIL ########"
run_expect fail "stock selection" 12 test_rdna3_w4a16_selection.py -k asymmetric -x
run_expect fail "stock numeric"   12 test_rdna3_w4a16.py -k asymmetric
echo "######## applying the three-line patch ########"
python /work/apply_patch.py
echo "######## PATCHED: everything must PASS ########"
run_expect pass "patched selection" 6 test_rdna3_w4a16_selection.py
run_expect pass "patched numeric"   8 test_rdna3_w4a16.py
echo TESTS_DONE
'
rc=$?
echo "container exit=$rc"
exit $rc
