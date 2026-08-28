#!/bin/bash
# The same before/after test run as on 0.23.1, but on the 0.27.0 image, so the
# patch is validated against the version it would actually be proposed for.
#
# Fixed 2026-08-28: this script had no exit-code check at all -- not even the
# broken `$?`-after-a-pipe that run_tests.sh had -- so nothing distinguished a
# passing patched run from a failing one. Same two-way assertion as there.
set -u -o pipefail
IMG=rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0
sudo docker run --rm --name w4a16-tests-027 --device /dev/kfd --device /dev/dri \
  --group-add video --ipc host --shm-size 8g -v /data/50603:/work \
  --entrypoint bash "$IMG" -c '
set -u
cd /work/tests027
python -c "import vllm, pytest; print(\"vllm\", vllm.__version__, \"pytest\", pytest.__version__)" || pip install -q pytest

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

echo "######## STOCK 0.27.0: the asymmetric tests must FAIL ########"
run_expect fail "stock selection" 6 test_rdna3_w4a16_selection.py -k asymmetric
run_expect fail "stock numeric"   6 test_rdna3_w4a16.py -k asymmetric
echo "######## applying the three-line patch ########"
python /work/tests027/apply_patch.py
echo "######## PATCHED 0.27.0: everything must PASS ########"
run_expect pass "patched selection" 5 test_rdna3_w4a16_selection.py
run_expect pass "patched numeric"   6 test_rdna3_w4a16.py
echo TESTS027_DONE
'
rc=$?
echo "container exit=$rc"
exit $rc
