#!/bin/bash
set -u
IMG=rocm/vllm:rocm7.14.0_rdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0
sudo docker run --rm --name w4a16-tests --device /dev/kfd --device /dev/dri \
  --group-add video --ipc host --shm-size 8g -v /data/50603:/work \
  --entrypoint bash "$IMG" -c '
set -u
cd /work/tests
python -c "import pytest; print(\"pytest\", pytest.__version__)" || pip install -q pytest
echo "######## STOCK: the new asymmetric tests must FAIL ########"
python -m pytest test_rdna3_w4a16_selection.py -k asymmetric -q -x --no-header 2>&1 | tail -12
echo "---- stock selection exit=$? ----"
python -m pytest test_rdna3_w4a16.py -k asymmetric -q --no-header 2>&1 | tail -12
echo "---- stock numeric exit=$? ----"
echo "######## applying the three-line patch ########"
python /work/apply_patch.py
echo "######## PATCHED: everything must PASS ########"
python -m pytest test_rdna3_w4a16_selection.py -q --no-header 2>&1 | tail -6
echo "---- patched selection exit=$? ----"
python -m pytest test_rdna3_w4a16.py -q --no-header 2>&1 | tail -8
echo "---- patched numeric exit=$? ----"
echo TESTS_DONE
'
