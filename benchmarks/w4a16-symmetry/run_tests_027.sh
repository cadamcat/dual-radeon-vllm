#!/bin/bash
# The same before/after test run as on 0.23.1, but on the 0.27.0 image, so the
# patch is validated against the version it would actually be proposed for.
set -u
IMG=rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0
sudo docker run --rm --name w4a16-tests-027 --device /dev/kfd --device /dev/dri \
  --group-add video --ipc host --shm-size 8g -v /data/50603:/work \
  --entrypoint bash "$IMG" -c '
set -u
cd /work/tests027
python -c "import vllm, pytest; print(\"vllm\", vllm.__version__, \"pytest\", pytest.__version__)" || pip install -q pytest
echo "######## STOCK 0.27.0: the asymmetric tests must FAIL ########"
python -m pytest test_rdna3_w4a16_selection.py -k asymmetric -q --no-header 2>&1 | tail -6
python -m pytest test_rdna3_w4a16.py -k asymmetric -q --no-header 2>&1 | tail -6
echo "######## applying the three-line patch ########"
python /work/tests027/apply_patch.py
echo "######## PATCHED 0.27.0: everything must PASS ########"
python -m pytest test_rdna3_w4a16_selection.py -q --no-header 2>&1 | tail -5
python -m pytest test_rdna3_w4a16.py -q --no-header 2>&1 | tail -6
echo TESTS027_DONE
'
