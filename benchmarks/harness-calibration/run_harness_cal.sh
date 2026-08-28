#!/bin/bash
# One container, one engine, three depths. The campaign's env and knobs.
set -u
fail=0   # cells that came back non-zero
cd /data/50603
[ -e harness-cal.jsonl ] && { echo "REFUSING: output exists"; exit 2; }
IMG=rocm/vllm:rocm7.14.0_rdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0
echo "########## harness calibration  $(date -u +%H:%M:%S) ##########"
sudo docker run --rm --name hcal --device /dev/kfd --device /dev/dri \
  --group-add video --ipc host --shm-size 16g -v /data:/data -v /data/50603:/work \
  -e VLLM_CLONE_MMAP=1 -e NCCL_P2P_DISABLE=1 -e HSA_ENABLE_SDMA=0 \
  --entrypoint bash "$IMG" -c "python -u /work/probe_harness_cal.py /work/harness-cal.jsonl" \
  > /data/50603/hcal.log 2>&1
rc=$?; fail=$((fail + (rc != 0)))
echo "exit=$rc $(date -u +%H:%M:%S)"
grep -aE "^vllm=|RESULT|Traceback|Error" /data/50603/hcal.log | tail -8
# a DONE marker that a poller greps for must not appear when a cell failed
[ "$fail" -eq 0 ] || { echo "HCAL_FAILED cells=$fail"; exit 1; }
echo HCAL_DONE
