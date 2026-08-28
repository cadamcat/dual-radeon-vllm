#!/bin/bash
# Hybrid vs patched RDNA3 on the same AWQ checkpoint, both on 0.27.0.
set -u
IMG=rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0
for arm in hybrid rdna3; do
  echo "########## arm=$arm  $(date -u +%H:%M:%S) ##########"
  sudo docker run --rm --name "a027-$arm" --device /dev/kfd --device /dev/dri \
    --group-add video --ipc host --shm-size 16g -v /data:/data -v /data/50603:/work \
    -w /work -e NCCL_P2P_DISABLE=1 -e HSA_ENABLE_SDMA=0 -e MAX_NUM_SEQS=16 --entrypoint python "$IMG" \
    -u /work/probe_027_ab.py "$arm" /work/w4a16-027.jsonl \
    > "/data/50603/a027-$arm.log" 2>&1
  echo "exit=$? arm=$arm"
  grep -aE "^PATCH|^ARM=|^RESULT|^ANSWER|^KERNELS|^CELL_DONE|Error|Traceback|AssertionError" \
    "/data/50603/a027-$arm.log" | tail -6
done
echo A027_DONE
