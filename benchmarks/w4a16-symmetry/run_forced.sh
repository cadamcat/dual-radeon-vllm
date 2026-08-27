#!/bin/bash
# One asymmetric checkpoint, two kernels. The patched arm may fail outright --
# that is a result, not an accident, so keep going and let the log hold it.
set -u
IMG=rocm/vllm:rocm7.14.0_rdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0
for arm in stock patched; do
  echo "########## arm=$arm  $(date -u +%H:%M:%S) ##########"
  sudo docker run --rm --name "forced-$arm" --device /dev/kfd --device /dev/dri \
    --group-add video --ipc host --shm-size 16g -v /data:/data -v /data/50603:/work \
    -e NCCL_P2P_DISABLE=1 -e HSA_ENABLE_SDMA=0 --entrypoint python "$IMG" \
    -u /work/probe_w4a16_forced.py "$arm" /work/w4a16-forced.jsonl \
    > "/data/50603/forced-$arm.log" 2>&1
  echo "exit=$? arm=$arm"
  grep -aE "^PATCH|^ARM=|^RESULT|^ANSWER|^KERNELS|^FORCED_CELL_DONE|Error|Traceback|assert" \
    "/data/50603/forced-$arm.log" | tail -8
done
echo FORCED_DONE
