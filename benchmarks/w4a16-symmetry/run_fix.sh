#!/bin/bash
# fixed        = all three changes
# layout_only  = changes 1 and 2, use_v2_format left at False. The control that
#                shows step 3 is load-bearing rather than decorative.
set -u
IMG=rocm/vllm:rocm7.14.0_rdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0
for arm in fixed layout_only; do
  echo "########## arm=$arm  $(date -u +%H:%M:%S) ##########"
  sudo docker run --rm --name "fix-$arm" --device /dev/kfd --device /dev/dri \
    --group-add video --ipc host --shm-size 16g -v /data:/data -v /data/50603:/work \
    -e NCCL_P2P_DISABLE=1 -e HSA_ENABLE_SDMA=0 --entrypoint python "$IMG" \
    -u /work/probe_w4a16_fix.py "$arm" /work/w4a16-fix.jsonl \
    > "/data/50603/fix-$arm.log" 2>&1
  echo "exit=$? arm=$arm"
  grep -aE "^PATCH|^RESULT|^ANSWER|^KERNELS|^FIX_CELL_DONE|Error|Traceback|assert" \
    "/data/50603/fix-$arm.log" | tail -6
done
echo FIX_DONE
