#!/bin/bash
# Repeat pass. One deliberate change from run A: the arms run in the opposite
# order within each context. Run A always measured stock before widened, so a
# monotonic drift over the 40 minutes -- the cards warming, say -- would land
# on the widened arm every time. Flipping the order puts that drift on the
# other arm; an effect that survives both orders is not drift.
set -u
cd /data/50603
[ -e stage3-027b.jsonl ] && { echo "REFUSING: stage3-027b.jsonl already exists"; exit 2; }
IMG=rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0
for ctx in 1024 8192 32768; do
  for arm in widened stock; do
    echo "########## B arm=$arm ctx=$ctx  $(date -u +%H:%M:%S) ##########"
    sudo docker run --rm --name "s3027b-$arm-$ctx" --device /dev/kfd --device /dev/dri \
      --group-add video --ipc host --shm-size 16g -v /data:/data -v /data/50603:/work \
      -e NCCL_P2P_DISABLE=1 -e HSA_ENABLE_SDMA=0 --entrypoint bash "$IMG" \
      /work/stage3_027b.sh "$arm" "$ctx" > "/data/50603/cell027b-$arm-$ctx.log" 2>&1
    echo "exit=$? $(date -u +%H:%M:%S)"
    grep -aE "ARM=|RESULT|Error|AssertionError|Traceback" "/data/50603/cell027b-$arm-$ctx.log" | tail -4
  done
done
echo STAGE3_027B_DONE
