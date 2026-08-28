#!/bin/bash
# Stage 3 on the 0.27 image. Same probe, same env flags, same six cells as the
# 0.23.1 run; only the image changes, so the two sweeps are comparable. The
# probe appends, and its route files append too, so the 0.23.1 route records
# are moved aside first rather than written into.
set -u
cd /data/50603
mkdir -p routes-023
for f in route-stock-*.txt route-widened-*.txt; do
  [ -e "$f" ] && mv "$f" routes-023/ && echo "preserved $f"
done
[ -e stage3-027.jsonl ] && { echo "REFUSING: stage3-027.jsonl already exists"; exit 2; }
IMG=rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0
for ctx in 1024 8192 32768; do
  for arm in stock widened; do
    echo "########## arm=$arm ctx=$ctx  $(date -u +%H:%M:%S) ##########"
    sudo docker run --rm --name "s3027-$arm-$ctx" --device /dev/kfd --device /dev/dri \
      --group-add video --ipc host --shm-size 16g -v /data:/data -v /data/50603:/work \
      -e NCCL_P2P_DISABLE=1 -e HSA_ENABLE_SDMA=0 --entrypoint bash "$IMG" \
      /work/stage3_027.sh "$arm" "$ctx" > "/data/50603/cell027-$arm-$ctx.log" 2>&1
    echo "exit=$? $(date -u +%H:%M:%S)"
    grep -aE "ARM=|RESULT|SAMPLE|Error|AssertionError|Traceback" "/data/50603/cell027-$arm-$ctx.log" | tail -6
  done
done
mkdir -p routes-027
for f in route-stock-*.txt route-widened-*.txt; do
  [ -e "$f" ] && mv "$f" routes-027/
done
echo STAGE3_027_DONE
