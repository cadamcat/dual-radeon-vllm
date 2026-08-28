#!/bin/bash
# Two cells, ctx=1024, one fresh container each. Routing does not depend on
# context length, so this is the cheapest shape that answers the question.
set -u
fail=0   # cells that came back non-zero
cd /data/50603
IMG=rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0
rm -f route027-*.txt route027-*.txt.err route-027.jsonl
for arm in stock widened; do
  echo "########## route arm=$arm  $(date -u +%H:%M:%S) ##########"
  sudo docker run --rm --name "r027-$arm" --device /dev/kfd --device /dev/dri \
    --group-add video --ipc host --shm-size 16g -v /data:/data -v /data/50603:/work \
    -e NCCL_P2P_DISABLE=1 -e HSA_ENABLE_SDMA=0 --entrypoint bash "$IMG" \
    -c "python -u /work/route_027.py $arm 1024 /work/route-027.jsonl" \
    > "/data/50603/route027-$arm.log" 2>&1
  rc=$?; fail=$((fail + (rc != 0)))
  echo "exit=$rc $(date -u +%H:%M:%S)"
  grep -aE "ARM=|ROUTES|^   pid=|RECORDER_ERRORS|Traceback|Error" "/data/50603/route027-$arm.log" | tail -12
done
# a DONE marker that a poller greps for must not appear when a cell failed
[ "$fail" -eq 0 ] || { echo "ROUTE_027_FAILED cells=$fail"; exit 1; }
echo ROUTE_027_DONE
