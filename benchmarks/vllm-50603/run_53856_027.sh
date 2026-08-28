#!/bin/bash
# Two arms: the image as shipped, and the same image with the PR's _rocm_C.
# The stock arm must reproduce the NaN; a clean patched arm alone proves nothing.
set -u
IMG=rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0
for arm in stock patched; do
  echo "########## arm=$arm  $(date -u +%H:%M:%S) ##########"
  if [ "$arm" = patched ]; then
    PRE='cp /work/rocm_C_53856.so /opt/python/lib/python3.14/site-packages/vllm/_rocm_C.abi3.so &&'
  else
    PRE=''
  fi
  sudo docker run --rm --name "v53856-$arm" --device /dev/kfd --device /dev/dri \
    --group-add video --ipc host --shm-size 8g -v /data/50603:/work -w /work \
    --entrypoint bash "$IMG" -c "$PRE python -u /work/probe_53856_027.py $arm /work/53856-027-$arm.jsonl" \
    > "/data/50603/v53856-$arm.log" 2>&1
  echo "exit=$? arm=$arm"
  grep -aE "^device=|PROBE_DONE|Traceback|Error" "/data/50603/v53856-$arm.log" | tail -3
done
echo V53856_DONE
