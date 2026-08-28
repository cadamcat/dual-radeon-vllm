#!/bin/bash
# Qwen3.8-27B on 0.27, stock against vllm#45916, three depths. One fresh
# container per cell so the file swap cannot leak between arms.
set -u
cd /data/50603
[ -e qwen38-027-depth.jsonl ] && { echo "REFUSING: output already exists"; exit 2; }
rm -f route38-*.txt route38-*.txt.err
IMG=rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0
for ctx in 1024 8192 32768; do
  for arm in stock splitkv; do
    echo "########## arm=$arm ctx=$ctx  $(date -u +%H:%M:%S) ##########"
    sudo docker run --rm --name "q38-$arm-$ctx" --device /dev/kfd --device /dev/dri \
      --group-add video --ipc host --shm-size 16g -v /data:/data -v /data/50603:/work \
      -e NCCL_P2P_DISABLE=1 -e HSA_ENABLE_SDMA=0 --entrypoint bash "$IMG" \
      -c "python -u /work/probe_027_depth.py $arm $ctx /work/qwen38-027-depth.jsonl" \
      > "/data/50603/q38-$arm-$ctx.log" 2>&1
    echo "exit=$? $(date -u +%H:%M:%S)"
    grep -aE "^ARM=|RESULT|RECORDER_ERRORS|AssertionError|Traceback|Error" \
      "/data/50603/q38-$arm-$ctx.log" | tail -5
  done
done
echo Q38_DEPTH_DONE
