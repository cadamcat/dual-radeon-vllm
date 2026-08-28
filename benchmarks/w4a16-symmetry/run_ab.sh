#!/bin/bash
# Fresh container per cell: probe_w4a16_ab.py patches the kernel-selection
# source on disk, so a shared container would carry the instrumentation across
# cells (harmless) and any future gate edit across arms (not harmless).
set -u
fail=0   # cells that came back non-zero
IMG=rocm/vllm:rocm7.14.0_rdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0
for ctx in 1024 8192 32768; do
  for arm in asym sym; do
    echo "########## arm=$arm ctx=$ctx  $(date -u +%H:%M:%S) ##########"
    sudo docker run --rm --name "ab-$arm-$ctx" --device /dev/kfd --device /dev/dri \
      --group-add video --ipc host --shm-size 16g -v /data:/data -v /data/50603:/work \
      -e NCCL_P2P_DISABLE=1 -e HSA_ENABLE_SDMA=0 --entrypoint python "$IMG" \
      -u /work/probe_w4a16_ab.py "$arm" "$ctx" /work/w4a16-ab.jsonl \
      > "/data/50603/ab-$arm-$ctx.log" 2>&1
    rc=$?; fail=$((fail + (rc != 0)))
    echo "exit=$rc arm=$arm ctx=$ctx"
    grep -E "ARM=|RESULT|KERNELS|Error|Traceback" "/data/50603/ab-$arm-$ctx.log" | tail -6
  done
done
# a DONE marker that a poller greps for must not appear when a cell failed
[ "$fail" -eq 0 ] || { echo "AB_FAILED cells=$fail"; exit 1; }
echo AB_DONE
