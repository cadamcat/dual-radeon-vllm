#!/bin/bash
# Stage 1 re-run on the 0.27 image. The question is whether the gfx11 gate's
# "gqa_ratio >= 3" still sits where the CK-vs-Triton crossover is; on 0.23.1
# CK won at every ratio from 1 to 4, which is what makes the 3 look arbitrary.
# The probe is byte-identical to the one that produced the 0.23.1 rows, so the
# two sweeps are directly comparable. Two rounds, to see the noise.
set -u
IMG=rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0
for r in 1 2; do
  echo "########## round $r  $(date -u +%H:%M:%S) ##########"
  sudo docker run --rm --name "s1-027-r$r" --device /dev/kfd --device /dev/dri \
    --group-add video --ipc host --shm-size 8g -v /data/50603:/work -w /work \
    --entrypoint bash "$IMG" -c "python -u /work/probe_50603.py /work/stage1-027-r$r.jsonl" \
    > "/data/50603/s1-027-r$r.log" 2>&1
  echo "exit=$? round=$r $(date -u +%H:%M:%S)"
  tail -2 "/data/50603/s1-027-r$r.log"
done
echo S1_027_DONE
