#!/bin/bash
# Repeat pass for the Qwen3.8-27B depth sweep, with two deliberate changes from
# run A. First, a discarded warm-up cell: the harness calibration found the
# first run after the GPU services stop reading up to 31% low, converging only
# by the third, so cell zero is thrown away rather than averaged in. Second,
# the arms run in the opposite order within each depth, so drift over the hour
# cannot land on the same arm both times.
#
# Everything else is run A's method unchanged -- one fresh container per cell,
# max_model_len = ctx + 512 per cell -- because changing it would cost the
# comparison.
set -u
cd /data/50603
[ -e qwen38-027-depth-b.jsonl ] && { echo "REFUSING: output exists"; exit 2; }
rm -f route38b-*.txt route38b-*.txt.err qwen38-warmup-discard.jsonl
IMG=rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0

run_cell () {  # arm ctx out tag
  sudo docker run --rm --name "q38b-$1-$2" --device /dev/kfd --device /dev/dri \
    --group-add video --ipc host --shm-size 16g -v /data:/data -v /data/50603:/work \
    -e NCCL_P2P_DISABLE=1 -e HSA_ENABLE_SDMA=0 --entrypoint bash "$IMG" \
    -c "python -u /work/probe_027_depth.py $1 $2 $3" > "/data/50603/q38b-$4.log" 2>&1
  echo "exit=$? $(date -u +%H:%M:%S)"
  grep -aE "RESULT|AssertionError|Traceback" "/data/50603/q38b-$4.log" | tail -3
}

echo "########## cell 0, DISCARDED warm-up  $(date -u +%H:%M:%S) ##########"
run_cell stock 1024 /work/qwen38-warmup-discard.jsonl warmup

for ctx in 1024 8192 32768; do
  for arm in splitkv stock; do
    echo "########## B arm=$arm ctx=$ctx  $(date -u +%H:%M:%S) ##########"
    run_cell "$arm" "$ctx" /work/qwen38-027-depth-b.jsonl "$arm-$ctx"
  done
done
echo Q38_DEPTH_B_DONE
