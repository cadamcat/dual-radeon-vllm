#!/bin/bash
# Two more passes at the one cell the ledger will not put on a chart:
# Qwen3.8-27B, ctx 8192, with vllm#45916. Runs A and B disagree by 15.8% there
# while the stock arm at the same depth agrees to 0.5%.
#
# No warm-up cell: that was needed for the single-container calibration probe,
# and this method was shown not to need it -- run B's discarded warm-up read
# 37.0398 against run A's first measured cell at 37.0397.
set -u
fail=0   # cells that came back non-zero
cd /data/50603
[ -e qwen38-8k-r3r4.jsonl ] && { echo "REFUSING: output exists"; exit 2; }
mv -f route38-splitkv-8192.txt route38-splitkv-8192.AB.txt 2>/dev/null
IMG=rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0
for r in 3 4; do
  echo "########## pass $r  splitkv ctx=8192  $(date -u +%H:%M:%S) ##########"
  sudo docker run --rm --name "q38-r$r-8192" --device /dev/kfd --device /dev/dri \
    --group-add video --ipc host --shm-size 16g -v /data:/data -v /data/50603:/work \
    -e NCCL_P2P_DISABLE=1 -e HSA_ENABLE_SDMA=0 --entrypoint bash "$IMG" \
    -c "python -u /work/probe_027_depth.py splitkv 8192 /work/qwen38-8k-r3r4.jsonl" \
    > "/data/50603/q38-r$r-8192.log" 2>&1
  rc=$?; fail=$((fail + (rc != 0)))
  echo "exit=$rc $(date -u +%H:%M:%S)"
  grep -aE "RESULT|AssertionError|Traceback" "/data/50603/q38-r$r-8192.log" | tail -2
done
# a DONE marker that a poller greps for must not appear when a cell failed
[ "$fail" -eq 0 ] || { echo "Q38_8K_FAILED cells=$fail"; exit 1; }
echo Q38_8K_DONE
