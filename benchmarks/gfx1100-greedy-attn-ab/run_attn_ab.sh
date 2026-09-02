#!/bin/bash
# The attention-backend A/B for vllm#50603. Same harness, same container, same
# models and depths as the published nondet.py cells; the only variable is which
# attention backend serves them, which is the axis the 36-cell set confounds
# with the W4A16 quantisation kernel. See nondet_attn.py for the confound.
#
# The quantisation kernel must NOT follow the backend, or the A/B tests two
# things at once; every run log is grepped for it below and the line is printed
# beside the result.
#
# TP is NOT a variable here and cannot be: Muse-Glimmer-30B-INT4 is 21 GB and
# gemma-3-27b-it-w4a16 is 19 GB on disk, against 19.98 GiB of card, so neither
# runs at TP=1 on this box. @AIwork4me's W7900D has 48 GB and can.
set -u
LOG=/data/rccl-build/nondet-attn-ab.log
BASE=27971584
exec > >(tee -a "$LOG") 2>&1

kill_v() {
  sudo -n docker exec vllm-tp2 bash -lc 'pkill -9 -f VLLM::; pkill -9 -f EngineCore; true' >/dev/null 2>&1
}
wait_idle() {
  for i in $(seq 1 40); do
    v1=$(cat /sys/class/drm/card1/device/mem_info_vram_used)
    v2=$(cat /sys/class/drm/card2/device/mem_info_vram_used)
    if [ "$v1" -lt $((BASE+20000000)) ] && [ "$v2" -lt $((BASE+20000000)) ]; then return 0; fi
    sleep 5
  done
  echo "  WARNING vram did not return to idle"
}

restore() {
  echo "===== restore $(date -Is) ====="
  kill_v; wait_idle
  sudo -n docker stop vllm-tp2 >/dev/null 2>&1
  sudo -n systemctl start ollama llamacpp-hub
  sleep 8
  echo "services: $(systemctl is-active ollama llamacpp-hub | tr '\n' ' ')"
  printf 'vram: '
  cat /sys/class/drm/card1/device/mem_info_vram_used /sys/class/drm/card2/device/mem_info_vram_used | tr '\n' ' '
  echo
  echo "=== ATTN-AB COMPLETE $(date -Is) ==="
}
trap restore EXIT

echo "===== start $(date -Is) ====="
PATCHSTATE=$(sudo -n docker exec vllm-tp2 bash -lc 'grep -c first_block /opt/python/lib/python3.14/site-packages/vllm/v1/attention/ops/chunked_prefill_paged_decode.py')
echo "patch state (first_block sites, left exactly as found): $PATCHSTATE"
sudo -n systemctl stop ollama llamacpp-hub
sleep 3
sudo -n docker start vllm-tp2 >/dev/null 2>&1
sleep 2

for which in muse gemma3; do
  for backend in ROCM_ATTN TRITON_ATTN; do
    kill_v; wait_idle
    echo "--- $which backend=$backend  $(date -Is) ---"
    RL=/data/rccl-build/nondet-attn-$which-$backend.log
    timeout 2400 sudo -n docker exec vllm-tp2 python3 /rb/nondet_attn.py "$which" 1 "$backend" > "$RL" 2>&1
    rc=$?
    grep -aE '^\[within\]|NONDET DONE' "$RL" | sed 's/^/  /'
    if [ $rc -ne 0 ]; then
      echo "  RUN FAILED rc=$rc; tail:"
      tail -5 "$RL" | sed 's/^/    /'
    fi
    echo "  backend in log: $(grep -aoE 'Using [A-Z_]+ backend|Overriding with [A-Z_]+' "$RL" | sort -u | tr '\n' ' ')"
    echo "  quant kernel:   $(grep -aoE 'Using [A-Za-z0-9]+LinearKernel' "$RL" | sort -u | tr '\n' ' ')"
  done
done
