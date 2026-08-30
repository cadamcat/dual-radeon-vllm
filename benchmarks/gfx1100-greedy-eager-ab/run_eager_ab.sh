#!/bin/bash
# The enforce_eager A/B for vllm#50603. Same harness, same container, same
# models and depths as the published nondet.py cells; the only variable is
# whether CUDA graphs are captured.
#
# TP is NOT a variable here and cannot be: Muse-Glimmer-30B-INT4 is 21 GB and
# gemma-3-27b-it-w4a16 is 19 GB on disk, against 19.98 GiB of card, so neither
# runs at TP=1 on this box. @AIwork4me's W7900D has 48 GB and can.
set -u
LOG=/data/rccl-build/nondet-eager-ab.log
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
  echo "=== EAGER-AB COMPLETE $(date -Is) ==="
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
  for eager in 0 1; do
    kill_v; wait_idle
    echo "--- $which enforce_eager=$eager  $(date -Is) ---"
    RL=/data/rccl-build/nondet-eager-$which-e$eager.log
    timeout 2400 sudo -n docker exec vllm-tp2 python3 /rb/nondet_eager.py "$which" 1 "$eager" > "$RL" 2>&1
    rc=$?
    grep -aE '^\[within\]|NONDET DONE' "$RL" | sed 's/^/  /'
    if [ $rc -ne 0 ]; then
      echo "  RUN FAILED rc=$rc; tail:"
      tail -5 "$RL" | sed 's/^/    /'
    fi
    echo "  graphs captured: $(grep -ac 'Capturing CUDA graph' "$RL"), log says: $(grep -aoE 'enforce_eager=[A-Za-z]+' "$RL" | head -1)"
  done
done
