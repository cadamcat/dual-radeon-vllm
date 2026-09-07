#!/bin/bash
# run.sh -- Qwen3.8-27B past 32 000 on vllm 0.27 + vllm#45916. Runs ON THE
# GUEST, detached. See runner.py's docstring for what this measures and why.
#
# Eight rungs: three that overlap `D8-27B-tp2-long` (0.23) so the two sittings
# can be compared, and the five that stack has never carried.
#
#   nohup bash /data/rccl-build/bench0906/run.sh >/dev/null 2>&1 &
#   tail -f /data/rccl-build/bench0906/PROGRESS.txt
set -u
H=/data/rccl-build/bench0906
mkdir -p "$H"
LOG=$H/run.log
BASE=27971584
MD5_SPLITKV=84c6d4f9b2dfe2714b3a8f43ee832b02
CPPD=/opt/python/lib/python3.14/site-packages/vllm/v1/attention/ops/chunked_prefill_paged_decode.py
exec > >(tee -a "$LOG") 2>&1
log() { echo "$(date -u +%H:%M:%S) | $*" | tee -a $H/PROGRESS.txt; }

wait_idle() {
  for i in $(seq 1 40); do
    v1=$(cat /sys/class/drm/card1/device/mem_info_vram_used)
    v2=$(cat /sys/class/drm/card2/device/mem_info_vram_used)
    if [ "$v1" -lt $((BASE+20000000)) ] && [ "$v2" -lt $((BASE+20000000)) ]; then return 0; fi
    sleep 5
  done
  log "WARNING vram did not return to idle"
}
restore() {
  log "restore: stopping vllm, returning the guest to its services"
  sudo -n docker exec vllm-027 bash -lc 'pkill -9 -f "vllm serve"; pkill -9 -f VLLM::; pkill -9 -f EngineCore; true' >/dev/null 2>&1
  wait_idle
  sudo -n docker stop vllm-027 >/dev/null 2>&1
  sudo -n systemctl start ollama llamacpp-hub
  sleep 8
  log "services: $(systemctl is-active ollama llamacpp-hub | tr '\n' ' ')"
  log "vram: $(cat /sys/class/drm/card1/device/mem_info_vram_used /sys/class/drm/card2/device/mem_info_vram_used | tr '\n' ' ')  (baseline $BASE)"
  log "===== END ====="
}
trap restore EXIT

log "===== start ====="
sudo -n systemctl stop ollama llamacpp-hub; sleep 3
log "services stopped: $(systemctl is-active ollama llamacpp-hub | tr '\n' ' ')"
sudo -n docker start vllm-027 >/dev/null 2>&1; sleep 3

# The stack, asserted before anything is measured. campaign-2026-09-02c exists
# because this file went missing once and the run reproduced the stock arm
# while reporting itself as the patched one.
GOT=$(sudo -n docker exec vllm-027 md5sum $CPPD | cut -d' ' -f1)
VER=$(sudo -n docker exec vllm-027 python -c 'import vllm;print(vllm.__version__)')
log "vllm=$VER  chunked_prefill_paged_decode.py=$GOT"
if [ "$GOT" != "$MD5_SPLITKV" ]; then
  log "REFUSING: expected $MD5_SPLITKV (vllm#45916 applied), got $GOT"
  exit 3
fi

cd "$H"
BENCH_CONTAINER=vllm-027 \
BENCH_TARGETS=8000,16000,32000,48000,64000,80000,96000,128000 \
  python3 "$H/runner.py"
rc=$?
log "runner exit=$rc"
exit $rc
