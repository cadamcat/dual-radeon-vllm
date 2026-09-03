#!/bin/bash
# run_c1_ab.sh -- C1: does vllm#54706's deterministic split-K epilogue remove
# the greedy non-determinism? Same harness, models, depths and backend as the
# published cells (nondet_attn.py, ROCM_ATTN); the only variable is which
# _rocm_C.abi3.so the container runs:
#   baseline  our own gfx1100 build of the container's vLLM commit, unpatched
#   pr54706   the same build with the PR's two kernel files
# The shipped wheel's kernel is what every published cell ran on; `baseline`
# is there so that a change between the wheel and `pr54706` can be attributed
# to the patch and not to our toolchain. The installed .so is backed up before
# the first swap and restored, md5-checked, at the end -- and on any exit.
#
# Runs ON THE GUEST, detached. Log: /data/rccl-build/c1ab/c1-ab.log
set -u
H=/data/rccl-build/c1ab
mkdir -p $H
LOG=$H/c1-ab.log
BASE=27971584
SO=/opt/python/lib/python3.14/site-packages/vllm/_rocm_C.abi3.so
BAK=/rb/c1ab/_rocm_C.abi3.so.shipped
exec > >(tee -a "$LOG") 2>&1
log() { echo "$(date -u +%H:%M:%S) | $*" | tee -a $H/PROGRESS.txt >/dev/null; echo "$(date -u +%H:%M:%S) | $*"; }
dx() { sudo -n docker exec vllm-tp2 bash -lc "$1"; }
kill_v() { dx 'pkill -9 -f VLLM::; pkill -9 -f EngineCore; true' >/dev/null 2>&1; }
wait_idle() {
  for i in $(seq 1 40); do
    v1=$(cat /sys/class/drm/card1/device/mem_info_vram_used); v2=$(cat /sys/class/drm/card2/device/mem_info_vram_used)
    if [ "$v1" -lt $((BASE+20000000)) ] && [ "$v2" -lt $((BASE+20000000)) ]; then return 0; fi
    sleep 5
  done
  log "WARNING vram did not return to idle"
}
restore() {
  log "restore: putting the shipped _rocm_C back"
  kill_v; wait_idle
  dx "cp -f $BAK $SO && md5sum $SO $BAK"
  sudo -n docker stop vllm-tp2 >/dev/null 2>&1
  sudo -n systemctl start ollama llamacpp-hub
  sleep 8
  log "services: $(systemctl is-active ollama llamacpp-hub | tr '\n' ' ')"
  log "vram: $(cat /sys/class/drm/card1/device/mem_info_vram_used /sys/class/drm/card2/device/mem_info_vram_used | tr '\n' ' ')"
  log "=== C1-AB COMPLETE ==="
}
trap restore EXIT

log "===== start ====="
sudo -n systemctl stop ollama llamacpp-hub; sleep 3
sudo -n docker start vllm-tp2 >/dev/null 2>&1; sleep 2
dx "test -f $BAK || cp -p $SO $BAK; md5sum $SO $BAK; ls -la $SO $BAK"
log "patch state (first_block sites, left as found): $(dx 'grep -c first_block /opt/python/lib/python3.14/site-packages/vllm/v1/attention/ops/chunked_prefill_paged_decode.py')"

for arm in baseline pr54706; do
  case $arm in baseline) SRC=/rb/c1c/build/_rocm_C.abi3.so;; pr54706) SRC=/rb/c1b/build/_rocm_C.abi3.so;; esac
  dx "cp -f $SRC $SO"
  log "arm=$arm installed: $(dx "md5sum $SO $SRC" | tr '\n' ' ')"
  for which in muse gemma3; do
    kill_v; wait_idle
    log "--- $arm $which backend=ROCM_ATTN ---"
    RL=$H/nondet-c1-$arm-$which-ROCM_ATTN.log
    timeout 2400 sudo -n docker exec vllm-tp2 python3 /rb/nondet_attn.py "$which" 1 ROCM_ATTN > "$RL" 2>&1
    rc=$?
    grep -aE '^\[within\]|NONDET DONE' "$RL" | sed 's/^/  /'
    [ $rc -ne 0 ] && { log "RUN FAILED rc=$rc"; tail -5 "$RL" | sed 's/^/    /'; }
    log "backend in log: $(grep -aoE 'Using [A-Z_]+ backend|Overriding with [A-Z_]+' "$RL" | sort -u | tr '\n' ' ')"
    log "quant kernel:   $(grep -aoE 'Using [A-Za-z0-9]+LinearKernel' "$RL" | sort -u | tr '\n' ' ')"
    cp -f /data/rccl-build/nondet-attn-$which-ROCM_ATTN-p1.json $H/nondet-c1-$arm-$which-ROCM_ATTN-p1.json 2>/dev/null || true
  done
done
