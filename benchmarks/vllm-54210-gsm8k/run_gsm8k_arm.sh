#!/bin/bash
# One arm of vllm#54210's gsm8k evaluation. Runs ON THE GUEST, detached.
#
#   nohup bash /data/gsm8k-run/run_gsm8k_arm.sh stock|widened [LIMIT] &
#
# The server runs in the container with the gate forced by
# gsm8k_serve_gate.py; lm_eval runs OUTSIDE it, in /data/lmeval-venv, over HTTP.
# That split is deliberate: this comparison rests on the container's file md5s,
# and pip resolving lm_eval's dependencies inside it could move torch or vllm.
#
# A score is worth nothing here unless the kernel under test was actually
# reached, so the route file is checked before the score is kept: the widened
# arm must show use_custom=True and the stock arm must not.
set -u
ARM=${1:?stock|widened}
LIMIT=${2:-}
H=/data/gsm8k-run
mkdir -p $H
LOG=$H/$ARM.log
BASE=27971584
MODEL=/models/gemma-3-27b-it-w4a16
# lm_eval runs OUTSIDE the container, where /models does not exist, and it loads
# a tokenizer from whatever it is given as the model name. Give it the host side
# of the same bind mount (/data/incoming -> /models) and assert the two are the
# same directory, so the harness cannot tokenise against a different checkpoint
# than the server serves.
TOKENIZER=/data/incoming/gemma-3-27b-it-w4a16
ROCM_PY=/opt/python/lib/python3.14/site-packages/vllm/platforms/rocm.py
CPPD_PY=/opt/python/lib/python3.14/site-packages/vllm/v1/attention/ops/chunked_prefill_paged_decode.py
CPPD_MD5=84c6d4f9b2dfe2714b3a8f43ee832b02
ROUTE=/rb/gsm8k-run/route-$ARM.txt
ROUTE_HOST=$H/route-$ARM.txt
exec > >(tee -a "$LOG") 2>&1
log() { echo "$(date -u +%H:%M:%S) | $*" | tee -a $H/PROGRESS.txt; }

wait_idle() {
  for i in $(seq 1 40); do
    v1=$(cat /sys/class/drm/card1/device/mem_info_vram_used)
    v2=$(cat /sys/class/drm/card2/device/mem_info_vram_used)
    [ "$v1" -lt $((BASE+20000000)) ] && [ "$v2" -lt $((BASE+20000000)) ] && return 0
    sleep 5
  done
  log "WARNING vram did not return to idle"
}
restore() {
  log "restore: stopping the server, returning the guest to its services"
  sudo -n docker exec vllm-027 bash -lc 'pkill -9 -f "vllm serve"; pkill -9 -f VLLM::; pkill -9 -f EngineCore; true' >/dev/null 2>&1
  # The arm rewrote two files inside the container. Put THOSE two back from the
  # backup taken before it started -- not by recreating the container, which
  # would also discard the vllm#45916 patch this container has carried since
  # 2026-09-02 and which campaign-2026-09-06, -09-07 and 09-02c all rest on.
  for f in "$ROCM_PY" "$CPPD_PY"; do
    sudo -n docker exec vllm-027 bash -lc "cp -f /rb/gsm8k-run/\$(basename $f).bak $f" 2>/dev/null
  done
  got=$(sudo -n docker exec vllm-027 md5sum "$CPPD_PY" 2>/dev/null | cut -d" " -f1)
  if [ "$got" = "$CPPD_MD5" ]; then log "container restored: cppd=$got"; else
    log "WARNING cppd is $got, expected $CPPD_MD5 -- run apply_45916.py before the next campaign"; fi
  sudo -n docker stop vllm-027 >/dev/null 2>&1
  # Only now: the pages are the container's, and they are freed when it stops.
  # Waiting before the stop timed out on every healthy run and said so.
  wait_idle
  sudo -n systemctl start ollama llamacpp-hub
  sleep 8
  log "services: $(systemctl is-active ollama llamacpp-hub | tr '\n' ' ')"
  v1=$(cat /sys/class/drm/card1/device/mem_info_vram_used)
  v2=$(cat /sys/class/drm/card2/device/mem_info_vram_used)
  if [ "$v1" = "$BASE" ] && [ "$v2" = "$BASE" ]; then log "vram: $v1 $v2, both at baseline"
  else log "WARNING vram: $v1 $v2, baseline is $BASE"; fi
  log "===== END $ARM ====="
}
trap restore EXIT

log "===== start $ARM${LIMIT:+ (limit $LIMIT)} ====="
sudo -n systemctl stop ollama llamacpp-hub; sleep 3
sudo -n docker start vllm-027 >/dev/null 2>&1; sleep 3
# Both sides of the copy. The probe opens the route file with 'a', so a file
# left from an earlier run of the same arm is silently inherited and the verdict
# is then read off two runs at once -- which is how a route that says
# use_custom=True can outlive the run that earned it. The 14:03 widened run
# reported the 13:45 run's pids alongside its own.
rm -f "$ROUTE_HOST" "/data/rccl-build/gsm8k-run/route-$ARM.txt"
mkdir -p /data/rccl-build/gsm8k-run
cp $H/gsm8k_serve_gate.py /data/rccl-build/gsm8k-run/

VER=$(sudo -n docker exec vllm-027 python -c 'import vllm;print(vllm.__version__)')
BEFORE=$(sudo -n docker exec vllm-027 md5sum "$CPPD_PY" | cut -d" " -f1)
log "vllm=$VER model=$MODEL arm=$ARM cppd=$BEFORE"
[ "$BEFORE" = "$CPPD_MD5" ] || { log "REFUSING: cppd is $BEFORE, expected $CPPD_MD5 (vllm#45916)"; exit 3; }
IN=$(sudo -n docker exec vllm-027 md5sum "$MODEL/config.json" | cut -d" " -f1)
OUT=$(md5sum "$TOKENIZER/config.json" | cut -d" " -f1)
[ "$IN" = "$OUT" ] || { log "REFUSING: $TOKENIZER ($OUT) is not $MODEL ($IN)"; exit 3; }
log "tokenizer=$TOKENIZER config.json=$OUT, the served checkpoint"
# back the two files up before either is rewritten, so restore puts back the
# state this container had rather than the image's
for f in "$ROCM_PY" "$CPPD_PY"; do
  sudo -n docker exec vllm-027 bash -lc "cp -f $f /rb/gsm8k-run/\$(basename $f).bak"
done

sudo -n docker exec -d vllm-027 bash -lc \
  "cd /rb/gsm8k-run && python -u gsm8k_serve_gate.py $ARM $ROUTE $MODEL > /rb/gsm8k-run/serve-$ARM.log 2>&1"

for i in $(seq 1 90); do
  code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health 2>/dev/null || true)
  [ "$code" = "200" ] && break
  sleep 10
done
[ "$code" = "200" ] || { log "REFUSING: server never became healthy"; tail -5 /data/rccl-build/gsm8k-run/serve-$ARM.log; exit 3; }
log "server healthy after $((i*10))s"

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 /data/lmeval-venv/bin/lm_eval \
  --model local-completions \
  --model_args "model=$MODEL,tokenizer=$TOKENIZER,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=1,max_retries=2,tokenized_requests=False" \
  --tasks gsm8k_local --include_path /data/lmeval-tasks \
  ${LIMIT:+--limit $LIMIT} \
  --output_path $H/out-$ARM --log_samples
rc=$?
log "lm_eval exit=$rc"

cp -f /data/rccl-build/gsm8k-run/route-$ARM.txt "$ROUTE_HOST" 2>/dev/null || true
log "routes: $(sort -u "$ROUTE_HOST" 2>/dev/null | tr '\n' ' ' || echo NONE)"
exit $rc
