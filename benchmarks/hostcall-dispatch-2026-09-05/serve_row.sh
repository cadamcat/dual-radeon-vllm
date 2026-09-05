#!/bin/bash
# The end-to-end half of one row: Qwen3-8B, TP=1 (no RCCL anywhere), served by
# vLLM 0.23 with the backend it would choose itself (ROCM_ATTN, the CK kernel)
# or forced to TRITON_ATTN, one request each. Runs on the GUEST HOST and TAKES
# THE LEASE [INV-012]. Revised twice after TASK-0007 / TASK-0007b.
#
#   serve_row.sh <row-label> <default|triton>
#
# 0.23 ignores VLLM_ATTENTION_BACKEND (gfx1100-greedy-attn-ab found this the
# hard way); the flag is --attention-backend, and one_request.py greps the log
# for the backend the engine actually chose and says whether it matches. The
# cell's `ok` is a MEASUREMENT (false is an expected outcome in the absent
# row); harness failures -- server never launched, client failed, no row --
# are ROW-FAILED. ROW-DONE is printed by the EXIT trap after restoration.
set -u
ROW="${1:?row label}"; BE="${2:?default|triton}"
B=/data/rccl-build/pa
C=vllm-tp2
IMAGE_MUST_CONTAIN="vllm_0.23.0"
BASELINE=27971584
MODEL=/models/Qwen3-8B
P=$B/SERVE-$ROW-$BE.txt
SLOG_IN=/rb/pa/serve-$ROW-$BE.log
SLOG=$B/serve-$ROW-$BE.log
CELL=$B/serve-$ROW-$BE.jsonl
OUT=$B/serve-cells.jsonl
case "$BE" in
  default) FLAG="" ;;
  triton)  FLAG="--attention-backend TRITON_ATTN" ;;
  *) echo "backend must be default or triton" >&2; exit 2 ;;
esac

say() { echo "$(date -u +%H:%M:%S) | $*" | tee -a "$P"; }
vram() { cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | tr '\n' ' '; }
at_baseline() {
  local vals n; vals=$(cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null); n=$(echo "$vals" | grep -c .)
  [ "$n" -eq 2 ] || return 1
  for v in $vals; do [ "$v" = "$BASELINE" ] || return 1; done
  return 0
}
container_state() {
  local r; r=$(sudo docker inspect -f '{{.State.Running}}' "$C" 2>/dev/null) || { echo unknown; return; }
  [ "$r" = true ] && echo running || echo stopped
}
stop_serve() {   # B1's b1_decode.sh: kill the whole tree, wait for the exact baseline on both cards
  sudo docker exec "$C" bash -lc "pkill -f 'vllm serve' >/dev/null 2>&1; sleep 3;
    pkill -9 -f 'vllm serve' >/dev/null 2>&1;
    pkill -9 -f 'VLLM::' >/dev/null 2>&1;
    pkill -9 -f 'from multiprocessing' >/dev/null 2>&1; true" >/dev/null 2>&1
  for _ in $(seq 1 40); do at_baseline && { say "serve stopped; VRAM back to baseline: $(vram)"; return 0; }; sleep 3; done
  say "WARNING: VRAM did not return to baseline after stopping the server: $(vram)"
  return 1
}

LEASE_TAKEN=0
ROW_OK=0
restore() {
  RC=$?
  trap - EXIT
  local ok=1 st
  say "--- restore (exit=$RC) ---"
  st=$(container_state)
  if [ "$st" != stopped ]; then
    stop_serve || ok=0
    if sudo docker stop "$C" >/dev/null 2>&1; then say "stopped $C"; else say "RESTORE-FAILED: docker stop $C"; ok=0; fi
    [ "$(container_state)" = stopped ] || { say "RESTORE-FAILED: $C still not stopped"; ok=0; }
  fi
  if [ "$LEASE_TAKEN" = 1 ]; then
    if sudo systemctl start ollama llamacpp-hub; then say "services start requested"; else say "RESTORE-FAILED: systemctl start"; ok=0; fi
    sleep 5
    for s in ollama llamacpp-hub; do
      [ "$(systemctl is-active "$s")" = active ] || { say "RESTORE-FAILED: $s is $(systemctl is-active "$s")"; ok=0; }
    done
  fi
  for _ in $(seq 1 40); do at_baseline && break; sleep 3; done
  if at_baseline; then say "VRAM back to baseline: $(vram)"; else say "RESTORE-FAILED: VRAM $(vram) (want $BASELINE on both cards)"; ok=0; fi
  if [ "$ok" = 1 ] && [ "$RC" = 0 ] && [ "$ROW_OK" = 1 ]; then
    say "ROW-DONE serve row=$ROW backend=$BE in $OUT (restore ok)"; say "===== SERVE row $ROW backend $BE ended (exit=0) ====="; exit 0
  elif [ "$ok" = 1 ]; then
    say "ROW-FAILED serve row=$ROW backend=$BE (exit=$RC, restore ok)"; say "===== SERVE row $ROW backend $BE ended (exit=$RC) ====="; exit "$RC"
  else
    say "RESTORE-FAILED serve row=$ROW backend=$BE (exit=$RC)"; say "===== SERVE row $ROW backend $BE ended (exit=$RC) RESTORE-FAILED ====="; exit 9
  fi
}
trap restore EXIT

: > "$P"
say "===== SERVE row=$ROW backend=$BE start ====="
[ -f "$B/one_request.py" ] || { say "FATAL: $B/one_request.py missing"; exit 2; }
case "$ROW" in
  atomics_present) WANT_CAPS=2; WANT_DMESG=0 ;;
  atomics_absent)  WANT_CAPS=0; WANT_DMESG=2 ;;
  *) say "FATAL: row label must be atomics_present or atomics_absent"; exit 2 ;;
esac
LSPCI=$(sudo lspci -vv 2>/dev/null) || { say "FATAL: lspci failed"; exit 5; }
NCAPLINES=$(printf '%s\n' "$LSPCI" | grep -c "AtomicOpsCap"); [ "$NCAPLINES" -ge 2 ] || { say "FATAL: lspci shows $NCAPLINES AtomicOpsCap lines"; exit 5; }
CAPS=$(printf '%s\n' "$LSPCI" | grep -c "AtomicOpsCap: Routing- 32bit+ 64bit+")
DMESG=$(sudo dmesg 2>/dev/null) || { say "FATAL: dmesg failed"; exit 5; }
DMESG_HITS=$(printf '%s\n' "$DMESG" | grep -c "PCIE atomic ops is not supported")
HOST=$(hostname); UUID=$(sudo cat /sys/class/dmi/id/product_uuid 2>/dev/null || echo unknown)
say "host=$HOST product_uuid=$UUID root ports 32bit+ 64bit+ : $CAPS (of $NCAPLINES) ; dmesg no-atomics lines: $DMESG_HITS"
if [ "$CAPS" -ne "$WANT_CAPS" ] || [ "$DMESG_HITS" -ne "$WANT_DMESG" ]; then
  say "FATAL: label $ROW wants caps=$WANT_CAPS dmesg=$WANT_DMESG; the platform says caps=$CAPS dmesg=$DMESG_HITS. Not taking the lease."
  exit 5
fi
at_baseline || { say "FATAL: VRAM not at baseline before start: $(vram)"; exit 5; }
[ "$(container_state)" = stopped ] || { say "FATAL: $C is $(container_state); someone else holds the box"; exit 5; }
IMAGE=$(sudo docker inspect -f '{{.Config.Image}}@{{.Image}}' "$C") || { say "FATAL: docker inspect $C"; exit 5; }
case "$IMAGE" in *"$IMAGE_MUST_CONTAIN"*) ;; *) say "FATAL: $C runs $IMAGE, expected *$IMAGE_MUST_CONTAIN*"; exit 5 ;; esac
say "container $C image $IMAGE"

say "stopping ollama and llamacpp-hub"
LEASE_TAKEN=1
sudo systemctl stop ollama llamacpp-hub || { say "FATAL: could not stop services"; exit 3; }
sleep 5
sudo docker start "$C" >/dev/null || { say "FATAL: could not start $C"; exit 3; }
sleep 5
sudo rm -f "$SLOG" "$CELL" || { say "FATAL: could not clear stale files"; exit 6; }
[ -e "$CELL" ] && { say "FATAL: stale $CELL survived rm"; exit 6; }

sudo docker exec -d "$C" bash -lc "export VLLM_CLONE_MMAP=1 HSA_ENABLE_SDMA=0;
  nohup vllm serve $MODEL --gpu-memory-utilization 0.9 --max-model-len 8192 --port 8000 $FLAG > $SLOG_IN 2>&1" \
  || { say "FATAL: docker exec -d for the server failed"; exit 3; }
sleep 5
[ -s "$SLOG" ] || sudo test -s "$SLOG" || say "WARNING: serve log empty after 5 s"
say "serve started (TP=1, flag='$FLAG'), log $SLOG_IN"
sudo timeout -k 15 900 docker exec "$C" bash -lc "cd /rb/pa && python3 one_request.py --row $ROW --backend $BE --serve-log $SLOG_IN --out /rb/pa/serve-$ROW-$BE.jsonl --caps $CAPS --dmesg $DMESG_HITS --host $HOST --image '$IMAGE'" 2>&1 | sed 's/^/    /' | tee -a "$P"
RC=${PIPESTATUS[0]}
say "client rc=$RC"
[ "$RC" -eq 0 ] || { say "ROW-FAILED: client exited $RC"; exit 6; }
NROWS=$(sudo grep -ac "\"row\": \"$ROW\", \"backend_requested\": \"$BE\"" "$CELL" 2>/dev/null || true); NROWS=${NROWS:-0}
DONE=$(grep -ac ONE_REQUEST_DONE "$P" || true)
if [ "$NROWS" -ne 1 ] || [ "$DONE" -ne 1 ]; then say "ROW-FAILED: serve cell wrote $NROWS row(s), done markers $DONE"; exit 6; fi
say "backend line: $(sudo grep -aoE 'Using [A-Z_]+ backend[^\n]{0,40}|Overriding with [A-Z_]+' "$SLOG" | head -1)"
say "fallback warning lines: $(sudo grep -ac 'falling back to Triton' "$SLOG")"
say "first error line: $(sudo grep -aoE 'hipError[A-Za-z]+|the operation cannot be performed in the present state|Pcie atomics not enabled, hostcall not supported|AQL dispatch failed|EngineCore.{0,60}(died|failed)' "$SLOG" | head -1)"
stop_serve || true
BEFORE=$(grep -ac "\"row\": \"$ROW\", \"backend_requested\": \"$BE\"" "$OUT" 2>/dev/null || true); BEFORE=${BEFORE:-0}
sudo cat "$CELL" >> "$OUT" || { say "ROW-FAILED: could not append to $OUT"; exit 6; }
AFTER=$(grep -ac "\"row\": \"$ROW\", \"backend_requested\": \"$BE\"" "$OUT" || true)
[ "$((AFTER - BEFORE))" -eq 1 ] || { say "ROW-FAILED: $OUT gained $((AFTER - BEFORE)) rows, not 1"; exit 6; }
ROW_OK=1
say "serve cell recorded (ok=$(sudo python3 -c "import json,sys; print(json.loads(open(sys.argv[1]).readline()).get('ok'))" "$CELL")); restoring"
