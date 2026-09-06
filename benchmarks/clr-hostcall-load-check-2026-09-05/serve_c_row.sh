#!/bin/bash
# The end-to-end cell of the opt-in experiment: Qwen3-8B, TP=2 (RCCL in the
# loop), served by vLLM 0.27 in the ROCm 10.0 container, one request, under
# the SDK's runtime or under A + the opt-in with HIP_HOSTCALL_ALLOW_MISSING=1.
# Runs on the GUEST HOST and TAKES THE LEASE [INV-012]; skeleton from
# hostcall-dispatch-2026-09-05/serve_row.sh, the runtime swap from
# clr_demo_row.sh (in place, md5-verified, undone in the EXIT trap).
#
#   serve_c_row.sh <atomics_present|atomics_absent> <stock|patched>
set -u
ROW="${1:?row label}"; RT="${2:?stock|patched}"
B=/data/rccl-build/pa
C=clr100
IMAGE_MUST_CONTAIN="vllm_0.27.0"
BASELINE=27971584
MODEL=/models/Qwen3-8B
TP=2
SP=/opt/python/lib/python3.14/site-packages
BUILD_HOST=/data/rccl-build/clr-rocm10c-build; BUILD_IN_C=/rb/clr-rocm10c-build
P=$B/SERVE-rocm10c-$ROW-$RT.txt
SLOG_IN=/rb/pa/serve-rocm10c-$ROW-$RT.log
SLOG=$B/serve-rocm10c-$ROW-$RT.log
CELL=$B/serve-rocm10c-$ROW-$RT.jsonl
OUT=$B/serve-rocm10c-cells.jsonl
case "$RT" in
  stock)   ENV="" ;;
  patched) ENV="HIP_HOSTCALL_ALLOW_MISSING=1 AMD_LOG_LEVEL=1" ;;
  *) echo "runtime must be stock or patched" >&2; exit 2 ;;
esac

say() { echo "$(date -u +%H:%M:%S) | $*" | tee -a "$P"; }
vram() { cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | tr '\n' ' '; }
at_baseline() { local v n; v=$(cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null); n=$(echo "$v" | grep -c .); [ "$n" -eq 2 ] || return 1; for x in $v; do [ "$x" = "$BASELINE" ] || return 1; done; }
container_state() { local r; r=$(sudo docker inspect -f '{{.State.Running}}' "$C" 2>/dev/null) || { echo unknown; return; }; [ "$r" = true ] && echo running || echo stopped; }
stop_serve() {
  sudo docker exec "$C" bash -lc "pkill -f 'vllm serve' >/dev/null 2>&1; sleep 3; pkill -9 -f 'vllm serve' >/dev/null 2>&1; pkill -9 -f 'VLLM::' >/dev/null 2>&1; pkill -9 -f 'from multiprocessing' >/dev/null 2>&1; true" >/dev/null 2>&1
  for _ in $(seq 1 40); do at_baseline && { say "serve stopped; VRAM back to baseline: $(vram)"; return 0; }; sleep 3; done
  say "WARNING: VRAM did not return to baseline after stopping the server: $(vram)"; return 1
}
SWAPPED=0; CORE_REAL=""; DEVEL_REAL=""; STOCK_CORE_MD5=""; STOCK_DEVEL_MD5=""; PATCHED_MD5=""
swap_in() {
  sudo docker exec "$C" bash -lc "cp -p $CORE_REAL $CORE_REAL.stock && cp -p $DEVEL_REAL $DEVEL_REAL.stock && cp $BUILD_IN_C/hipamd/lib/$PATCHED_BASE $CORE_REAL && cp $BUILD_IN_C/hipamd/lib/$PATCHED_BASE $DEVEL_REAL" || return 1
  SWAPPED=1
  local m; m=$(sudo docker exec "$C" md5sum "$CORE_REAL" | cut -c1-32); [ "$m" = "$PATCHED_MD5" ] || { say "swap_in: core md5 $m != patched $PATCHED_MD5"; return 1; }
}
swap_out() {
  [ "$SWAPPED" = 1 ] || return 0
  sudo docker exec "$C" bash -lc "cp -p $CORE_REAL.stock $CORE_REAL && cp -p $DEVEL_REAL.stock $DEVEL_REAL && rm -f $CORE_REAL.stock $DEVEL_REAL.stock" || return 1
  local m1 m2; m1=$(sudo docker exec "$C" md5sum "$CORE_REAL" | cut -c1-32); m2=$(sudo docker exec "$C" md5sum "$DEVEL_REAL" | cut -c1-32)
  [ "$m1" = "$STOCK_CORE_MD5" ] && [ "$m2" = "$STOCK_DEVEL_MD5" ] || { say "swap_out: md5 $m1/$m2 != stock $STOCK_CORE_MD5/$STOCK_DEVEL_MD5"; return 1; }
  SWAPPED=0; say "stock runtime restored (md5 verified)"
}

LEASE_TAKEN=0; ROW_OK=0
restore() {
  RC=$?; trap - EXIT; local ok=1
  say "--- restore (exit=$RC) ---"
  if [ "$(container_state)" != stopped ]; then
    stop_serve || ok=0
    swap_out || { say "RESTORE-FAILED: stock runtime not restored"; ok=0; }
    sudo docker stop "$C" >/dev/null 2>&1 && say "stopped $C" || { say "RESTORE-FAILED: docker stop"; ok=0; }
  fi
  if [ "$LEASE_TAKEN" = 1 ]; then
    sudo systemctl start ollama llamacpp-hub || { say "RESTORE-FAILED: systemctl start"; ok=0; }; sleep 5
    for s in ollama llamacpp-hub; do [ "$(systemctl is-active $s)" = active ] || { say "RESTORE-FAILED: $s"; ok=0; }; done
  fi
  for _ in $(seq 1 40); do at_baseline && break; sleep 3; done
  at_baseline && say "VRAM back to baseline: $(vram)" || { say "RESTORE-FAILED: VRAM $(vram)"; ok=0; }
  if [ "$ok" = 1 ] && [ "$RC" = 0 ] && [ "$ROW_OK" = 1 ]; then say "ROW-DONE serve-c row=$ROW runtime=$RT (restore ok)"; exit 0
  elif [ "$ok" = 1 ]; then say "ROW-FAILED serve-c row=$ROW runtime=$RT (exit=$RC, restore ok)"; exit "$RC"
  else say "RESTORE-FAILED serve-c row=$ROW runtime=$RT (exit=$RC)"; exit 9; fi
}
trap restore EXIT

: > "$P"
say "===== SERVE-C row=$ROW runtime=$RT start ====="
[ -f "$B/one_request_tp.py" ] || { say "FATAL: $B/one_request_tp.py missing"; exit 2; }
case "$ROW" in atomics_present) WANT_CAPS=2; WANT_DMESG=0 ;; atomics_absent) WANT_CAPS=0; WANT_DMESG=2 ;; *) say "FATAL: bad label"; exit 2 ;; esac
LSPCI=$(sudo lspci -vv 2>/dev/null) || { say "FATAL: lspci"; exit 5; }
NCAP=$(printf '%s\n' "$LSPCI" | grep -c AtomicOpsCap); CAPS=$(printf '%s\n' "$LSPCI" | grep -c "AtomicOpsCap: Routing- 32bit+ 64bit+")
DMESG_HITS=$(sudo dmesg 2>/dev/null | grep -c "PCIE atomic ops is not supported")
HOST=$(hostname)
say "host=$HOST kernel=$(uname -r) caps=$CAPS/$NCAP dmesg=$DMESG_HITS"
{ [ "$NCAP" -ge 2 ] && [ "$CAPS" -eq "$WANT_CAPS" ] && [ "$DMESG_HITS" -eq "$WANT_DMESG" ]; } || { say "FATAL: platform does not match label $ROW"; exit 5; }
at_baseline || { say "FATAL: VRAM not at baseline"; exit 5; }
[ "$(container_state)" = stopped ] || { say "FATAL: $C not stopped"; exit 5; }
IMAGE=$(sudo docker inspect -f '{{.Config.Image}}@{{.Image}}' "$C") || { say "FATAL: docker inspect"; exit 5; }
case "$IMAGE" in *"$IMAGE_MUST_CONTAIN"*) ;; *) say "FATAL: $C runs $IMAGE"; exit 5 ;; esac
PATCHED_FILE=$(ls $BUILD_HOST/hipamd/lib/libamdhip64.so.7.*-* 2>/dev/null | head -1); [ -n "$PATCHED_FILE" ] || { say "FATAL: patched runtime not built"; exit 5; }
PATCHED_BASE=$(basename "$PATCHED_FILE"); PATCHED_MD5=$(md5sum "$PATCHED_FILE" | cut -c1-32)
say "container $C image $IMAGE ; patched runtime $PATCHED_BASE md5=$PATCHED_MD5"

say "stopping services"; LEASE_TAKEN=1
sudo systemctl stop ollama llamacpp-hub || { say "FATAL: stop services"; exit 3; }; sleep 5
sudo docker start "$C" >/dev/null || { say "FATAL: docker start"; exit 3; }; sleep 5
CORE_REAL=$(sudo docker exec "$C" readlink -f $SP/_rocm_sdk_core/lib/libamdhip64.so.7); DEVEL_REAL=$(sudo docker exec "$C" readlink -f $SP/_rocm_sdk_devel/lib/libamdhip64.so.7)
STOCK_CORE_MD5=$(sudo docker exec "$C" md5sum "$CORE_REAL" | cut -c1-32); STOCK_DEVEL_MD5=$(sudo docker exec "$C" md5sum "$DEVEL_REAL" | cut -c1-32)
say "SDK runtime md5 core=$STOCK_CORE_MD5 devel=$STOCK_DEVEL_MD5"
RCCL_MD5=$(sudo docker exec "$C" md5sum $SP/_rocm_sdk_libraries/lib/librccl.so.1 | cut -c1-32); say "stock RCCL md5=$RCCL_MD5"
if [ "$RT" = patched ]; then swap_in || { say "FATAL: could not put the patched runtime in place"; exit 4; }; RUNTIME_MD5=$PATCHED_MD5; else RUNTIME_MD5=$STOCK_CORE_MD5; fi
sudo rm -f "$SLOG" "$CELL"
sudo docker exec -d "$C" bash -lc "export NCCL_P2P_DISABLE=1 HSA_ENABLE_SDMA=0 $ENV;
  nohup vllm serve $MODEL --tensor-parallel-size $TP --gpu-memory-utilization 0.9 --max-model-len 8192 --port 8000 > $SLOG_IN 2>&1" \
  || { say "FATAL: docker exec -d for the server failed"; exit 3; }
sleep 5
say "serve started (TP=$TP, env='$ENV'), log $SLOG_IN"
sudo timeout -k 15 900 docker exec "$C" bash -lc "cd /rb/pa && python3 one_request_tp.py --row $ROW --runtime $RT --tp $TP --serve-log $SLOG_IN --out /rb/pa/serve-rocm10c-$ROW-$RT.jsonl --caps $CAPS --dmesg $DMESG_HITS --host $HOST --image '$IMAGE' --env '$ENV' --runtime-md5 $RUNTIME_MD5" 2>&1 | sed 's/^/    /' | tee -a "$P"
RC=${PIPESTATUS[0]}
say "client rc=$RC"
[ "$RC" -eq 0 ] || { say "ROW-FAILED: client exited $RC"; exit 6; }
say "mapped runtime in the engine: $(sudo docker exec "$C" bash -lc "for p in /proc/[0-9]*; do c=\$(tr '\\0' ' ' < \$p/cmdline 2>/dev/null); case \"\$c\" in *VLLM::*|*'vllm serve'*) grep -m1 -o '[^ ]*libamdhip64[^ ]*' \$p/maps 2>/dev/null && break;; esac; done" 2>/dev/null | head -1)"
say "null-buffer lines: $(sudo grep -ac 'proceed with a null hostcall buffer' "$SLOG") ; refusals: $(sudo grep -ac 'refused: it declares a hostcall buffer' "$SLOG") ; generic errors: $(sudo grep -ac 'the operation cannot be performed in the present state' "$SLOG") ; memory faults: $(sudo grep -ac 'Memory access fault' "$SLOG")"
stop_serve || true
NROWS=$(sudo grep -ac "\"row\": \"$ROW\", \"runtime\": \"$RT\"" "$CELL" 2>/dev/null || true); NROWS=${NROWS:-0}
[ "$NROWS" -eq 1 ] || { say "ROW-FAILED: serve cell wrote $NROWS row(s)"; exit 6; }
sudo cat "$CELL" >> "$OUT" || { say "ROW-FAILED: could not append to $OUT"; exit 6; }
ROW_OK=1
say "serve cell recorded (ok=$(sudo python3 -c "import json,sys; print(json.loads(open(sys.argv[1]).readline()).get('ok'))" "$CELL")); restoring"
