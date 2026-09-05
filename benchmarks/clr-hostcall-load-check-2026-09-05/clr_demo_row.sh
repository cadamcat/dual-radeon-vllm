#!/bin/bash
# One platform state of the CLR load-check demonstration: the 57-line probe and
# the twelve collective cases under stock RCCL 2.30.4, each once under the
# stock HIP runtime and once under the patched one (rocm-systems 2b22ab01 +
# clr-hostcall-load-check.patch, built in /rb/clr-build). The patched runtime
# is put IN PLACE of the SDK's libamdhip64 for the duration of a cell (backup,
# copy, run, restore by md5), because rocm_sdk.preload_libraries() dlopens the
# SDK copy by absolute path at import and LD_LIBRARY_PATH never reaches it; the
# row records the library each process actually mapped. Runs on the GUEST
# HOST and TAKES THE LEASE [INV-012]; skeleton and restore from B2's
# capability_row.sh. The label is checked against lspci/dmesg first.
#
#   clr_demo_row.sh <atomics_present|atomics_absent>
set -u
ROW="${1:?row label}"
B=/data/rccl-build/b1
D=/data/rccl-build/pa
C=vllm-tp2
SP=/opt/python/lib/python3.14/site-packages
SDL=$SP/_rocm_sdk_libraries/lib
SDD=$SP/_rocm_sdk_devel/lib
DEPLOYED_IN_C=/rb/librccl-final.so
DEPLOYED_MD5=ab5b50f0d84806ed7fbe0f4f560151ff
STOCK=/rb/b1/librccl-stock2304.so
PATCHED=/rb/clr-build/hipamd/lib
BASELINE=27971584
OUT=$D/clr-demo.jsonl
P=$D/CLRDEMO-$ROW.txt

say() { echo "$(date -u +%H:%M:%S) | $*" | tee -a "$P"; }
vram() { cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | tr '\n' ' '; }
at_baseline() { local v n; v=$(cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null); n=$(echo "$v" | grep -c .); [ "$n" -eq 2 ] || return 1; for x in $v; do [ "$x" = "$BASELINE" ] || return 1; done; }
container_state() { local r; r=$(sudo docker inspect -f '{{.State.Running}}' "$C" 2>/dev/null) || { echo unknown; return; }; [ "$r" = true ] && echo running || echo stopped; }

LEASE_TAKEN=0; ROW_OK=0
restore() {
  RC=$?; trap - EXIT; local ok=1
  say "--- restore (exit=$RC) ---"
  if [ "$(container_state)" != stopped ]; then
    swap_out || { say "RESTORE-FAILED: stock runtime not restored"; ok=0; }
    sudo docker exec "$C" bash -lc "cp $DEPLOYED_IN_C $SDL/librccl.so.1 && cp $DEPLOYED_IN_C $SDD/librccl.so.1" 2>/dev/null
    M=$(sudo docker exec "$C" md5sum $SDL/librccl.so.1 2>/dev/null | awk '{print $1}')
    [ "$M" = "$DEPLOYED_MD5" ] && say "restored librccl md5=$M" || { say "RESTORE-FAILED: librccl md5=$M want $DEPLOYED_MD5"; ok=0; }
    sudo docker stop "$C" >/dev/null 2>&1 && say "stopped $C" || { say "RESTORE-FAILED: docker stop"; ok=0; }
  fi
  if [ "$LEASE_TAKEN" = 1 ]; then
    sudo systemctl start ollama llamacpp-hub || { say "RESTORE-FAILED: systemctl start"; ok=0; }; sleep 5
    for s in ollama llamacpp-hub; do [ "$(systemctl is-active $s)" = active ] || { say "RESTORE-FAILED: $s"; ok=0; }; done
  fi
  for _ in $(seq 1 40); do at_baseline && break; sleep 3; done
  at_baseline && say "VRAM back to baseline: $(vram)" || { say "RESTORE-FAILED: VRAM $(vram)"; ok=0; }
  if [ "$ok" = 1 ] && [ "$RC" = 0 ] && [ "$ROW_OK" = 1 ]; then say "ROW-DONE clr-demo row=$ROW (restore ok)"; exit 0
  elif [ "$ok" = 1 ]; then say "ROW-FAILED clr-demo row=$ROW (exit=$RC, restore ok)"; exit "$RC"
  else say "RESTORE-FAILED clr-demo row=$ROW (exit=$RC)"; exit 9; fi
}
trap restore EXIT

: > "$P"
say "===== CLR demo row=$ROW start ====="
case "$ROW" in atomics_present) WANT_CAPS=2; WANT_DMESG=0 ;; atomics_absent) WANT_CAPS=0; WANT_DMESG=2 ;; *) say "FATAL: bad label"; exit 2 ;; esac
LSPCI=$(sudo lspci -vv 2>/dev/null) || { say "FATAL: lspci"; exit 5; }
CAPS=$(printf '%s\n' "$LSPCI" | grep -c "AtomicOpsCap: Routing- 32bit+ 64bit+"); NCAP=$(printf '%s\n' "$LSPCI" | grep -c AtomicOpsCap)
DMESG_HITS=$(sudo dmesg 2>/dev/null | grep -c "PCIE atomic ops is not supported")
say "host=$(hostname) kernel=$(uname -r) caps=$CAPS/$NCAP dmesg=$DMESG_HITS"
{ [ "$NCAP" -ge 2 ] && [ "$CAPS" -eq "$WANT_CAPS" ] && [ "$DMESG_HITS" -eq "$WANT_DMESG" ]; } || { say "FATAL: platform does not match label $ROW"; exit 5; }
at_baseline || { say "FATAL: VRAM not at baseline"; exit 5; }
[ "$(container_state)" = stopped ] || { say "FATAL: $C not stopped"; exit 5; }
[ -f /data/rccl-build/clr-build/hipamd/lib/libamdhip64.so ] || { say "FATAL: patched runtime not built"; exit 5; }
PATCHED_MD5=$(md5sum /data/rccl-build/clr-build/hipamd/lib/libamdhip64.so.7.14.60850-0000000 | cut -c1-32)
say "patched libamdhip64 md5=$PATCHED_MD5"

say "stopping services"; LEASE_TAKEN=1
sudo systemctl stop ollama llamacpp-hub || { say "FATAL: stop services"; exit 3; }; sleep 5
sudo docker start "$C" >/dev/null || { say "FATAL: docker start"; exit 3; }; sleep 5

# the SDK's runtime files, resolved through their symlinks, and their stock md5s
CORE_REAL=$(sudo docker exec "$C" readlink -f $SP/_rocm_sdk_core/lib/libamdhip64.so.7)
DEVEL_REAL=$(sudo docker exec "$C" readlink -f $SP/_rocm_sdk_devel/lib/libamdhip64.so.7)
STOCK_CORE_MD5=$(sudo docker exec "$C" md5sum "$CORE_REAL" | cut -c1-32)
STOCK_DEVEL_MD5=$(sudo docker exec "$C" md5sum "$DEVEL_REAL" | cut -c1-32)
say "SDK runtime: core $CORE_REAL md5=$STOCK_CORE_MD5 ; devel $DEVEL_REAL md5=$STOCK_DEVEL_MD5"
SWAPPED=0
swap_in() {
  sudo docker exec "$C" bash -lc "cp -p $CORE_REAL $CORE_REAL.stock && cp -p $DEVEL_REAL $DEVEL_REAL.stock && cp $PATCHED/libamdhip64.so.7.14.60850-0000000 $CORE_REAL && cp $PATCHED/libamdhip64.so.7.14.60850-0000000 $DEVEL_REAL" || return 1
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

# the probe, compiled in the container once
sudo docker exec "$C" bash -lc "cd /rb/pa && [ -x hipgate3 ] || $SP/_rocm_sdk_devel/bin/hipcc -O1 hipgate3.cpp -o hipgate3" >> "$P" 2>&1 || { say "FATAL: hipgate3 did not compile"; exit 3; }

cell() {   # cell <runtime:stock|patched> <what:probe|collective> -> appends one JSON row
  local RT=$1
  local WHAT=$2
  local ENV="" RC ERR NAMED LOADMSG MAPPED PASSED
  local LOG="$D/clrdemo-$ROW-$RT-$WHAT.log"
  [ "$RT" = patched ] && ENV="AMD_LOG_LEVEL=1"
  say "----- cell row=$ROW runtime=$RT what=$WHAT -----"
  if [ "$RT" = patched ]; then swap_in || { say "FATAL: could not put the patched runtime in place"; exit 4; }; fi
  if [ "$WHAT" = probe ]; then
    sudo timeout -k 10 120 docker exec "$C" bash -lc "cd /rb/pa && env $ENV ./hipgate3" > "$LOG" 2>&1; RC=$?
  else
    sudo docker exec "$C" bash -lc "cp $STOCK $SDL/librccl.so.1 && cp $STOCK $SDD/librccl.so.1" || { say "FATAL: install stock rccl"; exit 4; }
    sudo timeout -k 15 300 docker exec -e AR_OUT=/rb/pa/clrdemo-cc.jsonl -e NCCL_P2P_DISABLE=1 -e HSA_ENABLE_SDMA=0 "$C" bash -lc "cd /rb/b1 && env $ENV torchrun --nproc_per_node 2 collective_correctness.py" > "$LOG" 2>&1; RC=$?
  fi
  # which runtime the measuring process actually mapped, with its md5: torch for the collective cell, the probe binary otherwise
  if [ "$WHAT" = collective ]; then
    MAPPED=$(sudo docker exec "$C" bash -lc "env $ENV python3 -c \"import torch,hashlib; p=sorted({l.split()[-1] for l in open('/proc/self/maps') if 'amdhip64' in l}); print(p[0], hashlib.md5(open(p[0],'rb').read()).hexdigest()) if p else print('none')\"" 2>/dev/null | tail -1)
  else
    MAPPED=$(sudo docker exec "$C" bash -lc "env $ENV LD_DEBUG=libs /rb/pa/hipgate3 2>&1 >/dev/null | grep -m1 -oE 'calling init: [^ ]*libamdhip64[^ ]*' | sed 's/calling init: //'" 2>/dev/null)
    [ -n "$MAPPED" ] && MAPPED="$MAPPED $(sudo docker exec "$C" md5sum "$MAPPED" 2>/dev/null | cut -c1-32)"
  fi
  if [ "$RT" = patched ]; then swap_out || { say "FATAL: stock runtime not restored after the cell"; exit 4; }; fi
  NAMED=$(grep -ac "hipErrorHostcallUnsupported" "$LOG"); LOADMSG=$(grep -ac "declares hidden_hostcall_buffer" "$LOG")
  ERR=$(grep -aoE "hipErrorHostcallUnsupported|hipErrorIllegalState|the operation cannot be performed in the present state|launch of [^ ]+ refused[^\"]{0,80}" "$LOG" | head -1)
  PASSED=$(grep -ao '[0-9]*/12 cases pass' "$LOG" | tail -1 | cut -d/ -f1)
  say "cell rc=$RC named_error_lines=$NAMED load_messages=$LOADMSG passed=${PASSED:-n/a} first_error=${ERR:-none}"
  sudo python3 - "$OUT" "$ROW" "$RT" "$WHAT" "$RC" "$NAMED" "$LOADMSG" "${PASSED:-}" "${ERR:-}" "$LOG" "$PATCHED_MD5" "$CAPS" "$DMESG_HITS" "${MAPPED:-}" <<'PY'
import json, sys
out, row, rt, what, rc, named, loadmsg, passed, err, log, pmd5, caps, dm, mapped = sys.argv[1:15]
tail = "".join(open(log, errors="replace").readlines()[-8:])
rec = {"kind": "clr_demo_cell", "row": row, "runtime": rt, "what": what, "rc": int(rc),
       "named_error_lines": int(named), "load_messages": int(loadmsg),
       "correctness_passed": int(passed) if passed else None, "error": err or None,
       "patched_libamdhip64_md5": pmd5, "root_ports_with_completer_support": int(caps),
       "dmesg_no_atomics_lines": int(dm), "mapped_amdhip64": mapped or None, "log_tail": tail}
open(out, "a").write(json.dumps(rec) + "\n")
PY
}
for RT in stock patched; do cell $RT probe; cell $RT collective; done
ROW_OK=1
say "four cells recorded; restoring"
