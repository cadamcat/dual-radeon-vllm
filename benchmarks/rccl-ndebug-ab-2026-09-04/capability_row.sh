#!/bin/bash
# B2, one row of the capability matrix: three libraries against whatever the
# platform currently advertises. Runs on the GUEST HOST and TAKES THE LEASE.
#
#   b2_row.sh <row-label>      row-label is "atomics_present" or "atomics_absent"
#
# The row label is NOT trusted: the script reads the root ports itself and
# records what it found, so a mislabelled run is visible in the data rather
# than silently wrong.
set -u
ROW="${1:?row label}"
B=/data/rccl-build/b1
C=vllm-tp2
SDL=/opt/python/lib/python3.14/site-packages/_rocm_sdk_libraries/lib
SDD=/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel/lib
DEPLOYED_IN_C=/rb/librccl-final.so
DEPLOYED_MD5=ab5b50f0d84806ed7fbe0f4f560151ff
BASELINE=27971584
OUT=$B/capability-matrix.jsonl
P=$B/B2-$ROW.txt

say() { echo "$(date -u +%H:%M:%S) | $*" | tee -a "$P"; }
vram() { cat /sys/class/drm/card*/device/mem_info_vram_used | tr '\n' ' '; }

LEASE_TAKEN=0
restore() {
  RC=$?
  say "--- restore (exit=$RC) ---"
  if sudo docker inspect "$C" >/dev/null 2>&1 && \
     [ "$(sudo docker inspect -f '{{.State.Running}}' "$C")" = true ]; then
    sudo docker exec "$C" bash -lc "cp $DEPLOYED_IN_C $SDL/librccl.so.1 && cp $DEPLOYED_IN_C $SDD/librccl.so.1"
    M=$(sudo docker exec "$C" md5sum $SDL/librccl.so.1 | awk '{print $1}')
    say "restored librccl md5=$M (want $DEPLOYED_MD5)"
    sudo docker stop "$C" >/dev/null 2>&1 && say "stopped $C"
  fi
  [ "$LEASE_TAKEN" = 1 ] && { sudo systemctl start ollama llamacpp-hub; sleep 5
    say "services: $(systemctl is-active ollama) $(systemctl is-active llamacpp-hub)"; }
  say "VRAM: $(vram)"
  say "===== B2 row $ROW ended (exit=$RC) ====="
}
trap restore EXIT

: > "$P"
say "===== B2 row=$ROW start ====="

# what the platform actually says, recorded rather than assumed
CAPS=$(sudo lspci -vv 2>/dev/null | grep -c "AtomicOpsCap: Routing- 32bit+ 64bit+")
DMESG_HITS=$(sudo dmesg 2>/dev/null | grep -c "PCIE atomic ops is not supported")
say "root ports reporting 32bit+ 64bit+ : $CAPS"
say "dmesg 'PCIE atomic ops is not supported' lines: $DMESG_HITS"
sudo lspci -vv 2>/dev/null | grep "AtomicOpsCap" | sed 's/^/    /' | tee -a "$P"

say "stopping ollama and llamacpp-hub"
sudo systemctl stop ollama llamacpp-hub || { say "FATAL: could not stop services"; exit 3; }
LEASE_TAKEN=1
sleep 5
sudo docker start "$C" >/dev/null || { say "FATAL: could not start $C"; exit 3; }
sleep 5

for CELL in stock2304:librccl-stock2304.so nondebug:librccl-nondebug-deploy.so ndebug:librccl-ndebug-deploy.so; do
  LIBNAME=${CELL#*:}; ARM=${CELL%%:*}
  MD5=$(md5sum "$B/$LIBNAME" | awk '{print $1}')
  say "----- cell row=$ROW library=$ARM md5=$MD5 -----"
  sudo docker exec "$C" bash -lc "cp /rb/b1/$LIBNAME $SDL/librccl.so.1 && cp /rb/b1/$LIBNAME $SDD/librccl.so.1" \
    || { say "FATAL: install failed"; exit 4; }
  GOT=$(sudo docker exec "$C" md5sum $SDL/librccl.so.1 | awk '{print $1}')
  [ "$GOT" = "$MD5" ] || { say "FATAL: installed $GOT want $MD5"; exit 4; }

  LOG="$B/b2-$ROW-$ARM.log"
  sudo rm -f "$B/b2cell.jsonl" "$B/b2cell.rank1.jsonl" "$LOG"
  sudo timeout 600 docker exec -e AR_OUT=/rb/b1/b2cell.jsonl \
    -e NCCL_P2P_DISABLE=1 -e HSA_ENABLE_SDMA=0 \
    "$C" bash -lc "cd /rb/b1 && torchrun --nproc_per_node 2 collective_correctness.py" \
    > "$LOG" 2>&1
  RC=$?
  PASSED=$(sudo grep -o '[0-9]*/12 cases pass' "$LOG" | tail -1 | cut -d/ -f1)
  ERRLINE=$(sudo grep -oE "(hipError[A-Za-z]*|the operation cannot be performed in the present state|RuntimeError: [^\"]{0,120}|invalid device function)" "$LOG" | head -1)
  say "cell rc=$RC passed=${PASSED:-0}/12 error=${ERRLINE:-none}"
  sudo python3 - "$OUT" "$ROW" "$ARM" "$MD5" "$RC" "${PASSED:-0}" "$CAPS" "$DMESG_HITS" "${ERRLINE:-}" "$LOG" <<'PY'
import json, sys, os
out, row, arm, md5, rc, passed, caps, dmesg_hits, err, log = sys.argv[1:11]
tail = ""
try:
    tail = "".join(open(log, errors="replace").readlines()[-6:])
except OSError:
    pass
rec = {"kind": "capability_cell", "row": row, "library": arm, "md5": md5,
       "rc": int(rc), "correctness_passed": int(passed), "correctness_total": 12,
       "dispatched": int(rc) == 0,
       "root_ports_with_completer_support": int(caps),
       "dmesg_no_atomics_lines": int(dmesg_hits),
       "error": err or None, "log_tail": tail}
with open(out, "a") as fh:
    fh.write(json.dumps(rec) + "\n")
PY
done
say "row $ROW complete"
