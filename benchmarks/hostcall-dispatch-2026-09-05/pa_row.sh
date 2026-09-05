#!/bin/bash
# One row of the paged-attention dispatch matrix: three arms of pa_probe.py
# against whatever the platform currently advertises. Runs on the GUEST HOST
# and TAKES THE LEASE [INV-012]. Skeleton from B2's capability_row.sh, revised
# twice after TASK-0007 / TASK-0007b (AGENTS/reviews/REVIEW-0007.md).
#
#   pa_row.sh <row-label>      atomics_present | atomics_absent
#
# Nothing is trusted: the label is checked against lspci/dmesg (and the reads
# themselves are checked) before the lease is taken; the image is checked; each
# arm writes its own file which must hold exactly one row for (row, arm) with a
# done marker; the probe's own attribute reading must agree with the label; the
# row's cells are appended and counted; and ROW-DONE is printed by the EXIT
# trap only after restoration succeeded, so it cannot precede RESTORE-FAILED.
set -u
ROW="${1:?row label}"
B=/data/rccl-build/pa
C=vllm-tp2
IMAGE_MUST_CONTAIN="vllm_0.23.0"
BASELINE=27971584
OUT=$B/pa-cells.jsonl
P=$B/PA-$ROW.txt
ARMS="as_shipped ck_forced triton_forced"
PROFILE_FLAG=${PA_PROFILE:+--profile}     # PA_PROFILE=1: record the kernel names each arm ran (torch.profiler)

say() { echo "$(date -u +%H:%M:%S) | $*" | tee -a "$P"; }
vram() { cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | tr '\n' ' '; }
at_baseline() {   # exactly two cards, both at the exact baseline
  local vals n; vals=$(cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null); n=$(echo "$vals" | grep -c .)
  [ "$n" -eq 2 ] || return 1
  for v in $vals; do [ "$v" = "$BASELINE" ] || return 1; done
  return 0
}
container_state() {   # running | stopped | unknown
  local r; r=$(sudo docker inspect -f '{{.State.Running}}' "$C" 2>/dev/null) || { echo unknown; return; }
  [ "$r" = true ] && echo running || echo stopped
}
kill_probes() {   # leftover probe processes inside the container; prints the number killed, or ERR
  # The pattern is written as [p]a_probe.py so that this scanner's own command line, which
  # contains the bracketed form, does not match itself (TASK-0007b N1).
  sudo docker exec "$C" bash -lc 'n=0; for p in /proc/[0-9]*; do pid=${p#/proc/}; [ "$pid" = "$$" ] && continue; [ "$pid" = "$PPID" ] && continue; if tr "\0" " " < $p/cmdline 2>/dev/null | grep -q "python3 [p]a_probe.py"; then kill -9 $pid 2>/dev/null && n=$((n+1)); fi; done; echo $n' 2>/dev/null || echo ERR
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
    K=$(kill_probes); say "leftover probe processes killed: $K (container state: $st)"
    [ "$K" = ERR ] && { say "RESTORE-FAILED: could not scan the container for leftover probes"; ok=0; }
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
    say "ROW-DONE row=$ROW cells=3 in $OUT (restore ok)"; say "===== PA row $ROW ended (exit=0) ====="; exit 0
  elif [ "$ok" = 1 ]; then
    say "ROW-FAILED row=$ROW (exit=$RC, restore ok)"; say "===== PA row $ROW ended (exit=$RC) ====="; exit "$RC"
  else
    say "RESTORE-FAILED row=$ROW (exit=$RC)"; say "===== PA row $ROW ended (exit=$RC) RESTORE-FAILED ====="; exit 9
  fi
}
trap restore EXIT

: > "$P"
say "===== PA row=$ROW start ====="
for f in pa_probe.py hipattr.cpp; do [ -f "$B/$f" ] || { say "FATAL: $B/$f missing"; exit 2; }; done
case "$ROW" in
  atomics_present) WANT_CAPS=2; WANT_DMESG=0; WANT_ATTR="[1, 1]" ;;
  atomics_absent)  WANT_CAPS=0; WANT_DMESG=2; WANT_ATTR="[0, 0]" ;;
  *) say "FATAL: row label must be atomics_present or atomics_absent"; exit 2 ;;
esac
# the platform reads themselves are checked: lspci must succeed and show AtomicOpsCap lines at all
LSPCI=$(sudo lspci -vv 2>/dev/null) || { say "FATAL: lspci failed"; exit 5; }
NCAPLINES=$(printf '%s\n' "$LSPCI" | grep -c "AtomicOpsCap"); [ "$NCAPLINES" -ge 2 ] || { say "FATAL: lspci shows $NCAPLINES AtomicOpsCap lines"; exit 5; }
CAPS=$(printf '%s\n' "$LSPCI" | grep -c "AtomicOpsCap: Routing- 32bit+ 64bit+")
DMESG=$(sudo dmesg 2>/dev/null) || { say "FATAL: dmesg failed"; exit 5; }
DMESG_HITS=$(printf '%s\n' "$DMESG" | grep -c "PCIE atomic ops is not supported")
HOST=$(hostname); UUID=$(sudo cat /sys/class/dmi/id/product_uuid 2>/dev/null || echo unknown)
say "host=$HOST kernel=$(uname -r) product_uuid=$UUID"
say "root ports reporting 32bit+ 64bit+ : $CAPS (of $NCAPLINES AtomicOpsCap lines)"
say "dmesg 'PCIE atomic ops is not supported' lines: $DMESG_HITS"
printf '%s\n' "$LSPCI" | grep "AtomicOpsCap" | sed 's/^/    /' | tee -a "$P"
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
K=$(kill_probes); say "stale probe processes killed before start: $K"; [ "$K" = ERR ] && { say "FATAL: cannot scan the container"; exit 3; }

# hipattr: the platform's own answer, compiled in the container against the SDK's runtime (CPU only)
SP=/opt/python/lib/python3.14/site-packages
sudo docker exec "$C" bash -lc "cd /rb/pa && rm -f hipattr && $SP/_rocm_sdk_devel/lib/llvm/bin/clang++ -O1 -D__HIP_PLATFORM_AMD__ -I$SP/_rocm_sdk_devel/include hipattr.cpp $SP/_rocm_sdk_core/lib/libamdhip64.so.7 -Wl,-rpath,$SP/_rocm_sdk_core/lib -o hipattr && ./hipattr" > "$B/hipattr-$ROW.txt" 2>&1 \
  || { say "FATAL: hipattr did not build or run: $(tail -3 "$B/hipattr-$ROW.txt" | tr '\n' ' ')"; exit 3; }
grep -q "HIPATTR_DONE devices=2" "$B/hipattr-$ROW.txt" || { say "FATAL: hipattr did not see two devices: $(cat "$B/hipattr-$ROW.txt" | tr '\n' ' ')"; exit 3; }
say "hipattr: $(grep -a '^device=' "$B/hipattr-$ROW.txt" | tr '\n' ';')"

for ARM in $ARMS; do
  say "----- cell row=$ROW arm=$ARM -----"
  CELL="$B/pa-$ROW-$ARM.jsonl"; LOG="$B/pa-$ROW-$ARM.log"
  sudo rm -f "$CELL" "$LOG" "$B/pa-$ROW-$ARM.amdlog.jsonl" "$B/amdlog-$ROW-$ARM.txt" "$B/amdlog-$ROW-$ARM.full" "$B/amdlog-$ROW-$ARM.full.gz" || { say "FATAL: could not clear stale files for $ARM"; exit 6; }
  [ -e "$CELL" ] && { say "FATAL: stale $CELL survived rm"; exit 6; }
  sudo timeout -k 15 330 docker exec "$C" bash -lc "cd /rb/pa && timeout -k 10 280 python3 pa_probe.py --arm $ARM --row $ROW --out /rb/pa/pa-$ROW-$ARM.jsonl --caps $CAPS --dmesg $DMESG_HITS --host $HOST --image '$IMAGE' --vm 101@pve --product-uuid $UUID $PROFILE_FLAG" > "$LOG" 2>&1
  RC=$?
  K=$(kill_probes); [ "$K" != 0 ] && say "WARNING: leftover probe processes after arm $ARM: $K"
  grep -a "^row=" "$LOG" | tail -1 | sed 's/^/    /' | tee -a "$P"
  say "cell rc=$RC"
  NROWS=$(sudo grep -ac "\"row\": \"$ROW\", \"arm\": \"$ARM\"" "$CELL" 2>/dev/null || true); NROWS=${NROWS:-0}
  DONE=$(grep -ac PA_PROBE_DONE "$LOG" || true)
  if [ "$NROWS" -ne 1 ] || [ "$DONE" -ne 1 ]; then
    say "ROW-FAILED: arm=$ARM wrote $NROWS row(s), done markers $DONE"; exit 6
  fi
  OUTCOME=$(sudo python3 -c "import json,sys; r=json.loads(open(sys.argv[1]).readline()); print(r['outcome'])" "$CELL") || { say "ROW-FAILED: cannot parse $CELL"; exit 6; }
  ATTR=$(sudo python3 -c "import json,sys; r=json.loads(open(sys.argv[1]).readline()); print(json.dumps(r.get('host_native_atomic_supported')), r.get('attr_matches_platform'), r.get('hipattr_ok'), r.get('ck_op_wrapped'), r.get('ck_op_calls'), r.get('fallback_warning_seen'))" "$CELL")
  say "outcome=$OUTCOME attr/match/hipattr_ok/op_wrapped/op_calls/fallback: $ATTR"
  case "$ATTR" in "$WANT_ATTR True True"*) ;; *) say "ROW-FAILED: arm=$ARM the probe's attribute reading does not match the label $ROW (want $WANT_ATTR, attribute and props agreeing, hipattr ok)"; exit 6 ;; esac
  # PA_FORCE_AMDLOG=1 captures the ShaderName for every arm, refused or not: which
  # kernel instantiation the runtime actually dispatched is evidence either way.
  # (An earlier version put ${PA_FORCE_AMDLOG:+a|b} inside the case pattern; the
  # expanded "|" is a literal there, not an alternation, and nothing matched.)
  WANT_AMDLOG=0
  case "$OUTCOME" in
    harness_error|gate_declined) say "ROW-FAILED: arm=$ARM $OUTCOME"; exit 6 ;;
    refused|not_launched|launched_no_output|dispatch_error) WANT_AMDLOG=1 ;;
    dispatched_ok|dispatched_wrong|dispatched_nan) [ -n "${PA_FORCE_AMDLOG:-}" ] && WANT_AMDLOG=1 ;;
  esac
  case "$WANT_AMDLOG" in
    1)
      say "arm $ARM: $OUTCOME -- capturing an AMD_LOG_LEVEL=4 excerpt from a second run, separate file"
      FULL="$B/amdlog-$ROW-$ARM.full"
      sudo timeout -k 15 330 docker exec -e AMD_LOG_LEVEL=4 -e AMD_LOG_MASK=0xFFFFFFFF "$C" bash -lc "cd /rb/pa && timeout -k 10 280 python3 pa_probe.py --arm $ARM --row $ROW --out /rb/pa/pa-$ROW-$ARM.amdlog.jsonl --caps $CAPS --dmesg $DMESG_HITS --host $HOST --image '$IMAGE' --vm 101@pve --product-uuid $UUID $PROFILE_FLAG" > "$FULL" 2>&1
      K=$(kill_probes); [ "$K" != 0 ] && say "WARNING: leftover probe processes after the AMD-log rerun: $K"
      grep -aE "ShaderName|Pcie atomics not enabled|hostcall not supported|AQL dispatch failed|hipErrorIllegalState|Returned hipError|^row=" "$FULL" | head -200 > "$B/amdlog-$ROW-$ARM.txt"
      say "excerpt: $(wc -l < "$B/amdlog-$ROW-$ARM.txt") lines; full log $(du -h "$FULL" | cut -f1), gzipped"
      gzip -f "$FULL" ;;
  esac
done
BEFORE=$(grep -ac "\"row\": \"$ROW\"" "$OUT" 2>/dev/null || true); BEFORE=${BEFORE:-0}
for ARM in $ARMS; do sudo cat "$B/pa-$ROW-$ARM.jsonl" >> "$OUT" || { say "ROW-FAILED: could not append $ARM to $OUT"; exit 6; }; done
AFTER=$(grep -ac "\"row\": \"$ROW\"" "$OUT" || true)
[ "$((AFTER - BEFORE))" -eq 3 ] || { say "ROW-FAILED: $OUT gained $((AFTER - BEFORE)) rows, not 3"; exit 6; }
[ "$BEFORE" -ne 0 ] && say "NOTE: $OUT already held $BEFORE row(s) for $ROW from an earlier run; analyze.py keys by (row, arm), later rows win"
ROW_OK=1
say "all three arms recorded; restoring"
