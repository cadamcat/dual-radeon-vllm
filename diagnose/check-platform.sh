#!/usr/bin/env bash
# check-platform.sh — one-shot triage for the "no PCIe atomics -> no hostcall" bug.
#
# Answers three questions without compiling anything:
#   1. Does the kernel say PCIe atomics are unavailable to the GPUs?
#   2. Do the upstream root ports advertise AtomicOp routing?
#   3. Does the RCCL you are about to use require a hostcall?
#
# Exit code: 0 = you are probably NOT affected, 1 = you probably ARE affected,
#            2 = inconclusive (missing tools / could not locate a library).
#
# No root required for 1 and 3; 2 needs privileges for full lspci output.

set -uo pipefail
verdict_atomics="unknown"
verdict_hostcall="unknown"

hr() { printf '%s\n' "------------------------------------------------------------"; }
say() { printf '%s\n' "$*"; }

say "== 1. Kernel view: does amdgpu have PCIe atomics? =="
if command -v dmesg >/dev/null 2>&1; then
  hits=$(dmesg 2>/dev/null | grep -i "atomic ops is not supported" || true)
  if [ -n "$hits" ]; then
    say "$hits"
    say ">> amdgpu reports PCIe atomics UNAVAILABLE."
    verdict_atomics="absent"
  elif dmesg 2>/dev/null | grep -qi "amdgpu"; then
    say "   no 'atomic ops is not supported' line found for amdgpu."
    say ">> atomics are probably available (or the message scrolled out of the ring buffer)."
    verdict_atomics="present"
  else
    say "   no amdgpu lines visible (need root, or the buffer wrapped)."
    say "   retry with: sudo dmesg | grep -i 'atomic ops'"
  fi
else
  say "   dmesg not available."
fi
hr

say "== 2. PCIe view: do the GPUs' upstream ports route AtomicOps? =="
if command -v lspci >/dev/null 2>&1; then
  gpus=$(lspci -D 2>/dev/null | grep -iE 'VGA|Display|3D controller' | grep -i 'AMD/ATI' | awk '{print $1}')
  if [ -z "$gpus" ]; then
    say "   no AMD GPU found via lspci."
  else
    for g in $gpus; do
      say "   GPU $g"
      # walk one hop up the sysfs device tree to the upstream port
      up=$(basename "$(dirname "$(readlink -f "/sys/bus/pci/devices/$g" 2>/dev/null)")" 2>/dev/null)
      if [ -n "${up:-}" ] && [ -e "/sys/bus/pci/devices/$up" ]; then
        cap=$(lspci -vvs "$up" 2>/dev/null | grep -i "AtomicOpsCap" || true)
        ctl=$(lspci -vvs "$up" 2>/dev/null | grep -i "AtomicOpsCtl" || true)
        say "     upstream port $up"
        [ -n "$cap" ] && say "       ${cap#"${cap%%[![:space:]]*}"}" || say "       (AtomicOpsCap not reported — try running as root)"
        [ -n "$ctl" ] && say "       ${ctl#"${ctl%%[![:space:]]*}"}"
        case "$cap" in
          *Routing-*) say "     >> Routing- : this port does NOT route AtomicOps." ;;
          *Routing+*) say "     >> Routing+ : this port routes AtomicOps." ;;
        esac
      else
        say "     (could not resolve upstream port)"
      fi
    done
    say "   note: on a QEMU guest the upstream port is an emulated pcie-root-port,"
    say "   which does not implement AtomicOp routing at all."
  fi
else
  say "   lspci not available (install pciutils)."
fi
hr

say "== 3. Library view: does your RCCL need a hostcall? =="
LIB="${1:-}"
if [ -z "$LIB" ]; then
  LIB=$(ldconfig -p 2>/dev/null | awk '/librccl\.so/{print $NF; exit}')
  [ -z "$LIB" ] && LIB=$(find /opt/rocm* /usr/lib* -name 'librccl.so*' -type f 2>/dev/null | head -1)
fi
if [ -z "$LIB" ] || [ ! -f "$LIB" ]; then
  say "   no librccl found. Pass one explicitly:  $0 /path/to/librccl.so.1"
else
  say "   inspecting: $LIB"
  RE=$(command -v llvm-readelf || command -v readelf || true)
  OD=$(command -v llvm-objdump || true)
  if [ -z "$OD" ] || [ -z "$RE" ]; then
    say "   need llvm-objdump + llvm-readelf (ROCm's llvm/bin) to inspect the device image."
  else
    tmp=$(mktemp -d); cp "$LIB" "$tmp/lib.so" 2>/dev/null
    ( cd "$tmp" && "$OD" --offloading lib.so >/dev/null 2>&1 || true )
    dev=$(ls "$tmp"/lib.so.*gfx* 2>/dev/null | head -1)
    if [ -n "$dev" ]; then
      n=$("$RE" --notes "$dev" 2>/dev/null | grep -ic hidden_hostcall_buffer || echo 0)
      say "   hidden_hostcall_buffer occurrences: $n"
      if [ "$n" -gt 0 ]; then
        say "   >> This RCCL REQUIRES a hostcall. On a platform without atomics it will fail."
        verdict_hostcall="required"
      else
        say "   >> This RCCL needs no hostcall. It will dispatch even without atomics."
        verdict_hostcall="none"
      fi
    else
      nobits=$("$RE" -S "$LIB" 2>/dev/null | grep -i 'hip_fatbin' | grep -ci NOBITS || true)
      kdir=""
      for c in "$(dirname "$LIB")/../.kpack" "$(dirname "$LIB")/.kpack"; do
        [ -d "$c" ] && kdir=$(cd "$c" && pwd) && break
      done
      if [ "${nobits:-0}" -gt 0 ] || [ -n "$kdir" ]; then
        say "   this install uses the split 'kpack' layout — device code is not in the .so."
        [ -n "$kdir" ] && say "   (it lives in $kdir/rccl_lib_gfx<arch>.kpack, an opaque format)"
        say "   >> static inspection impossible here; use the runtime probe hipgate3.cpp."
      else
        say "   could not extract a device image (no bundled gfx target?)."
      fi
    fi
    rm -rf "$tmp"
  fi
fi
hr

say "== Verdict =="
if [ "$verdict_atomics" = "absent" ] && [ "$verdict_hostcall" = "required" ]; then
  say "AFFECTED. No PCIe atomics + an RCCL that needs hostcall = guaranteed failure"
  say "at the first cross-GPU collective."
  say "Confirm decisively:  hipcc --offload-arch=\$(your arch) -O2 hipgate3.cpp -o hipgate3 && ./hipgate3"
  say "Then see ../docs/deploy-vllm.md"
  exit 1
elif [ "$verdict_atomics" = "absent" ] && [ "$verdict_hostcall" = "none" ]; then
  say "Not affected with THIS library: no atomics, but your RCCL needs no hostcall."
  exit 0
elif [ "$verdict_atomics" = "present" ]; then
  say "Probably not affected: your platform appears to route AtomicOps."
  exit 0
else
  say "INCONCLUSIVE — run hipgate3.cpp, it needs neither root nor lspci:"
  say "  hipcc --offload-arch=gfx1100 -O2 hipgate3.cpp -o hipgate3 && ./hipgate3"
  exit 2
fi
