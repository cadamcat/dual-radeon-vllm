#!/usr/bin/env bash
# check-platform.sh — one-shot triage for the "no PCIe atomics -> no hostcall" bug.
#
# Answers three questions without compiling anything:
#   1. Does the kernel say PCIe atomics are unavailable to the GPUs?
#   2. Can AtomicOps reach each GPU, and if not, which port stops them?
#   3. Does the RCCL you are about to use require a hostcall?
#
# Question 2 walks the whole bridge chain rather than one hop, because *where*
# it breaks decides whether reslotting the card can help. Thanks to @adderek in
# ROCm/ROCm#6520 for the case that shows why: 00:01.2 Routing+ above 03:00.0
# Routing- means every chipset-fed slot is equally dead, and only CPU-direct
# lanes are worth trying.
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

say "== 2. PCIe view: can AtomicOps reach each GPU? =="
# This mirrors pci_enable_atomic_ops_to_root() in drivers/pci/pci.c, which is
# what amdgpu calls with COMP32|COMP64 to set have_atomics_support. Two
# different bits are involved and they are easy to confuse:
#
#   root port          must advertise 32-bit and 64-bit AtomicOp COMPLETER
#                      support. Its own Routing bit is about peer-to-peer
#                      between root ports and is NOT consulted here.
#   switch ports       every upstream/downstream port between the GPU and the
#   in between         root port must advertise AtomicOp ROUTING, and upstream
#                      ports must not have egress blocking enabled.
#
# Checking Routing on the root port, as an earlier version of this script did,
# reports healthy machines as broken: consumer root complexes commonly show
# "Routing- 32bit+ 64bit+", which passes.
if command -v lspci >/dev/null 2>&1; then
  gpus=$(lspci -D 2>/dev/null | grep -iE 'VGA|Display|3D controller' | grep -i 'AMD/ATI' | awk '{print $1}')
  if [ -z "$gpus" ]; then
    say "   no AMD GPU found via lspci."
  else
    for g in $gpus; do
      say "   GPU $g"
      # walk the sysfs device tree up to the host bridge, collecting every PCI
      # bridge on the way. prepending puts the chain in CPU-first order.
      chain=""
      p=$(readlink -f "/sys/bus/pci/devices/$g" 2>/dev/null)
      while [ -n "${p:-}" ] && [ "$p" != "/" ]; do
        p=$(dirname "$p")
        b=$(basename "$p")
        case "$b" in
          *:*:*.*) [ -e "/sys/bus/pci/devices/$b" ] || break; chain="$b $chain" ;;
          *) break ;;
        esac
      done
      if [ -z "$chain" ]; then
        say "     (could not resolve the bridge chain)"
        continue
      fi
      root_fail=""; switch_break=""; unknown=0
      for b in $chain; do
        # lspci honours only the LAST -s, so query one device per invocation
        info=$(lspci -vvs "$b" 2>/dev/null)
        typ=$(printf '%s\n' "$info" | grep -oE 'Express \(v[0-9]\) [A-Za-z]+ Port' | head -1)
        typ=${typ##*) }
        cap=$(printf '%s\n' "$info" | grep -i "AtomicOpsCap" | head -1)
        cap=${cap#"${cap%%[![:space:]]*}"}
        ctl=$(printf '%s\n' "$info" | grep -i "AtomicOpsCtl" | head -1)
        if [ -z "$cap" ]; then
          unknown=$((unknown + 1))
          say "     $b  $(printf %-17s "${typ:-bridge}")AtomicOpsCap not reported (needs root)"
          continue
        fi
        case "$typ" in
          "Root Port")
            # amdgpu asks for COMP32|COMP64; the Routing bit here is irrelevant
            case "$cap" in
              *32bit+*64bit+*) say "     $b  $(printf %-17s "root port")$cap" ;;
              *) root_fail="$b"; say "     $b  $(printf %-17s "root port")$cap   <- no 32/64-bit AtomicOp completer" ;;
            esac ;;
          "Upstream Port"|"Downstream Port")
            case "$cap" in
              *Routing+*)
                say "     $b  $(printf %-17s "$(echo "$typ" | tr "[:upper:]" "[:lower:]")")$cap"
                case "$ctl" in *EgressBlck+*) [ -z "$switch_break" ] && switch_break="$b"
                  say "         ^ egress blocking is ENABLED here, which stops AtomicOps" ;; esac ;;
              *) [ -z "$switch_break" ] && switch_break="$b"
                 say "     $b  $(printf %-17s "$(echo "$typ" | tr "[:upper:]" "[:lower:]")")$cap   <- does not route AtomicOps" ;;
            esac ;;
          *)
            say "     $b  $(printf %-17s "${typ:-bridge}")$cap" ;;
        esac
      done
      if [ -n "$root_fail" ]; then
        say "     >> AtomicOps cannot reach this GPU: root port $root_fail does not"
        say "        advertise 32/64-bit AtomicOp completer support, so nothing below"
        say "        it can use them. No slot on this host changes that."
        [ "$verdict_atomics" = "unknown" ] && verdict_atomics="absent"
      elif [ -n "$switch_break" ]; then
        say "     >> AtomicOps cannot reach this GPU: switch port $switch_break blocks"
        say "        them. The root port is fine, so every slot below $switch_break is"
        say "        equally affected while lanes that bypass it (usually CPU-direct)"
        say "        are not. Moving the card to a CPU-attached slot should work."
        [ "$verdict_atomics" = "unknown" ] && verdict_atomics="absent"
      elif [ "$unknown" -gt 0 ]; then
        say "     >> inconclusive: $unknown bridge(s) did not report AtomicOpsCap."
        say "        rerun as root for the full lspci output."
      else
        say "     >> the path can carry AtomicOps: root port completes them and every"
        say "        switch in between routes them."
      fi
    done
    say "   note: QEMU advertises completer support on an emulated root port"
    say "   automatically since 8.1.0, but declines under any of seven conditions."
    say "   The one that bites on the Proxmox default is a multifunction device:"
    say "   hostpci0: 0b:00 rather than 0b:00.0. If yours is already .0 and the"
    say "   port still shows 32bit- 64bit-, one of the other six applies — see"
    say "   docs/vfio-atomics.md."
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
      # `grep -c` prints 0 AND exits 1 when it matches nothing, so `|| echo 0`
      # appends a second line and the comparison below dies with "integer
      # expression expected" — landing in the else branch, which reports the
      # good verdict. deploy-tp2.sh already gets this right; this copy did not.
      # A failed readelf has to be distinguished from a genuine zero as well.
      if notes=$("$RE" --notes "$dev" 2>/dev/null); then
        n=$(printf '%s\n' "$notes" | grep -ic hidden_hostcall_buffer || true)
        n=${n:-0}
        say "   hidden_hostcall_buffer occurrences: $n"
        if [ "$n" -gt 0 ]; then
          say "   >> This RCCL REQUIRES a hostcall. On a platform without atomics it will fail."
          verdict_hostcall="required"
        else
          say "   >> This RCCL needs no hostcall. It will dispatch even without atomics."
          verdict_hostcall="none"
        fi
      else
        say "   >> llvm-readelf could not read the device image. Hostcall state UNKNOWN."
        verdict_hostcall="unknown"
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
