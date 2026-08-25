#!/bin/bash
# deploy-tp2.sh — inject the no-hostcall RCCL into a fresh rocm7.14 vLLM container
# so tensor-parallel (TP2) works on the VFIO dual-7900XT guest (no PCIe atomics).
#
# Root cause (see vllm-tp2-调查档案.md): stock RCCL 2.30.4 device kernels carry a
# hidden_hostcall_buffer (from device assert()), which ROCr refuses to dispatch
# when PCIe atomics are unavailable (VFIO guest) -> "operation cannot be performed
# in the present state". Fix = RCCL rebuilt with -DNDEBUG (0 hostcall).
#
# Prereqs (durable, in /data/rccl-build): librccl-final.so (NDEBUG + patchelf
# --remove-needed librocm_smi64 + patchelf --add-needed librsmi_stub.so.1 +
# ncclCommDump export), rsmi_stub.c, sitecustomize.py.
#
# Usage: run INSIDE a rocm7.14 vLLM container, from a checkout of this repository:
#
#     ./deploy/deploy-tp2.sh /path/to/librccl-final.so
#
# rsmi_stub.c and sitecustomize.py are taken from this script's own directory.
# librccl-final.so is a build artefact (see build/build-rccl-nohostcall.sh) and is
# not in the repository, so it must be given as an argument or via LIBRCCL.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
LIBRCCL="${1:-${LIBRCCL:-}}"
ARCH="${ARCH:-gfx1100}"

if [ -z "$LIBRCCL" ] || [ ! -f "$LIBRCCL" ]; then
  echo "error: give the path to your no-hostcall librccl build:" >&2
  echo "       $0 /path/to/librccl-final.so   (or set LIBRCCL=...)" >&2
  exit 2
fi
for f in "$HERE/rsmi_stub.c" "$HERE/sitecustomize.py"; do
  [ -f "$f" ] || { echo "error: missing $f — run this from a repo checkout" >&2; exit 2; }
done

SDL=/opt/python/lib/python3.14/site-packages/_rocm_sdk_libraries/lib
SDD=/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel/lib
SP=/opt/python/lib/python3.14/site-packages
CLANG=$SDD/lib/llvm/bin/clang
[ -x "$CLANG" ] || CLANG=$SDD/llvm/bin/clang

echo "[1/4] build rsmi stub (weak no-op rocm_smi so librccl loads w/o real librocm_smi64)"
$CLANG -shared -fPIC -o $SDL/librsmi_stub.so.1 "$HERE/rsmi_stub.c"

echo "[2/4] swap no-hostcall librccl into the runtime path torch loads"
cp -a $(readlink -f $SDL/librccl.so.1) $SDL/librccl.so.1.orig 2>/dev/null || true
cp "$LIBRCCL" $SDL/librccl.so.1
cp -a $(readlink -f $SDD/librccl.so.1) $SDD/librccl.so.1.orig 2>/dev/null || true
cp "$LIBRCCL" $SDD/librccl.so.1

echo "[3/4] install sitecustomize (pre-init amdsmi BEFORE torch loads librccl+stub)"
cp "$HERE/sitecustomize.py" $SP/sitecustomize.py

echo "[4/4] verify: 0 hostcall + device_count"
export PATH=$SDD/llvm/bin:$PATH
llvm-objdump --offloading $SDL/librccl.so.1 >/dev/null 2>&1
DEV=$(ls $SDL/librccl.so.1.*${ARCH} 2>/dev/null | head -1)
if [ -z "$DEV" ]; then
  echo "  ERROR: no device image for ${ARCH} — nothing was verified." >&2
  echo "         set ARCH=<your gfx target> and re-run." >&2
  rm -f $SDL/librccl.so.1.*gfx* $SDL/librccl.so.1.*host* 2>/dev/null || true
  exit 3
fi
# grep -c exits 1 when the count is 0, which is the outcome we want, so it must
# not be the last command under `set -e` and its status must not be trusted.
# Capture the number, then decide. Getting this backwards let a good library
# abort the deploy and a bad one reach DONE.
# grep -c prints "0" and exits 1 on no match, so `|| true` cannot tell a genuine
# zero from a readelf that never ran — and ${HC:-1} does not fire either, because
# HC is "0" rather than empty. Check the reader's own status first.
if ! RAW=$(llvm-readelf --notes "$DEV" 2>/dev/null); then
  echo "ERROR: llvm-readelf could not read the device image; hostcall state unknown." >&2
  exit 4
fi
HC=$(printf '%s\n' "$RAW" | grep -ic hidden_hostcall_buffer || true)
HC=${HC:-1}
rm -f $SDL/librccl.so.1.*gfx* $SDL/librccl.so.1.*host* 2>/dev/null || true
echo "  hidden_hostcall_buffer (want 0): $HC"
if [ "${HC:-1}" -ne 0 ]; then
  echo "  ERROR: this library still carries a hostcall buffer. ROCr will refuse" >&2
  echo "         to dispatch it on a platform without PCIe atomics. See" >&2
  echo "         docs/root-cause.md §5 — NDEBUG must reach the DEVICE pass." >&2
  exit 4
fi

DC=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
echo "  device_count (want >= 2): $DC"
if [ "${DC:-0}" -lt 2 ]; then
  echo "  ERROR: torch sees $DC device(s); tensor-parallel needs at least 2." >&2
  echo "         Check /dev/kfd, /dev/dri and the container's --group-add." >&2
  exit 5
fi

echo "DONE. Launch: NCCL_P2P_DISABLE=1 HSA_ENABLE_SDMA=0 vllm serve <model> --tensor-parallel-size 2 ..."
