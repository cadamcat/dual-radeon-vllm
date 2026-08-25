#!/bin/bash
#
# ===========================================================================
# NOTE FOR READERS
#
# This is the ACTUAL script that produced our working library, kept verbatim
# (original comments included, some in Chinese) rather than tidied up, so you
# can see exactly what was done. Paths are specific to our container — adapt:
#
#   SRC=...     RCCL source tree. Get the verified version with:
#                 git clone --depth 1 -b release/rocm-rel-7.1.1.1 \
#                   https://github.com/ROCm/rccl.git
#               That is 2.27.7. Do NOT use 2.30.4 from the rocm-systems
#               monorepo — NDEBUG is not sufficient there, and we have tested
#               that on hardware. See ../docs/open-questions.md section 0.
#
#   PATH=...    points at a pip-installed ROCm SDK. On a conventional ROCm
#               install use $ROCM_PATH/bin and $ROCM_PATH/lib/llvm/bin.
#
#   /work       our bind-mount. Anywhere writable works.
#
# To target GPUs other than gfx1100, add to the cmake/configure step:
#   -DGPU_TARGETS="gfx1100;gfx1101;gfx1102" -DAMDGPU_TARGETS="<same>"
# Use explicit architecture names (RCCL's newer device linker rejects
# generic targets such as gfx11-generic; see ../docs/deploy-vllm.md).
#
# Verify the result before deploying:
#   ./verify-nohostcall.sh <lib>     hostcall must be 0
#   ./check-symbols.sh    <lib>      all torch symbols must be present
# ===========================================================================
#
# build-rccl-nohostcall.sh — rebuild RCCL 2.27.7 (b38 血统) with device asserts
# compiled OUT (-DNDEBUG) so its kernels carry NO hidden_hostcall_buffer and thus
# dispatch fine on the VFIO guest that lacks PCIe atomics. Produces librccl-final.so.
#
# WHY: stock RCCL device compile uses RCCL's own -O3 target flags that BYPASS
# CMAKE_CXX_FLAGS_RELEASE, so -DNDEBUG never reaches the device pass -> device
# assert() -> __assert_fail + __ockl_fprintf -> hidden_hostcall_buffer in every
# kernel -> ROCr refuses AQL dispatch without PCIe atomics ("present state").
#
# RUN INSIDE the `rccl-build` container (has 7.14 toolchain + RCCL source at
# /work/rccl-rocm-7.1.1, /work = /data/rccl-build on host). ~85 min (device LTO).
set -e
SRC=/work/rccl-rocm-7.1.1
ARCH="${ARCH:-gfx1100}"
export PATH=/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel/bin:/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel/lib/llvm/bin:$PATH

# 1. THE FIX: force -DNDEBUG onto every compile (incl. the device kernel pass)
cd $SRC
cp -n CMakeLists.txt CMakeLists.txt.bak || true
grep -q 'add_compile_definitions(NDEBUG)' CMakeLists.txt || \
  sed -i '/^project(rccl CXX)/a add_compile_definitions(NDEBUG) # VFIO-hostcall-fix' CMakeLists.txt

# 2. reconfigure + verify NDEBUG now reaches the device compile (must print > 0)
#    A tree that has never been configured has no build/ and no cached cmake
#    arguments, so `cmake .` there does nothing useful. Configure it first if so;
#    these are the options this build used.
if [ ! -f build/build.ninja ]; then
  echo "no configured build/ — configuring one"
  cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGPU_TARGETS="$ARCH" \
    -DCMAKE_PREFIX_PATH="$(dirname "$(command -v hipcc)")/.."
fi
cd build && cmake . >/dev/null
echo "NDEBUG count in build.ninja (want >0): $(grep -c NDEBUG build.ninja)"

# 3. rebuild — ~85 min device LTO link (peak ~18.5G RAM; keep an 8G swap as insurance)
ninja

# 4. verify device kernels carry NO hostcall + KEEP ncclCommDump
llvm-objdump --offloading librccl.so.1.0 >/dev/null 2>&1
DEV=$(ls librccl.so.1.0.*${ARCH} 2>/dev/null | head -1)
if [ -z "$DEV" ]; then
  echo "ERROR: no device image for ${ARCH}. Nothing was verified." >&2
  echo "       If you built for another target, re-run with ARCH=<gfx target>." >&2
  exit 3
fi
# grep -c exits 1 on a zero count, and a command substitution inside echo hides
# that status entirely, so these used to print and continue no matter what.
# Capture, print, then assert.
# grep -c prints "0" and exits 1 on no match, so `|| true` cannot tell a genuine
# zero from a readelf that never ran — and ${HC:-1} does not fire either, because
# HC is "0" rather than empty. Check the reader's own status first.
if ! RAW=$(llvm-readelf --notes "$DEV" 2>/dev/null); then
  echo "ERROR: llvm-readelf could not read the device image; hostcall state unknown." >&2
  exit 4
fi
HC=$(printf '%s\n' "$RAW" | grep -ic hidden_hostcall_buffer || true)
HC=${HC:-1}
CD=$(llvm-objdump -T librccl.so.1.0 2>/dev/null | grep -c ncclCommDump || true)
rm -f librccl.so.1.0.*${ARCH} librccl.so.1.0.*host* 2>/dev/null || true
echo "hidden_hostcall_buffer (want 0): $HC"
echo "ncclCommDump export (want >= 1): $CD"
if [ "${HC:-1}" -ne 0 ]; then
  echo "ERROR: device image still carries a hostcall buffer. NDEBUG did not reach" >&2
  echo "       the DEVICE pass; see docs/root-cause.md §5." >&2
  exit 4
fi
if [ "${CD:-0}" -lt 1 ]; then
  echo "ERROR: ncclCommDump is not exported; torch will refuse to load this." >&2
  echo "       Build the companion dump stub and patchelf --add-needed it, or" >&2
  echo "       keep the symbol in the build. See NOTICE.md and the release page." >&2
  exit 5
fi

# 5. patchelf: drop real librocm_smi64 (poisons torch's amdsmi -> device_count=0),
#    add the rsmi stub as the provider of the 9 rsmi_* symbols librccl imports
cp librccl.so.1.0 /work/librccl-final.so
patchelf --remove-needed librocm_smi64.so.1 /work/librccl-final.so
patchelf --add-needed  librsmi_stub.so.1    /work/librccl-final.so
echo "DONE -> /work/librccl-final.so  (= /data/rccl-build/librccl-final.so). Deploy with deploy-tp2.sh."
