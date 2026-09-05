#!/bin/bash
# CLR (the HIP runtime) from the rocm-systems commit a rocm/vllm image's SDK was
# built from, configured against the wheel SDK inside that image. CPU only.
#   clr_build.sh [patchfile]    applies the patch to the source tree before configuring
# Two trees, selected by CLR_TAG:
#   (unset)          rocm-systems 2b22ab01 (TheRock 7.14) in the vLLM 0.23 image:
#                    /rb/clr-src.tgz -> /rb/clr-src, /rb/clr-build, /rb/clr-build.log
#   CLR_TAG=rocm10   rocm-systems 6b0e43f3 (TheRock 10.0) in the vLLM 0.27 image:
#                    /rb/clr-rocm10-src.tgz -> /rb/clr-rocm10-src, /rb/clr-rocm10-build,
#                    /rb/clr-rocm10-build.log
#   CLR_TAG=rocm10a  the same source (a hard link of the tarball) with PR A alone,
#                    clr-hostcall-load-check-a.patch -> /rb/clr-rocm10a-build
# ROCM_KPACK_ENABLED=ON is what the SDK's own build has and the default lacks:
# torch's device code is kpack-split, and a runtime built without it cannot load
# a single torch kernel (SIGSEGV on the first op, 2026-09-05).
# Two build-only accommodations for the wheel layout, both logged: the SDK's
# comgr is a newer major than the 3.0 ROCclrLC.cmake asks for, and the image has
# no GL headers (rocclr's GL interop is REQUIRED), so they are apt-installed.
# The 7.14 SDK ships rocm-kpack's imported-target files without a *-config.cmake
# (a one-line stand-in is written); the 10.0 SDK ships the config and it is used.
set -u
SP=/opt/python/lib/python3.14/site-packages; DEV=$SP/_rocm_sdk_devel; CORE=$SP/_rocm_sdk_core
TAG=${CLR_TAG:-}; X=${TAG:+-$TAG}
SRC=/rb/clr$X-src; BUILD=/rb/clr$X-build; TGZ=$SRC.tgz; L=/rb/clr$X-build.log; : > $L
say() { echo "$(date -u +%H:%M:%S) | $*" | tee -a $L; }
say "tree=$SRC build=$BUILD source=$TGZ"
# CLR_FRESH=1 discards the source and build trees; otherwise the tree is kept
# (already patched) and the build is incremental.
if [ -n "${CLR_FRESH:-}" ] || [ ! -f $SRC/.patched ]; then
  rm -rf $SRC $BUILD; mkdir -p $SRC
  say "unpack"; tar xzf $TGZ -C $SRC || { say "UNPACK-FAILED"; exit 2; }
  if [ -n "${1:-}" ]; then
    say "apply $1 (md5 $(md5sum "$1" | cut -c1-32))"; (cd $SRC && patch -p1 --forward < "$1") >> $L 2>&1 || { say "PATCH-FAILED"; tail -5 $L; exit 5; }
    cp "$1" $SRC/.patched
  fi
else
  say "source tree kept (patched: $(md5sum $SRC/.patched | cut -c1-32)); incremental build"
fi
CV=$(sed -n 's/.*set(PACKAGE_VERSION "\([0-9]*\)\..*/\1/p' $DEV/lib/cmake/amd_comgr/amd_comgr-config-version.cmake | head -1)
say "SDK amd_comgr major version: ${CV:-?}"
if [ -n "$CV" ] && [ "$CV" != 3 ]; then
  sed -i "s/find_package(amd_comgr 3.0 REQUIRED CONFIG/find_package(amd_comgr $CV.0 REQUIRED CONFIG/" $SRC/projects/clr/rocclr/cmake/ROCclrLC.cmake
  say "ROCclrLC.cmake: requested comgr 3.0 -> $CV.0 (build-only accommodation)"
fi
say "GL and zstd headers via apt (mirrors.aliyun.com), if missing"
sed -i 's#archive.ubuntu.com#mirrors.aliyun.com#g; s#security.ubuntu.com#mirrors.aliyun.com#g' /etc/apt/sources.list.d/ubuntu.sources /etc/apt/sources.list 2>/dev/null
if [ ! -f /usr/include/GL/gl.h ] || [ ! -f /usr/include/zstd.h ]; then
  (apt-get update -qq && apt-get install -y -qq libgl-dev libegl-dev libglx-dev libzstd-dev) >> $L 2>&1 || { say "APT-FAILED"; tail -5 $L; exit 6; }
fi
ls /usr/include/GL/gl.h /usr/include/zstd.h >> $L 2>&1 || { say "APT-FAILED: headers missing"; exit 6; }
say "CppHeaderParser (hipamd generates hip_prof_str.h with it)"
python3 -c "import CppHeaderParser" 2>/dev/null || pip3 install -q -i https://mirrors.aliyun.com/pypi/simple/ CppHeaderParser >> $L 2>&1 || pip3 install -q CppHeaderParser >> $L 2>&1 || { say "PIP-FAILED"; exit 6; }
if [ -f $DEV/lib/cmake/rocm-kpack/rocm-kpack-config.cmake ]; then
  KPACK_DIR=$DEV/lib/cmake/rocm-kpack; say "rocm-kpack config: the SDK's own"
else
  mkdir -p /rb/kpack-cmake
  printf 'include("%s/lib/cmake/rocm-kpack/rocm-kpack-targets.cmake")\n' "$DEV" > /rb/kpack-cmake/rocm-kpack-config.cmake
  KPACK_DIR=/rb/kpack-cmake; say "rocm-kpack config: one-line stand-in including the SDK's targets file"
fi
say "configure"
cmake -G Ninja -S $SRC/projects/clr -B $BUILD \
  -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF -DROCM_KPACK_ENABLED=ON -Drocm-kpack_DIR=$KPACK_DIR \
  -DHIP_COMMON_DIR=$SRC/projects/hip -DHIPCC_BIN_DIR=$DEV/bin -DHIP_PLATFORM=amd \
  -DROCM_PATH=$DEV -DCMAKE_PREFIX_PATH="$DEV;$CORE" \
  -DCMAKE_C_COMPILER=$DEV/lib/llvm/bin/clang -DCMAKE_CXX_COMPILER=$DEV/lib/llvm/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/rb/clr$X-install >> $L 2>&1
rc=$?; say "configure rc=$rc"; [ $rc -eq 0 ] || { say "CONFIGURE-FAILED"; grep -iE "CMake Error|Could NOT|could not|No such" $L | head -20 | tee -a $L; exit 3; }
say "CONFIGURE-OK; build (ninja -j8 amdhip64)"
ninja -C $BUILD -j8 amdhip64 >> $L 2>&1
rc=$?; say "build rc=$rc"; [ $rc -eq 0 ] || { say "BUILD-FAILED"; grep -E "error:|FAILED:" $L | head -20 | tee -a $L; exit 4; }
ls -la $BUILD/hipamd/lib/libamdhip64.so* | tee -a $L
say "BUILD-OK"
