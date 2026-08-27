#!/bin/bash
# Build the reporter's Reproducer 1 (rccl-tests) inside our container image.
# Compile only, no GPU needed, so this can run while the GPU arms are busy.
set -eu
SDD=/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel
SDL=/opt/python/lib/python3.14/site-packages/_rocm_sdk_libraries/lib
export PATH=$SDD/bin:$SDD/llvm/bin:$PATH
export ROCM_PATH=$SDD HIP_PATH=$SDD
cd /work
if [ ! -d rccl-tests ]; then
  git clone --depth 1 https://github.com/ROCm/rccl-tests
fi
cd rccl-tests
git log --oneline -1 | sed 's/^/rccl-tests commit: /'
make MPI=0 HIP_HOME=$SDD RCCL_HOME=$SDL NCCL_HOME=$SDL GPU_TARGETS=gfx1100 -j"$(nproc)" 2>&1 | tail -25
echo "--- built binaries ---"
ls -la build/ 2>/dev/null | head -20
echo "BUILD_DONE"
