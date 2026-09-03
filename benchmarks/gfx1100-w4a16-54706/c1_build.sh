#!/bin/bash
# c1_build.sh -- build vLLM's ROCm extension (_rocm_C) for gfx1100 with
# vllm#54706 applied, inside the vllm-tp2 container, at the container's own
# vLLM commit (9ddef7117). Builds only; swaps nothing. Runs ON THE GUEST,
# detached, and logs to $H/build.log.
#
# The container's wheel is 0.23.1.dev1+g9ddef7117.d20260715 from the rocm/vllm
# image, and 9ddef7117 is a commit of the ROCm/vllm fork, not of
# vllm-project/vllm's main: the first attempt cloned upstream, found no such
# commit, and went on building main's kernels (caught in the log). This one
# fetches the fork and refuses to continue unless HEAD is that commit.
set -u
H=${C1_DIR:-/data/rccl-build/c1b}
C=/rb/$(basename $H)
mkdir -p $H
log() { echo "$(date -u +%H:%M:%S) | $*" | tee -a $H/PROGRESS.txt; }

sudo docker start vllm-tp2 >/dev/null 2>&1 || true
log "source: ROCm/vllm at 9ddef7117"
sudo docker exec -e C1_NOPATCH="${C1_NOPATCH:-}" vllm-tp2 bash -lc "
set -e
cd $C
if [ ! -d vllm-src/.git ]; then git clone -q https://github.com/ROCm/vllm vllm-src; fi
cd vllm-src
git fetch -q origin 9ddef7117 || git fetch -q origin
git checkout -q 9ddef7117
test \"\$(git rev-parse --short=9 HEAD)\" = 9ddef7117 || { echo 'HEAD is not 9ddef7117'; exit 3; }
git rev-parse HEAD
if [ -n \"\${C1_NOPATCH:-}\" ]; then echo 'baseline: PR not applied'; else git apply --check --exclude='tests/*' /rb/c1/pr54706.diff && git apply --exclude='tests/*' /rb/c1/pr54706.diff; fi
git status --short | head
md5sum csrc/rocm/q_gemm_rdna3.cu csrc/rocm/q_gemm_rdna3_wmma.cu
" 2>&1 | tee -a $H/build.log
rc=${PIPESTATUS[0]}
[ $rc -eq 0 ] || { log "source step failed rc=$rc"; log "build exit $rc"; exit $rc; }
log "configure"
sudo docker exec -e PYTORCH_ROCM_ARCH=gfx1100 -e VLLM_TARGET_DEVICE=rocm -e MAX_JOBS=12 vllm-tp2 bash -lc "
set -e
cd $C/vllm-src
python3 -c 'import torch; print(torch.__version__, torch.utils.cmake_prefix_path)'
cmake -S . -B $C/build -G Ninja \
  -DVLLM_TARGET_DEVICE=rocm -DVLLM_PYTHON_EXECUTABLE=\$(which python3) \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=\$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)') \
  -DPYTORCH_ROCM_ARCH=gfx1100
" 2>&1 | tee -a $H/build.log
rc=${PIPESTATUS[0]}
[ $rc -eq 0 ] || { log "configure failed rc=$rc"; log "build exit $rc"; exit $rc; }
log "build _rocm_C"
sudo docker exec -e PYTORCH_ROCM_ARCH=gfx1100 -e VLLM_TARGET_DEVICE=rocm vllm-tp2 bash -lc "
set -e
cd $C/vllm-src
cmake --build $C/build --target _rocm_C -j 12
find $C/build -name '_rocm_C*.so' -exec ls -la {} \; -exec md5sum {} \;
" 2>&1 | tee -a $H/build.log
rc=${PIPESTATUS[0]}
log "build exit $rc"
exit $rc
