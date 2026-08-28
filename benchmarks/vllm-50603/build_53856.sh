#!/bin/bash
# Build _rocm_C with vllm#53856 applied, gfx1100 only, and keep the .so.
# The image ships /app/vllm at the same commit it was built from (f46a9dfe2),
# and csrc/rocm/attention.cu has not changed upstream since 2026-07-31, so the
# PR's diff against its own base applies here unchanged.
set -eu
IMG=rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0
sudo docker run --rm --name build53856 -v /data/50603:/work \
  --entrypoint bash "$IMG" -c '
set -eux
cd /app/vllm
git apply /work/53856-attn.diff
grep -c mask_v_cache_padding csrc/rocm/attention.cu

export PYTORCH_ROCM_ARCH=gfx1100
export VLLM_TARGET_DEVICE=rocm
TORCH_CMAKE=$(python -c "import torch;print(torch.utils.cmake_prefix_path)")

cmake -S . -B /tmp/b -GNinja \
  -DVLLM_TARGET_DEVICE=rocm \
  -DVLLM_PYTHON_EXECUTABLE=$(command -v python) \
  -DCMAKE_PREFIX_PATH="$TORCH_CMAKE" \
  -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -25

cmake --build /tmp/b --target _rocm_C -j"$(nproc)" 2>&1 | tail -25
find /tmp/b -name "_rocm_C*.so" -exec cp -v {} /work/rocm_C_53856.so \;
ls -la /work/rocm_C_53856.so
'
echo BUILD53856_DONE
