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
# Usage: run INSIDE a rocm7.14 vLLM container that has /data/rccl-build mounted at /rb.
set -e
SDL=/opt/python/lib/python3.14/site-packages/_rocm_sdk_libraries/lib
SDD=/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel/lib
SP=/opt/python/lib/python3.14/site-packages
CLANG=/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel/lib/llvm/bin/clang

echo "[1/4] build rsmi stub (weak no-op rocm_smi so librccl loads w/o real librocm_smi64)"
$CLANG -shared -fPIC -o $SDL/librsmi_stub.so.1 /rb/vfio-tp2/rsmi_stub.c

echo "[2/4] swap no-hostcall librccl into the runtime path torch loads"
cp -a $(readlink -f $SDL/librccl.so.1) $SDL/librccl.so.1.orig 2>/dev/null || true
cp /rb/vfio-tp2/librccl-final.so $SDL/librccl.so.1
cp -a $(readlink -f $SDD/librccl.so.1) $SDD/librccl.so.1.orig 2>/dev/null || true
cp /rb/vfio-tp2/librccl-final.so $SDD/librccl.so.1

echo "[3/4] install sitecustomize (pre-init amdsmi BEFORE torch loads librccl+stub)"
cp /rb/vfio-tp2/sitecustomize.py $SP/sitecustomize.py

echo "[4/4] verify: 0 hostcall + device_count"
export PATH=$SDD/llvm/bin:$PATH
llvm-objdump --offloading $SDL/librccl.so.1 >/dev/null 2>&1
DEV=$(ls $SDL/librccl.so.1.*gfx1100 2>/dev/null | head -1)
echo -n "  hidden_hostcall_buffer (want 0): "; llvm-readelf --notes "$DEV" 2>/dev/null | grep -ic hidden_hostcall_buffer
rm -f $SDL/librccl.so.1.*gfx1100 $SDL/librccl.so.1.*host* 2>/dev/null || true
python3 -c "import torch; print('  device_count (want 2):', torch.cuda.device_count())"
echo "DONE. Launch: NCCL_P2P_DISABLE=1 HSA_ENABLE_SDMA=0 vllm serve <model> --tensor-parallel-size 2 ..."
