#!/bin/bash
# env-var sweep against the 30s all_reduce repro. Each row: label + extra env.
# Baseline always: NCCL_P2P_DISABLE=1 HSA_ENABLE_SDMA=0 (the crashing config).
cd "$(dirname "$0")"   # ar.py lives next to this script
PORT=29600
run() {
  local label="$1"; shift
  PORT=$((PORT+1))
  local out
  out=$(env "$@" NCCL_DEBUG=WARN timeout 90 torchrun --nproc_per_node=2 --master_port=$PORT ar.py 2>&1)
  if echo "$out" | grep -q "all_reduce OK"; then
    echo "PASS  | $label"
  elif echo "$out" | grep -q "present state"; then
    echo "FAIL-presentstate | $label"
  elif echo "$out" | grep -qi "no transport\|Could not find\|unsupported\|no such transport"; then
    echo "FAIL-notransport | $label :: $(echo "$out" | grep -im1 'no transport\|Could not find\|unsupported')"
  else
    echo "FAIL-other | $label :: $(echo "$out" | grep -iE 'WARN|Error|failure' | grep -v 'RCCL_USE_AMD_SMI\|libnccl\|librccl-net\|GIN\|IB' | head -1)"
  fi
}

echo "=== control (baseline crashing config) ==="
run "control: P2P_DISABLE=1 SDMA=0" NCCL_P2P_DISABLE=1 HSA_ENABLE_SDMA=0
echo "=== single-knob adds on baseline ==="
run "+ HSA_FORCE_FINE_GRAIN_PCIE=1" NCCL_P2P_DISABLE=1 HSA_ENABLE_SDMA=0 HSA_FORCE_FINE_GRAIN_PCIE=1
run "+ GPU_MAX_HW_QUEUES=1"         NCCL_P2P_DISABLE=1 HSA_ENABLE_SDMA=0 GPU_MAX_HW_QUEUES=1
run "+ AMD_SERIALIZE_KERNEL=3"      NCCL_P2P_DISABLE=1 HSA_ENABLE_SDMA=0 AMD_SERIALIZE_KERNEL=3
run "+ HSA_ENABLE_INTERRUPT=0"      NCCL_P2P_DISABLE=1 HSA_ENABLE_SDMA=0 HSA_ENABLE_INTERRUPT=0
run "+ RCCL_USE_AMD_SMI_LIB=1"      NCCL_P2P_DISABLE=1 HSA_ENABLE_SDMA=0 RCCL_USE_AMD_SMI_LIB=1
run "+ NCCL_CUMEM_ENABLE=1"         NCCL_P2P_DISABLE=1 HSA_ENABLE_SDMA=0 NCCL_CUMEM_ENABLE=1
run "+ NCCL_SHM_DISABLE=1"          NCCL_P2P_DISABLE=1 HSA_ENABLE_SDMA=0 NCCL_SHM_DISABLE=1
echo "=== transport variants ==="
run "P2P allowed (SDMA=0)"          HSA_ENABLE_SDMA=0
run "SDMA on (P2P_DISABLE=1)"       NCCL_P2P_DISABLE=1
run "+ FINEGRAIN + SERIALIZE combo" NCCL_P2P_DISABLE=1 HSA_ENABLE_SDMA=0 HSA_FORCE_FINE_GRAIN_PCIE=1 AMD_SERIALIZE_KERNEL=3 GPU_MAX_HW_QUEUES=1
echo "=== DONE ==="
