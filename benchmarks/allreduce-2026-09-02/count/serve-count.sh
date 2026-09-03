#!/bin/bash
set -u
export VLLM_CLONE_MMAP=1
export NCCL_P2P_DISABLE=1
export HSA_ENABLE_SDMA=0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL
export NCCL_DEBUG_FILE=/rb/ar0902/count/rccl.%h.%p
exec vllm serve /models/Qwen3-8B --tensor-parallel-size 2 --gpu-memory-utilization 0.85 \
  --max-model-len 4096 --enforce-eager --port 8000 > /rb/ar0902/count/serve.log 2>&1
