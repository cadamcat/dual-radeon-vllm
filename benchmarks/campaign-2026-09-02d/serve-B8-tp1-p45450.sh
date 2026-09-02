#!/bin/bash
set -u
export VLLM_CLONE_MMAP=1
export NCCL_P2P_DISABLE=1
export HSA_ENABLE_SDMA=0
exec vllm serve /models/Qwen3-8B --tensor-parallel-size 1 --gpu-memory-utilization 0.9 --max-model-len 15792 --port 8000 > /rb/bench0902d/serve-logs/B8-tp1-p45450.log 2>&1
