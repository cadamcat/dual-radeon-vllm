#!/bin/bash
set -u
export VLLM_CLONE_MMAP=1
export NCCL_P2P_DISABLE=1
export HSA_ENABLE_SDMA=0
exec vllm serve /models/gemma-4-31B-it-qat-w4a16-ct --tensor-parallel-size 2 --gpu-memory-utilization 0.85 --max-model-len 82558 --port 8000 > /rb/bench0903/serve-logs/C-31B-tp2-long.log 2>&1
