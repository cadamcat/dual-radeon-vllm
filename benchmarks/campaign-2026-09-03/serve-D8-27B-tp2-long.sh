#!/bin/bash
set -u
export VLLM_CLONE_MMAP=1
export NCCL_P2P_DISABLE=1
export HSA_ENABLE_SDMA=0
exec vllm serve /models/Qwen3.8-27B-AWQ-INT4 --tensor-parallel-size 2 --gpu-memory-utilization 0.85 --max-model-len 122633 --port 8000 --max-num-seqs 161 > /rb/bench0903/serve-logs/D8-27B-tp2-long.log 2>&1
