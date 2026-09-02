#!/bin/bash
set -u
export VLLM_CLONE_MMAP=1
export NCCL_P2P_DISABLE=1
export HSA_ENABLE_SDMA=0
exec vllm serve /models/Qwen3.8-27B-AWQ-INT4 --tensor-parallel-size 2 --gpu-memory-utilization 0.92 --max-model-len 33000 --port 8000 --max-num-seqs 16 --attention-backend TRITON_ATTN > /rb/bench0902c/serve-logs/Q38-triton-tp2-x16.log 2>&1
