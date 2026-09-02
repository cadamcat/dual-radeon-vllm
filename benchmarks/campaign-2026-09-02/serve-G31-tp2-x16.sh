#!/bin/bash
set -u
export VLLM_CLONE_MMAP=1
export NCCL_P2P_DISABLE=1
export HSA_ENABLE_SDMA=0
exec vllm serve /models/gemma-4-31B-it-qat-w4a16-ct --tensor-parallel-size 2 --gpu-memory-utilization 0.92 --max-model-len 33000 --port 8000 --max-num-seqs 16 --hf-overrides '{"allow_global_per_layer_attribute_access": true}' > /rb/bench0902a/serve-logs/G31-tp2-x16.log 2>&1
