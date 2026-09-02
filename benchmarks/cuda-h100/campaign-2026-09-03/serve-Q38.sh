#!/bin/bash
set -u
exec vllm serve /models/Qwen3.8-27B-AWQ-INT4 --max-model-len 132000 --port 8000 --no-enable-prefix-caching --gpu-memory-utilization 0.9 > /work/h100-2026-09-03b/serve-Q38.log 2>&1
