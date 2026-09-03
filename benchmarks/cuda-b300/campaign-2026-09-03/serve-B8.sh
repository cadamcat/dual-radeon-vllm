#!/bin/bash
set -u
exec vllm serve /models/Qwen3-8B --max-model-len 33000 --port 8000 --no-enable-prefix-caching --gpu-memory-utilization 0.9 > /work/b300-2026-09-03/serve-B8.log 2>&1
