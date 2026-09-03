#!/bin/bash
set -u
exec vllm serve /models/Muse-Glimmer-30B-INT4 --max-model-len 131072 --port 8000 --no-enable-prefix-caching --gpu-memory-utilization 0.9 > /work/b300-2026-09-03/serve-MG30.log 2>&1
