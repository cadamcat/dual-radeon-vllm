#!/bin/bash
set -u
exec vllm serve /models/gemma-4-26B-A4B-AWQ --max-model-len 132000 --port 8000 --no-enable-prefix-caching --gpu-memory-utilization 0.9 --tensor-parallel-size 4 --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' > /work/h100-tp4-2026-09-03/serve-G26A4B.log 2>&1
