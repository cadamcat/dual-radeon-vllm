#!/bin/bash
set -u
exec vllm serve /models/gemma-4-31B-it-qat-w4a16-ct --max-model-len 33000 --port 8000 --no-enable-prefix-caching --gpu-memory-utilization 0.9 --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' > /work/h100-2026-09-03c/serve-G31-mml33.log 2>&1
