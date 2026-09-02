#!/bin/bash
set -u
exec vllm serve /models/gemma-4-26B-A4B-AWQ --max-model-len 132000 --port 8000 --no-enable-prefix-caching --gpu-memory-utilization 0.9 --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' > /work/h100-2026-09-03b/serve-G26A4B.log 2>&1
