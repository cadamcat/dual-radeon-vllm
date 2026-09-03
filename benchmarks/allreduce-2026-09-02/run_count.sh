#!/bin/bash
# run_count.sh -- count the all-reduces one decode step issues (count_collectives.py).
#
# Runs ON THE GUEST. Stops the two GPU services, starts the vllm-tp2 container,
# serves Qwen3-8B at TP=2 with RCCL's per-collective logging on and CUDA graphs
# off, differences two requests, then restores everything and prints both
# cards' VRAM so the caller can confirm the 27971584-byte baseline.
#
# Paths: the container mounts /data/rccl-build as /rb and /data/incoming as
# /models. Everything this run writes goes under /rb/ar0902/count.
set -u
H=/data/rccl-build/ar0902          # guest path
C=/rb/ar0902                        # the same directory inside the container
mkdir -p $H/count
log() { echo "$(date -u +%H:%M:%S) | $*" | tee -a $H/count/PROGRESS.txt; }

log "stopping ollama llamacpp-hub"
sudo systemctl stop ollama llamacpp-hub
sudo docker start vllm-tp2 >/dev/null && log "vllm-tp2 started"

cat > $H/count/serve-count.sh <<'EOF'
#!/bin/bash
set -u
export VLLM_CLONE_MMAP=1
export NCCL_P2P_DISABLE=1
export HSA_ENABLE_SDMA=0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL
export NCCL_DEBUG_FILE=/rb/ar0902/count/rccl.%h.%p
exec vllm serve /models/Qwen3-8B --tensor-parallel-size 2 --gpu-memory-utilization 0.85 \
  --max-model-len 4096 --enforce-eager --port 8000 > /rb/ar0902/count/serve.log 2>&1
EOF
sudo docker exec vllm-tp2 pkill -f 'vllm serve' 2>/dev/null || true
sudo docker exec -d vllm-tp2 bash $C/count/serve-count.sh
log "serve launched (eager, NCCL_DEBUG_SUBSYS=COLL, log file per rank)"

# the script waits for /health itself (up to 15 min), then sends the pair
sudo docker exec -e CC_DIR=$C/count -e CC_OUT=$C/collectives.jsonl -e BENCH_MACHINE='RX 7900 XT' \
  vllm-tp2 python3 $C/count_collectives.py 2>&1 | tee -a $H/count/count.out
rc=${PIPESTATUS[0]}
log "count_collectives.py exit $rc"

log "stopping the server and the container"
sudo docker exec vllm-tp2 pkill -f 'vllm serve' 2>/dev/null || true
sleep 5
sudo docker stop vllm-tp2 >/dev/null && log "vllm-tp2 stopped"
sudo systemctl start ollama llamacpp-hub && log "ollama llamacpp-hub started"
sleep 5
for c in /sys/class/drm/card*/device/mem_info_vram_used; do log "$c $(cat $c)"; done
ls -la $H/count/ | tee -a $H/count/count.out
exit $rc
