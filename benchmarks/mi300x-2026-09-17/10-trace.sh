#!/bin/bash
# Item 1: on CDNA, is a declaring paged-attention kernel the one that runs?
# Two witnesses, both named by launchtrace.so (this runtime prints no
# ShaderName at any AMD_LOG_LEVEL, which is why the shim exists):
#   A  vLLM's paged-attention op driven directly, the 2026-09-05 probe's method
#   B  a real serve answering two concurrent requests
# Joined against the image's own declaration list from 05-scan.sh.
set -euo pipefail
: "${BENCH_WORK:=/work}" "${BENCH_MODELS:=/models}"
ARCH=${BENCH_ARCH:-gfx942}
TSV="$BENCH_WORK/rocm_C-$ARCH-kernels.tsv"
# Which model to trace. On gfx942 the declaring family in this image is not the
# paged-attention one (the 05-scan run found 0 of 1920 declaring) but the int4
# skinny-GEMM wvSplitK family, so a quantised model is the one that can dispatch
# a declaring kernel. Run this script once per model and keep both joins.
MODEL=${BENCH_TRACE_MODEL:-Qwen3-8B}
TAG=$(echo "$MODEL" | tr "/." "__")
cd "$BENCH_WORK"
exec > >(tee -a "$BENCH_WORK/trace.log") 2>&1
echo "== 10-trace $(date -Is)"
[ -f "$TSV" ] || { echo "FATAL: $TSV missing — run 05-scan.sh first"; exit 2; }

# hipcc's driver does not find the HIP headers for a bare shared object here;
# the 2026-09-05 row built its probes the same way, with the include path named.
SP=$(python3 -c 'import site;print(site.getsitepackages()[0])')
INC=$( { ls -d "$SP"/_rocm_sdk_devel/include /opt/rocm/include 2>/dev/null || true; } | head -1)
CXX=$( { ls "$SP"/_rocm_sdk_devel/lib/llvm/bin/clang++ /opt/rocm/llvm/bin/clang++ 2>/dev/null || true; } | head -1)
"$CXX" -O1 -shared -fPIC -D__HIP_PLATFORM_AMD__ -I"$INC" -x c \
  repo/benchmarks/hostcall-dispatch-2026-09-05/launchtrace.c -o launchtrace.so -ldl
cp repo/benchmarks/hostcall-dispatch-2026-09-05/pa_probe.py .

# A. the op itself. Qwen3-8B's shape: head 128, 32 query heads over 8 KV heads,
#    gqa 4 — the ratio whose CDNA instantiation is the declaring one on gfx1100.
LD_PRELOAD=./launchtrace.so python3 pa_probe.py --arm as_shipped --row "mi300x-$ARCH" \
  --out "$BENCH_WORK/pa-probe-$TAG.jsonl" --ctx 4096 --heads 32 --kv-heads 8 \
  > "$BENCH_WORK/trace-probe-$TAG.txt" 2>&1 || echo "pa_probe exit $? (it was written for the 0.23 container; read trace-probe.txt before concluding anything)"

# B. a serve, two concurrent requests. Its own short serve so the trace never
#    touches a timing row; 8192 is enough for the 4 096-token prompt.
pkill -f '[v]llm serve' || true; sleep 3
LD_PRELOAD=./launchtrace.so nohup vllm serve "$BENCH_MODELS/$MODEL" \
  --max-model-len 8192 --max-num-seqs 16 --port 8000 \
  > "$BENCH_WORK/trace-serve-$TAG.log" 2>&1 &
for i in $(seq 1 180); do
  grep -q "Application startup complete" "$BENCH_WORK/trace-serve-$TAG.log" && break
  sleep 5
done
grep -q "Application startup complete" "$BENCH_WORK/trace-serve-$TAG.log" || { echo "FATAL: serve did not start"; tail -40 "$BENCH_WORK/trace-serve-$TAG.log"; exit 3; }
python3 - <<'PY'
import json, os, threading, urllib.request
W, M = os.environ.get("BENCH_WORK", "/work"), os.environ.get("BENCH_MODELS", "/models")
MODEL = os.environ.get("BENCH_TRACE_MODEL", "Qwen3-8B")
prompt = "word " * 3800          # ~4 000 tokens, the probe's context
def one(i, out):
    body = json.dumps({"model": f"{M}/{MODEL}", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 64, "temperature": 0.8, "ignore_eos": True}).encode()
    r = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", data=body,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=600) as resp:
        out[i] = json.loads(resp.read())["usage"]
out = {}
ts = [threading.Thread(target=one, args=(i, out)) for i in range(2)]
[t.start() for t in ts]; [t.join() for t in ts]
print("serve answered:", json.dumps(out))
PY
pkill -f '[v]llm serve' || true; sleep 3

# Join both traces against the declaration list. join_trace.py keys on the
# mangled symbol with the metadata's `.kd` suffix stripped; the first version
# keyed on the demangled name and matched nothing.
python3 join_trace.py "$TAG" "$BENCH_WORK/trace-probe-$TAG.txt" "$BENCH_WORK/trace-serve-$TAG.log"
echo "== trace done"
