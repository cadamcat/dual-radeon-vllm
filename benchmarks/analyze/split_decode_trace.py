"""Separate decode-step kernel calls from prefill ones in a vLLM trace.

The long-context trace mixes both: chunked prefill runs 4 iterations before the
11 decode steps, and paged attention is called in each. Averaging over all of
them understates or overstates the decode cost depending on which dominates, so
attribute each kernel call to the engine step whose window it falls in.

Each engine step emits its execute_context annotation *twice* under the same
name: a `user_annotation` for the CPU-side launch (~5 ms) and a
`gpu_user_annotation` for the device work (~237 ms at 32K). Counting both gives
22 windows where there are 11 steps and halves every per-step figure, so this
takes the GPU one. The check that it is right: the count must match the layer
arithmetic — 176 paged-attention calls / 16 full-attention layers = 11, and 528
linear-attention calls / 48 layers = 11 — and torch's own aggregation in
profiler_out_*.txt reports the same 11 calls at 236.899 ms.
"""
import json, gzip, sys
from collections import defaultdict

path = sys.argv[1]
op = gzip.open if path.endswith(".gz") else open
with op(path, "rt") as f:
    d = json.load(f)
ev = d["traceEvents"] if isinstance(d, dict) else d

# engine-step windows, named by vLLM's execute_context annotation
windows = []
for e in ev:
    if e.get("ph") != "X" or e.get("cat") != "gpu_user_annotation":
        continue                    # the CPU-side twin would double the count
    n = e.get("name", "")
    if n.startswith("execute_context_"):
        ts, dur = e.get("ts", 0), e.get("dur", 0) or 0
        kind = "decode" if "generation_1" in n else "prefill"
        windows.append((ts, ts + dur, kind, n))
windows.sort()
print(f"engine-step windows: {len(windows)}")
for kind in ("prefill", "decode"):
    ws = [w for w in windows if w[2] == kind]
    if ws:
        tot = sum(b - a for a, b, _, _ in ws) / 1000
        print(f"  {kind:8} n={len(ws):3}  total={tot:10.1f} ms  avg={tot/len(ws):9.3f} ms")

def bucket(ts):
    for a, b, kind, _ in windows:
        if a <= ts <= b:
            return kind
    return "outside"

TARGETS = ("kernel_paged_attention_2d",
           "fused_recurrent_gated_delta_rule_packed_decode",
           "_causal_conv1d_update_kernel",
           "triton_w4a16_gemm_kernel",
           "ncclDevKernel_Generic")

agg = defaultdict(lambda: [0.0, 0])
for e in ev:
    if e.get("ph") != "X" or (e.get("cat") or "").lower() != "kernel":
        continue
    n = e.get("name", "")
    m = next((t for t in TARGETS if n.startswith(t)), None)
    if not m:
        continue
    a = agg[(m, bucket(e.get("ts", 0)))]
    a[0] += e.get("dur", 0) or 0
    a[1] += 1

ndec = len([w for w in windows if w[2] == "decode"]) or 1
print(f"\n{'kernel':52} {'phase':8} {'total_ms':>10} {'calls':>7} {'us/call':>10} {'ms/step':>9}")
print("-" * 100)
for (name, phase), (dur, cnt) in sorted(agg.items()):
    per_step = dur / 1000 / ndec if phase == "decode" else float("nan")
    print(f"{name[:52]:52} {phase:8} {dur/1000:10.3f} {cnt:7d} {dur/cnt:10.3f} "
          f"{per_step:9.3f}" if phase == "decode" else
          f"{name[:52]:52} {phase:8} {dur/1000:10.3f} {cnt:7d} {dur/cnt:10.3f} {'-':>9}")
