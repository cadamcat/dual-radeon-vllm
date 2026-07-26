#!/usr/bin/env python3
"""analyze.py — derived metrics across configs: TP2/TP1 speedup, MBU, cross-model view."""
import json, os, glob

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("BENCH_RESULTS") or os.path.join(_HERE, "..", "results.jsonl")
# The per-token weight-traffic section reads safetensors headers, so it only works
# on a machine that holds the models; point MODELS_DIR at them (default: the server path).
MODELS_DIR = os.environ.get("MODELS_DIR", "/data/incoming")

RES = RESULTS
rows = [json.loads(l) for l in open(RES) if l.strip()]
agg = {}
for r in rows:
    if r["kind"] in ("prefill", "decode"):
        k = agg.setdefault(r["cfg"], {}).setdefault(r["target"], {"prefill": [], "decode": [], "pw": []})
        if r["kind"] == "prefill":
            k["prefill"].append(r["prefill_tps"])
        else:
            if r.get("decode_tps"):
                k["decode"].append(r["decode_tps"])
            if r.get("p1_max") is not None:
                k["pw"].append((r.get("p1_max", 0), r.get("p2_max", 0)))

def avg(x):
    return sum(x) / len(x) if x else None

def best(x):   # max of the 2 rounds = least-contaminated measurement
    return max(x) if x else None

TARGETS = [500, 1000, 2000, 4000, 6000, 8000, 12000, 16000, 20000, 24000, 32000]

print("### decode tok/s (mean of 2 rounds)")
cfgs = [c for c in ["B-8B-tp1", "B-8B-tp2", "A-12B-tp1", "A-12B-tp2", "C-31B-tp2",
                    "D-27B-tp2", "E-26B-tp2", "E-26B-tp2-eagerfb"] if c in agg]
print("  ctx | " + " | ".join(f"{c:>12}" for c in cfgs))
for t in TARGETS:
    cells = []
    for c in cfgs:
        v = avg(agg[c].get(t, {}).get("decode", []))
        cells.append(f"{v:>12.1f}" if v else f"{'-':>12}")
    print(f"{t:>5} | " + " | ".join(cells))

print("\n### prefill tok/s (best of 2 rounds)")
print("  ctx | " + " | ".join(f"{c:>12}" for c in cfgs))
for t in TARGETS:
    cells = []
    for c in cfgs:
        v = best(agg[c].get(t, {}).get("prefill", []))
        cells.append(f"{v:>12.0f}" if v else f"{'-':>12}")
    print(f"{t:>5} | " + " | ".join(cells))

print("\n### TP2/TP1 speedup (decode)")
for m, (t1, t2) in {"8B BF16": ("B-8B-tp1", "B-8B-tp2"),
                    "12B w4a16": ("A-12B-tp1", "A-12B-tp2")}.items():
    if t1 not in agg or t2 not in agg:
        continue
    print(f"  {m}:")
    for t in TARGETS:
        a, b = avg(agg[t1].get(t, {}).get("decode", [])), avg(agg[t2].get(t, {}).get("decode", []))
        if a and b:
            print(f"    {t:>6}: TP1 {a:5.1f} -> TP2 {b:5.1f} = {b/a:.2f}x  (eff {b/a/2*100:.0f}%)")

print("\n### per-token weight traffic (from checkpoint headers)")
import struct

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("BENCH_RESULTS") or os.path.join(_HERE, "..", "results.jsonl")

if not os.path.isdir(MODELS_DIR):
    print(f"  (skipped: MODELS_DIR={MODELS_DIR} not present — this section reads safetensors")
    print("   headers, so it only runs on the machine that holds the models)")
    raise SystemExit(0)

for d, lbl in [("Qwen3-8B", "8B BF16"), ("gemma-4-12B-it-qat-w4a16-ct", "12B w4a16"),
               ("gemma-4-31B-it-qat-w4a16-ct", "31B w4a16")]:
    tot = 0
    for fp in sorted(glob.glob(os.path.join(MODELS_DIR, d, "*.safetensors"))):
        with open(fp, "rb") as f:
            L = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(L))
        for k, v in h.items():
            if k == "__metadata__":
                continue
            tot += v["data_offsets"][1] - v["data_offsets"][0]
    cfgp = os.path.join(MODELS_DIR, d, "config.json")
    cj = json.load(open(cfgp)) if os.path.exists(cfgp) else {}
    tc = cj.get("text_config", cj)
    print(f"  {lbl:>10}: ckpt {tot/2**30:6.2f} GiB  hidden={tc.get('hidden_size')} "
          f"vocab={tc.get('vocab_size')} tie_emb={cj.get('tie_word_embeddings', tc.get('tie_word_embeddings'))}")
