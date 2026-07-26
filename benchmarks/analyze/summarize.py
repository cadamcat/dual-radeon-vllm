#!/usr/bin/env python3
"""summarize.py [cfg...] — aggregate bench0725 results.jsonl into per-config tables."""
import json, sys, os

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("BENCH_RESULTS") or os.path.join(_HERE, "..", "results.jsonl")


RES = RESULTS
rows = [json.loads(l) for l in open(RES) if l.strip()]
want = sys.argv[1:] or None

meta = {}
for r in rows:
    if r["kind"] == "model_meta":
        meta[r["cfg"]] = r
    elif r["kind"] in ("config_complete", "config_failed"):
        meta.setdefault(r["cfg"], {})[r["kind"]] = r

data = {}
for r in rows:
    if r["kind"] in ("prefill", "decode"):
        data.setdefault(r["cfg"], {}).setdefault(r["target"], {}).setdefault(r["kind"], []).append(r)

for cfg in data:
    if want and cfg not in want:
        continue
    m = meta.get(cfg, {})
    st = "COMPLETE" if "config_complete" in m else ("FAILED" if "config_failed" in m else "running")
    print(f"\n=== {cfg} [{st}] mml={m.get('mml')} util={m.get('util')} "
          f"KV={m.get('kv_tokens')}tok/{m.get('kv_gib')}GiB conc={m.get('concurrency')}x "
          f"load={m.get('weights_s')}s+init {m.get('init_engine_s')}s ===")
    print(f"{'ctx':>6} {'ptok':>6} | {'prefill r1':>10} {'r2':>8} | {'decode r1':>9} {'r2':>7} | "
          f"{'W(c1+c2)':>10} | {'VRAM':>12}")
    for t in sorted(data[cfg]):
        p = data[cfg][t].get("prefill", [])
        c = data[cfg][t].get("decode", [])
        pt = p[0]["prompt_tokens"] if p else (c[0]["prompt_tokens"] if c else 0)
        pf = [f"{x['prefill_tps']:.0f}" for x in p] + ["-"] * (2 - len(p))
        dc = [f"{(x.get('decode_tps') or 0):.1f}" for x in c] + ["-"] * (2 - len(c))
        pw = f"{c[0].get('p1_max','?')}+{c[0].get('p2_max','?')}" if c else "-"
        vr = f"{c[0].get('v1_g','?')}+{c[0].get('v2_g','?')}G" if c else "-"
        print(f"{t:>6} {pt:>6} | {pf[0]:>10} {pf[1]:>8} | {dc[0]:>9} {dc[1]:>7} | {pw:>10} | {vr:>12}")

errs = [r for r in rows if r["kind"] == "error"]
if errs:
    print(f"\n!! {len(errs)} errors; last: {errs[-1].get('cfg')} {errs[-1].get('step')} {errs[-1]['err'][:90]}")
fails = [r for r in rows if r["kind"] == "config_failed"]
for f in fails:
    print(f"!! FAILED {f['cfg']}: {f.get('why')}")
