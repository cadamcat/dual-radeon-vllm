#!/usr/bin/env python3
"""decode_slope.py — how much does one token of context cost each model at decode time?
For a pure-attention model this is the KV-read growth; for a linear-attention (SSM)
model it should be ~0. Also reports prefill flatness (SSM's expected win)."""
import json

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("BENCH_RESULTS") or os.path.join(_HERE, "..", "results.jsonl")


rows = [json.loads(l) for l in open(RESULTS) if l.strip()]
# Group by `target`, NOT by `prompt_tokens`. The random anti-cache prefix makes the
# rounds of one point differ by a few tokens (e.g. 32010 vs 32013); grouping on the
# measured length would split them and silently report a single round.
dec, pre, ptok = {}, {}, {}
for r in rows:
    if r["kind"] == "decode" and r.get("decode_tps"):
        dec.setdefault(r["cfg"], {}).setdefault(r["target"], []).append(r["decode_tps"])
    if r["kind"] == "prefill":
        pre.setdefault(r["cfg"], {}).setdefault(r["target"], []).append(r["prefill_tps"])
    if r["kind"] in ("decode", "prefill"):
        ptok.setdefault(r["cfg"], {}).setdefault(r["target"], []).append(r["prompt_tokens"])

def actual(c, t):                      # mean measured prompt length for a target
    v = ptok[c][t]
    return sum(v) / len(v)

ORDER = ["B-8B-tp2", "A-12B-tp2", "C-31B-tp2", "D-27B-tp2", "E-26B-tp2", "E-26B-tp2-eagerfb"]
print(f"{'config':>18} | {'decode@short':>12} {'decode@long':>11} {'drop':>6} | "
      f"{'us per ctx-tok':>14} | {'prefill@short':>13} {'prefill@long':>12} {'drop':>6}")
for c in ORDER:
    if c not in dec:
        continue
    S = sorted(dec[c])
    lo, hi = S[0], S[-1]
    d_lo = sum(dec[c][lo]) / len(dec[c][lo])
    d_hi = sum(dec[c][hi]) / len(dec[c][hi])
    # slope uses the *measured* prompt lengths, not the nominal targets
    slope = (1000 / d_hi - 1000 / d_lo) / (actual(c, hi) - actual(c, lo)) * 1000
    P = sorted(pre.get(c, {}))
    p_lo = max(pre[c][P[0]]) if P else 0
    p_hi = max(pre[c][P[-1]]) if P else 0
    print(f"{c:>18} | {d_lo:>12.1f} {d_hi:>11.1f} {(d_hi/d_lo-1)*100:>5.1f}% | "
          f"{slope:>14.3f} | {p_lo:>13.0f} {p_hi:>12.0f} {(p_hi/p_lo-1)*100:>5.1f}%   "
          f"(rounds {len(dec[c][lo])}/{len(dec[c][hi])}, ctx {actual(c,lo):.0f}->{actual(c,hi):.0f})")

print("\n# linearity check for D-27B (is the SSM part really O(1) per token?)")
c = "D-27B-tp2"
if c in dec:
    for t_ in sorted(dec[c]):
        ms = 1000 / (sum(dec[c][t_]) / len(dec[c][t_]))
        print(f"  ctx {actual(c, t_):>6.0f}: {ms:7.2f} ms/token")
