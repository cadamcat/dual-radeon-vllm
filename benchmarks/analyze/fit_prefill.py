#!/usr/bin/env python3
"""Fit T(S) = a + b*S + c*S^2 to measured prefill TTFT; explains where prefill tok/s peaks.

SUPERSEDED 2026-08-30 by `build_prefill.py --fits`. Kept because the numbers
published in docs/benchmarks.md section 4 came from this script and this is what
reproduces them; do not use it for new work. Two reasons:

* It buckets by the measured `prompt_tokens` and takes `min()` of each bucket,
  commented "min of the 2 rounds". That is true on CUDA, where both rounds of a
  rung report the same count, and false on ROCm, where they differ by one to
  three tokens -- so most buckets hold one sample and the minimum is that
  sample. `A-12B-tp1` is fitted on nineteen mostly-unpaired points rather than
  eleven paired ones. Every row carries `target`, which is exact on both.
* It fits whatever is in one file, so a configuration measured in two campaigns
  is fitted as one curve belonging to neither.

It also does not gate on repeatability, which is what makes it report a `b` of
4.1 us/tok for the A100's 2026-08-29 prefill -- that campaign ran with prefix
caching on, every rung is a prefix of the next, and `min()` selects the cached
round. See benchmarks/cuda-a100/campaign-2026-08-30/README.md.

`a` and `S*` are the outputs this affects; `b` and `c` move by under 3 %.
"""
import json, sys

import os
from paths import RESULTS


cfg = sys.argv[1] if len(sys.argv) > 1 else "B-8B-tp2"
rows = [json.loads(l) for l in open(RESULTS) if l.strip()]
pts = {}
for r in rows:
    if r["kind"] == "prefill" and r["cfg"] == cfg:
        pts.setdefault(r["prompt_tokens"], []).append(r["ttft"])
S = sorted(pts)
T = [min(pts[s]) for s in S]          # min of the 2 rounds = least noise
n = len(S)
if n < 4:
    print(f"{cfg}: only {n} points, skip"); sys.exit()

def solve(A, y):
    m = len(A)
    M = [row[:] + [y[i]] for i, row in enumerate(A)]
    for col in range(m):
        p = max(range(col, m), key=lambda r: abs(M[r][col]))
        M[col], M[p] = M[p], M[col]
        for r in range(m):
            if r != col and M[col][col]:
                f = M[r][col] / M[col][col]
                for k in range(col, m + 1):
                    M[r][k] -= f * M[col][k]
    return [M[i][m] / M[i][i] for i in range(m)]

P = [[sum(s ** (i + j) for s in S) for j in range(3)] for i in range(3)]
q = [sum(T[k] * S[k] ** i for k in range(n)) for i in range(3)]
a, b, c = solve(P, q)
print(f"=== {cfg} ===")
print(f"fit  T(S) = {a*1000:.0f} ms  +  {b*1e6:.1f} us/tok * S  +  {c*1e9:.2f} ns/tok^2 * S^2")
print(f"peak throughput at S* = sqrt(a/c) = {(a/c)**0.5:.0f} tokens")
print()
hdr = ("S", "T_meas", "T_fit", "fixed", "linear", "quad", "quad%", "tok/s")
print("{:>6} {:>8} {:>7} | {:>6} {:>7} {:>7} {:>6} | {:>6}".format(*hdr))
for k in range(n):
    s = S[k]; f = a; l = b * s; qd = c * s * s; tot = f + l + qd
    print(f"{s:>6} {T[k]:>8.3f} {tot:>7.3f} | {f:>6.3f} {l:>7.3f} {qd:>7.3f} "
          f"{qd/tot*100:>5.1f}% | {s/T[k]:>6.0f}")
