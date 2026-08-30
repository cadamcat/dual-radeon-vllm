#!/usr/bin/env python3
"""Read every number in README.md back out of results.jsonl.

Nothing here trusts the prose. Run from this directory:  python3 check.py
"""
import json, os, statistics, sys

RANGE_CUT = 8.0
D = os.path.dirname(os.path.abspath(__file__))
A, B, C, Dd = ("Q38-rocm-nopatch-tp2", "Q38-rocm-45916-tp2",
               "Q38-triton-stock-tp2", "Q38-triton-52684-tp2")

def solve(P, q):
    n = len(P)
    M = [row[:] + [q[i]] for i, row in enumerate(P)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[p] = M[p], M[i]
        for r in range(n):
            if r != i and M[i][i]:
                f = M[r][i] / M[i][i]
                for c in range(i, n + 1):
                    M[r][c] -= f * M[i][c]
    return [M[i][n] / M[i][i] for i in range(n)]

def load(kind):
    by = {}
    for line in open(os.path.join(D, "results.jsonl")):
        line = line.strip()
        if not line:
            continue
        j = json.loads(line)
        if j.get("kind") != kind:
            continue
        k = (j["cfg"], j["target"])
        d = by.setdefault(k, {"v": [], "tok": []})
        d["v"].append(j["ttft"] if kind == "prefill" else j["decode_tps"])
        d["tok"].append(j["prompt_tokens"])
    out = {}
    for (cfg, t), d in by.items():
        v = sorted(d["v"]); tok = round(statistics.mean(d["tok"]))
        rng = (v[-1] - v[0]) / statistics.mean(v) * 100 if len(v) > 1 else None
        row = {"values": v, "runs": len(v), "prompt_tokens": tok, "range_pct": rng,
               "chart_grade": len(v) >= 2 and rng is not None and rng <= RANGE_CUT}
        row["rate"] = (tok / statistics.mean(v)) if kind == "prefill" else statistics.mean(v)
        out[(cfg, t)] = row
    return out

def fit(pre, cfg):
    rs = [r for (c, _), r in sorted(pre.items()) if c == cfg and r["chart_grade"]]
    S = [r["prompt_tokens"] for r in rs]; T = [min(r["values"]) for r in rs]
    n = len(S)
    P = [[sum(s ** (i + j) for s in S) for j in range(3)] for i in range(3)]
    q = [sum(T[k] * S[k] ** i for k in range(n)) for i in range(3)]
    a, b, c = solve(P, q)
    pred = [a + b * s + c * s * s for s in S]
    mt = statistics.mean(T)
    r2 = 1 - sum((T[k] - pred[k]) ** 2 for k in range(n)) / sum((t - mt) ** 2 for t in T)
    return a, b, c, r2, n, sum(1 for (cc, _) in pre if cc == cfg)

pre, dec = load("prefill"), load("decode")
ok = fail = 0
def chk(label, got, want, tol=0.006):
    global ok, fail
    good = abs(got - want) <= tol * max(abs(want), 1e-9)
    print(("  ok   " if good else "  FAIL ") + f"{label}: {got:.4g} vs README {want:.4g}")
    ok, fail = ok + good, fail + (not good)

print("decode table")
for ctx, a, b, c, d, ca, db in [
        (500, 38.45, 49.62, 49.23, 49.33, 1.28, 0.994),
        (8000, 12.38, 45.82, 47.25, 47.28, 3.82, 1.032),
        (16000, 7.17, 42.18, 45.27, 45.37, 6.31, 1.076),
        (32000, 3.90, 36.43, 41.91, 41.95, 10.74, 1.151)]:
    for cfg, want in ((A, a), (B, b), (C, c), (Dd, d)):
        r = dec[(cfg, ctx)]
        assert r["chart_grade"], f"{cfg}@{ctx} is not chart-grade"
        chk(f"decode {cfg} @{ctx}", r["rate"], want)
    chk(f"decode C/A @{ctx}", dec[(C, ctx)]["rate"] / dec[(A, ctx)]["rate"], ca, 0.005)
    chk(f"decode D/B @{ctx}", dec[(Dd, ctx)]["rate"] / dec[(B, ctx)]["rate"], db, 0.005)

print("prefill @32K and the fits")
for cfg, p32, aa, bb, cc, r2, rungs in [
        (A, 964.8, 52.5, 942.4, 2.79, 1.0000, 11),
        (B, 965.8, 219.6, 919.8, 3.23, 0.9999, 10),
        (C, 692.0, 358.7, 848.2, 18.17, 1.0000, 10),
        (Dd, 923.4, 260.8, 886.0, 5.84, 1.0000, 10)]:
    chk(f"prefill {cfg} @32000", pre[(cfg, 32000)]["rate"], p32)
    fa, fb, fc, fr2, n, tot = fit(pre, cfg)
    chk(f"fit a  {cfg}", fa * 1e3, aa, 0.01)
    chk(f"fit b  {cfg}", fb * 1e6, bb, 0.01)
    chk(f"fit c  {cfg}", fc * 1e9, cc, 0.01)
    chk(f"fit r2 {cfg}", fr2, r2, 0.001)
    chk(f"rungs  {cfg}", n, rungs, 0)
    assert tot == 11, f"{cfg} has {tot} prefill rungs, expected 11"

print("counts")
for cfg in (A, B, C, Dd):
    n = sum(1 for line in open(os.path.join(D, "results.jsonl"))
            if line.strip() and json.loads(line).get("cfg") == cfg
            and json.loads(line).get("kind") in ("prefill", "decode"))
    chk(f"measurements {cfg}", n, 44, 0)
    chk(f"decode chart-grade {cfg}",
        sum(1 for (c, _), r in dec.items() if c == cfg and r["chart_grade"]), 11, 0)
nerr = sum(1 for line in open(os.path.join(D, "results.jsonl"))
           if line.strip() and json.loads(line).get("kind") == "error")
chk("error rows", nerr, 0, 0)

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
