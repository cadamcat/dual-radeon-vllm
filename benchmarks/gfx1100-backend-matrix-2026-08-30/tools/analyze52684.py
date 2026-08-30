#!/usr/bin/env python3
"""Aggregate and fit the 52684 A/B, by build_prefill.py's own rules.

Cells are the mean of a rung's rounds; a rung is chart-grade when it has >= 2
rounds agreeing within RANGE_CUT = 8%; the a/b/c fit takes the FASTER round of
each chart-grade rung. Nothing here is read from the projections -- it is
recomputed from the raw campaign rows so the two can be compared.
"""
import json, statistics, sys, collections

RANGE_CUT = 8.0

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

def load(path, kind):
    by = collections.defaultdict(lambda: {"v": [], "tok": []})
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        j = json.loads(line)
        if j.get("kind") != kind:
            continue
        k = (j["cfg"], j["target"])
        if kind == "prefill":
            by[k]["v"].append(j["ttft"])
        else:
            by[k]["v"].append(j["decode_tps"])
        by[k]["tok"].append(j["prompt_tokens"])
    rows = []
    for (cfg, target), d in sorted(by.items()):
        v = sorted(d["v"])
        tok = round(statistics.mean(d["tok"]))
        rng = (v[-1] - v[0]) / statistics.mean(v) * 100 if len(v) > 1 else None
        row = {"cfg": cfg, "ctx": target, "runs": len(v), "values": v,
               "prompt_tokens": tok, "range_pct": rng,
               "chart_grade": len(v) >= 2 and rng is not None and rng <= RANGE_CUT}
        if kind == "prefill":
            row["ttft_s"] = statistics.mean(v)
            row["prefill_tok_s"] = tok / row["ttft_s"]
        else:
            row["decode_tok_s"] = statistics.mean(v)
        rows.append(row)
    return rows

def fit(rows):
    rs = [r for r in rows if r["chart_grade"]]
    S = [r["prompt_tokens"] for r in rs]
    T = [min(r["values"]) for r in rs]
    n = len(S)
    if n < 4:
        return None, n, len(rows)
    P = [[sum(s ** (i + j) for s in S) for j in range(3)] for i in range(3)]
    q = [sum(T[k] * S[k] ** i for k in range(n)) for i in range(3)]
    a, b, c = solve(P, q)
    pred = [a + b * s + c * s * s for s in S]
    ss_res = sum((T[k] - pred[k]) ** 2 for k in range(n))
    mt = statistics.mean(T)
    ss_tot = sum((t - mt) ** 2 for t in T)
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return (a, b, c, r2), n, len(rows)

def report(path):
    for kind in ("prefill", "decode"):
        rows = load(path, kind)
        cfgs = sorted(set(r["cfg"] for r in rows))
        if not cfgs:
            continue
        print(f"\n########## {kind} ##########")
        ctxs = sorted(set(r["ctx"] for r in rows))
        key = "prefill_tok_s" if kind == "prefill" else "decode_tok_s"
        hdr = f"{'ctx':>6} | " + " | ".join(f"{c[:26]:>26}" for c in cfgs)
        print(hdr); print("-" * len(hdr))
        idx = {(r["cfg"], r["ctx"]): r for r in rows}
        for ctx in ctxs:
            cells = []
            for c in cfgs:
                r = idx.get((c, ctx))
                cells.append("-" * 26 if r is None else
                             f"{r[key]:11.2f} {'ok ' if r['chart_grade'] else 'BAD'} {r['range_pct']:6.2f}%")
            print(f"{ctx:>6} | " + " | ".join(cells))
        if kind == "prefill":
            print()
            for c in cfgs:
                f, n, tot = fit([r for r in rows if r["cfg"] == c])
                if f is None:
                    print(f"  {c}: {n}/{tot} chart-grade, too few to fit")
                else:
                    a, b, cc, r2 = f
                    print(f"  {c}: rungs {n}/{tot}  a={a*1e3:8.1f} ms  b={b*1e6:8.1f} us/tok  "
                          f"c={cc*1e9:7.2f} ns/tok^2  r2={r2:.4f}")

if __name__ == "__main__":
    report(sys.argv[1])
