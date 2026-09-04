#!/usr/bin/env python3
"""The +-NDEBUG A/B, rolled up. Every figure the README quotes comes from here.

Six sweeps, interleaved A B A B A B so that drift over the half hour cannot be
read as an arm effect. Per cell the three runs are averaged, and the comparison
is reported against the measured noise floor rather than against zero: with a
per-cell spread of ~2 % a 2 % ratio means nothing on its own, so the statistic
that carries the result is the DIRECTION, counted across cells and checked in
each sweep separately.
"""
import json, os, statistics as st
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ("ndebug", "nondebug")
REPEATS = (1, 2, 3)
METRICS = ("t_graph_us", "t_stream_us", "t_sync_us_median")


def cells(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("kind") == "allreduce":
            out[(r["hidden"], r["ntok"])] = r
    return out


def load():
    runs = {(a, r): cells(os.path.join(HERE, f"ar2-{a}-r{r}.jsonl"))
            for a in ARMS for r in REPEATS}
    keys = sorted(set.intersection(*[set(v) for v in runs.values()]))
    return runs, keys


def meta(path):
    with open(path) as fh:
        return json.loads(fh.readline())


def sign_p(k, n):
    """Two-sided exact binomial at p=0.5."""
    k = max(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def means(runs, key, metric):
    return tuple(st.fmean([runs[(a, r)][key][metric] for r in REPEATS])
                 for a in ARMS)


def noise(runs, keys, metric):
    """Within-arm (max-min)/mean across the three repeats, both arms pooled."""
    out = []
    for a in ARMS:
        for k in keys:
            v = [runs[(a, r)][k][metric] for r in REPEATS]
            out.append((max(v) - min(v)) / st.fmean(v))
    return out


def block(runs, keys, metric):
    ratios, slower = [], 0
    for k in keys:
        a, b = means(runs, k, metric)
        ratios.append(b / a)
        slower += b > a
    return ratios, slower, sign_p(slower, len(keys))


# ---------------------------------------------------------------- decode -----

def decode_rows():
    return [json.loads(l) for l in open(os.path.join(HERE, "decode.jsonl"))]


def decode_cells(metric="decode_tps"):
    """{(model, depth): {arm: [values]}} and the order the sessions ran in."""
    rows = decode_rows()
    order = {(m["label"], m["arm"]): i
             for i, m in enumerate(r for r in rows if r["kind"] == "decode_meta")}
    cells = {}
    for r in rows:
        if r["kind"] != "decode" or metric not in r:
            continue
        cells.setdefault((r["label"], r["depth"]), {}).setdefault(r["arm"], []).append(r[metric])
    return cells, order


def decode_report():
    rows = decode_rows()
    metas = [r for r in rows if r["kind"] == "decode_meta"]
    runs = [r for r in rows if r["kind"] == "decode"]
    print(f"\n=== end to end: {len(runs)} runs, "
          f"{sum(1 for r in runs if 'error' in r)} errors, "
          f"{sum(1 for m in metas if m['library_matches'])}/{len(metas)} sessions "
          f"served the library they were given ===")
    for m in metas:
        print(f"  session {m['label']:4s} arm={m['arm']:9s} load {m['server_load_s']}s")

    for metric in ("decode_tps", "prefill_tps"):
        cells, order = decode_cells(metric)
        print(f"\n  {metric}")
        print(f"    {'model':5} {'depth':>6} {'ndebug':>9} {'nondebug':>9}"
              f" {'by arm':>8} {'noise%':>7} {'2nd/1st':>8}")
        for (lab, dep), arms in sorted(cells.items()):
            a, b = arms["ndebug"], arms["nondebug"]
            ma, mb = st.fmean(a), st.fmean(b)
            noise = max((max(a) - min(a)) / ma, (max(b) - min(b)) / mb) * 100
            first, second = sorted((order[(lab, x)], x) for x in ("ndebug", "nondebug"))
            m1 = st.fmean(arms[first[1]]); m2 = st.fmean(arms[second[1]])
            print(f"    {lab:5} {dep:>6} {ma:>9.2f} {mb:>9.2f} {mb/ma:>8.4f}"
                  f" {noise:>6.2f}% {m2/m1:>8.4f}")


def order_effect(metric, drop_depth=None):
    """Ratio of the second session of a pair to the first, per cell."""
    cells, order = decode_cells(metric)
    out = {}
    for (lab, dep), arms in cells.items():
        if drop_depth and dep in drop_depth:
            continue
        first, second = sorted((order[(lab, x)], x) for x in ("ndebug", "nondebug"))
        out[(lab, dep)] = st.fmean(arms[second[1]]) / st.fmean(arms[first[1]])
    return out


def arm_effect(metric, drop_depth=None):
    cells, _ = decode_cells(metric)
    return {(lab, dep): st.fmean(a["nondebug"]) / st.fmean(a["ndebug"])
            for (lab, dep), a in cells.items()
            if not (drop_depth and dep in drop_depth)}


# ----------------------------------------------------------- capability ------

def capability_cells():
    """{(row, library): record} for the six-cell capability matrix."""
    rows = [json.loads(l) for l in
            open(os.path.join(HERE, "capability-matrix.jsonl"))]
    return {(r["row"], r["library"]): r for r in rows}


HOSTCALL = {"stock2304": 13, "nondebug": 6, "ndebug": 0}


def capability_report():
    cells = capability_cells()
    print("\n=== the capability matrix: 2 platform states x 3 libraries ===")
    cols = ("stock2304", "nondebug", "ndebug")
    print(f"    {'':17}" + "".join(f"{c + ' (' + str(HOSTCALL[c]) + ')':>22}" for c in cols))
    for row in ("atomics_present", "atomics_absent"):
        line = f"    {row:17}"
        for c in cols:
            k = cells[(row, c)]
            line += f"{(str(k['correctness_passed']) + '/12' if k['dispatched'] else 'REFUSED'):>22}"
        print(line)
    for (row, lib), r in sorted(cells.items()):
        if r["error"]:
            print(f"    {row} / {lib}: {r['error']}")
    for row in ("atomics_present", "atomics_absent"):
        r = cells[(row, "ndebug")]
        print(f"    {row:17} root ports with completer support "
              f"{r['root_ports_with_completer_support']}, "
              f"amdgpu complaints {r['dmesg_no_atomics_lines']}")


def main():
    runs, keys = load()
    print(f"cells shared by all six sweeps: {len(keys)}")
    for a in ARMS:
        m = meta(os.path.join(HERE, f"ar2-{a}-r1.jsonl"))["rccl_loaded"]
        print(f"  arm={a:9s} md5={m['md5']} hostcall={m['hidden_hostcall_buffer']}"
              f" via {m.get('hidden_hostcall_buffer_method')}")

    for metric in METRICS:
        n = noise(runs, keys, metric)
        ratios, slower, p = block(runs, keys, metric)
        print(f"\n=== {metric} ===")
        print(f"  noise  (max-min)/mean within an arm: median {st.median(n)*100:.2f} %"
              f"  p90 {sorted(n)[int(.9*len(n))]*100:.2f} %")
        print(f"  pooled median ratio {st.median(ratios):.4f}"
              f"  nondebug slower in {slower}/{len(keys)}  sign p={p:.4f}")
        for r in REPEATS:
            k = sum(1 for x in keys
                    if runs[("nondebug", r)][x][metric] > runs[("ndebug", r)][x][metric])
            print(f"    sweep {r}: slower in {k}/{len(keys)}  sign p={sign_p(k, len(keys)):.4f}")

    print("\n=== t_graph_us by token count (the shape of the effect) ===")
    print(f"{'ntok':>6} {'median ratio':>13} {'slower':>8} {'ndebug us':>11} {'nondebug us':>12}")
    for nt in sorted({k[1] for k in keys}):
        ks = [k for k in keys if k[1] == nt]
        rs, sl, av, bv = [], 0, [], []
        for k in ks:
            a, b = means(runs, k, "t_graph_us")
            rs.append(b / a); sl += b > a; av.append(a); bv.append(b)
        print(f"{nt:>6} {st.median(rs):>13.4f} {sl:>6}/{len(ks)}"
              f" {st.fmean(av):>11.2f} {st.fmean(bv):>12.2f}")

    for label, sel in (("ntok <= 16", lambda k: k[1] <= 16),
                       ("ntok >= 256", lambda k: k[1] >= 256)):
        ks = [k for k in keys if sel(k)]
        rs, sl, p = block(runs, ks, "t_graph_us")
        print(f"  {label:12s} n={len(ks):2d} median {st.median(rs):.4f}"
              f" slower {sl}/{len(ks)} sign p={p:.4f}")

    ok = 0
    for a in ARMS:
        rows = [json.loads(l) for l in open(os.path.join(HERE, f"correct2-{a}.jsonl"))]
        cases = [r for r in rows if r.get("kind") == "correctness"]
        passed = sum(1 for r in cases if r["ok"])
        ok += passed == len(cases) == 12
        print(f"correctness arm={a}: {passed}/{len(cases)} pass")
    print(f"both arms correct: {ok == 2}")
    decode_report()
    capability_report()
    print("\n=== prefill: does the difference follow the arm or the session order? ===")
    oe = order_effect("prefill_tps", drop_depth={500})
    ae = arm_effect("prefill_tps", drop_depth={500})
    for k in sorted(oe):
        print(f"    {k[0]:4s} {k[1]:>6}  2nd/1st {oe[k]:.4f}   nondebug/ndebug {ae[k]:.4f}")
    print(f"    cells where the SECOND session is faster: "
          f"{sum(1 for v in oe.values() if v > 1)}/{len(oe)}")
    print(f"    cells where the unfixed arm is faster:    "
          f"{sum(1 for v in ae.values() if v > 1)}/{len(ae)}")


if __name__ == "__main__":
    main()
