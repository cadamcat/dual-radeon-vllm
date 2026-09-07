#!/usr/bin/env python3
"""Figures for the depth-cost article.

Everything is recomputed from committed rows -- benchmarks/decode.jsonl and
benchmarks/prefill.jsonl for the ladders, campaign-2026-09-07/results.jsonl for
the drift control -- and nothing is typed.

The article's subject is that a retention percentage is not a property of a
model, shown twice: across six checkpoints, where the percentage ranking and
the slope ranking disagree, and within one checkpoint, whose slope spans 3.2x
across software that leaves its baseline alone.

The unit throughout is milliseconds per token against context, which is what
`docs/hybrid-decode-on-rdna.md` 6.5 already uses for this quantity. Its slope
is microseconds per context token; its intercept is what a step costs before
any context is read.
"""
import json
import pathlib
import statistics

R = pathlib.Path(__file__).resolve().parents[2]
B = R / "benchmarks"

DEC = [json.loads(l) for l in open(B / "decode.jsonl")]
PRE = [json.loads(l) for l in open(B / "prefill.jsonl")]

# The six arms of the pair's own long campaign, and the two this checkpoint
# gained on 2026-09-07. `-long` is the campaign's own suffix.
PAIR_DATE = "2026-09-03"
NEW_DATE = "2026-09-07b"
A27 = "D8-27B-tp2-long-027b"          # 0.27, the backend vLLM picks
B27 = "D8-27B-tp2-triton-long-027b"   # 0.27, TRITON_ATTN forced
C27 = "D8-27B-tp2-long-027c"          # the drift control
OLD = "D8-27B-tp2-long"               # 0.23.1, the published arm


def ladder(cfg, date, rows=None, key="decode_tok_s"):
    rs = [r for r in (rows or DEC)
          if r["machine"] == "RX 7900 XT" and r["cfg"] == cfg and r["date"] == date]
    return {r["ctx"]: r[key] for r in rs}


def fit(d, lo=0, hi=10 ** 9):
    """ms per token against context: (slope us/context token, intercept ms, r2, n).

    A linear fit is the right shape only when the cost grows with context. Muse
    bounds its attention at 2 048 tokens, so its curve bends and r2 says so --
    which is why r2 is carried into the figure rather than dropped.
    """
    xs = sorted(k for k in d if lo <= k <= hi)
    ys = [1000.0 / d[k] for k in xs]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    b = (sum((k - mx) * (y - my) for k, y in zip(xs, ys))
         / sum((k - mx) ** 2 for k in xs))
    i = my - b * mx
    ss = sum((y - my) ** 2 for y in ys)
    rs = sum((y - (i + b * k)) ** 2 for k, y in zip(xs, ys))
    return b * 1000, i, 1 - rs / ss, n


# ---- fig1: the two rankings, on the span all six share ----------------------
# Their deepest rungs differ (32 000 to 128 000) and a slope taken over a
# different span is a different number, so the ranking is computed where every
# arm has rungs. That span is the shallowest arm's ceiling.
six = {}
for r in DEC:
    if (r["machine"] == "RX 7900 XT" and r["date"] == PAIR_DATE
            and str(r["cfg"]).endswith("-long")):
        six.setdefault(r["cfg"], {})[r["ctx"]] = r["decode_tok_s"]
MODEL = {r["cfg"]: r["model"] for r in DEC if r["cfg"] in six}
ARCH = {r["cfg"]: r["arch"] for r in DEC if r["cfg"] in six}
SHARED = min(max(d) for d in six.values())

rank = []
for cfg, d in six.items():
    b, i, r2, n = fit(d, 0, SHARED)
    xs = sorted(k for k in d if k <= SHARED)
    rank.append({"cfg": cfg, "model": MODEL[cfg], "arch": ARCH[cfg],
                 "slope_us_tok": b, "intercept_ms": i, "r2": r2, "rungs": n,
                 "from_ctx": xs[0], "to_ctx": xs[-1],
                 "pct": (d[xs[-1]] / d[xs[0]] - 1) * 100})
by_pct = sorted(rank, key=lambda x: -x["pct"])        # flattest first
by_slope = sorted(rank, key=lambda x: x["slope_us_tok"])
for i, x in enumerate(by_pct):
    x["rank_pct"] = i + 1
for i, x in enumerate(by_slope):
    x["rank_slope"] = i + 1
# the claim the figure is drawn to make
_inv = max(rank, key=lambda x: x["rank_slope"] - x["rank_pct"])
assert _inv["rank_pct"] == 1 and _inv["rank_slope"] == len(rank), \
    (_inv["model"], _inv["rank_pct"], _inv["rank_slope"])
# and that the baselines are what the percentage is dividing by
_int = [x["intercept_ms"] for x in rank]

fig1 = {
    "shared_to": SHARED,
    "rows": sorted(rank, key=lambda x: x["slope_us_tok"]),
    "worst_inversion": {"model": _inv["model"], "cfg": _inv["cfg"],
                        "rank_pct": _inv["rank_pct"],
                        "rank_slope": _inv["rank_slope"],
                        "pct": _inv["pct"], "slope_us_tok": _inv["slope_us_tok"],
                        "intercept_ms": _inv["intercept_ms"]},
    "intercept_spread": max(_int) / min(_int),
    "worst_r2": min(x["r2"] for x in rank),
    "worst_r2_model": min(rank, key=lambda x: x["r2"])["model"],
    "best_r2_of_the_rest": min(x["r2"] for x in rank
                               if x["r2"] > min(x["r2"] for x in rank)),
}

# ---- fig2: one checkpoint, three stacks ------------------------------------
stacks = []
for cfg, date, label, note in (
        (OLD, PAIR_DATE, "vLLM 0.23.1 · ROCM_ATTN", "the published arm"),
        (A27, NEW_DATE, "vLLM 0.27.1 · ROCM_ATTN", "what 0.27 picks on its own"),
        (B27, NEW_DATE, "vLLM 0.27.1 · TRITON_ATTN", "one serve flag further")):
    d = ladder(cfg, date)
    p = ladder(cfg, date, PRE, "prefill_tok_s")
    b, i, r2, n = fit(d)
    xs = sorted(d)
    bs, isl, r2s, ns = fit(d, 0, SHARED)
    stacks.append({
        "cfg": cfg, "label": label, "note": note, "date": date,
        "slope_us_tok": b, "intercept_ms": i, "r2": r2, "rungs": n,
        "slope_on_shared": bs,
        "from_ctx": xs[0], "to_ctx": xs[-1],
        "pct": (d[xs[-1]] / d[xs[0]] - 1) * 100,
        "points": [{"ctx": k, "tok_s": d[k], "ms_tok": 1000.0 / d[k]} for k in xs],
        "prefill": [{"ctx": k, "tok_s": p[k]} for k in sorted(p)],
    })
_sl = [s["slope_on_shared"] for s in stacks]
fig2 = {"stacks": stacks,
        "slope_span": max(_sl) / min(_sl),
        "slope_hi": max(_sl), "slope_lo": min(_sl),
        # where the two new arms land among the six, by the same fit
        "placed": sorted(
            [{"label": s["label"], "slope": s["slope_on_shared"], "new": True}
             for s in stacks]
            + [{"label": x["model"], "slope": x["slope_us_tok"], "new": False}
               for x in rank if x["cfg"] != OLD],
            key=lambda x: x["slope"])}

# ---- fig3: the trade, and the drift control that licenses reading it --------
da, db = ladder(A27, NEW_DATE), ladder(B27, NEW_DATE)
pa, pb = ladder(A27, NEW_DATE, PRE, "prefill_tok_s"), ladder(B27, NEW_DATE, PRE, "prefill_tok_s")
trade = [{"ctx": k, "decode": db[k] / da[k], "prefill": pb[k] / pa[k]}
         for k in sorted(da) if k in db]
assert all(b["decode"] >= a["decode"] - 1e-9 for a, b in zip(trade, trade[1:])), \
    "the decode ratio is drawn as monotone and is not"

C = [json.loads(l) for l in open(B / "campaign-2026-09-07" / "results.jsonl")]
drift = []
for t in sorted({r["target"] for r in C
                 if r.get("cfg") == C27 and r.get("kind") == "decode"}):
    med = lambda cfg: statistics.median(
        [r["decode_tps"] for r in C if r.get("cfg") == cfg
         and r.get("kind") == "decode" and r.get("target") == t])
    drift.append({"ctx": t, "first": med(A27), "again": med(C27),
                  "pct": (med(C27) / med(A27) - 1) * 100})
fig3 = {"trade": trade,
        "widest_decode": max(x["decode"] for x in trade),
        "widest_prefill": min(x["prefill"] for x in trade),
        "drift": drift,
        "worst_drift_pct": max(abs(x["pct"]) for x in drift),
        "gap_pct": (max(x["decode"] for x in trade) - 1) * 100}

out = {
    "_what": "The depth-cost article. One checkpoint family on one pair of cards: "
             "what a retention percentage measures, what a slope measures, and "
             "how much of each belongs to the software. Derived by "
             "site/src/genfig-depth.py from benchmarks/decode.jsonl, "
             "benchmarks/prefill.jsonl and benchmarks/campaign-2026-09-07/.",
    "fig1": fig1, "fig2": fig2, "fig3": fig3,
}
(pathlib.Path(__file__).parent / "figures-depth.json").write_text(
    json.dumps(out, indent=1, ensure_ascii=False) + "\n")

print(f"fig1 shared span 500-{SHARED}, {len(rank)} arms; "
      f"{fig1['worst_inversion']['model']} is #{fig1['worst_inversion']['rank_pct']} by "
      f"percentage and #{fig1['worst_inversion']['rank_slope']} by slope; "
      f"intercepts span {fig1['intercept_spread']:.1f}x")
print(f"fig2 slope span {fig2['slope_span']:.2f}x "
      f"({fig2['slope_lo']:.3f} to {fig2['slope_hi']:.3f} us/tok)")
print(f"fig3 decode widens to {fig3['widest_decode']:.2f}x, prefill to "
      f"{fig3['widest_prefill']:.2f}x, drift at most {fig3['worst_drift_pct']:.2f} %")
