"""Figures for a100-vs-two-radeons.html.

The Radeon side comes from benchmarks/speculative-decoding/, the A100 side from
the leg logs in benchmarks/cuda-a100/45450-validation/logs/, parsed the same way
verify_doc_figures.py parses them. Bandwidth utilisation is recomputed from the
July campaign's rates and the per-GPU bytes/token docs/benchmarks.md states.
"""
import json, pathlib, re, sys
R = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "benchmarks" / "analyze"))
import verify_doc_figures as V

SP = R / "benchmarks" / "speculative-decoding"
VD = R / "benchmarks" / "cuda-a100" / "45450-validation" / "logs"
lad = lambda fn: {r["depth"]: r["tok_per_s"] for r in json.load(open(SP / fn))["rows"]}
leg = lambda fn: float(re.search(r"RESULT decode_tok_s=([\d.]+)",
                                 open(VD / fn).read()).group(1))

# nominal ceilings: two RX 7900 XT at 800 GB/s each against one A100-SXM4-80GB
NOMINAL = {"radeons_gb_s": 1600.0, "a100_gb_s": 2039.0}

# ---- fig1: the ladder, and the U it makes --------------------------------
# The Radeons' 32K rung is compared against the A100's 30K leg, as the source
# table states; every other rung is matched.
PAIRS = [(1024, "D1K.log", 1024), (8192, "D8K.log", 8192),
         (16384, "D16K.log", 16384), (32768, "D30.log", 30000)]
p45 = lad("mtp-31b-p45450.json")
fig1 = {"rows": [{"ctx": ctx, "a100_ctx": actx, "radeons": p45[ctx],
                  "a100": leg(fn), "advantage": leg(fn) / p45[ctx],
                  "matched": ctx == actx}
                 for ctx, fn, actx in PAIRS],
        "nominal_ratio": NOMINAL["a100_gb_s"] / NOMINAL["radeons_gb_s"],
        "nominal": NOMINAL, "runs_per_cell": 1}
adv = [r["advantage"] for r in fig1["rows"]]
fig1["min_advantage"] = min(adv)
fig1["min_at"] = fig1["rows"][adv.index(min(adv))]["ctx"]
fig1["u_shaped"] = adv[0] > min(adv) < adv[-1]
fig1["below_nominal"] = [r["ctx"] for r in fig1["rows"]
                         if r["advantage"] < fig1["nominal_ratio"]]

# ---- fig1: the 2026-08-29 ladder, both arms, both machines ----------------
# This is the figure. Eleven matched rungs, two rounds a cell, both arms, one
# session on each machine -- and it carries the arm the four probe points above
# do not have, which is the one where nobody is speculating and which is what
# the article's question actually asks about.
#
# The four points are kept, as an option, because the article was written
# against them and because a reader should be able to see what the weaker
# measurement said. They are a different configuration and not more points on
# this one: k=1 against k=3, probe-t8t64 against the campaign harness, one run
# a cell against two, and one of their four cells compares 32 768 tokens
# against 30 000 because that was the longest leg the A100 side had.
import statistics, collections

def campaign(path, cfgs):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for line in open(path):
        r = json.loads(line)
        if r.get("kind") == "decode" and r.get("decode_tps") and r["cfg"] in cfgs:
            by[r["cfg"]][r["target"]].append(r["decode_tps"])
    out = {}
    for cfg, d in by.items():
        out[cfg] = {t: {"tok_s": statistics.mean(v), "runs": len(v),
                        "range_pct": (max(v) - min(v)) / statistics.mean(v) * 100.0}
                    for t, v in d.items()}
    return out

CR = campaign(R / "benchmarks/campaign-2026-08-29/results.jsonl",
              {"G31-tp2", "G31-mtp-p45450-tp2"})
CA = campaign(R / "benchmarks/cuda-a100/campaign-2026-08-29/results.jsonl",
              {"A100-G31", "A100-G31-mtp-p45450"})
rungs = sorted(CR["G31-tp2"])
assert rungs == sorted(CA["A100-G31"]), "the two halves must share a ladder"

def cell(d, t):
    c = d[t]
    return {"tok_s": round(c["tok_s"], 2), "runs": c["runs"],
            "range_pct": round(c["range_pct"], 2)}

fig1["campaign"] = {
    "date": "2026-08-29", "k": 3, "harness": "campaign-server", "runs_per_cell": 2,
    "matched": True,
    "rows": [{"ctx": t,
              "radeons_nospec": cell(CR["G31-tp2"], t),
              "radeons_mtp": cell(CR["G31-mtp-p45450-tp2"], t),
              "a100_nospec": cell(CA["A100-G31"], t),
              "a100_mtp": cell(CA["A100-G31-mtp-p45450"], t),
              "advantage_nospec": CA["A100-G31"][t]["tok_s"] / CR["G31-tp2"][t]["tok_s"],
              "advantage_mtp": (CA["A100-G31-mtp-p45450"][t]["tok_s"]
                                / CR["G31-mtp-p45450-tp2"][t]["tok_s"])}
             for t in rungs],
    "source": {"radeons": "benchmarks/campaign-2026-08-29/results.jsonl",
               "a100": "benchmarks/cuda-a100/campaign-2026-08-29/results.jsonl"},
}
_c = fig1["campaign"]
_c["nospec_min"] = min(r["advantage_nospec"] for r in _c["rows"])
_c["nospec_max"] = max(r["advantage_nospec"] for r in _c["rows"])
_c["mtp_min"] = min(r["advantage_mtp"] for r in _c["rows"])
_c["mtp_at_500"] = _c["rows"][0]["advantage_mtp"]
_c["mtp_at_32k"] = _c["rows"][-1]["advantage_mtp"]
# the A100 leads on every rung without speculation, and does not with it
_c["nospec_always_ahead"] = all(r["advantage_nospec"] > 1.0 for r in _c["rows"])
_c["mtp_behind_at"] = [r["ctx"] for r in _c["rows"] if r["advantage_mtp"] < 1.0]

# ---- what the prose beside fig1 claims, computed rather than typed ---------
# Acceptance comes from vLLM's own `SpecDecoding metrics` lines, aligned to each
# rung's measurement window by the timestamps both sides carry. Step cost is
# derived: a speculative step yields `acceptance` tokens and a plain one yields
# 1, so its cost in plain steps is (acceptance / spec_rate) / (1 / plain_rate).
import datetime, re as _re

_ACC = _re.compile(r"INFO (\d\d)-(\d\d) (\d\d):(\d\d):(\d\d).*?"
                   r"Mean acceptance length: ([0-9.]+)")

def accept_by_rung(log, decodes):
    """{ctx: mean acceptance} for the rungs a metrics line falls inside."""
    pts = sorted(decodes.items())
    year = datetime.datetime.fromtimestamp(pts[0][1]["ts"],
                                           datetime.timezone.utc).year
    marks = []
    for line in open(log, errors="replace"):
        m = _ACC.search(line)
        if not m:
            continue
        mo, d, hh, mm, ss = (int(x) for x in m.groups()[:5])
        marks.append((datetime.datetime(year, mo, d, hh, mm, ss,
                                        tzinfo=datetime.timezone.utc).timestamp(),
                      float(m.group(6))))
    out, prev = {}, pts[0][1]["ts"] - 60
    for ctx, c in pts:
        inside = [v for t, v in marks if prev < t <= c["ts"]]
        prev = c["ts"]
        if inside:
            out[ctx] = sum(inside) / len(inside)
    return out

def with_ts(path, cfg):
    by = collections.defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        if r.get("kind") == "decode" and r.get("decode_tps") and r["cfg"] == cfg:
            by[r["target"]].append((r["decode_tps"], r["ts"]))
    return {t: {"tok_s": statistics.mean([v for v, _ in vs]),
                "ts": max(ts for _, ts in vs)} for t, vs in by.items()}

_RS = with_ts(R / "benchmarks/campaign-2026-08-29/results.jsonl", "G31-mtp-p45450-tp2")
_AS = with_ts(R / "benchmarks/cuda-a100/campaign-2026-08-29/results.jsonl", "A100-G31-mtp-p45450")
_RA = accept_by_rung(R / "benchmarks/campaign-2026-08-29/logs/G31-mtp-p45450-tp2.log", _RS)
_AA = accept_by_rung(R / "benchmarks/cuda-a100/campaign-2026-08-29/logs/serve-A100-G31-mtp-p45450.log", _AS)

def economics(acc, spec, plain):
    """cost of a speculative step in plain steps, at every rung acceptance covers"""
    out = {}
    for ctx, a in sorted(acc.items()):
        if ctx in spec and ctx in plain:
            out[ctx] = {"acceptance": round(a, 2),
                        "step_cost": round(a * plain[ctx]["tok_s"] / spec[ctx]["tok_s"], 2)}
    return out

_shared = sorted(set(_RA) & set(_AA))
fig1["economics"] = {
    "rungs": _shared,
    "why": "a speculative step yields `acceptance` tokens where a plain one "
           "yields 1, so its cost in plain steps is "
           "(acceptance / spec_rate) / (1 / plain_rate)",
    "radeons": economics(_RA, _RS, CR["G31-tp2"]),
    "a100": economics(_AA, _AS, CA["A100-G31"]),
}
for who in ("radeons", "a100"):
    d = fig1["economics"][who]
    rows = [d[c] for c in _shared if c in d]
    fig1["economics"][who + "_acceptance_range"] = [min(r["acceptance"] for r in rows),
                                                    max(r["acceptance"] for r in rows)]
    fig1["economics"][who + "_step_cost_range"] = [min(r["step_cost"] for r in rows),
                                                   max(r["step_cost"] for r in rows)]
# the point of the block: the machine that accepts more is the one that loses
fig1["economics"]["a100_accepts_more"] = (
    fig1["economics"]["a100_acceptance_range"][0]
    > fig1["economics"]["radeons_acceptance_range"][1])

# ---- fig2: the same starved path, twice as starved on two cards -----------
stk = lad("mtp-31b-stock45450.json")
fig2 = {"retention": [
    {"who": "radeons", "short": stk[1024], "long": stk[32768],
     "short_ctx": 1024, "long_ctx": 32768,
     "pct": stk[32768] / stk[1024] * 100.0,
     "source": "benchmarks/speculative-decoding/mtp-31b-stock45450.json"},
    {"who": "a100", "short": leg("C1K.log"), "long": leg("C30.log"),
     "short_ctx": 1024, "long_ctx": 30000,
     "pct": leg("C30.log") / leg("C1K.log") * 100.0,
     "source": "benchmarks/cuda-a100/45450-validation/logs/"}],
    # the 2D launch grid is (q blocks, kv heads); TP halves the KV heads a rank
    # sees, so the count is a property of the split, not of the vendor
    "kv_heads": {"model_total": 16, "radeons_per_rank": 8, "a100_per_rank": 16},
    "grid": "(total_num_q_blocks, num_kv_heads)"}
fig2["ratio"] = (fig2["retention"][1]["pct"] / fig2["retention"][0]["pct"])

# ---- fig3: tensor parallelism erodes speculation's economics ---------------
nospec = lad("splitkv-31b-stock.json")
mat = json.load(open(R / "benchmarks/cuda-a100/gemma4-mtp-backend-matrix.json"))
a100_nospec_30k = mat["decode_tok_s"]["30000"]["triton_forced"]["nospec"]
fig3 = {"cases": [
    {"who": "radeons", "ctx": 32768, "nospec": nospec[32768], "spec": p45[32768],
     "gain_pct": (p45[32768] / nospec[32768] - 1) * 100.0},
    {"who": "a100", "ctx": 30000, "nospec": a100_nospec_30k, "spec": leg("D30.log"),
     "gain_pct": (leg("D30.log") / a100_nospec_30k - 1) * 100.0}],
    "why": "every draft step pays the all-reduce floor, which does not shrink "
           "with the drafter"}
fig3["ratio"] = fig3["cases"][1]["gain_pct"] / fig3["cases"][0]["gain_pct"]

# ---- the three steps: one card, the second card, and the pair -------------
# The old shape of this article was "two cards against one", which conflates
# three questions and can answer none of them separately. This round measured
# one 7900 XT, one A100 and one L4 on the same ladder, so the three come apart:
# what one consumer card is worth against one datacentre card, what the second
# consumer card then buys, and only then what the pair is worth.
#
# Rule for picking a run: **decode and prefill for a machine and a model come
# from the same session.** This figure's whole claim is that the second card is
# worth something different at prefill than at decode, and reading the two off
# different runs would put a session boundary inside the comparison. That is
# what fixes the A100 on 2026-08-30 rather than 2026-08-29: the earlier session
# has decode but its prefill is 0 of 11 rungs chart-grade -- prefix caching was
# on and every rung is a strict prefix of the next. The front page draws A100
# decode from the earlier session and A100 prefill from this one; that is its
# rule, not this one's, and the two disagree by 2.2 % on the MoE.
sys.path.insert(0, str(R / "benchmarks" / "analyze"))
import build_prefill as _bp

DEC = [json.loads(l) for l in open(R / "benchmarks" / "decode.jsonl")]
PRE = [json.loads(l) for l in open(R / "benchmarks" / "prefill.jsonl")]
FITS = {(f["machine"], f["cfg"], f["date"]): f for f in _bp.fits(PRE)}

# machine id -> (machine as the projections name it, how many cards)
CARDS = {"one": ("RX 7900 XT", 1), "two": ("RX 7900 XT", 2),
         "a100": ("A100-SXM4-80GB", 1), "l4": ("L4", 1)}
# (machine id, model) -> (cfg, date). One arm each, stock only.
ARMS = {
    ("one",  "gemma-4-12B-it"):  ("A-12B-tp1",   "2026-08-24"),
    ("two",  "gemma-4-12B-it"):  ("A-12B-tp2",   "2026-08-24"),
    ("a100", "gemma-4-12B-it"):  ("G12",         "2026-08-30"),
    ("l4",   "gemma-4-12B-it"):  ("G12",         "2026-08-30"),
    ("one",  "gemma-4-26B-A4B"): ("E26-tp1-u95", "2026-08-30"),
    ("two",  "gemma-4-26B-A4B"): ("E-26B-tp2",   "2026-08-24"),
    ("a100", "gemma-4-26B-A4B"): ("G26A4B",      "2026-08-30"),
    ("l4",   "gemma-4-26B-A4B"): ("G26A4B",      "2026-08-30"),
}
MODELS = ["gemma-4-12B-it", "gemma-4-26B-A4B"]


def dec_ladder(mid, model):
    machine, _ = CARDS[mid]
    cfg, date = ARMS[(mid, model)]
    rows = sorted([r for r in DEC if r["machine"] == machine and r["cfg"] == cfg
                   and r["date"] == date and r["spec"] is None],
                  key=lambda r: r["ctx"])
    assert rows, (mid, model, cfg, date)
    return rows


def pre_fit(mid, model):
    machine, _ = CARDS[mid]
    cfg, date = ARMS[(mid, model)]
    f = FITS.get((machine, cfg, date))
    assert f and "b_us_tok" in f, (mid, model, cfg, date, f and f.get("note"))
    return f


# Step one and step three read off the same object: every arm's ladder, and its
# ratio against one 7900 XT at each depth the two share. One card is the
# denominator throughout, so a reader compares everything to the thing they
# might already own.
ladders = []
for model in MODELS:
    base = {r["ctx"]: r["decode_tok_s"] for r in dec_ladder("one", model)}
    for mid in ("one", "two", "a100", "l4"):
        rows = dec_ladder(mid, model)
        cfg, date = ARMS[(mid, model)]
        f = FITS.get((CARDS[mid][0], cfg, date))
        ladders.append({
            "machine": mid, "machine_name": CARDS[mid][0], "cards": CARDS[mid][1],
            "model": model, "cfg": cfg, "date": date, "tp": rows[0]["tp"],
            "vllm": rows[0]["vllm"], "prefix_caching": rows[0]["prefix_caching"],
            "source": rows[0]["source"],
            # null past 12 000 on the MoE: one card cannot reach those depths,
            # so there is no ratio to state rather than a ratio of 1
            "points": [{"ctx": r["ctx"], "tok_s": r["decode_tok_s"],
                        "runs": r["runs"], "range_pct": r["range_pct"],
                        "graded": r["chart_grade"],
                        "vs_one": (r["decode_tok_s"] / base[r["ctx"]]
                                   if r["ctx"] in base else None)}
                       for r in rows],
            "fit": ({"b_us_tok": f["b_us_tok"], "c_ns_tok2": f["c_ns_tok2"],
                     "r2": f["r2"], "rungs": f["rungs"]}
                    if f and "b_us_tok" in f else None),
        })

# Step two, and the reason this article was rewritten. The second card is one
# thing at decode and another at prefill, and neither is the 2x a reader who has
# not measured it expects. Decode is per rung, because the gain is not flat and
# quoting either end alone hides that it rises before it falls.
second = []
for model in MODELS:
    one = {p["ctx"]: p["tok_s"] for p in
           next(l for l in ladders if l["machine"] == "one" and l["model"] == model)["points"]}
    two = {p["ctx"]: p["tok_s"] for p in
           next(l for l in ladders if l["machine"] == "two" and l["model"] == model)["points"]}
    shared = sorted(set(one) & set(two))
    gains = [{"ctx": c, "gain": two[c] / one[c]} for c in shared]
    fo, ft = pre_fit("one", model), pre_fit("two", model)
    second.append({
        "model": model, "decode": gains,
        "decode_min": min(g["gain"] for g in gains),
        "decode_max": max(g["gain"] for g in gains),
        "decode_at_shortest": gains[0]["gain"], "decode_at_deepest": gains[-1]["gain"],
        "b_gain": fo["b_us_tok"] / ft["b_us_tok"],
        "c_gain": fo["c_ns_tok2"] / ft["c_ns_tok2"],
        "depths": len(shared),
        # A coefficient ratio is not what anyone waits for. This is: the two
        # terms put back together at the deepest rung both arms reached, which
        # is the prompt a reader would actually send. `a` is left out because
        # this ladder cannot measure it -- it moved 9.8 to 99.8 ms across two
        # campaigns of the same arm -- and it is the term that matters least at
        # depth, where b*S and c*S^2 are seconds and a is milliseconds.
        "wall": {"ctx": shared[-1],
                 "one_s": fo["b_us_tok"] * 1e-6 * shared[-1]
                          + fo["c_ns_tok2"] * 1e-9 * shared[-1] ** 2,
                 "two_s": ft["b_us_tok"] * 1e-6 * shared[-1]
                          + ft["c_ns_tok2"] * 1e-9 * shared[-1] ** 2},
    })
    w = second[-1]["wall"]
    w["gain"] = w["one_s"] / w["two_s"]

# What one card is worth against one card, on the two terms prefill separates
# into. b is the linear term -- GEMM throughput, the compute -- and c is the
# quadratic one, which is how badly attention scales. On the dense model they
# come apart by a factor of two; on the mixture-of-experts, whose 4B active
# parameters shrink the compute term, they do not come apart at all. That
# contrast is the reason this figure is per model rather than per machine.
percard = []
for model in MODELS:
    ref = pre_fit("one", model)
    for mid in ("a100", "l4"):
        f = pre_fit(mid, model)
        percard.append({"model": model, "machine": mid,
                        "b_ratio": ref["b_us_tok"] / f["b_us_tok"],
                        "c_ratio": ref["c_ns_tok2"] / f["c_ns_tok2"]})
for r in percard:
    r["terms_separate"] = r["c_ratio"] / r["b_ratio"]

# One list for the figure that draws it: every arm against one 7900 XT, on both
# of prefill's terms. The second card is in here beside the two other cards on
# purpose -- the question the article asks is what each of them is worth against
# the card a reader might already have, and "buy a second one" is one of the
# answers a reader can act on.
terms = []
for model in MODELS:
    ref = pre_fit("one", model)
    for mid in ("two", "a100", "l4"):
        f = pre_fit(mid, model)
        terms.append({"model": model, "machine": mid,
                      "b": ref["b_us_tok"] / f["b_us_tok"],
                      "c": ref["c_ns_tok2"] / f["c_ns_tok2"],
                      "b_us_tok": f["b_us_tok"], "c_ns_tok2": f["c_ns_tok2"],
                      "rungs": f["rungs"], "r2": f["r2"]})
terms_max = max(max(t["b"], t["c"]) for t in terms)

# Two more numbers the prose states. Retention is what the L4 has instead of
# speed, and it is a property of a small card with slow memory rather than an
# advantage. The crossover is the consequence of b and c pointing opposite ways:
# put both terms back together at 32 K and the L4 wins the prompt it lost the
# coefficient on.
retention = [{"machine": l["machine"], "model": l["model"],
              "pct": l["points"][-1]["tok_s"] / l["points"][0]["tok_s"] * 100.0,
              "from_ctx": l["points"][0]["ctx"], "to_ctx": l["points"][-1]["ctx"]}
             for l in ladders]
_S = 32000
_wall = lambda f: f["b_us_tok"] * 1e-6 * _S + f["c_ns_tok2"] * 1e-9 * _S ** 2
crossover = {"ctx": _S, "model": "gemma-4-12B-it",
             "one_s": _wall(pre_fit("one", "gemma-4-12B-it")),
             "l4_s": _wall(pre_fit("l4", "gemma-4-12B-it"))}
crossover["l4_gain"] = crossover["one_s"] / crossover["l4_s"]

split = {"ladders": ladders, "second": second, "percard": percard,
         "retention": retention, "crossover": crossover,
         "terms": terms, "terms_max": terms_max,
         "one_ref": {model: {"b_us_tok": pre_fit("one", model)["b_us_tok"],
                             "c_ns_tok2": pre_fit("one", model)["c_ns_tok2"]}
                     for model in MODELS},
         "models": MODELS, "arms": {f"{k[0]}|{k[1]}": list(v) for k, v in ARMS.items()},
         # 31B is the model the rest of this article is about and it has no
         # single-card row anywhere, because it cannot have one: the checkpoint
         # is 22 GB of weights and the card is 20 GB. Arithmetic, not a gap.
         "no_single_card": {"model": "gemma-4-31B-it", "weights_gb": 22,
                            "card_gb": 20}}

# ---- the realized-bandwidth side, recomputed rather than quoted -----------
jul = V.decode(str(R / "benchmarks" / "results.jsonl"))
GIB = 1024 ** 3 / 1e9          # GiB -> GB
BYTES = {"B-8B-tp2": ("Qwen3-8B BF16", 7.01), "C-31B-tp2": ("gemma-4-31B w4a16", 10.84),
         "A-12B-tp2": ("gemma-4-12B w4a16", 4.78)}
util = []
for cfg, (name, gib) in BYTES.items():
    rate = V.tps(jul, cfg, 500)
    util.append({"cfg": cfg, "model": name, "gib_per_token": gib, "tok_s": rate,
                 "gb_s": gib * GIB * rate,
                 "pct": gib * GIB * rate / 800.0 * 100.0})
util.sort(key=lambda u: -u["pct"])
bw = {"rows": util, "peak_gb_s": 800.0,
      "subject": "C-31B-tp2",
      "subject_pct": next(u["pct"] for u in util if u["cfg"] == "C-31B-tp2")}

out = {"_what": "Every figure in a100-vs-two-radeons.html. Derived from "
                "benchmarks/speculative-decoding/, benchmarks/cuda-a100/ and "
                "benchmarks/results.jsonl by site/src/genfig-a100.py.",
       "fig1": fig1, "fig2": fig2, "fig3": fig3, "bandwidth": bw,
       "split": split}
json.dump(out, open(pathlib.Path(__file__).parent / "figures-a100.json", "w"),
          ensure_ascii=False, indent=1)
print("fig1:", [(r["ctx"], round(r["advantage"], 2)) for r in fig1["rows"]],
      f'| nominal {fig1["nominal_ratio"]:.2f} | U {fig1["u_shaped"]}',
      "| below nominal at", fig1["below_nominal"])
print("fig2 retention:", [(r["who"], round(r["pct"], 1)) for r in fig2["retention"]],
      f'ratio {fig2["ratio"]:.2f}')
print("fig3 gains:", [(c["who"], round(c["gain_pct"], 1)) for c in fig3["cases"]],
      f'ratio {fig3["ratio"]:.2f}')
print("bandwidth:", [(u["model"], round(u["pct"], 1)) for u in util])
print("bytes:", len(json.dumps(out)))
