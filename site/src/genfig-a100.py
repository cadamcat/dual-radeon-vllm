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

# ---- fig1's second layer: the 2026-08-29 ladder, both arms, both machines --
# A different configuration, not more points on the same one: k=3 against the
# k=1 above, the campaign harness against probe-t8t64, eleven matched rungs
# against four with one of them mismatched, and two rounds a cell against one.
# It is drawn as its own toggle for exactly that reason. What it adds is the
# arm the figure above does not have: without speculation the A100 leads
# everywhere, and turning MTP on is what erases the lead.
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
       "fig1": fig1, "fig2": fig2, "fig3": fig3, "bandwidth": bw}
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
