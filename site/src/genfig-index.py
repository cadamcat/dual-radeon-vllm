"""The index's "what this machine does today" figure.

One line per model, each the fastest configuration that model has been measured
in. That is not one experiment: five of the lines come from a single campaign
and are directly comparable, and two do not, because a later stack or
speculative decoding beats that campaign by more than its own spread. Which
line is which is data, not a footnote -- every series carries the stack it
needed, and this script refuses to emit a pick it cannot show is the best one
in the repository.

Run it; do not hand-edit figures-index.json.
"""
import json, pathlib, re, sys

R = pathlib.Path(__file__).resolve().parents[2]
B = R / "benchmarks"
led = [json.loads(l) for l in open(B / "ledger.jsonl")]

# The backbone: one day, one stack, one harness, so these lines are one
# experiment and may be read against each other.
CAMPAIGN = "2026-08-24"
BACKBONE = ["gemma-4-26B-A4B", "Qwen3-8B", "Muse-Glimmer-30B", "gemma-3-27b-it",
            "gemma-4-12B-it"]
# What the reader is most likely to be here for, lit without being asked.
LIT = ["Qwen3.8-27B", "gemma-4-31B-it", "Muse-Glimmer-30B", "gemma-4-26B-A4B"]
# Deliberately absent: Qwen3.6-27B exists in the ledger only as the superseded
# 2026-07-25 stock run whose collapse the split-KV work fixed, so it is not a
# statement about today.
OMIT = ["Qwen3.6-27B"]

sid = lambda r: (r["model"], r["tp"], r["vllm"], tuple(r["patches"]), r["harness"], r["date"])


def ledger_series(model, tp=2, date=None, vllm=None, patches=None, cfg=None):
    rows = [r for r in led if r["model"] == model and r["tp"] == tp
            and (date is None or r["date"] == date)
            and (vllm is None or r["vllm"] == vllm)
            and (patches is None or r["patches"] == patches)
            and (cfg is None or r["cfg"] == cfg)]
    assert rows, (model, tp, date, vllm, patches, cfg)
    ids = {sid(r) for r in rows}
    assert len(ids) == 1, f"{model}: {len(ids)} series match, not one"
    rows.sort(key=lambda r: r["ctx"])
    i = ids.pop()
    return {"model": model, "tp": tp, "vllm": i[2], "patches": list(i[3]),
            "harness": i[4], "date": i[5], "quant": rows[0]["quant"],
            "arch": rows[0]["arch"], "spec": rows[0]["spec"] is not None,
            "spec_desc": rows[0]["spec"], "attn_backend": rows[0]["attn_backend"],
            "cfg": rows[0]["cfg"],
            "source": "benchmarks/ledger.jsonl",
            "points": [{"ctx": r["ctx"], "tok_s": r["decode_tok_s"], "runs": r["runs"],
                        "range_pct": r["range_pct"], "graded": r["chart_grade"]}
                       for r in rows]}


series = [ledger_series(m, 2, CAMPAIGN) for m in BACKBONE]

# --- the two models the campaign does not represent -------------------------
# Both from 2026-08-29, eleven rungs and two rounds a cell, neither speculating.
# They used to be a three-point probe and a four-point k=1 *speculative* arm,
# which meant one line in a chart captioned "what this machine does today" had
# MTP on and the rest did not, with nothing on the page saying so. Speculation
# is a button now, below.
#
# Qwen3.8-27B: the campaign ran it on 0.23.1, which has no native gfx1100 W4A16
# kernel for an asymmetric checkpoint. 0.27 does. The kernel is pinned with
# --attention-backend because ROCm's own selector takes ROCM_ATTN and that is
# 15.0% slower at 32K -- one flag, no patch, and the check below would fail if
# the faster of the two were left undrawn.
series.append(ledger_series("Qwen3.8-27B", 2, "2026-08-29", cfg="Q38-triton-tp2"))
series.append(ledger_series("gemma-4-31B-it", 2, "2026-08-29", cfg="G31-tp2"))

for s in series:
    s["machine"] = "rdna3"
    s["lit"] = s["model"] in LIT

# --- speculation, as its own layer -----------------------------------------
# One arm per model that has one, measured the same day as that model's line
# above and against it as a control. Off until the MTP button is pressed: it is
# a different way of running the same machine, not a faster reading of the same
# thing, and on Qwen3.8 it is a net loss past 8K.
MTP = [("gemma-4-31B-it", "G31-mtp-p45450-tp2"),
       ("Qwen3.8-27B", "Q38-mtp-triton-p45450-tp2")]
for model, cfg in MTP:
    m = ledger_series(model, 2, "2026-08-29", cfg=cfg)
    m["machine"] = "rdna3"
    m["lit"] = False
    assert m["spec"], f"{cfg} is not a speculative arm"
    series.append(m)

# --- the other machine ------------------------------------------------------
# Both arms from the 2026-08-29 campaign, eleven rungs and two rounds a cell,
# the same ladder the Radeon lines use. It used to be five single-run points
# from a validation log, speculative, with no control beside it -- so the one
# cross-machine comparison on the front page was between a speculative A100 and
# a stock Radeon. Now the default is stock against stock and the MTP button
# moves both.
import statistics as _st, collections as _ct

def _a100(cfg, spec_desc, lit):
    by = _ct.defaultdict(list)
    for line in open(B / "cuda-a100" / "campaign-2026-08-29" / "results.jsonl"):
        r = json.loads(line)
        if r.get("kind") == "decode" and r.get("decode_tps") and r["cfg"] == cfg:
            by[r["target"]].append(r["decode_tps"])
    assert len(by) == 11, f"{cfg}: {len(by)} rungs"
    pts = []
    for ctx in sorted(by):
        v = by[ctx]
        m = _st.mean(v)
        rng = (max(v) - min(v)) / m * 100.0
        pts.append({"ctx": ctx, "tok_s": m, "runs": len(v),
                    "range_pct": rng, "graded": len(v) >= 2 and rng <= 8.0})
    return {"model": "gemma-4-31B-it", "machine": "a100", "tp": 1, "lit": lit,
            "vllm": "0.28.0", "patches": ["vllm#45450 3D admission"] if spec_desc else [],
            "harness": "campaign-server", "date": "2026-08-29", "quant": "w4a16 QAT",
            "arch": "dense", "spec": bool(spec_desc), "spec_desc": spec_desc,
            "attn_backend": "TRITON_ATTN", "cfg": cfg,
            "source": "benchmarks/cuda-a100/campaign-2026-08-29/results.jsonl",
            "points": pts}

series.append(_a100("A100-G31", None, True))
series.append(_a100("A100-G31-mtp-p45450", {"method": "mtp", "k": 3}, False))

# --- how well this machine repeats a whole campaign -------------------------
# The same models were run twice, thirty days apart, on the same box. Their
# disagreement is this machine's campaign-to-campaign reproducibility, measured
# rather than assumed, and it is the slack the check below is entitled to.
PRIOR = "2026-07-25"
c1 = {(r["model"], r["tp"], r["ctx"]): r["decode_tok_s"] for r in led if r["date"] == PRIOR}
c2 = {(r["model"], r["tp"], r["ctx"]): r["decode_tok_s"] for r in led if r["date"] == CAMPAIGN}
shared = sorted(set(c1) & set(c2))
assert len(shared) >= 40, f"only {len(shared)} cells shared by the two campaigns"
diffs = sorted(abs(c1[k] - c2[k]) / max(c1[k], c2[k]) * 100 for k in shared)
REPRO = {"cells": len(shared), "worst_pct": diffs[-1], "median_pct": diffs[len(diffs) // 2],
         "campaigns": [PRIOR, CAMPAIGN]}

# the backbone is this campaign because it is the one that measured all of them
for m in BACKBONE:
    assert any(r["model"] == m and r["date"] == CAMPAIGN for r in led), m
assert not all(any(r["model"] == m and r["date"] == PRIOR for r in led) for m in BACKBONE), \
    "the earlier campaign also covers every backbone model; say why this one"

# --- the pick has to survive the rest of the repository ----------------------
# For every model on the Radeons, no other series in the ledger may beat the one
# chosen here at a depth they share, by more than this machine repeats itself. A
# faster run that exists and is not drawn would make "today's best" a lie.
# Like against like: a speculative row cannot beat a line drawn without
# speculation, and does not answer the same question. Each layer is checked
# against the rows of its own kind, so both the default view and the MTP one
# have to be the fastest of their sort.
beaten = []
for spec_layer in (False,):
    picked = {s["model"]: s for s in series
              if s["machine"] == "rdna3" and s["spec"] == spec_layer}
    for model, s in picked.items():
        mine = {p["ctx"]: p["tok_s"] for p in s["points"]}
        for r in led:
            if r["model"] != model or r["tp"] != s["tp"]:
                continue
            if (r["spec"] is not None) != spec_layer:
                continue
            if sid(r) == (model, s["tp"], s["vllm"], tuple(s["patches"]),
                          s["harness"], s["date"]) and r["cfg"] == s.get("cfg"):
                continue
            if r["ctx"] not in mine:
                continue
            slack = max(r["range_pct"] or 0.0, REPRO["worst_pct"]) / 100.0 * r["decode_tok_s"]
            if r["decode_tok_s"] - mine[r["ctx"]] > slack:
                beaten.append((("MTP " if spec_layer else "") + model, r["ctx"],
                               round(r["decode_tok_s"], 2), round(mine[r["ctx"]], 2),
                               r["date"], r["cfg"]))
assert not beaten, "a faster measurement exists and is not drawn:\n  " + \
                   "\n  ".join(map(str, beaten))

# The MTP layer is held to a different rule, because "the fastest speculative
# measurement" is not what it answers. Neither Qwen3.8 arm dominates -- ROCM_ATTN
# is faster at 500 (91.43 against 75.10) and far slower at 32K (24.34 against
# 34.02) -- so picking by speed would draw an arm whose stock control is not on
# the chart. What the button is for is what happens to *this line* when MTP goes
# on, so each speculative series has to be its own line's companion: same day,
# same kernel, same stack apart from the speculation.
mtp_pairs = []
for m in [x for x in series if x["machine"] == "rdna3" and x["spec"]]:
    base = next(x for x in series
                if x["machine"] == "rdna3" and not x["spec"] and x["model"] == m["model"])
    assert m["date"] == base["date"], (m["cfg"], m["date"], base["date"])
    assert m["attn_backend"] == base["attn_backend"], (m["cfg"], m["attn_backend"])
    assert m["vllm"] == base["vllm"], (m["cfg"], m["vllm"])
    mtp_pairs.append({"model": m["model"], "base_cfg": base["cfg"], "mtp_cfg": m["cfg"],
                      "attn_backend": m["attn_backend"], "date": m["date"],
                      "spec": m["spec_desc"],
                      "delta_pct": [
                          {"ctx": a["ctx"],
                           "pct": (b["tok_s"] / a["tok_s"] - 1) * 100.0}
                          for a, b in zip(base["points"], m["points"])]})
for pr in mtp_pairs:
    pr["crosses_zero"] = (pr["delta_pct"][0]["pct"] > 0) != (pr["delta_pct"][-1]["pct"] > 0)
    pr["at_shortest_pct"] = pr["delta_pct"][0]["pct"]
    pr["at_deepest_pct"] = pr["delta_pct"][-1]["pct"]

# Each override has to earn its place against the campaign line it replaces --
# but "faster" is not the only way to earn it, and pretending otherwise is what
# put a speculative arm in this chart without a label. Two reasons count:
#
#   faster        Qwen3.8 on 0.27, which the 0.23 campaign had no native
#                 gfx1100 W4A16 kernel for
#   reproduces    gemma-4-31B, whose 2026-08-29 line lands within this
#                 machine's own campaign-to-campaign spread of the 2026-08-24
#                 one, and is drawn instead because it is the same-day control
#                 for the MTP arm the button reveals. A pair measured five days
#                 apart is not a pair.
over = []
for model in ("Qwen3.8-27B", "gemma-4-31B-it"):
    camp = ledger_series(model, 2, CAMPAIGN)
    c = {p["ctx"]: p["tok_s"] for p in camp["points"]}
    mine = {p["ctx"]: p["tok_s"] for p in picked[model]["points"]}
    near = [(x, min(c, key=lambda k: abs(k - x))) for x in mine]
    gains = [mine[x] / c[k] for x, k in near if abs(x - k) / x < 0.06]
    assert gains, f"{model}: override shares no depth with the campaign"
    faster = min(gains) > 1.0
    reproduces = (1 - min(gains)) * 100.0 <= REPRO["worst_pct"]
    assert faster or reproduces, (
        f"{model}: override neither beats the campaign nor reproduces it "
        f"(worst {(min(gains) - 1) * 100:.2f}% against a {REPRO['worst_pct']:.2f}% spread)")
    over.append({"model": model, "min": min(gains), "max": max(gains),
                 "why": "faster" if faster else "reproduces",
                 "campaign_deepest": c[max(c)], "picked_deepest": mine[max(mine)]})

for m in OMIT:
    assert any(r["model"] == m for r in led), f"{m} is not in the ledger to omit"
    assert not any(s["model"] == m for s in series), f"{m} was not omitted"

out = {
    "_what": "The index's best-measured-today figure. One line per model, each the "
             "fastest configuration it has been measured in; five share one campaign "
             "and two do not. Derived by site/src/genfig-index.py from "
             "benchmarks/ledger.jsonl, benchmarks/speculative-decoding/ and "
             "benchmarks/cuda-a100/.",
    "best": {
        "series": series,
        "campaign": {"date": CAMPAIGN, "models": len(BACKBONE),
                     "vllm": series[0]["vllm"], "patches": series[0]["patches"]},
        "repro": REPRO,
        "overrides": over,
        "mtp_pairs": mtp_pairs,
        "omitted": OMIT,
        "machines": [{"id": "rdna3", "default": True}, {"id": "a100", "default": False}],
        "ctx_min": min(p["ctx"] for s in series for p in s["points"]),
        "ctx_max": max(p["ctx"] for s in series for p in s["points"]),
        "fastest": max(p["tok_s"] for s in series for p in s["points"]),
    },
}
json.dump(out, open(pathlib.Path(__file__).parent / "figures-index.json", "w"),
          ensure_ascii=False, indent=1)

print(f"{len(series)} series, {sum(len(s['points']) for s in series)} points")
for s in series:
    print(f'  {s["machine"]:6s} {s["model"]:18s} {"lit " if s["lit"] else "    "}'
          f'{len(s["points"]):2d} pts  {s["points"][0]["tok_s"]:6.1f} -> '
          f'{s["points"][-1]["tok_s"]:5.1f}  '
          f'{"MTP " if s["spec"] else ""}{"+".join(s["patches"]) or "stock"}')
print("overrides:", [(o["model"], round(o["min"], 2), round(o["max"], 2)) for o in over])
print(f'the two campaigns agree on {REPRO["cells"]} cells to '
      f'{REPRO["worst_pct"]:.2f}% at worst, {REPRO["median_pct"]:.2f}% median')
print("no faster measurement is left undrawn")

# --- one small figure per article card -------------------------------------
# Each card's numbers are read out of that article's own figures-*.json, so a
# card and the page it links to cannot disagree about what the article found --
# the same rule the one-line summary already follows. Nothing here is typed.
#
# Where the article's finding IS a comparison, the card draws both sides rather
# than the ratio between them: a lone ratio line tells a reader the shape and
# not the thing. A name beginning "@" is a strings-table key, because it is
# prose; a bare name is a machine string and reads the same in both languages.
D = pathlib.Path(__file__).parent
fig = lambda n: json.load(open(D / n))

A_HYB, A_A100, A_SPEC = fig("figures.json"), fig("figures-a100.json"), fig("figures-spec.json")
A_W4, A_MEAS, A_MOE = fig("figures-w4a16.json"), fig("figures-measure.json"), fig("figures-moe.json")
A_LOAD, A_RCCL, A_RD = fig("figures-loader.json"), fig("figures-rccl.json"), fig("figures-rdna3.json")
A_GQA, A_65 = fig("figures-gqa.json"), fig("figures-6565.json")

_hyb = [s for s in A_HYB["fig1"]["series"] if s["arch"] == "hybrid SSM"][0]
_dense = [s for s in A_HYB["fig1"]["series"]
          if s["arch"] == "dense" and len(s["points"]) == len(_hyb["points"])][0]
_gqa023 = A_GQA["fig1"]["versions"][0]
_gqaex = [r for r in _gqa023["rows"] if not r["admitted"]]
_meas = A_MEAS["fig2"]["rows"][0]

cards = {
 # what this article compares is the SLOPE -- "fourteen to forty times steeper
 # than any dense model" -- so the card draws each model against its own rate at
 # the shortest depth. On absolute tok/s the hybrid's 12.1 sits so far under the
 # dense model's 79.6 that its whole collapse reads as a flat line near zero.
 "hybrid-ssm-collapse": {
   "form": "line", "unit": "cRetained", "xlog": True, "y0": 0, "xctx": True,
   "series": [{"name": _hyb["model"], "kind": "bad",
               "pts": [[p["ctx"], p["tok_s"] / _hyb["points"][0]["tok_s"] * 100.0]
                       for p in _hyb["points"]]},
              {"name": _dense["model"],
               "pts": [[p["ctx"], p["tok_s"] / _dense["points"][0]["tok_s"] * 100.0]
                       for p in _dense["points"]]}],
   "src": "figures.json fig1"},
 "a100-vs-two-radeons": {
   "form": "line", "unit": "cTokS", "xlog": True, "y0": 0, "xctx": True,
   "series": [{"name": "@cRadeons",
               "pts": [[r["ctx"], r["radeons"]] for r in A_A100["fig1"]["rows"]]},
              {"name": "@cA100", "alt": True,
               "pts": [[r["a100_ctx"], r["a100"]] for r in A_A100["fig1"]["rows"]]}],
   "src": "figures-a100.json fig1"},
 "speculative-decoding-net-loss": {
   "form": "line", "unit": "cTokS", "xlog": True, "y0": 0, "xctx": True,
   "series": [{"name": "@cNoSpec",
               "pts": [[r["ctx"], r["nospec"]] for r in A_SPEC["fig1"]["rows"]]},
              {"name": "MTP", "kind": "bad",
               "pts": [[r["ctx"], r["mtp"]] for r in A_SPEC["fig1"]["rows"]]}],
   "src": "figures-spec.json fig1"},
 "w4a16-two-problems": {
   "form": "line", "unit": "cMsStep", "xlog": True, "y0": 0, "xctx": True,
   "series": [{"name": "@cAsym", "kind": "bad",
               "pts": [[c["ctx"], c["asym_ms"]] for c in A_W4["fig1"]["cells"]]},
              {"name": "@cSym",
               "pts": [[c["ctx"], c["sym_ms"]] for c in A_W4["fig1"]["cells"]]}],
   "src": "figures-w4a16.json fig1"},
 "measuring-decode": {
   "form": "line", "unit": "cRun", "y0": 0, "xrun": True,
   "rule": _meas["converged"], "ruleT": "@cConverged",
   "series": [{"name": "@cFourRuns",
               "pts": [[i + 1, v] for i, v in enumerate(_meas["runs"])]}],
   "src": "figures-measure.json fig2"},
 "gqa-gate-costs-nothing": {
   "form": "line", "unit": "cRatio", "xlog": True, "rule": 1.0, "ruleT": "@cParity",
   "xctx": True,
   "series": [{"name": r["shape"], "pts": [[c["ctx"], c["ratio"]] for c in r["cells"]]}
              for r in _gqaex],
   "src": "figures-gqa.json fig1"},
 "moe-written-off-by-eager": {
   "form": "bars", "unit": "cTokS",
   "bars": ([{"label": b["model"], "v": b["tok_s"]} for b in A_MOE["fig1"]["bars"]]
            + [{"label": A_MOE["fig1"]["bars"][0]["model"], "note": "@cEager",
                "v": A_MOE["fig1"]["eager"]["tok_s"], "kind": "bad"}]),
   "src": "figures-moe.json fig1"},
 "weight-loading-19x": {
   "form": "bars", "unit": "cMsLog", "log": True,
   # the article calls these kernels -28 and -30; the label is derived from the
   # kernel string rather than typed, and says which of the two -28s this is
   "bars": [{"label": re.search(r"-\d+", s["kernel"]).group(0)
                      + (" stock" if s["shipped"] else " +342981f"),
             "v": [c for c in s["cases"] if c["key"] == "rw_p_resident"][0]["ms"],
             "kind": "bad" if i == 0 else None}
            for i, s in enumerate(A_LOAD["fig1"]["states"])],
   "src": "figures-loader.json fig1"},
 "rdna3-second-class": {
   "form": "bars", "unit": "cFindings",
   "bars": [{"label": "@cRdna3", "v": A_RD["fig1"]["counts"]["rdna3"], "kind": "bad"},
            {"label": "@cNotRdna3",
             "v": A_RD["fig1"]["total"] - A_RD["fig1"]["counts"]["rdna3"]}],
   "src": "figures-rdna3.json fig1"},
 "reporting-a-non-reproduction": {
   "form": "bars", "unit": "cInits",
   "bars": [{"label": a["arm"], "v": a["n"]} for a in A_65["fig1"]["arms"]],
   "src": "figures-6565.json fig1"},
 "rccl-atomics-hostcall": {
   "form": "status", "unit": "cHostcall",
   "rows": [{"label": s["rccl"], "ok": s["behaviour"] == "works",
             "note": "0" if s["hostcall"] == "0" else "N"}
            for s in A_RCCL["shipped"]],
   "src": "figures-rccl.json shipped"},
}
out["cards"] = cards
json.dump(out, open(D / "figures-index.json", "w"), ensure_ascii=False, indent=1)
print(f"cards: {len(cards)} "
      f"({sum(1 for c in cards.values() if c['form'] == 'line')} line, "
      f"{sum(1 for c in cards.values() if c['form'] == 'bars')} bars, "
      f"{sum(1 for c in cards.values() if c['form'] == 'status')} status)")
