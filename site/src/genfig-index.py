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


def ledger_series(model, tp=2, date=None, vllm=None, patches=None):
    rows = [r for r in led if r["model"] == model and r["tp"] == tp
            and (date is None or r["date"] == date)
            and (vllm is None or r["vllm"] == vllm)
            and (patches is None or r["patches"] == patches)]
    assert rows, (model, tp, date, vllm, patches)
    ids = {sid(r) for r in rows}
    assert len(ids) == 1, f"{model}: {len(ids)} series match, not one"
    rows.sort(key=lambda r: r["ctx"])
    i = ids.pop()
    return {"model": model, "tp": tp, "vllm": i[2], "patches": list(i[3]),
            "harness": i[4], "date": i[5], "quant": rows[0]["quant"],
            "arch": rows[0]["arch"], "spec": False,
            "source": "benchmarks/ledger.jsonl",
            "points": [{"ctx": r["ctx"], "tok_s": r["decode_tok_s"], "runs": r["runs"],
                        "range_pct": r["range_pct"], "graded": r["chart_grade"]}
                       for r in rows]}


series = [ledger_series(m, 2, CAMPAIGN) for m in BACKBONE]

# --- the two models the campaign does not represent -------------------------
# Qwen3.8-27B: the campaign ran it on 0.23.1, which has no native gfx1100 W4A16
# kernel for an asymmetric checkpoint. 0.27 does.
q38 = ledger_series("Qwen3.8-27B", 2, "2026-08-28", patches=["vllm#45916 split-KV"])
series.append(q38)

# gemma-4-31B: speculation is a net loss on the stock attention gate and a net
# win once vllm#45450 admits the verify step to the 3D path.
spec = json.load(open(B / "speculative-decoding" / "mtp-31b-p45450.json"))
series.append({
    "model": "gemma-4-31B-it", "tp": 2, "vllm": "0.23.1.dev1+g9ddef7117",
    "patches": ["vllm#45450 3D admission"], "harness": "probe-t8t64",
    "date": "2026-08-26", "quant": "w4a16 QAT", "arch": "dense", "spec": True,
    "source": "benchmarks/speculative-decoding/mtp-31b-p45450.json",
    "points": [{"ctx": r["depth"], "tok_s": r["tok_per_s"], "runs": 1,
                "range_pct": None, "graded": True} for r in spec["rows"]]})

for s in series:
    s["machine"] = "rdna3"
    s["lit"] = s["model"] in LIT

# --- the other machine ------------------------------------------------------
# Only one model has been measured on both, and it was measured in the same
# configuration on both, which is what makes the overlay a comparison rather
# than two unrelated ladders on one pair of axes.
VD = B / "cuda-a100" / "45450-validation" / "logs"
leg = lambda fn: float(re.search(r"RESULT decode_tok_s=([\d.]+)",
                                 open(VD / fn).read()).group(1))
A100 = [(1024, "D1K.log"), (8192, "D8K.log"), (16384, "D16K.log"),
        (30000, "D30.log"), (50000, "D50.log")]
series.append({
    "model": "gemma-4-31B-it", "machine": "a100", "tp": 1, "lit": True,
    "vllm": "0.28.0", "patches": ["vllm#45450 3D admission"], "harness": "probe-ids",
    "date": "2026-08-26", "quant": "w4a16 QAT", "arch": "dense", "spec": True,
    "source": "benchmarks/cuda-a100/45450-validation/logs/",
    "points": [{"ctx": c, "tok_s": leg(fn), "runs": 1, "range_pct": None,
                "graded": True} for c, fn in A100]})

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
picked = {s["model"]: s for s in series if s["machine"] == "rdna3"}
beaten = []
for model, s in picked.items():
    mine = {p["ctx"]: p["tok_s"] for p in s["points"]}
    for r in led:
        if r["model"] != model or r["tp"] != s["tp"] or sid(r) == (
                model, s["tp"], s["vllm"], tuple(s["patches"]), s["harness"], s["date"]):
            continue
        if r["ctx"] not in mine:
            continue
        slack = max(r["range_pct"] or 0.0, REPRO["worst_pct"]) / 100.0 * r["decode_tok_s"]
        if r["decode_tok_s"] - mine[r["ctx"]] > slack:
            beaten.append((model, r["ctx"], round(r["decode_tok_s"], 2),
                           round(mine[r["ctx"]], 2), r["date"], "+".join(r["patches"])))
assert not beaten, "a faster measurement exists and is not drawn:\n  " + \
                   "\n  ".join(map(str, beaten))

# and the two overrides must actually beat the campaign they replace
over = []
for model in ("Qwen3.8-27B", "gemma-4-31B-it"):
    camp = ledger_series(model, 2, CAMPAIGN)
    c = {p["ctx"]: p["tok_s"] for p in camp["points"]}
    mine = {p["ctx"]: p["tok_s"] for p in picked[model]["points"]}
    near = [(x, min(c, key=lambda k: abs(k - x))) for x in mine]
    gains = [mine[x] / c[k] for x, k in near if abs(x - k) / x < 0.06]
    assert gains and min(gains) > 1.0, f"{model}: override does not beat the campaign"
    over.append({"model": model, "min": min(gains), "max": max(gains),
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
