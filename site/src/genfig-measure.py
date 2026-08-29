"""Figures for measuring-decode.html.

Four committed sources: the harness calibration rounds, the two campaign files
the cross-campaign control is computed from, the ledger's own spread
distribution, and the greedy-nondeterminism record.
"""
import json, pathlib, statistics, sys
R = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "benchmarks" / "analyze"))
import verify_doc_figures as V
import build_ledger

C = R / "benchmarks" / "harness-calibration"
TARGETS = [500, 8000, 32000]

# ---- fig1 and fig2: four identical runs, and what they converge to --------
cal = {}
for i in (1, 2, 3, 4):
    for line in open(C / f"harness-cal-r{i}.jsonl"):
        r = json.loads(line)
        cal[(i, r["campaign_target"])] = r
aug = [json.loads(l) for l in open(R / "benchmarks" / "results-2026-08-24.jsonl")]
camp = {t: [r["decode_tps"] for r in aug if r.get("kind") == "decode"
            and r.get("cfg") == "B-8B-tp2" and r.get("target") == t]
        for t in TARGETS}
conv = lambda t, k: (cal[(3, t)][k] + cal[(4, t)][k]) / 2

fig1 = {"rows": [{"ctx": t, "campaign": sum(camp[t]) / len(camp[t]),
                  "campaign_runs": len(camp[t]),
                  "probe_64": conv(t, "tps_64"), "probe_512": conv(t, "tps_512"),
                  "delta_64_pct": (conv(t, "tps_64") / (sum(camp[t]) / len(camp[t])) - 1) * 100,
                  "delta_512_pct": (conv(t, "tps_512") / (sum(camp[t]) / len(camp[t])) - 1) * 100,
                  "prompt_tokens": cal[(3, t)]["prompt_tokens_got"],
                  "depth_matched": cal[(3, t)]["prompt_tokens_got"]
                                   == cal[(3, t)]["prompt_tokens_wanted"]}
                 for t in TARGETS],
        "model": "Qwen3-8B",
        "why_this_model": ["head_dim 128, so vllm#45916's split-KV cannot apply",
                           "no sliding window, so the block-skip patch cannot apply",
                           "bf16, so no W4A16 kernel selection",
                           "gqa_ratio 4, which the gfx11 gate admits either way"]}
fig1["worst_delta_pct"] = max(abs(r["delta_64_pct"]) for r in fig1["rows"])
fig1["worst_delta_512_pct"] = max(abs(r["delta_512_pct"]) for r in fig1["rows"])
fig1["all_depths_matched"] = all(r["depth_matched"] for r in fig1["rows"])

fig2 = {"rows": [{"ctx": t, "runs": [cal[(i, t)]["tps_64"] for i in (1, 2, 3, 4)],
                  "converged": conv(t, "tps_64"),
                  "vs_converged_pct": [(cal[(i, t)]["tps_64"] / conv(t, "tps_64") - 1) * 100
                                       for i in (1, 2, 3, 4)]}
                 for t in TARGETS]}
fig2["first_run_worst_pct"] = min(r["vs_converged_pct"][0] for r in fig2["rows"])
fig2["second_run_worst_pct"] = min(r["vs_converged_pct"][1] for r in fig2["rows"])
fig2["converged_spread_pct"] = {
    "min": min(abs(cal[(3, t)]["tps_64"] / cal[(4, t)]["tps_64"] - 1) * 100 for t in TARGETS),
    "max": max(abs(cal[(3, t)]["tps_64"] / cal[(4, t)]["tps_64"] - 1) * 100 for t in TARGETS)}
# the deficit is largest where the machine is fastest, which is why it read as
# a harness difference before the repeats existed
fig2["worst_at"] = min(fig2["rows"], key=lambda r: r["vs_converged_pct"][0])["ctx"]

# ---- fig3: the same configurations, thirty days apart ---------------------
jul, augd = V.decode(str(R / "benchmarks/results.jsonl")), V.decode(
    str(R / "benchmarks/results-2026-08-24.jsonl"))
NAME = {"A-12B-tp1": "gemma-4-12B · TP=1", "A-12B-tp2": "gemma-4-12B · TP=2",
        "B-8B-tp1": "Qwen3-8B · TP=1", "B-8B-tp2": "Qwen3-8B · TP=2",
        "C-31B-tp2": "gemma-4-31B · TP=2", "E-26B-tp2": "gemma-4-26B · TP=2"}
shared = sorted(c for c in jul if c in augd)
fig3 = {"controls": [{"cfg": c, "name": NAME[c], "offset_pct": V.offset(jul, augd, c),
                      "depths": len([t for t in sorted(augd[c]) if t in jul[c]])}
                     for c in shared],
        "band_pct": 0.25}
fig3["within_band"] = sum(1 for c in fig3["controls"]
                          if abs(c["offset_pct"]) <= fig3["band_pct"])
fig3["outside"] = [c["name"] for c in fig3["controls"]
                   if abs(c["offset_pct"]) > fig3["band_pct"]]

# ---- fig4: what the ledger records, and where it cuts ---------------------
led = [json.loads(l) for l in open(R / "benchmarks" / "ledger.jsonl")]
spreads = sorted(r["range_pct"] for r in led if r["range_pct"] is not None)
fig4 = {"rows": len(led), "with_range": len(spreads),
        "cut": build_ledger.RANGE_CUT,
        "spreads": spreads,
        "median": statistics.median(spreads),
        "p95": spreads[int(len(spreads) * 0.95)],
        "above_cut": [r for r in spreads if r > build_ledger.RANGE_CUT],
        "ungraded": sum(1 for r in led if not r["chart_grade"]),
        "single_run": sum(1 for r in led if r["runs"] < 2)}
# the gap the cut sits in, which is why it is a cut and not a convention
tail = [s for s in spreads if s > 2.5]
fig4["tail"] = tail

# ---- what the 2026-08-29 campaign is a case of ----------------------------
# The article's subject is that a range travels with a point. This campaign is
# the cleanest case of why: eight arms, one ladder, and the ungraded rungs all
# land in one place. Not "speculation is unstable" -- gemma-4's speculative arm
# is graded at every rung. Speculation on *this model* is.
_c29 = [r for r in led if r["date"] == "2026-08-29"]
_arms = {}
for r in _c29:
    a = _arms.setdefault(r["cfg"], {"cfg": r["cfg"], "model": r["model"],
                                    "spec": r["spec"] is not None,
                                    "attn_backend": r["attn_backend"],
                                    "rungs": 0, "ungraded": 0, "worst_range_pct": 0.0})
    a["rungs"] += 1
    a["ungraded"] += 0 if r["chart_grade"] else 1
    a["worst_range_pct"] = max(a["worst_range_pct"], r["range_pct"] or 0.0)
_arms = [_arms[k] for k in sorted(_arms)]
_nospec = [a for a in _arms if not a["spec"]]
_spec = [a for a in _arms if a["spec"]]
fig4["campaign"] = {
    "date": "2026-08-29", "arms": _arms,
    "nospec_arms": len(_nospec),
    "nospec_rungs": sum(a["rungs"] for a in _nospec),
    "nospec_ungraded": sum(a["ungraded"] for a in _nospec),
    "nospec_worst_range_pct": max(a["worst_range_pct"] for a in _nospec),
    "spec_arms": len(_spec),
    "spec_rungs": sum(a["rungs"] for a in _spec),
    "spec_ungraded": sum(a["ungraded"] for a in _spec),
    # the one speculative arm that is graded throughout, which is what stops
    # this being a statement about speculation as such
    "spec_arms_fully_graded": [a["cfg"] for a in _spec if a["ungraded"] == 0],
    "ungraded_all_one_model": len({a["model"] for a in _spec if a["ungraded"]}) == 1,
    "ungraded_model": sorted({a["model"] for a in _spec if a["ungraded"]})[0],
}
bad = [r for r in led if not r["chart_grade"]]
fig4["ungraded_cells"] = [{"model": r["model"], "ctx": r["ctx"], "runs": r["runs"],
                           "range_pct": r["range_pct"], "values": sorted(r["values"]),
                           "patches": r["patches"]} for r in bad]
# adding runs to the bimodal cell made its range WIDER, which is the point.
# Run order is not recoverable from the ledger's sorted values, so the first
# two are read from the files that produced them.
H = R / "benchmarks" / "hybrid-splitkv-027"
cell8k = lambda fn: [json.loads(l) for l in open(H / fn)
                     if json.loads(l).get("ctx") == 8192
                     and json.loads(l).get("arm") == "splitkv"]
order = ([r["decode_tok_s"] for r in cell8k("qwen38-027-depth.jsonl")]
         + [r["decode_tok_s"] for r in cell8k("qwen38-027-depth-b.jsonl")]
         + [r["decode_tok_s"] for r in cell8k("qwen38-8k-r3r4.jsonl")])
rng = lambda v: (max(v) - min(v)) / statistics.mean(v) * 100
b = bad[0]
fig4["bimodal"] = {"model": b["model"], "ctx": b["ctx"], "runs": b["runs"],
                   "range_pct": b["range_pct"],
                   "values": sorted(b["values"]),
                   "in_run_order": order,
                   "low_cluster": sorted(b["values"])[:2],
                   "high_cluster": sorted(b["values"])[2:],
                   "range_at_two": rng(order[:2]),
                   "range_at_four": rng(order),
                   "widened_by_adding_runs": rng(order) > rng(order[:2]),
                   # what a standard deviation would have said instead
                   "stdev_at_two": statistics.stdev(order[:2]),
                   "stdev_at_four": statistics.stdev(order)}
fig4["why_range_not_stddev"] = {
    "runs_distribution": {n: sum(1 for r in led if r["runs"] == n)
                          for n in sorted({r["runs"] for r in led})}}

# ---- the reason output comparison is not a correctness test here ----------
nd = json.load(open(R / "benchmarks" / "gfx1100-greedy-nondeterminism.json"))
fig5 = {"cells": nd["result"]["cells"],
        "varying": nd["result"]["cells_with_more_than_one_output"],
        "by_state": nd["result"]["split_by_kernel_state"],
        "within_process": len(nd["within_process"]["cells"]),
        "within_process_varying": sum(1 for c in nd["within_process"]["cells"]
                                      if c["distinct_outputs"] > 1),
        "worst_distinct_of_8": max(c["distinct_outputs"]
                                   for c in nd["within_process"]["cells"]),
        "kernel_fixed_input_agreed": True}

out = {"_what": "Every figure in measuring-decode.html. Derived from "
                "benchmarks/harness-calibration/, the two campaign files, "
                "benchmarks/ledger.jsonl and gfx1100-greedy-nondeterminism.json "
                "by site/src/genfig-measure.py.",
       "fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5}
json.dump(out, open(pathlib.Path(__file__).parent / "figures-measure.json", "w"),
          ensure_ascii=False, indent=1)
print("fig1 deltas:", [(r["ctx"], round(r["delta_64_pct"], 2), round(r["delta_512_pct"], 2))
                       for r in fig1["rows"]], "depths matched", fig1["all_depths_matched"])
print("fig2 first run:", [(r["ctx"], round(r["vs_converged_pct"][0], 1)) for r in fig2["rows"]],
      "worst at", fig2["worst_at"],
      f'converged {fig2["converged_spread_pct"]["min"]:.2f}-{fig2["converged_spread_pct"]["max"]:.2f}%')
print("fig3:", [(c["cfg"], round(c["offset_pct"], 3)) for c in fig3["controls"]],
      "within band", fig3["within_band"], "of", len(fig3["controls"]))
print(f'fig4 {fig4["rows"]} rows, cut {fig4["cut"]}, median {fig4["median"]:.2f}, '
      f'p95 {fig4["p95"]:.2f}, tail {[round(x,1) for x in fig4["tail"]]}')
print("fig4 bimodal:", {k: (round(v, 2) if isinstance(v, float) else v)
                        for k, v in fig4["bimodal"].items() if k != "values"})
print("fig4 runs:", fig4["why_range_not_stddev"]["runs_distribution"])
print("fig5:", fig5)
print("bytes:", len(json.dumps(out)))
