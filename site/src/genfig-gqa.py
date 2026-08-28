"""Figures for gqa-gate-costs-nothing.html.

Everything here is recomputed from benchmarks/vllm-50603/*.jsonl. The stage-1
grid is the argument: sixty cells, no exception, and the excluded band overlaps
the admitted one on both vLLM versions.
"""
import json, pathlib, statistics
R = pathlib.Path(__file__).resolve().parents[2]
G = R / "benchmarks" / "vllm-50603"
jl = lambda fn: [json.loads(l) for l in open(G / fn)]

CTXS = [1024, 2048, 4096, 8192, 16384, 32768]
SHAPES = [(8, 8), (8, 4), (32, 16), (12, 4), (16, 4)]

# ---- fig1: the grid, on both versions ------------------------------------
def grid(rows):
    by = {(r["num_heads"], r["num_kv_heads"], r["ctx_len"]): r for r in rows}
    out = []
    for h, kv in SHAPES:
        r0 = by[(h, kv, CTXS[0])]
        out.append({"shape": f"{h}/{kv}", "gqa": r0["gqa_ratio"],
                    "admitted": r0["gate_as_shipped"],
                    "cells": [{"ctx": c,
                               "ratio": (by[(h, kv, c)]["triton"]["median_ms"]
                                         / by[(h, kv, c)]["ck"]["median_ms"]),
                               "triton_ms": by[(h, kv, c)]["triton"]["median_ms"],
                               "ck_ms": by[(h, kv, c)]["ck"]["median_ms"]}
                              for c in CTXS]})
    return out


s1 = jl("stage1-rocm-paths.jsonl")
s27a, s27b = jl("stage1-027-r1.jsonl"), jl("stage1-027-r2.jsonl")
fig1 = {"versions": [
    {"id": "023", "vllm": "0.23.1.dev1+g9ddef7117", "rocm": "7.14",
     "rows": grid(s1), "source": "stage1-rocm-paths.jsonl"},
    {"id": "027", "vllm": "0.27.1.dev5+gf46a9dfe2", "rocm": "10.0",
     "rows": grid(s27a), "source": "stage1-027-r1.jsonl"}],
    "ctxs": CTXS}
allr = [c["ratio"] for v in fig1["versions"] for r in v["rows"] for c in r["cells"]]
fig1["cells"] = len(allr)
fig1["never_slower"] = sum(1 for x in allr if x > 1.0)
fig1["min"], fig1["max"] = min(allr), max(allr)
# the bands are quoted over every round measured on that version, not over the
# one round the grid happens to draw
ROUNDS = {"023": [grid(s1)], "027": [grid(s27a), grid(s27b)]}
for v in fig1["versions"]:
    ex = [c["ratio"] for g in ROUNDS[v["id"]] for r in g if not r["admitted"]
          for c in r["cells"]]
    ad = [c["ratio"] for g in ROUNDS[v["id"]] for r in g if r["admitted"]
          for c in r["cells"]]
    v["excluded_band"] = [min(ex), max(ex)]
    v["admitted_band"] = [min(ad), max(ad)]
    v["bands_overlap"] = max(ex) > min(ad) and max(ad) > min(ex)
    v["rounds"] = len(ROUNDS[v["id"]])
# the two 0.27 rounds, so the reader can see what a repeat is worth here
ga, gb = grid(s27a), grid(s27b)
spread = [abs(cb["ratio"] / ca["ratio"] - 1) * 100.0
          for ra, rb in zip(ga, gb) for ca, cb in zip(ra["cells"], rb["cells"])]
fig1["round_spread_pct"] = {"max": max(spread), "median": statistics.median(spread)}

# ---- fig2: accuracy, and the control that makes a flat band mean something -
def med_by_ctx(rows, path):
    return [{"ctx": c,
             "rel": statistics.median([r[path]["max_rel_err"] for r in rows
                                       if r["ctx_len"] == c])} for c in CTXS]


cuda = jl("stage2-cuda-control.jsonl")
fig2 = {"series": [
    {"id": "rocm_triton", "points": med_by_ctx(s1, "triton")},
    {"id": "rocm_ck", "points": med_by_ctx(s1, "ck")},
    {"id": "a100_triton", "points": med_by_ctx(cuda, "triton")}]}
# band and median over every individual cell, not over the per-depth medians:
# ninety cells on two architectures is what the claim is about
allcells = ([r["triton"]["max_rel_err"] for r in s1]
            + [r["ck"]["max_rel_err"] for r in s1]
            + [r["triton"]["max_rel_err"] for r in cuda])
fig2["cells"] = len(allcells)
fig2["band"] = [min(allcells), max(allcells)]
fig2["median"] = statistics.median(allcells)
at = lambda c: statistics.median(
    [r["triton"]["max_rel_err"] for r in s1 if r["ctx_len"] == c]
    + [r["ck"]["max_rel_err"] for r in s1 if r["ctx_len"] == c]
    + [r["triton"]["max_rel_err"] for r in cuda if r["ctx_len"] == c])
fig2["context_move"] = at(CTXS[-1]) / at(CTXS[0])
fig2["context_span"] = CTXS[-1] // CTXS[0]
# the stronger statement: the widest gap between any two depths, not just ends
meds = [at(c) for c in CTXS]
fig2["context_worst_ratio"] = max(meds) / min(meds)
# the figure the source README quotes: the two shortest depths pooled against
# the two longest, whose centres are about 16x apart
bucket = lambda cs: statistics.median(
    [r[a]["max_rel_err"] for r in s1 for a in ("triton", "ck") if r["ctx_len"] in cs]
    + [r["triton"]["max_rel_err"] for r in cuda if r["ctx_len"] in cs])
fig2["bucket_move"] = bucket(CTXS[-2:]) / bucket(CTXS[:2])
fig2["bucket_span"] = ((CTXS[-1] + CTXS[-2]) / 2) / ((CTXS[0] + CTXS[1]) / 2)
fig2["per_depth_median"] = [{"ctx": c, "rel": at(c)} for c in CTXS]
# the positive control: does the harness see corruption when it is there?
ctrl = jl("stage1b-tail-control.jsonl")
fig2["control"] = {
    "rows": len(ctrl),
    "cases": [{"fill": r["fill"], "ctx": r["ctx_len"], "aligned": r["block_aligned"],
               "tail": r["tail_slots"],
               "triton_finite": r["triton"]["all_finite"],
               "ck_finite": r["ck"]["all_finite"]} for r in ctrl]}
nan_rows = [r for r in ctrl if r["fill"] == "nan"]
fig2["control"]["nan_rows"] = len(nan_rows)
fig2["control"]["poisoned"] = sum(1 for r in nan_rows if not r["triton"]["all_finite"])
fig2["control"]["all_poisoned_straddle"] = all(
    not r["block_aligned"] for r in nan_rows if not r["triton"]["all_finite"])
fig2["control"]["aligned_all_clean"] = all(
    r["triton"]["all_finite"] for r in nan_rows if r["block_aligned"])
fig2["control"]["ck_matches_triton"] = all(
    r["ck"]["all_finite"] == r["triton"]["all_finite"] for r in ctrl)
fig2["control"]["garbage_clean"] = all(
    r["triton"]["all_finite"] for r in ctrl if r["fill"] != "nan")

# ---- fig3: end to end, one model, both versions, both run orders ----------
def e2e(fn):
    d = {(r["arm"], r["ctx"]): r for r in jl(fn)}
    return {c: {"stock": d[("stock", c)]["decode_tok_s"],
                "widened": d[("widened", c)]["decode_tok_s"],
                "ratio": d[("widened", c)]["decode_tok_s"] / d[("stock", c)]["decode_tok_s"],
                "gate_flipped": (not d[("stock", c)]["gate_gqa2"])
                                and d[("widened", c)]["gate_gqa2"]}
            for c in (1024, 8192, 32768)}


runs = {"023": e2e("stage3-endtoend.jsonl"), "027A": e2e("stage3-027.jsonl"),
        "027B": e2e("stage3-027b.jsonl")}
fig3 = {"depths": [1024, 8192, 32768], "runs": runs,
        "pooled_027": {c: (runs["027A"][c]["ratio"] + runs["027B"][c]["ratio"]) / 2
                       for c in (1024, 8192, 32768)},
        "order": {"027A": "stock before widened at every depth",
                  "027B": "the reverse"}}
fig3["gate_flipped_everywhere"] = all(runs[k][c]["gate_flipped"]
                                      for k in runs for c in fig3["depths"])
fig3["worst_pass_spread_pct"] = max(
    abs(runs["027B"][c][a] / runs["027A"][c][a] - 1) * 100.0
    for c in fig3["depths"] for a in ("stock", "widened"))
fig3["grows_with_context"] = all(
    fig3["pooled_027"][a] < fig3["pooled_027"][b]
    for a, b in zip(fig3["depths"], fig3["depths"][1:]))

# ---- fig4: the fault widening the gate would also admit -------------------
st, pa = jl("53856-027-stock.jsonl"), jl("53856-027-patched.jsonl")
key = lambda r: (r["dtype"], r["gqa_ratio"], r["poison"], r["ctx_len"])
pb = {key(r): r for r in pa}
cells = []
for r in st:
    p = pb[key(r)]
    cells.append({"dtype": r["dtype"], "gqa": r["gqa_ratio"], "poison": r["poison"],
                  "ctx": r["ctx_len"], "tail": r["tail_slots"],
                  "admitted": r["gate_as_shipped"],
                  "stock_ok": r["as_shipped"]["all_finite"],
                  "patched_ok": p["as_shipped"]["all_finite"],
                  "stock_forced_ok": r["ck_forced"]["all_finite"],
                  "patched_forced_ok": p["ck_forced"]["all_finite"],
                  "triton_ok": r["triton_forced"]["all_finite"]})
fig4 = {"cells": cells, "rows_per_arm": len(st),
        "poisoned_stock": sum(1 for c in cells if not c["stock_ok"]),
        "fixed": sum(1 for c in cells if not c["stock_ok"] and c["patched_ok"]),
        "newly_broken": sum(1 for c in cells if c["stock_ok"] and not c["patched_ok"]),
        "poisoned_forced": sum(1 for c in cells if not c["stock_forced_ok"]),
        "fixed_forced": sum(1 for c in cells
                            if not c["stock_forced_ok"] and c["patched_forced_ok"]),
        "triton_clean": sum(1 for c in cells if c["triton_ok"]),
        "k_only_ever_poisons": sum(1 for c in cells
                                   if c["poison"] == "k_only" and not c["stock_forced_ok"]),
        "ck_ran": sum(1 for r in st + pa if r["gqa_ratio"] == 4
                      and r["as_shipped"]["used_ck_kernel"]),
        "ck_expected": sum(1 for r in st + pa if r["gqa_ratio"] == 4)}
fig4["all_poisoned_straddle"] = all(not c["ctx"] % 16 == 0 for c in cells
                                    if not c["stock_forced_ok"])

out = {"_what": "Every figure in gqa-gate-costs-nothing.html, recomputed from "
                "benchmarks/vllm-50603/*.jsonl by site/src/genfig-gqa.py.",
       "fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4}
json.dump(out, open(pathlib.Path(__file__).parent / "figures-gqa.json", "w"),
          ensure_ascii=False, indent=1)
print(f'fig1 {fig1["cells"]} cells, {fig1["never_slower"]} with CK faster, '
      f'range {fig1["min"]:.2f}-{fig1["max"]:.2f}x')
for v in fig1["versions"]:
    print(f'  {v["id"]}: excluded {v["excluded_band"][0]:.2f}-{v["excluded_band"][1]:.2f} '
          f'admitted {v["admitted_band"][0]:.2f}-{v["admitted_band"][1]:.2f} '
          f'overlap {v["bands_overlap"]}')
print(f'fig1 round spread max {fig1["round_spread_pct"]["max"]:.1f}%, '
      f'median {fig1["round_spread_pct"]["median"]:.1f}%')
print(f'fig2 {fig2["cells"]} cells, band {fig2["band"][0]:.2e}-'
      f'{fig2["band"][1]:.2e}, median {fig2["median"]:.2e}, '
      f'{fig2["context_span"]}x context moves it {fig2["context_move"]:.3f}x, '
      f'bucketed {fig2["bucket_span"]:.0f}x moves it {fig2["bucket_move"]:.3f}x')
print("fig2 control:", {k: v for k, v in fig2["control"].items() if k != "cases"})
print("fig3 pooled 0.27:", {c: round(r, 3) for c, r in fig3["pooled_027"].items()},
      "gate flipped everywhere", fig3["gate_flipped_everywhere"],
      f'worst pass spread {fig3["worst_pass_spread_pct"]:.1f}%')
print("fig4:", {k: v for k, v in fig4.items() if k != "cells"})
print("bytes:", len(json.dumps(out)))
