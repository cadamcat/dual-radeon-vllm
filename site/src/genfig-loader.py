"""Figures for weight-loading-19x.html.

Two data files, both committed: hmm-kernel-three-states.json for the reproducer
across three kernel states, and loader-flag-kernel-30.json for what the writable
mapping still costs end to end on the kernel Canonical ships. The opening
question's numbers are the 2026-07-24 ones, which have no committed data file
and are extracted from docs/open-questions.md instead.
"""
import json, pathlib, re, statistics
R = pathlib.Path(__file__).resolve().parents[2]
B = R / "benchmarks"

hmm = json.load(open(B / "hmm-kernel-three-states.json"))
ld = json.load(open(B / "loader-flag-kernel-30.json"))
oq = (R / "docs/open-questions.md").read_text()

# ---- fig1: one reproducer, three kernel states ---------------------------
CASES = [("r_p_resident", "r--p", True), ("rw_p_not_resident", "rw-p", False),
         ("rw_p_resident", "rw-p", True)]
med = lambda v: statistics.median(v) if isinstance(v, list) else float(v)
states = []
for s in hmm["states"]:
    row = {"kernel": s["kernel"], "amdgpu": s["amdgpu"], "date": s["date"],
           "shipped": "rebuilt" not in s["amdgpu"], "cases": []}
    for key, mapping, resident in CASES:
        v = s[key]
        row["cases"].append({"key": key, "mapping": mapping, "resident": resident,
                             "ms": med(v),
                             "runs": len(v) if isinstance(v, list) else 1,
                             "values": v if isinstance(v, list) else [v]})
    states.append(row)
bad = lambda st: [c for c in st["cases"] if c["key"] == "rw_p_resident"][0]["ms"]
ro = lambda st: [c for c in st["cases"] if c["key"] == "r_p_resident"][0]["ms"]
fig1 = {"states": states, "unit": hmm["unit"], "what": hmm["what"],
        "timeout_ms": 1000,
        "stock28": bad(states[0]), "reverted": bad(states[1]), "shipped30": bad(states[2]),
        # the arithmetic that identified the mechanism: whole timeout windows
        "windows": bad(states[0]) / 1000.0,
        "residual_ms": bad(states[0]) - 1000.0 * int(bad(states[0]) / 1000.0),
        "fix_factor": bad(states[0]) / bad(states[2]),
        # what survives the fix: the writable mapping itself
        "writable_penalty_30": bad(states[2]) / ro(states[2])}

# ---- fig2: what the writable mapping still costs, end to end -------------
MODES = ["baseline", "eager", "flag", "pread"]
rows = {(r["model"], r["cache"], r["mode"]): r for r in ld["medians_seconds"]}
groups = []
for model, cache in [(m, c) for m in ["gemma-4-12B-w4a16", "Qwen3-8B",
                                      "gemma-4-31B-w4a16", "gemma-4-26B-A4B-MoE"]
                     for c in ("warm", "cold")]:
    cells = [dict(mode=m, **{k: rows[(model, cache, m)][k]
                             for k in ("median_s", "min_s", "max_s", "n")})
             for m in MODES if (model, cache, m) in rows]
    if not cells:
        continue
    base = [c for c in cells if c["mode"] == "baseline"][0]["median_s"]
    for c in cells:
        c["vs_baseline"] = base / c["median_s"]
    groups.append({"model": model, "cache": cache, "cells": cells})
fig2 = {"groups": groups, "modes": MODES,
        "storage": ld["machine"]["storage"], "ram_gib": ld["machine"]["ram_gib"],
        "kernel": ld["machine"]["kernel"],
        "best_flag": max(c["vs_baseline"] for g in groups for c in g["cells"]
                         if c["mode"] == "flag"),
        "worst_flag": min(c["vs_baseline"] for g in groups for c in g["cells"]
                          if c["mode"] == "flag"),
        # the counterexample the PR's own wording did not allow for
        "moe_flag": [c["vs_baseline"] for g in groups if "MoE" in g["model"]
                     for c in g["cells"] if c["mode"] == "flag"][0]}
# the end-to-end control: a real server start, not the isolated harness
e2e = {}
for r in ld["end_to_end_loading_weights_took_seconds"]:
    e2e.setdefault((r["cache"], r["mode"]), []).append(r["loading_weights_took_s"])
fig2["end_to_end"] = [{"cache": c, "mode": m, "reps": v,
                       "mean_s": sum(v) / len(v)} for (c, m), v in sorted(e2e.items())]
# the isolated harness has to be checkable against a real server start, or it
# is only measuring itself. 12B, the one checkpoint both phases cover.
emean = {(r["cache"], r["mode"]): r["mean_s"] for r in fig2["end_to_end"]}
fig2["e2e_vs_harness"] = []
for cache in ("warm", "cold"):
    served = emean[(cache, "baseline")] / emean[(cache, "flag")]
    harness = [c["vs_baseline"] for g_ in groups
               if g_["model"] == "gemma-4-12B-w4a16" and g_["cache"] == cache
               for c in g_["cells"] if c["mode"] == "flag"][0]
    fig2["e2e_vs_harness"].append({"cache": cache, "served": served,
                                   "harness": harness,
                                   "harness_optimistic_pct": (harness / served - 1) * 100.0})

# ---- fig3: where the memory goes, which is the mechanism ------------------
fig3 = {"rows": [{"model": r["model"], "cache": r["cache"], "mode": r["mode"],
                  "anon": r["peak_rss_anon_mib"], "file": r["peak_rss_file_mib"],
                  "peak_gib": r["peak_rss_gib"]}
                 for r in ld["resident_set_split_mib"]]}
by = {(r["model"], r["cache"], r["mode"]): r for r in fig3["rows"]}
b31, f31 = by[("gemma-4-31B-w4a16", "cold", "baseline")], by[("gemma-4-31B-w4a16", "cold", "flag")]
fig3["swap"] = {"model": "gemma-4-31B-w4a16",
                "baseline": [b31["anon"], b31["file"]],
                "flag": [f31["anon"], f31["file"]]}
fig3["sharded"] = by[("Qwen3-8B", "cold", "baseline")]["anon"]
fig3["pread_peak_gib"] = [r["peak_gib"] for r in fig3["rows"] if r["mode"] == "pread"]

# ---- the question as it was asked, from the document that still carries it
hist = oq.split("**Measured on the verified configuration (2026-07-24):**", 1)[1]
dd = re.search(r"\|\s*Disk, raw sequential read[^|]*\|\s*([\d.]+) GB/s", hist)
q8 = re.search(r"\|\s*Qwen3-8B[^|]*\|\s*(\d+) s\s*\|\s*\*\*(\d+) MB/s", hist)
q12 = re.search(r"\|\s*gemma-4-12B[^|]*\|\s*(\d+) s\s*\|\s*\*\*(\d+) MB/s", hist)
opening = {"disk_gb_s": float(dd.group(1)),
           "rows": [{"model": "Qwen3-8B", "seconds": int(q8.group(1)),
                     "mb_s": int(q8.group(2))},
                    {"model": "gemma-4-12B", "seconds": int(q12.group(1)),
                     "mb_s": int(q12.group(2))}],
           "reproducible_from_repo": False,
           "source": "docs/open-questions.md §8, historical record, 2026-07-24"}
for r in opening["rows"]:
    r["times_slower"] = opening["disk_gb_s"] * 1000.0 / r["mb_s"]
opening["range"] = [min(r["times_slower"] for r in opening["rows"]),
                    max(r["times_slower"] for r in opening["rows"])]

out = {"_what": "Every figure in weight-loading-19x.html. fig1 from "
                "benchmarks/hmm-kernel-three-states.json, fig2 and fig3 from "
                "benchmarks/loader-flag-kernel-30.json; the opening question is "
                "extracted from docs/open-questions.md, which is the only record "
                "of it. Derived by site/src/genfig-loader.py.",
       "opening": opening, "fig1": fig1, "fig2": fig2, "fig3": fig3}
json.dump(out, open(pathlib.Path(__file__).parent / "figures-loader.json", "w"),
          ensure_ascii=False, indent=1)
print("opening:", [(r["model"], r["mb_s"], round(r["times_slower"])) for r in opening["rows"]],
      "range", [round(x) for x in opening["range"]])
print(f'fig1 stock {fig1["stock28"]} -> reverted {fig1["reverted"]} -> shipped '
      f'{fig1["shipped30"]}  ({fig1["fix_factor"]:.0f}x)')
print(f'fig1 windows {fig1["windows"]:.4f}, residual {fig1["residual_ms"]:.1f} ms, '
      f'writable penalty on -30 {fig1["writable_penalty_30"]:.2f}x')
print("fig2 groups:", [(g["model"], g["cache"], len(g["cells"])) for g in groups])
print(f'fig2 flag range {fig2["worst_flag"]:.2f}x to {fig2["best_flag"]:.2f}x, '
      f'MoE {fig2["moe_flag"]:.3f}x')
print("fig3 31B swap:", fig3["swap"], "8B sharded anon", fig3["sharded"])
print("bytes:", len(json.dumps(out)))
