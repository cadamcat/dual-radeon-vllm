"""Figures for moe-written-off-by-eager.html.

The compiled side is recomputed from benchmarks/results.jsonl. The eager side
has no committed raw output, so it is extracted from the table in
docs/benchmarks.md rather than retyped here -- an eager row added to the
document and not to the article fails the build check.
"""
import json, pathlib, re, sys, collections
R = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "benchmarks" / "analyze"))
import verify_doc_figures as V

JUL = R / "benchmarks" / "results.jsonl"
rows = [json.loads(l) for l in open(JUL)]
jul = V.decode(str(JUL))

NAMES = {"E-26B-tp2": ("gemma-4-26B-A4B", "MoE, 128 experts"),
         "B-8B-tp2": ("Qwen3-8B", "dense"),
         "A-12B-tp2": ("gemma-4-12B-it", "dense"),
         "C-31B-tp2": ("gemma-4-31B-it", "dense"),
         "D-27B-tp2": ("Qwen3.6-27B", "hybrid SSM")}
# the TP=1 starts are the same checkpoints at a different parallelism, and the
# figure has to name them the same way or it reads as eight different models
ALL_NAMES = dict(NAMES, **{"B-8B-tp1": ("Qwen3-8B", "dense"),
                           "A-12B-tp1": ("gemma-4-12B-it", "dense")})

# ---- the eager side, extracted from the document that still carries it ----
bm = (R / "docs/benchmarks.md").read_text()


def table_after(heading, doc):
    body = doc.split(heading, 1)[1]
    out = []
    for line in body.split("\n"):
        if line.startswith("|") and not re.match(r"^\|[\s|:-]+\|$", line):
            out.append([c.strip() for c in line.strip().strip("|").split("|")])
        elif out and not line.startswith("|"):
            break
    return out[1:]


strip = lambda s: re.sub(r"\s+", " ", re.sub(r"[*`]", "", s)).strip()
moe_tbl = [[strip(c) for c in r] for r in
           table_after("## 1. The MoE was written off because nobody waited",
                       bm)]
eager_tps = float(re.search(r"~([\d.]+) tok/s", moe_tbl[0][1]).group(1))
eager_pw = [int(x) for x in re.findall(r"(\d+) W", moe_tbl[2][1])]
# the 12B repeat of the same mistake, from the paragraph under that table
tail = bm.split("## 1. The MoE was written off because nobody waited", 1)[1]
eager_12b = float(re.search(r"a flat ([\d.]+) tok/s", tail).group(1))

# ---- fig1: the ranking the eager number would have produced ---------------
fig1 = {"bars": [], "eager": {"tok_s": eager_tps,
                              "reproducible_from_repo": False,
                              "source": "docs/benchmarks.md §1"}}
for cfg, (name, arch) in NAMES.items():
    fig1["bars"].append({"cfg": cfg, "model": name, "arch": arch,
                         "tok_s": V.tps(jul, cfg, 500),
                         "runs": len(jul[cfg][500]["tps"]),
                         "ntok": V.ntok(jul, cfg, 500)})
fig1["bars"].sort(key=lambda b: -b["tok_s"])
fig1["ratio"] = fig1["bars"][0]["tok_s"] / eager_tps
fig1["ratio_12b"] = V.tps(jul, "A-12B-tp2", 500) / eager_12b
fig1["eager_12b"] = eager_12b

# ---- fig2: what the 26 minutes is, per configuration ----------------------
meta = [r for r in rows if r.get("kind") == "model_meta"]
seen = collections.Counter()
starts = []
for r in meta:
    seen[r["cfg"]] += 1
    starts.append({"cfg": r["cfg"], "nth": seen[r["cfg"]],
                   "init_engine_s": float(r["init_engine_s"]),
                   "model_load_s": float(r["model_load_s"]),
                   "tp": 2 if r["cfg"].endswith("tp2") else 1,
                   "model": ALL_NAMES[r["cfg"]][0],
                   "load_share_pct": float(r["model_load_s"])
                                     / float(r["init_engine_s"]) * 100.0})
fig2 = {"starts": starts,
        "slowest": max(starts, key=lambda s: s["init_engine_s"]),
        "over_20min": [s["cfg"] for s in starts if s["init_engine_s"] > 1200],
        # the only repeat start in the campaign, and it is not one of those two
        "repeat": {"cfg": "A-12B-tp1",
                   "cold": next(s["init_engine_s"] for s in starts
                                if s["cfg"] == "A-12B-tp1" and s["nth"] == 1),
                   "warm": next(s["init_engine_s"] for s in starts
                                if s["cfg"] == "A-12B-tp1" and s["nth"] == 2)}}
fig2["repeat"]["ratio"] = fig2["repeat"]["cold"] / fig2["repeat"]["warm"]
fig2["no_warm_start_for"] = [s["cfg"] for s in starts
                             if s["init_engine_s"] > 1200
                             and sum(1 for x in starts if x["cfg"] == s["cfg"]) == 1]

# ---- fig3: the two qualitative artefacts eager produced -------------------
dec = [r for r in rows if r.get("kind") == "decode" and r["cfg"] == "E-26B-tp2"]
by_depth = collections.defaultdict(list)
for r in dec:
    by_depth[r["target"]].append(r)
pts = []
for t in sorted(by_depth):
    v = by_depth[t]
    p1, p2 = max(x["p1_max"] for x in v), max(x["p2_max"] for x in v)
    pts.append({"ctx": t, "tok_s": sum(x["decode_tps"] for x in v) / len(v),
                "p1": p1, "p2": p2, "runs": len(v),
                "asym_pct": abs(p1 - p2) / max(p1, p2) * 100.0,
                "vram_equal": all(x["v1_g"] == x["v2_g"] for x in v)})
asym = [abs(r["p1_max"] - r["p2_max"]) / max(r["p1_max"], r["p2_max"]) * 100.0
        for r in dec]
fig3 = {"points": pts, "rows": len(dec),
        "worst_asym_pct": max(asym), "median_asym_pct": sorted(asym)[len(asym) // 2],
        "vram_equal_everywhere": all(r["v1_g"] == r["v2_g"] for r in dec),
        "retained_pct": V.retained(jul, "E-26B-tp2"),
        "eager_power": {"w": eager_pw, "reproducible_from_repo": False,
                        "asym_pct": (max(eager_pw) - min(eager_pw)) / max(eager_pw) * 100.0,
                        "source": "docs/benchmarks.md §1"}}

out = {"_what": "Every figure in moe-written-off-by-eager.html. The compiled "
                "side is recomputed from benchmarks/results.jsonl; the eager "
                "side is extracted from docs/benchmarks.md, which is the only "
                "record of it. Derived by site/src/genfig-moe.py.",
       "fig1": fig1, "fig2": fig2, "fig3": fig3}
json.dump(out, open(pathlib.Path(__file__).parent / "figures-moe.json", "w"),
          ensure_ascii=False, indent=1)
print("fig1 ranking:", [(b["model"], round(b["tok_s"], 2)) for b in fig1["bars"]])
print(f'fig1 eager {eager_tps} -> {fig1["ratio"]:.2f}x ; 12B {eager_12b} -> {fig1["ratio_12b"]:.2f}x')
print("fig2 starts:", [(s["cfg"], s["nth"], round(s["init_engine_s"])) for s in starts])
print("fig2 over 20 min:", fig2["over_20min"], "no warm start measured for",
      fig2["no_warm_start_for"])
print(f'fig2 repeat: {fig2["repeat"]["cold"]} -> {fig2["repeat"]["warm"]} '
      f'({fig2["repeat"]["ratio"]:.2f}x)')
print(f'fig3 {fig3["rows"]} rows, worst asym {fig3["worst_asym_pct"]:.2f}%, '
      f'median {fig3["median_asym_pct"]:.2f}%, eager claim '
      f'{fig3["eager_power"]["asym_pct"]:.1f}%')
print(f'fig3 retained {fig3["retained_pct"]:.1f}%')
print("bytes:", len(json.dumps(out)))
