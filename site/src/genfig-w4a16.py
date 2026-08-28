"""Figures for w4a16-two-problems.html.

Every number is read out of the committed measurement files, never typed.
fig1 and fig2 come from benchmarks/w4a16-symmetry/, fig3 pools that directory's
0.23.1 cells with the ledger's 0.27 rows -- and carries each arm's full stack
identity, because those two are different images and are not an A/B.
"""
import json, pathlib, sys
R = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "benchmarks" / "analyze"))
import verify_doc_figures as V           # noqa: F401  (kept for helper reuse)

W = R / "benchmarks" / "w4a16-symmetry"
jl = lambda p: [json.loads(l) for l in open(p)]

DEPTHS = (1024, 8192, 32768)

# ---- fig1: one model family, two packagings, three depths -----------------
ab = {(r["arm"], r["ctx"]): r for r in jl(W / "w4a16-ab.jsonl")}
ms = lambda arm, c: 1000.0 / ab[(arm, c)]["decode_tok_s"]
cells = []
for c in DEPTHS:
    cells.append({
        "ctx": c,
        "asym_tok_s": ab[("asym", c)]["decode_tok_s"],
        "sym_tok_s": ab[("sym", c)]["decode_tok_s"],
        "asym_ms": ms("asym", c), "sym_ms": ms("sym", c),
        "penalty_ms": ms("asym", c) - ms("sym", c),
        "ratio": ab[("sym", c)]["decode_tok_s"] / ab[("asym", c)]["decode_tok_s"],
        # one container start per cell: no repeat, so no range to quote
        "runs": 1, "range_pct": None, "graded": True,
    })
pen = [d["penalty_ms"] for d in cells]
rat = [d["ratio"] for d in cells]
fig1 = {"cells": cells,
        "penalty_spread_pct": (max(pen) - min(pen)) / min(pen) * 100.0,
        "ratio_decline_pct": (rat[0] - rat[-1]) / rat[0] * 100.0,
        # the only repeat this experiment has, from the first attempt's sym arm
        "sym_repeat_worst_pct": max(
            abs(r["decode_tok_s"] / ab[("sym", r["ctx"])]["decode_tok_s"] - 1) * 100.0
            for r in jl(W / "logs" / "w4a16-ab-firstattempt-symonly.jsonl"))}

# ---- fig2: what decides which kernel runs --------------------------------
x2 = {r["corner"]: r for r in json.load(open(W / "w4a16-selection-2x2.json"))}
fig2 = {"corners": [{"corner": k, "symmetric": k.startswith("sym"),
                     "group_size": int(k.split("g")[1]), "chosen": v["chosen"],
                     "native": v["chosen"] == "RDNA3W4A16LinearKernel"}
                    for k, v in sorted(x2.items())],
        # the campaign's own checkpoints: the fastest model on this box shares
        # its group size with the slowest, so group size is not the axis
        "campaign": [{"checkpoint": r["checkpoint"], "symmetric": r["symmetric"],
                      "group_size": r["group_size"], "chosen": r["chosen"]}
                     for r in sorted(json.load(open(W / "w4a16-campaign-selection.json")),
                                     key=lambda r: r["checkpoint"])],
        "worker_record": {
            "asym": "RDNA3W4A16LinearKernel rejected (uint4) -> TritonW4A16LinearKernel",
            "sym": "RDNA3W4A16LinearKernel selected (uint4b8)"},
        "asym_cells_on_triton": sum(1 for (a, _), r in ab.items()
                                    if a == "asym" and "TritonW4A16LinearKernel" in r["kernels"]),
        "sym_cells_on_native": sum(1 for (a, _), r in ab.items()
                                   if a == "sym" and "uint4b8" in r["kernels"])}

# ---- fig3: two problems, two upstream fixes, neither of them ours ---------
led = jl(R / "benchmarks" / "ledger.jsonl")


def ledger_row(patches, ctx):
    rows = [r for r in led if r["model"] == "Qwen3.8-27B" and r["quant"] == "AWQ int4"
            and r["tp"] == 2 and r["vllm"] == "0.27.1.dev5+gf46a9dfe2"
            and r["patches"] == patches and r["harness"] == "probe-t8t64"
            and r["date"] == "2026-08-28" and r["ctx"] == ctx]
    assert len(rows) == 1, (patches, ctx, len(rows))
    return rows[0]


def pt(r):
    return {"ctx": r["ctx"], "tok_s": r["decode_tok_s"], "runs": r["runs"],
            "range_pct": r["range_pct"], "graded": r["chart_grade"]}


# every axis the ledger varies travels with the series, because two arms that
# share (model, ctx) and differ in patches would otherwise merge silently
IDENT = {"model": "Qwen3.8-27B", "quant": "AWQ int4", "arch": "hybrid SSM", "tp": 2}
fig3 = {"arms": [
    dict(IDENT, id="023", vllm="0.23.1.dev", kernel="TritonW4A16LinearKernel",
         patches=[], max_num_seqs=128, harness="probe-t8t64", date="2026-08-27",
         source="benchmarks/w4a16-symmetry/w4a16-ab.jsonl",
         points=[{"ctx": c, "tok_s": ab[("asym", c)]["decode_tok_s"],
                  "runs": 1, "range_pct": None, "graded": True} for c in DEPTHS]),
    dict(IDENT, id="027", vllm="0.27.1.dev5+gf46a9dfe2",
         kernel="RDNAHybridW4A16LinearKernel", patches=[], max_num_seqs=16,
         harness="probe-t8t64", date="2026-08-28", source="benchmarks/ledger.jsonl",
         points=[pt(ledger_row([], c)) for c in DEPTHS]),
    dict(IDENT, id="027p", vllm="0.27.1.dev5+gf46a9dfe2",
         kernel="RDNAHybridW4A16LinearKernel", patches=["vllm#45916 split-KV"],
         max_num_seqs=16, harness="probe-t8t64", date="2026-08-28",
         source="benchmarks/ledger.jsonl",
         points=[pt(ledger_row(["vllm#45916 split-KV"], c)) for c in DEPTHS])]}
# the cross-campaign control: the w4a16 pair's hybrid arm is a third
# independent container start of the same cell the split-KV campaign measured
h27 = {r["arm"]: r for r in jl(W / "w4a16-027.jsonl")}
fig3["control"] = {
    "w4a16_hybrid_1k": h27["hybrid"]["decode_tok_s"],
    "ledger_stock_1k": ledger_row([], 1024)["decode_tok_s"],
    "apart_pct": abs(h27["hybrid"]["decode_tok_s"] / ledger_row([], 1024)["decode_tok_s"] - 1) * 100.0,
    "forced_stock_1k": next(r for r in jl(W / "w4a16-forced.jsonl"))["decode_tok_s"]}
fig3["gain"] = {c: {"short_fix": fig3["arms"][1]["points"][i]["tok_s"]
                    / fig3["arms"][0]["points"][i]["tok_s"],
                    "long_fix": fig3["arms"][2]["points"][i]["tok_s"]
                    / fig3["arms"][1]["points"][i]["tok_s"]}
               for i, c in enumerate(DEPTHS)}

# ---- fig4: the twelve asymmetric configurations, and what serves them -----
cg = json.load(open(W / "coverage-gap.json"))
REGION = {"overlap": "overlap", "GAP:": "triton", "GAP+": "none"}
fig4 = {"configs": [{"group_size": r["group_size"], "act_order": r["has_g_idx"],
                     "served_by": ("RDNAHybridW4A16LinearKernel" if r["hybrid_accepts"]
                                   else "TritonW4A16LinearKernel" if r["triton_accepts"]
                                   else None),
                     "region": next(v for k, v in REGION.items() if r["region"].startswith(k)),
                     "reason": r["hybrid_reason"]} for r in cg]}
fig4["counts"] = {k: sum(1 for c in fig4["configs"] if c["region"] == k)
                  for k in ("overlap", "triton", "none")}
fig4["act_order_unserved"] = sum(1 for c in fig4["configs"]
                                 if c["act_order"] and c["served_by"] is None)
fig4["act_order_total"] = sum(1 for c in fig4["configs"] if c["act_order"])

out = {"_what": "Every figure in w4a16-two-problems.html. Derived from "
                "benchmarks/w4a16-symmetry/ and benchmarks/ledger.jsonl by "
                "site/src/genfig-w4a16.py; edit the data, not this file.",
       "fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4}
json.dump(out, open(pathlib.Path(__file__).parent / "figures-w4a16.json", "w"),
          ensure_ascii=False, indent=1)
print("fig1 penalty ms:", [round(d["penalty_ms"], 2) for d in cells],
      f'spread {fig1["penalty_spread_pct"]:.1f}%')
print("fig1 ratios    :", [round(d["ratio"], 3) for d in cells],
      f'decline {fig1["ratio_decline_pct"]:.1f}%')
print("fig2 corners   :", {c["corner"]: c["native"] for c in fig2["corners"]})
print("fig3 arms      :", [(a["id"], [round(p["tok_s"], 2) for p in a["points"]])
                           for a in fig3["arms"]])
print("fig3 control   :", f'{fig3["control"]["apart_pct"]:.2f}% apart')
print("fig4 regions   :", fig4["counts"], "act-order unserved",
      f'{fig4["act_order_unserved"]}/{fig4["act_order_total"]}')
print("bytes:", len(json.dumps(out)))
