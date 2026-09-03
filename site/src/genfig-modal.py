"""Figures for mem-busy-orders-five-settings.html.

Seven rented machine configurations in one night, and one counter read on
one card that orders what more bandwidth, more cards and slower memory are
worth in five settings that share no hardware. Everything here is recomputed
from the committed rows -- decode.jsonl for the ladders, the campaign files
for the telemetry, allreduce-2026-09-03/ for the collective -- and nothing is
typed except two list prices, which are marked as such on the figure.
"""
import collections, json, pathlib, statistics
R = pathlib.Path(__file__).resolve().parents[2]
B = R / "benchmarks"


def rows(p):
    return [json.loads(l) for l in open(B / p, encoding="utf-8") if l.strip()]


DEC = rows("decode.jsonl")
PRE = rows("prefill.jsonl")
SWEEP = "2026-09-03"
H100, H200, B300, PRO, H100X2 = ("H100-80GB-HBM3", "H200-143GB-HBM3e", "B300-SXM6",
                                 "RTX-PRO-6000-Blackwell", "H100-80GB-HBM3-x2")
MODELS = collections.OrderedDict([
    ("B8", "Qwen3-8B"), ("G31", "gemma-4-31B-it"), ("Q38", "Qwen3.8-27B"),
    ("G12", "gemma-4-12B-it"), ("MG30", "Muse-Glimmer-30B"), ("G26A4B", "gemma-4-26B-A4B")])


def dec(machine, cfg, ctx, date=SWEEP):
    got = [r for r in DEC if r["machine"] == machine and r["cfg"] == cfg
           and r["ctx"] == ctx and r["date"] == date]
    assert len(got) == 1, (machine, cfg, ctx, date, len(got))
    return got[0]


def ladder(machine, cfg, date=SWEEP):
    return sorted([r for r in DEC if r["machine"] == machine and r["cfg"] == cfg
                   and r["date"] == date], key=lambda r: r["ctx"])


# ---- mem_busy at the 500 rung, read off the raw rows, both cards ----------
def mem_busy(paths, cfg, ctx=500):
    """mean over the rounds of the cell's mem_busy_pct_max -- the counter the
    2026-09-02d finding is about, on the machine the ratio is measured against"""
    v = []
    for p in paths:
        for r in rows(p):
            if r.get("kind") == "decode" and r["cfg"] == cfg and r["target"] == ctx:
                v.append(r["mem_busy_pct_max"])
    assert len(v) == 2, (paths, cfg, ctx, v)
    return statistics.mean(v)


H100_RAW = ["cuda-h100/campaign-2026-09-03/results.jsonl",
            "cuda-h100/campaign-2026-09-03/results-q38.jsonl",
            "cuda-h100/campaign-2026-09-03b/results.jsonl"]
H200_RAW = ["cuda-h200/campaign-2026-09-03/results.jsonl"]
RAD_RAW = ["campaign-2026-09-02d/results.jsonl"]
mb_h100 = {c: mem_busy(H100_RAW, c) for c in MODELS}
mb_h200 = {c: mem_busy(H200_RAW, c) for c in MODELS}
assert all(mb_h200[c] < mb_h100[c] for c in MODELS), "faster memory should idle the controller more"

# ---- fig1: five settings, ordered by one counter ---------------------------
# The two ends of the H100's mem_busy range are Qwen3-8B (bf16) and the MoE.
# Each setting divides one configuration by another and the model is held
# fixed, so a ratio is what the setting is worth to that model.
END_HI, END_LO = "B8", "G26A4B"
assert max(mb_h100, key=mb_h100.get) == END_HI and min(mb_h100, key=mb_h100.get) == END_LO


def ratio(num_m, den_m, cfg, ctx=500):
    return dec(num_m, cfg, ctx)["decode_tok_s"] / dec(den_m, cfg, ctx)["decode_tok_s"]


# The pair's own setting: TP=2 against TP=1, 2026-09-02d, whose two models are
# the same bf16 8B and the w4a16 12B -- the MoE has no TP=1 row on this box
# past 12 000 and none in that campaign. Its mem_busy is its own single card's.
rad = {}
for c1, c2, name in (("B8-tp1-p45450", "B8-tp2-p45450", "Qwen3-8B"),
                     ("A12-tp1-p45450", "A12-tp2-p45450", "gemma-4-12B-it")):
    one, two = (dec("RX 7900 XT", c1, 500, "2026-09-02"), dec("RX 7900 XT", c2, 500, "2026-09-02"))
    rad[name] = {"ratio": two["decode_tok_s"] / one["decode_tok_s"],
                 "mem_busy": mem_busy(RAD_RAW, c1), "one_cfg": c1, "two_cfg": c2}
SETTINGS = [
    {"id": "radeon2", "what": "second RX 7900 XT", "kind": "second card, no P2P",
     "date": "2026-09-02", "base": "one RX 7900 XT",
     "models": [{"model": "Qwen3-8B", "cfg": "B8", "mem_busy": rad["Qwen3-8B"]["mem_busy"],
                 "ratio": rad["Qwen3-8B"]["ratio"], "ctx": 500},
                {"model": "gemma-4-12B-it", "cfg": "G12",
                 "mem_busy": rad["gemma-4-12B-it"]["mem_busy"],
                 "ratio": rad["gemma-4-12B-it"]["ratio"], "ctx": 500}]},
]
for sid, what, kind, num in (("h200", "H200 against H100", "faster memory, same compute", H200),
                             ("b300", "B300 against H100", "a newer card", B300),
                             ("pro6000", "RTX PRO 6000 against H100", "slower memory", PRO),
                             ("h100x2", "second H100", "second card, NVLink", H100X2)):
    ms = []
    for cfg in (END_HI, END_LO):
        ms.append({"model": MODELS[cfg], "cfg": cfg, "mem_busy": mb_h100[cfg],
                   "ratio": ratio(num, H100, cfg), "ctx": 500})
    SETTINGS.append({"id": sid, "what": what, "kind": kind, "date": SWEEP,
                     "base": "one H100", "models": ms})
# the ordering claim: in every setting the memory-bound end moves further from
# 1.0 than the compute-bound end, in whichever direction the setting moves
for s in SETTINGS:
    hi, lo = s["models"][0], s["models"][1]
    assert hi["mem_busy"] > lo["mem_busy"], s["id"]
    s["ordered"] = abs(hi["ratio"] - 1) > abs(lo["ratio"] - 1)
    s["direction"] = "faster" if hi["ratio"] > 1 else "slower"
assert all(s["ordered"] for s in SETTINGS), [s["id"] for s in SETTINGS if not s["ordered"]]

# ---- fig2: the prediction, committed before the data ------------------------
mach = {m["gpu_arg"]: m for m in rows("modal-2026-09-02/machines.jsonl")}
mclk = lambda k: float(mach[k]["clocks.max.memory"].split()[0])
R_CLK = mclk("H200") / mclk("H100")
assert abs(R_CLK - 1.222) < 0.001, R_CLK


def predicted(f):
    """the form PREDICTION.md applied, to the whole-percent mem_busy it applied
    it to -- the H100 README's table -- so what is recomputed is the prediction
    as it was committed, not a sharper one made afterwards"""
    f = round(f) / 100.0
    return 1.0 / ((1 - f) + f / R_CLK)


def first_agreeing_rung(cfg, tol_pct=1.0):
    """the shallowest rung whose two rounds agree within tol on BOTH machines,
    the rule PREDICTION.md fixed before looking"""
    for ctx in sorted({r["ctx"] for r in ladder(H100, cfg)}):
        a, b = dec(H100, cfg, ctx), dec(H200, cfg, ctx)
        if a["range_pct"] <= tol_pct and b["range_pct"] <= tol_pct:
            return ctx
    raise AssertionError(cfg)


pred = []
for cfg in sorted(MODELS, key=lambda c: -mb_h100[c]):
    ctx = first_agreeing_rung(cfg)
    pred.append({"cfg": cfg, "model": MODELS[cfg], "mem_busy_h100": mb_h100[cfg],
                 "mem_busy_h200": mb_h200[cfg], "predicted": predicted(mb_h100[cfg]),
                 "measured": ratio(H200, H100, cfg, ctx), "ctx": ctx})
spread_pred = pred[0]["predicted"] - pred[-1]["predicted"]
spread_meas = pred[0]["measured"] - pred[-1]["measured"]
ends_ordered = (max(pred, key=lambda p: p["measured"])["cfg"] == END_HI
                and min(pred, key=lambda p: p["measured"])["cfg"] == END_LO)
above_r = [p["cfg"] for p in pred if p["measured"] > R_CLK]
fig2 = {"r_clock": R_CLK, "mclk_h100": mclk("H100"), "mclk_h200": mclk("H200"),
        "rows": pred, "spread_predicted": spread_pred, "spread_measured": spread_meas,
        "ends_ordered": ends_ordered, "exceed_r": above_r,
        "mem_busy_falls_on_every_model": all(mb_h200[c] < mb_h100[c] for c in MODELS),
        "prediction_file": "benchmarks/cuda-h200/campaign-2026-09-03/PREDICTION.md"}
assert ends_ordered and above_r == ["B8"], (ends_ordered, above_r)

# ---- fig3: the collective, both ends -----------------------------------------
AR = B / "allreduce-2026-09-03"
# which configurations have a point-to-point link is typed here, and it is the
# one thing on this figure the rows do not carry: nvidia-smi's nvlink status
# was read at the time and recorded in the README, not in a row
COLL = [("H100-80GB-HBM3-x2-results.jsonl", "H100 ×2", "NVLink"),
        ("H100-80GB-HBM3-x4-results.jsonl", "H100 ×4", "NVLink"),
        ("H200-143GB-HBM3e-x4-results.jsonl", "H200 ×4", "NVLink"),
        ("B300-SXM6-x2-results.jsonl", "B300 ×2", "NVLink"),
        ("A100-SXM4-80GB-x2-results.jsonl", "A100 ×2", "NVLink"),
        ("RTX-PRO-6000-Blackwell-x2-results.jsonl", "RTX PRO 6000 ×2", "PCIe, no NVLink"),
        ("RTX-PRO-6000-Blackwell-x4-results.jsonl", "RTX PRO 6000 ×4", "PCIe, no NVLink")]


def ar_cell(path, ntok, hidden=4096):
    for r in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip()):
        if r.get("kind") == "allreduce" and r["hidden"] == hidden and r["ntok"] == ntok:
            return r["t_graph_us"]
    raise AssertionError((path, ntok))


coll = []
for fn, label, link in COLL:
    coll.append({"label": label, "link": link, "cards": 4 if "x4" in fn else 2,
                 "n1_us": ar_cell(AR / fn, 1), "n16384_us": ar_cell(AR / fn, 16384),
                 "source": f"benchmarks/allreduce-2026-09-03/{fn}"})
RADAR = B / "allreduce-2026-09-02" / "results.jsonl"
coll.append({"label": "RX 7900 XT ×2", "link": "PCIe 3.0, no P2P", "cards": 2,
             "n1_us": ar_cell(RADAR, 1), "n16384_us": ar_cell(RADAR, 16384),
             "source": "benchmarks/allreduce-2026-09-02/results.jsonl"})
for c in coll:
    c["ratio"] = c["n16384_us"] / c["n1_us"]
n1 = [c["n1_us"] for c in coll]
nB = [c["n16384_us"] for c in coll]
pairs1 = [c["n1_us"] for c in coll if c["cards"] == 2]
by_label = {c["label"]: c for c in coll}
fig3 = {"rows": coll, "hidden": 4096,
        "bandwidth_range": max(nB) / min(nB), "latency_range": max(n1) / min(n1),
        "latency_range_pairs": max(pairs1) / min(pairs1),
        "fourth_card": [
            {"family": "H100", "link": "NVLink",
             "n1": by_label["H100 ×4"]["n1_us"] / by_label["H100 ×2"]["n1_us"],
             "n16384": by_label["H100 ×4"]["n16384_us"] / by_label["H100 ×2"]["n16384_us"]},
            {"family": "RTX PRO 6000", "link": "PCIe",
             "n1": by_label["RTX PRO 6000 ×4"]["n1_us"] / by_label["RTX PRO 6000 ×2"]["n1_us"],
             "n16384": by_label["RTX PRO 6000 ×4"]["n16384_us"]
                       / by_label["RTX PRO 6000 ×2"]["n16384_us"]}],
        "h200_vs_h100_x4": {"n1": by_label["H200 ×4"]["n1_us"] / by_label["H100 ×4"]["n1_us"],
                            "n16384": by_label["H200 ×4"]["n16384_us"]
                                      / by_label["H100 ×4"]["n16384_us"]}}

# ---- fig4: three backends nobody asked for ------------------------------------
SINGLE = [("H100", H100), ("H200", H200), ("B300", B300), ("RTX PRO 6000", PRO)]
backends = []
for cfg in MODELS:
    row = {"cfg": cfg, "model": MODELS[cfg], "cells": []}
    for short, m in SINGLE:
        rs = ladder(m, cfg)
        assert rs, (m, cfg)
        pr = [r for r in PRE if r["machine"] == m and r["cfg"] == cfg and r["date"] == SWEEP]
        # decode.jsonl carries the arm table's backend and prefill.jsonl the
        # log's; the log is the authority, so that is what is drawn
        be = {r["attn_backend"] for r in pr}
        assert len(be) == 1, (m, cfg, be)
        qk = {(r.get("route") or {}).get("quant_kernel") for r in rs}
        assert len(qk) == 1, (m, cfg, qk)
        row["cells"].append({"machine": short, "attn_backend": be.pop(),
                             "quant_kernel": qk.pop()})
    backends.append(row)
distinct = sorted({c["attn_backend"] for r in backends for c in r["cells"]})
assert distinct == ["FLASHINFER", "FLASH_ATTN", "TRITON_ATTN"], distinct
# within one card the split is clean: the PRO 6000's three Triton arms fall
# at depth and its two FlashAttention arms hold
pro = []
for cfg in ("G12", "G26A4B", "G31", "Q38", "MG30"):
    rs = ladder(PRO, cfg)
    be = next(r["attn_backend"] for r in PRE if r["machine"] == PRO and r["cfg"] == cfg)
    pro.append({"cfg": cfg, "model": MODELS[cfg], "attn_backend": be,
                "change_pct": (rs[-1]["decode_tok_s"] / rs[0]["decode_tok_s"] - 1) * 100,
                "from_ctx": rs[0]["ctx"], "to_ctx": rs[-1]["ctx"]})
tri = [p["change_pct"] for p in pro if p["attn_backend"] == "TRITON_ATTN"]
fla = [p["change_pct"] for p in pro if p["attn_backend"] == "FLASH_ATTN"]
assert max(tri) < min(fla), (tri, fla)
fig4 = {"machines": [s for s, _ in SINGLE], "rows": backends, "distinct": distinct,
        "pro6000": pro, "pro6000_split": {"triton_worst": min(tri), "triton_best": max(tri),
                                          "flash_worst": min(fla), "flash_best": max(fla)}}

# ---- fig5: past 32 000, what makes a curve flat --------------------------------
STRUCT = {"MG30": "attention through a 2 048-token window",
          "G12": "sliding 1 024, one global layer in six",
          "G26A4B": "the same, mixture of experts",
          "Q38": "hybrid SSM: recurrent state plus attention layers",
          "G31": "sliding 1 024, one global layer in six"}
deep = []
for cfg in ("MG30", "G12", "G26A4B", "Q38", "G31"):
    rs = ladder(H100, cfg)
    assert rs[0]["ctx"] == 500 and rs[-1]["ctx"] == 128000, cfg
    a100 = dec("A100-SXM4-80GB", cfg if cfg != "G12" else "G12", 32000, "2026-08-30")
    deep.append({"cfg": cfg, "model": MODELS[cfg], "structure": STRUCT[cfg],
                 "change_pct": (rs[-1]["decode_tok_s"] / rs[0]["decode_tok_s"] - 1) * 100,
                 "a100_to_h100_at_32k": dec(H100, cfg, 32000)["decode_tok_s"]
                                        / a100["decode_tok_s"],
                 "points": [{"ctx": r["ctx"], "tok_s": r["decode_tok_s"]} for r in rs]})
deep.sort(key=lambda d: -d["change_pct"])
assert deep[0]["cfg"] == "MG30", deep[0]
fig5 = {"machine": "H100", "rows": deep,
        "prediction": "Muse-Glimmer was written down, before the run, as landing between "
                      "the attention models and the SSM; it landed flatter than both",
        "prediction_file": "benchmarks/cuda-h100/campaign-2026-09-03b/run.py",
        "ssm_falls_as_far_as_dense": abs(next(d["change_pct"] for d in deep if d["cfg"] == "Q38")
                                         - next(d["change_pct"] for d in deep if d["cfg"] == "G31"))
                                     < 1.0}

# ---- fig6: the controls, and the price ------------------------------------------
ctrl = []
for ctx in (500, 8000, 16000, 32000):
    modal = dec("A100-SXM4-80GB", "G12", ctx)
    colab = {d: dec("A100-SXM4-80GB", "G12" if d == "2026-08-30" else "A100-G12", ctx, d)
             for d in ("2026-08-29", "2026-08-30")}
    ctrl.append({"ctx": ctx, "modal": modal["decode_tok_s"],
                 "colab_0829": colab["2026-08-29"]["decode_tok_s"],
                 "colab_0830": colab["2026-08-30"]["decode_tok_s"],
                 "delta_pct": (modal["decode_tok_s"] / colab["2026-08-30"]["decode_tok_s"] - 1) * 100})
l4 = {"modal": dec("L4", "G12", 32000)["decode_tok_s"],
      "colab": [dec("L4", "G12", 32000, d)["decode_tok_s"] for d in ("2026-08-30", "2026-09-02")]}
# Modal's list prices on 2026-09-03, per card-hour, as cuda-modal/README.md
# records them. Typed: a price list is not a measurement and is marked so.
PRICE = {"H100": 3.95, "B300": 7.10}
price = []
for cfg in ("B8", "G12", "G26A4B"):
    h, b = dec(H100, cfg, 500)["decode_tok_s"], dec(B300, cfg, 500)["decode_tok_s"]
    price.append({"cfg": cfg, "model": MODELS[cfg], "mem_busy_h100": mb_h100[cfg],
                  "h100": h, "b300": b, "ratio": b / h,
                  "per_dollar": (b / PRICE["B300"]) / (h / PRICE["H100"])})
fig6 = {"a100": ctrl, "worst_delta_pct": max(abs(c["delta_pct"]) for c in ctrl),
        "l4": l4, "price": price, "price_usd_per_card_hour": PRICE,
        "price_source": "Modal list prices, 2026-09-03, typed"}
assert fig6["worst_delta_pct"] < 0.1, fig6["worst_delta_pct"]

out = {
 "_what": "Every figure in mem-busy-orders-five-settings.html. Recomputed from "
          "benchmarks/decode.jsonl, benchmarks/prefill.jsonl, the campaign files' "
          "telemetry and benchmarks/allreduce-2026-09-03/ by site/src/genfig-modal.py; "
          "edit the data, not this. The only typed numbers are two list prices.",
 "fig1": {"settings": SETTINGS, "ends": {"hi": MODELS[END_HI], "lo": MODELS[END_LO]},
          "mem_busy_h100": {MODELS[c]: mb_h100[c] for c in MODELS}},
 "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5, "fig6": fig6,
}
json.dump(out, open(pathlib.Path(__file__).parent / "figures-modal.json", "w"),
          ensure_ascii=False, indent=1)
for s in SETTINGS:
    print(f'fig1 {s["id"]:8s} ' + "  ".join(f'{m["model"]:16s} {m["mem_busy"]:5.1f}% {m["ratio"]:.3f}x'
                                          for m in s["models"]))
print("fig2 r", round(R_CLK, 3), "spread pred", round(spread_pred, 3), "meas", round(spread_meas, 3),
      [(p["cfg"], p["ctx"], round(p["predicted"], 3), round(p["measured"], 3)) for p in pred])
print("fig3 bw range", round(fig3["bandwidth_range"], 1), "lat", round(fig3["latency_range"], 2),
      "pairs", round(fig3["latency_range_pairs"], 2),
      [(f["family"], round(f["n1"], 2), round(f["n16384"], 2)) for f in fig3["fourth_card"]])
print("fig4", distinct, [(p["cfg"], p["attn_backend"], round(p["change_pct"], 1)) for p in pro])
print("fig5", [(d["cfg"], round(d["change_pct"], 1), round(d["a100_to_h100_at_32k"], 2)) for d in deep])
print("fig6 worst", round(fig6["worst_delta_pct"], 3), [(p["cfg"], round(p["ratio"], 3), round(p["per_dollar"], 2)) for p in price])
print("bytes", len(json.dumps(out)))
