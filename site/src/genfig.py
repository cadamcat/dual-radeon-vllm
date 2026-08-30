import json, collections, pathlib, sys
R = pathlib.Path(__file__).resolve().parents[2]
# reuse the verifier's own helpers so the slope here is the number it asserts:
# it measures at the prompt lengths the server actually reported and from
# unrounded rates, which differs from the ledger's 2-dp values in the 3rd digit
sys.path.insert(0, str(R / "benchmarks" / "analyze"))
import verify_doc_figures as V
led = [json.loads(l) for l in open(R/"benchmarks/ledger.jsonl")]

def key(r):
    # `spec` and `attn_backend` are in the key for the same reason they are in
    # gen_best_charts': without them a stock arm and a speculative one measured
    # the same day, on the same stack and the same patch list, are one group.
    # The 2026-08-29 campaign is exactly that shape. Neither figure below
    # selects those rows today, and every older row has both fields null, so
    # this changes no grouping that exists -- it stops one from forming.
    return (r["model"], r["quant"], r["arch"], r["tp"], r["vllm"],
            tuple(r["patches"]), r["harness"], r["date"],
            json.dumps(r.get("spec"), sort_keys=True), r.get("attn_backend"))
series = collections.defaultdict(list)
for r in led: series[key(r)].append(r)
for v in series.values(): v.sort(key=lambda r: r["ctx"])

def pack(k, rows):
    return {"model": k[0], "quant": k[1], "arch": k[2], "tp": k[3], "vllm": k[4],
            "patches": list(k[5]), "harness": k[6], "date": k[7],
            "spec": json.loads(k[8]), "attn_backend": k[9],
            "points": [{"ctx": r["ctx"], "tok_s": r["decode_tok_s"], "runs": r["runs"],
                        "range_pct": r["range_pct"], "graded": r["chart_grade"]} for r in rows]}

fig1 = [pack(k, v) for k, v in sorted(series.items())
        if k[7] == "2026-07-25" and k[3] == 2 and not k[5]]
fig4 = [pack(k, v) for k, v in sorted(series.items()) if k[7] == "2026-08-28"]

# The same hybrid model on the two kernels the engine can serve its 16 full
# attention layers with, measured 2026-08-29 without speculation on either.
# Appended to fig1 rather than given a figure: it is the same quantity on the
# same axis, and the legend already starts a series unlit unless the default
# regex names it -- which it does not, so these are off until asked for.
def _one(cfg, label):
    rows = [r for r in led if r["date"] == "2026-08-29" and r["cfg"] == cfg]
    assert len(rows) == 11, f"{cfg}: {len(rows)} rungs"
    rows.sort(key=lambda r: r["ctx"])
    k = key(rows[0])
    d = pack(k, rows)
    d["label"] = label
    d["campaign"] = True
    d["attn_backend"] = rows[0]["attn_backend"]
    d["retained_pct"] = rows[-1]["decode_tok_s"] / rows[0]["decode_tok_s"] * 100.0
    return d

_bk = [_one("Q38-tp2", "Qwen3.8-27B · ROCM_ATTN"),
       _one("Q38-triton-tp2", "Qwen3.8-27B · TRITON_ATTN")]
_bk_gain = [{"ctx": a["points"][i]["ctx"],
             "pct": (b["points"][i]["tok_s"] / a["points"][i]["tok_s"] - 1) * 100.0}
            for a, b in [(_bk[0], _bk[1])] for i in range(len(a["points"]))]
fig1 = fig1 + _bk

def slope_us(pts):
    """microseconds of decode time added per token of context, over the span."""
    (c0, t0), (c1, t1) = pts[0], pts[-1]
    return (1000.0/t1 - 1000.0/t0) / (c1 - c0) * 1000.0

lc = {}
for be in ("rocm", "vulkan"):
    d = json.load(open(R/f"benchmarks/llamacpp-depth-sweep-{be}.json"))
    pts = [(r["n_depth"], r["avg_ts"]) for r in d]
    lc[be] = {"points": [{"depth": p[0], "tok_s": round(p[1], 2),
                          "stddev_ts": round(r["stddev_ts"], 3)}
                         for p, r in zip(pts, d)],
              "slope_us": round(slope_us(pts), 3)}

jul = V.decode(str(R / "benchmarks/results.jsonl"))
fig3 = {"vllm_hybrid_slope_us": round(V.slope_us(jul, "D-27B-tp2"), 3),
        "vllm_hybrid_retained_pct": round(V.retained(jul, "D-27B-tp2"), 1),
        "llamacpp": lc,
        "dense_band_us": [0.118, 0.339]}

# --- section 6's prefill claim, which had no data behind it -----------------
# "Prefill behaves as the architecture promises: throughput improves with
# length, 805 -> 880 tok/s" was two numbers typed into the prose, from one
# model, on one machine, on one stack. Both numbers are right -- 805.0 and
# 882.6, best of each rung's two rounds, which is how this repository reports
# prefill -- and the generalisation from them is not.
#
# Qwen3.8-27B is the same architecture in the ledger's own `arch` column and it
# declines: -7.5 % on these cards and -8.1 % on an A100. So what §6 measured is
# one hybrid model on one stack, and the contrast it draws with dense models is
# the half that holds -- dense loses 35-43 % on both machines.
#
# Built from prefill.jsonl so the claim is data rather than typing, and so the
# A100 half exists at all: that machine's 2026-08-29 prefill was measured
# through a warm prefix cache and only became usable on 2026-08-30.
sys.path.insert(0, str(R / "benchmarks" / "analyze"))
import build_prefill as _bp
_PF = [json.loads(l) for l in open(R / "benchmarks" / "prefill.jsonl")]
_fits = {(f["machine"], f["cfg"], f["date"]): f for f in _bp.fits(_PF)}

_FIG5 = [("hybrid SSM", "Qwen3.6-27B", "RX 7900 XT", "D-27B-tp2", "2026-07-25"),
         ("hybrid SSM", "Qwen3.8-27B", "RX 7900 XT", "Q38-tp2", "2026-08-29"),
         ("hybrid SSM", "Qwen3.8-27B", "A100-SXM4-80GB", "Q38", "2026-08-30"),
         ("dense", "gemma-4-31B-it", "RX 7900 XT", "G31-tp2", "2026-08-29"),
         ("dense", "gemma-4-31B-it", "A100-SXM4-80GB", "G31", "2026-08-30"),
         ("dense", "gemma-4-12B-it", "A100-SXM4-80GB", "G12", "2026-08-30")]
fig5 = []
for arch, model, machine, cfg, date in _FIG5:
    rs = sorted([r for r in _PF if r["cfg"] == cfg and r["machine"] == machine
                 and r["date"] == date and r["chart_grade"]], key=lambda r: r["ctx"])
    assert rs, (machine, cfg, date)
    f = _fits[(machine, cfg, date)]
    lo, hi = rs[0], rs[-1]
    fig5.append({"arch": arch, "model": model, "machine": machine, "cfg": cfg,
                 "date": date, "attn_backend": rs[0]["attn_backend"],
                 "shallow_ctx": lo["ctx"], "deep_ctx": hi["ctx"],
                 "shallow_tok_s": lo["prefill_tok_s"],
                 "deep_tok_s": hi["prefill_tok_s"],
                 "change_pct": (hi["prefill_tok_s"] / lo["prefill_tok_s"] - 1) * 100.0,
                 "c_ns_tok2": f.get("c_ns_tok2")})
# the one that improves is one of three hybrid lines
_rise = [x for x in fig5 if x["arch"] == "hybrid SSM" and x["change_pct"] > 0]
assert len(_rise) == 1 and _rise[0]["model"] == "Qwen3.6-27B", _rise
assert all(x["change_pct"] < -30 for x in fig5 if x["arch"] == "dense"), fig5

out = {
 "_what": "Every figure in hybrid-ssm-collapse.html. Checked against the data "
          "files by benchmarks/analyze/verify_doc_figures.py; edit the data, not this.",
 "fig1": {"caption_source": "benchmarks/ledger.jsonl", "series": fig1,
          "backends": {"date": "2026-08-29", "spec": None,
                       "why": "the 16 full-attention layers are what falls over "
                              "with depth; these are the two kernels the engine "
                              "can serve them with, same checkpoint, same day",
                       "retained_pct": {b["attn_backend"]: b["retained_pct"] for b in _bk},
                       "gain_pct": _bk_gain,
                       "gain_at_deepest_pct": _bk_gain[-1]["pct"],
                       "worst_range_pct": max(p["range_pct"] for b in _bk
                                              for p in b["points"] if p["range_pct"])}},
 "fig2": {"caption_source": "torch profile, raw output not committed",
          "reproducible_from_repo": False,
          "kernels": [
            {"name": "fused_recurrent_gated_delta_rule_packed_decode", "layers": "48 linear", "us_1k": 8.466, "us_32k": 8.038},
            {"name": "_causal_conv1d_update_kernel", "layers": "48 linear", "us_1k": 2.260, "us_32k": 2.139},
            {"name": "triton_w4a16_gemm_kernel", "layers": "all", "us_1k": 275.792, "us_32k": 251.447},
            {"name": "kernel_paged_attention_2d", "layers": "16 full", "us_1k": 356.664, "us_32k": 10095.188}]},
 "fig3": fig3,
 "fig4": {"caption_source": "benchmarks/hybrid-splitkv-027/", "series": fig4},
 "fig5": {"caption_source": "benchmarks/prefill.jsonl", "series": fig5,
          "hybrid_that_rises": _rise[0]["model"],
          "hybrid_lines": sum(1 for x in fig5 if x["arch"] == "hybrid SSM")},
}
json.dump(out, open(pathlib.Path(__file__).parent / "figures.json", "w"), indent=1)
print(f"fig1 series: {[s['model'] for s in fig1]}")
print(f"fig4 series: {[(s['model'], '+'.join(s['patches']) or 'stock') for s in fig4]}")
print(f"fig3: vllm={fig3['vllm_hybrid_slope_us']}  rocm={lc['rocm']['slope_us']}  vulkan={lc['vulkan']['slope_us']}")
for x in fig5:
    print(f"fig5 {x['arch']:<11} {x['model']:<15} {x['machine']:<15} "
          f"{x['shallow_tok_s']:6.0f} -> {x['deep_tok_s']:6.0f}  {x['change_pct']:+6.1f}%")
print(f"bytes: {len(json.dumps(out))}")
