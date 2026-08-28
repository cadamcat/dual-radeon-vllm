import json, collections, pathlib, sys
R = pathlib.Path(__file__).resolve().parents[2]
# reuse the verifier's own helpers so the slope here is the number it asserts:
# it measures at the prompt lengths the server actually reported and from
# unrounded rates, which differs from the ledger's 2-dp values in the 3rd digit
sys.path.insert(0, str(R / "benchmarks" / "analyze"))
import verify_doc_figures as V
led = [json.loads(l) for l in open(R/"benchmarks/ledger.jsonl")]

def key(r):
    return (r["model"], r["quant"], r["arch"], r["tp"], r["vllm"],
            tuple(r["patches"]), r["harness"], r["date"])
series = collections.defaultdict(list)
for r in led: series[key(r)].append(r)
for v in series.values(): v.sort(key=lambda r: r["ctx"])

def pack(k, rows):
    return {"model": k[0], "quant": k[1], "arch": k[2], "tp": k[3], "vllm": k[4],
            "patches": list(k[5]), "harness": k[6], "date": k[7],
            "points": [{"ctx": r["ctx"], "tok_s": r["decode_tok_s"], "runs": r["runs"],
                        "range_pct": r["range_pct"], "graded": r["chart_grade"]} for r in rows]}

fig1 = [pack(k, v) for k, v in sorted(series.items())
        if k[7] == "2026-07-25" and k[3] == 2 and not k[5]]
fig4 = [pack(k, v) for k, v in sorted(series.items()) if k[7] == "2026-08-28"]

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

out = {
 "_what": "Every figure in hybrid-ssm-collapse.html. Checked against the data "
          "files by benchmarks/analyze/verify_doc_figures.py; edit the data, not this.",
 "fig1": {"caption_source": "benchmarks/ledger.jsonl", "series": fig1},
 "fig2": {"caption_source": "torch profile, raw output not committed",
          "reproducible_from_repo": False,
          "kernels": [
            {"name": "fused_recurrent_gated_delta_rule_packed_decode", "layers": "48 linear", "us_1k": 8.466, "us_32k": 8.038},
            {"name": "_causal_conv1d_update_kernel", "layers": "48 linear", "us_1k": 2.260, "us_32k": 2.139},
            {"name": "triton_w4a16_gemm_kernel", "layers": "all", "us_1k": 275.792, "us_32k": 251.447},
            {"name": "kernel_paged_attention_2d", "layers": "16 full", "us_1k": 356.664, "us_32k": 10095.188}]},
 "fig3": fig3,
 "fig4": {"caption_source": "benchmarks/hybrid-splitkv-027/", "series": fig4},
}
json.dump(out, open(pathlib.Path(__file__).parent / "figures.json", "w"), indent=1)
print(f"fig1 series: {[s['model'] for s in fig1]}")
print(f"fig4 series: {[(s['model'], '+'.join(s['patches']) or 'stock') for s in fig4]}")
print(f"fig3: vllm={fig3['vllm_hybrid_slope_us']}  rocm={lc['rocm']['slope_us']}  vulkan={lc['vulkan']['slope_us']}")
print(f"bytes: {len(json.dumps(out))}")
