#!/usr/bin/env python3
"""verify_doc_figures.py — recompute every headline figure the prose quotes.

The documents in this repository make quantitative claims and the point of the
repository is that each one is derivable from a committed data file. This script
is the check. Each entry names where the figure appears, what the prose says, and
how to recompute it from the JSONL. Exit status is non-zero if any disagrees, so
this can gate a commit.

    python3 verify_doc_figures.py            # both campaigns
    python3 verify_doc_figures.py -v         # show every check, not only failures

A figure is checked against half a unit in its own last quoted place: 4.2 admits
anything in [4.15, 4.25], 0.391 admits [0.3905, 0.3915]. That is what quoting a
rounded number means, and it is a tighter test than a percentage for the large
figures and a looser one for the small. Pass an explicit tol= to override.
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
JULY = os.path.join(HERE, "..", "results.jsonl")
AUG = os.path.join(HERE, "..", "results-2026-08-24.jsonl")


def decode(path):
    """per cfg, per nominal target: the decode rates and the prompt lengths the
    server actually reported. Nothing here uses the nominal length for anything
    but keying, which is what benchmarks/README.md promises."""
    d = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("kind") == "decode" and r.get("decode_tps"):
            cell = d.setdefault(r["cfg"], {}).setdefault(r["target"], {"tps": [], "ntok": []})
            cell["tps"].append(r["decode_tps"])
            cell["ntok"].append(r["prompt_tokens"])
    return d


mean = lambda v: sum(v) / len(v)


def tps(d, cfg, ctx):
    return mean(d[cfg][ctx]["tps"])


def ntok(d, cfg, ctx):
    return mean(d[cfg][ctx]["ntok"])


def slope_us(d, cfg):
    """microseconds of decode time added per token of context, measured between
    the shortest and longest rungs at the lengths the server reported, not at the
    nominal ones. The difference is 0.01 microseconds on the steepest curve here,
    which is the third significant figure the prose quotes."""
    lo, hi = tps(d, cfg, 500), tps(d, cfg, 32000)
    span = ntok(d, cfg, 32000) - ntok(d, cfg, 500)
    return (1000 / hi - 1000 / lo) / span * 1000


def retained(d, cfg):
    return tps(d, cfg, 32000) / tps(d, cfg, 500) * 100


def offset(a, b, cfg):
    """mean percentage difference of cfg across every shared context point"""
    ts = [t for t in sorted(b[cfg]) if t in a.get(cfg, {})]
    return sum((tps(b, cfg, t) - tps(a, cfg, t)) / tps(a, cfg, t) * 100 for t in ts) / len(ts)


def main():
    verbose = "-v" in sys.argv
    jul, aug = decode(JULY), decode(AUG)
    checks = []

    def ck(where, claim, value, tol=None):
        """claim is the figure AS WRITTEN, quoted, so that a trailing zero counts.
        0.390 as a float is 0.39 and would be given ten times the slack it should
        have; "0.390" keeps the third place the prose committed to."""
        text = str(claim)
        if tol is None:
            decimals = len(text.split(".")[1]) if "." in text else 0
            allowed = 0.5 * 10 ** -decimals
        else:
            allowed = abs(value) * tol
        num = float(text)
        checks.append((abs(num - value) <= allowed + 1e-12, where, text, value, allowed))

    # --- benchmarks.md, decode table, 2026-07-25 (moved off the README
    # front page on 2026-08-26; the July campaign lives in benchmarks.md) ----
    ck("benchmarks.md decode table, 26B MoE 500", "107.8", tps(jul, "E-26B-tp2", 500))
    ck("benchmarks.md decode table, 26B MoE 32K", "72.8", tps(jul, "E-26B-tp2", 32000))
    ck("benchmarks.md decode table, 8B 500", "79.6", tps(jul, "B-8B-tp2", 500))
    ck("benchmarks.md decode table, 31B 32K", "29.5", tps(jul, "C-31B-tp2", 32000))
    ck("benchmarks.md decode table, 27B SSM 32K", "4.2", tps(jul, "D-27B-tp2", 32000))

    # --- README.md, the single decode table (2026-08-24 campaign) -----------
    for cfg, name, claims in (
        ("E-26B-tp2", "26B MoE", ("107.7", "92.6", "72.9")),
        ("B-8B-tp2", "8B", ("79.5", "73.4", "61.4")),
        ("A-12B-tp2", "12B", ("59.9", "52.0", "41.4")),
        ("C-31B-tp2", "31B", ("42.8", "36.6", "29.3")),
        ("G-30B-tp2", "Muse", ("43.7", "37.8", "37.4")),
        ("D8-27B-tp2", "Qwen3.8", ("12.3", "11.7", "10.7")),
    ):
        for target, claim in zip((500, 8000, 32000), claims):
            ck(f"README decode table 08-24, {name} {target}", claim, tps(aug, cfg, target))
    ck("README gemma-3 note, 500", "44.8", tps(aug, "F-27B-tp2", 500))
    ck("README gemma-3 note, 8K", "34.6", tps(aug, "F-27B-tp2", 8000))
    ck("README gemma-3 note, 32K", "22.1", tps(aug, "F-27B-tp2", 32000))

    # --- benchmarks.md §2, slopes on stock vLLM -----------------------------
    ck("benchmarks.md §2 slope, 8B", "0.118", slope_us(jul, "B-8B-tp2"))
    ck("benchmarks.md §2 slope, 26B", "0.142", slope_us(jul, "E-26B-tp2"))
    ck("benchmarks.md §2 slope, 12B", "0.228", slope_us(jul, "A-12B-tp2"))
    ck("benchmarks.md §2 slope, 31B", "0.339", slope_us(jul, "C-31B-tp2"))
    ck("benchmarks.md §2 slope, 27B SSM", "4.840", slope_us(jul, "D-27B-tp2"))

    # --- benchmarks.md §6, the patched campaign -----------------------------
    ck("benchmarks.md §6 Qwen3.6 retained %", "35.1", retained(jul, "D-27B-tp2"))
    ck("benchmarks.md §6 Qwen3.8 retained %", "86.8", retained(aug, "D8-27B-tp2"))
    ck("benchmarks.md §6 Qwen3.8 slope", "0.390", slope_us(aug, "D8-27B-tp2"))
    ck("benchmarks.md §6 32K speedup",
       "2.51", tps(aug, "D8-27B-tp2", 32000) / tps(jul, "D-27B-tp2", 32000))
    ck("benchmarks.md §6 slope ratio 12.4x",
       "12.4", slope_us(jul, "D-27B-tp2") / slope_us(aug, "D8-27B-tp2"))
    ck("benchmarks.md §6 Muse slope", "0.122", slope_us(aug, "G-30B-tp2"))
    ck("benchmarks.md §6 gemma-3 slope", "0.731", slope_us(aug, "F-27B-tp2"))
    ck("benchmarks.md §6 Muse 500->32K %", "-14.4", (tps(aug, "G-30B-tp2", 32000) / tps(aug, "G-30B-tp2", 500) - 1) * 100)
    ck("benchmarks.md §6 gemma-3 500->32K %", "-50.7", (tps(aug, "F-27B-tp2", 32000) / tps(aug, "F-27B-tp2", 500) - 1) * 100)
    ck("benchmarks.md §6 gemma-3 32K ms/token",
       "45.34", 1000 / tps(aug, "F-27B-tp2", 32000))
    ck("benchmarks.md §6 Muse flat at 2000", "37.99", tps(aug, "G-30B-tp2", 2000))

    # --- benchmarks.md §6, the control offsets ------------------------------
    ck("benchmarks.md §6 control, 8B", "-0.10", offset(jul, aug, "B-8B-tp2"))
    ck("benchmarks.md §6 control, 12B", "-0.02", offset(jul, aug, "A-12B-tp2"))
    ck("benchmarks.md §6 control, 26B", "-0.23", offset(jul, aug, "E-26B-tp2"))
    ck("benchmarks.md §6 control, 31B", "-0.85", offset(jul, aug, "C-31B-tp2"))
    ck("benchmarks.md §6 control, 8B TP=1", "0.01", offset(jul, aug, "B-8B-tp1"))
    ck("benchmarks.md §6 control, 12B TP=1", "-0.82", offset(jul, aug, "A-12B-tp1"))
    # the prose says six of the nine August configurations are July reruns
    ck("benchmarks.md §6 control count", "6", len(set(jul) & set(aug)))
    ck("benchmarks.md §6 new configuration count", "3", len(set(aug) - set(jul)))
    # "steepest curve on the patched machine" — true of August only, and the
    # July hybrid must stay above it or that qualifier is doing no work
    ck("§6 gemma-3 steepest in August", "0.731", max(slope_us(aug, c) for c in aug if 32000 in aug[c]))
    ck("§6 July hybrid still steeper", "4.840", slope_us(jul, "D-27B-tp2"))

    # --- benchmarks.md §6, the gemma-4-31B offset investigation --------------
    off_file = os.path.join(HERE, "..", "gemma-4-31b-campaign-offset.json")
    if os.path.exists(off_file):
        off = json.load(open(off_file))
        rep = off["reproducibility"]
        ck("§6 offset, run1", "-0.85", rep["offset_against_july_pct"]["run1_in_campaign"])
        ck("§6 offset, run2 cold", "-0.90", rep["offset_against_july_pct"]["run2_cold"])
        ck("§6 offset, run3 warm", "-0.79", rep["offset_against_july_pct"]["run3_warm"])
        ck("§6 offset, RSD mean", "0.077", rep["relative_stddev_across_the_three_pct"]["mean"])
        ck("§6 offset, RSD max", "0.146", rep["relative_stddev_across_the_three_pct"]["max"])
        ck("§6 offset, warm minus cold",
           "0.11", off["temperature_is_not_the_cause"]["warm_minus_cold_pct"]["mean"])
        ab = off["atomics_ab_2026_08_26"]
        ck("§6 A/B run4 fresh boot", "-0.26",
           ab["offset_against_july_pct"]["run4_fresh_boot"])
        ck("§6 A/B run5 warm", "-0.78", ab["offset_against_july_pct"]["run5_warm"])
        ck("§6 A/B run5 vs August mean", "0.07", ab["run5_vs_august_mean_pct"])
        ck("§6 A/B run4-run5 gap", "0.52", ab["run4_vs_run5_offset_gap_pct"])

    # --- sliding-window kernel block, read out of the committed trace tables ---
    swin = os.path.join(HERE, "..", "sliding-window-block-skip.json")
    if os.path.exists(swin):
        kl = json.load(open(swin))["kernel_level"]
        def row(side, frag):
            hits = [r for r in kl[side]["kernels"] if frag in r["name"]]
            assert len(hits) == 1, f"{side}/{frag}: {len(hits)} rows"
            return hits[0]
        for side in ("before", "after"):
            step = kl[side]["decode_step_ms"]
            for r in kl[side]["kernels"]:
                ck(f"swin {side} {r['name'][:22]} % of step",
                   str(r["pct_of_decode_step"]), r["decode_only_ms"] / step * 100)
        # the patched side must be measured throughout: that trace has no prefill
        ck("swin patched rows all measured", "1",
           1 if all(r["basis"] == "measured" for r in kl["after"]["kernels"]) else 0)
        ck("swin GEMM is the largest patched kernel", "1",
           1 if kl["after"]["kernels"][0]["name"].startswith("void vllm::gptq_rdna3") else 0)
        ck("swin GEMM 260 calls per step", "260",
           row("after", "gemm_q4_kernel_rdna3")["calls_per_decode_step"])
        # the document quotes the profiler's own Self CUDA column, 44.67. Dividing
        # by the printed 1.575 s gives 44.68 because that denominator is rounded,
        # so this asserts the two agree to a tenth rather than to the last place.
        ck("swin GEMM pct of self cuda", "44.7",
           row("after", "gemm_q4_kernel_rdna3")["decode_only_ms"] / 1575.0 * 100)
        ck("swin attention 10.3x per call", "10.3", 1.589e3 / 154.796)
        ck("swin decode step 2.98x per call", "2.98", 85.567 / 28.668)

    # --- figures the 2026-08-25 review found wrong, so they cannot drift back ---
    swin_f = os.path.join(HERE, "..", "sliding-window-block-skip.json")
    if os.path.exists(swin_f):
        sw = json.load(open(swin_f))
        g3 = [r for r in sw["models_affected"]["gemma-3-27b-it-quantized.w4a16"]["depth_curve_n3"]
              if r["depth"] == 32768][0]
        ck("gemma-3 unpatched 8.05 tok/s", "8.05", 1000 / g3["before_median_ms"])
        ck("gemma-3 patched 22.09 tok/s", "22.09", 1000 / g3["after_median_ms"])
        g4 = [r for r in sw["controls"]["gemma-4-31B-it-qat-w4a16-ct"]["depth_curve_n2"]
              if r["depth"] == 32768][0]
        ck("gemma-4-31B comparator 30.21", "30.21", 1000 / g4["before_median_ms"])
    hmm_f = os.path.join(HERE, "..", "hmm-kernel-three-states.json")
    if os.path.exists(hmm_f):
        st = json.load(open(hmm_f))["states"]
        ck("hmm control r--p 3.2", "3.2", st[2]["r_p_resident"])
        ck("hmm control rw-p 13.1", "13.1", st[2]["rw_p_not_resident"])

    # --- figures the 2026-08-25 re-check corrected, pinned the same way -----
    # benchmarks.md §3 used to derive its T columns from the rounded tok/s in
    # the table above them (1000/46.7 = 21.41); these assert the raw-data
    # derivations that replaced them.
    ck("benchmarks.md §3 T(TP1) 8B", "21.42", 1000 / tps(jul, "B-8B-tp1", 500))
    ck("benchmarks.md §3 T(TP2) 8B", "12.57", 1000 / tps(jul, "B-8B-tp2", 500))
    ck("benchmarks.md §3 T(TP1) 12B", "19.87", 1000 / tps(jul, "A-12B-tp1", 500))
    ck("benchmarks.md §3 12B saved", "3.18",
       1000 / tps(jul, "A-12B-tp1", 500) - 1000 / tps(jul, "A-12B-tp2", 500))
    ck("benchmarks.md §3 12B efficiency %", "60",
       tps(jul, "A-12B-tp2", 500) / tps(jul, "A-12B-tp1", 500) / 2 * 100)
    pre_j = {}
    for line in open(JULY):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("kind") == "prefill":
            pre_j.setdefault(r["cfg"], {}).setdefault(r["target"], []).append(r["prefill_tps"])
    bj = lambda c, t: max(pre_j[c][t])
    ck("§2 27B prefill 500", "805", bj("D-27B-tp2", 500))
    ck("§2 27B prefill 32K", "883", bj("D-27B-tp2", 32000))
    ck("§2 27B prefill rise %", "9.6",
       (bj("D-27B-tp2", 32000) / bj("D-27B-tp2", 500) - 1) * 100)
    ck("§2 MoE prefill rise %", "24",
       (bj("E-26B-tp2", 32000) / bj("E-26B-tp2", 500) - 1) * 100)
    # "measured peak" is the best-of-rounds argmax, which for the 27B is 4000
    # by one tok/s over 6000 - a margin inside the noise, but it is what the
    # best-of rule the table states actually yields.
    ck("§4 27B measured prefill peak", "4000",
       max(pre_j["D-27B-tp2"], key=lambda t: bj("D-27B-tp2", t)))

    # --- README.zh.md is a condensed mirror; pin the cells it adds ----------
    ck("README.zh 8B 32K", "61.4", tps(jul, "B-8B-tp2", 32000))
    ck("README.zh 12B 500", "59.9", tps(jul, "A-12B-tp2", 500))
    ck("README.zh 12B 32K", "41.9", tps(jul, "A-12B-tp2", 32000))
    ck("README.zh 31B 500", "43.2", tps(jul, "C-31B-tp2", 500))
    ck("README.zh 27B 500", "12.1", tps(jul, "D-27B-tp2", 500))
    ck("README.zh TP2 speedup", "1.70",
       tps(jul, "B-8B-tp2", 500) / tps(jul, "B-8B-tp1", 500))
    ck("README.zh 12B speedup", "1.19",
       tps(jul, "A-12B-tp2", 500) / tps(jul, "A-12B-tp1", 500))

    # --- claims README makes about the charts it now shows ------------------
    slopes = sorted(slope_us(aug, c) for c in
                    ("E-26B-tp2", "G-30B-tp2", "B-8B-tp2", "A-12B-tp2",
                     "C-31B-tp2", "D8-27B-tp2"))
    ck("README Qwen3.8 is the steepest plotted", "0.390", slopes[-1])
    ck("README patched dense band top", "0.344", slopes[-2])
    pre_a = {}
    for line in open(AUG):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("kind") == "prefill":
            pre_a.setdefault(r["cfg"], {}).setdefault(r["target"], []).append(r["prefill_tps"])
    best = lambda c, t: max(pre_a[c][t])
    ck("README prefill 8B leads MoE at 500 by 2.1x", "2.1",
       best("B-8B-tp2", 500) / best("E-26B-tp2", 500))
    ck("README prefill MoE passes 8B by 32K", "1",
       1 if best("E-26B-tp2", 32000) > best("B-8B-tp2", 32000) else 0)

    # --- CUDA A100 annex (benchmarks/cuda-a100/) ----------------------------
    # Chain: committed probe output -> matrix JSON -> the ratios the two
    # READMEs quote. The logs are the raw layer; session A survives as
    # notebook-output extracts (see the annex README), session B as full logs.
    annex = json.load(open(os.path.join(HERE, "..", "cuda-a100",
                                        "gemma4-mtp-backend-matrix.json")))
    M = annex["decode_tok_s"]
    tf30, fi30 = M["30000"]["triton_forced"], M["30000"]["flashinfer_explicit"]
    au30 = M["30000"]["auto_selector_47547"]
    tf50, fi50 = M["50000"]["triton_forced"], M["50000"]["flashinfer_explicit"]

    def probe_result(relpath):
        """the RESULT line the probe printed, from the committed log/extract"""
        text = open(os.path.join(HERE, "..", "cuda-a100", "logs", relpath)).read()
        return float(re.search(r"RESULT decode_tok_s=([\d.]+)", text).group(1))

    for relpath, figure in [
        ("session-a/leg1-triton-mtp.txt", tf30["mtp"]),
        ("session-a/leg2-triton-nospec.txt", tf30["nospec"]),
        ("session-a/leg3c-flashinfer-explicit-mtp.txt", fi30["mtp"]),
        ("session-b/leg4c-flashinfer-nospec.log", fi30["nospec"]),
        ("session-b/leg3d-auto-mixed-mtp-coldcache.log", au30["mtp"]),
        ("session-b/leg4d-auto-mixed-nospec-coldcache.log", au30["nospec"]),
        ("session-b/leg5c-triton-mtp-50k.log", tf50["mtp"]),
        ("session-b/leg6c-triton-nospec-50k.log", tf50["nospec"]),
        ("session-b/leg7c-flashinfer-mtp-50k.log", fi50["mtp"]),
        ("session-b/leg8c-flashinfer-nospec-50k.log", fi50["nospec"]),
    ]:
        ck(f"annex JSON vs {os.path.basename(relpath)}", f"{figure:.2f}",
           probe_result(relpath))

    pct = lambda new, base: (new / base - 1) * 100
    ck("annex+main README, MTP delta on default 30K", "-28.2", pct(tf30["mtp"], tf30["nospec"]))
    ck("annex+main README, MTP delta on default 50K", "-61.1", pct(tf50["mtp"], tf50["nospec"]))
    ck("annex README, MTP delta healthy 30K (auto)", "35.0", pct(au30["mtp"], au30["nospec"]))
    ck("annex README, FlashInfer MTP vs off 50K", "-8.7", pct(fi50["mtp"], fi50["nospec"]))
    ck("annex+main README, MTP-off speedup on default 50K", "2.57", tf50["nospec"] / tf50["mtp"])
    ck("annex README, FlashInfer over default MTP 50K", "2.63", fi50["mtp"] / tf50["mtp"])

    RD = annex["readings"]
    ck("annex JSON readings, default delta 30K", str(RD["mtp_delta_on_triton_pct"]["30000"]), pct(tf30["mtp"], tf30["nospec"]))
    ck("annex JSON readings, default delta 50K", str(RD["mtp_delta_on_triton_pct"]["50000"]), pct(tf50["mtp"], tf50["nospec"]))
    ck("annex JSON readings, healthy delta 30K", str(RD["mtp_delta_on_healthy_backend_pct"]["30000_auto"]), pct(au30["mtp"], au30["nospec"]))
    ck("annex JSON readings, healthy delta 50K", str(RD["mtp_delta_on_healthy_backend_pct"]["50000_flashinfer"]), pct(fi50["mtp"], fi50["nospec"]))
    ck("annex JSON readings, MTP-off speedup", str(RD["default_mtp_off_speedup_50k"]), tf50["nospec"] / tf50["mtp"])
    ck("annex JSON readings, FlashInfer speedup", str(RD["flashinfer_over_default_mtp_50k"]), fi50["mtp"] / tf50["mtp"])

    # --- 45450 validation (benchmarks/cuda-a100/45450-validation/) ----------
    # Chain: leg logs -> the README's table, ratios, and the 8/8 identity.
    VDIR = os.path.join(HERE, "..", "cuda-a100", "45450-validation", "logs")

    def leg_result(name):
        text = open(os.path.join(VDIR, name)).read()
        return float(re.search(r"RESULT decode_tok_s=([\d.]+)", text).group(1))

    def leg_ids(name):
        """the two greedy token-id lists an ids-probe log carries"""
        text = open(os.path.join(VDIR, name)).read()
        return [json.loads(m) for m in re.findall(r"^IDS\d (\[.*\])$", text, re.M)]

    def marker_count(name):
        return open(os.path.join(VDIR, name)).read().count("PROBE_3D_SPEC_ACTIVE")

    c30, c50 = leg_result("C30.log"), leg_result("C50.log")
    d30, d50 = leg_result("D30.log"), leg_result("D50.log")
    ck("45450 README, stock 30K", "29.75", c30)
    ck("45450 README, stock 50K", "14.10", c50)
    ck("45450 README, ported 30K", "61.03", d30)
    ck("45450 README, ported 50K", "37.91", d50)
    ck("45450 README, ratio 30K", "2.05", d30 / c30)
    ck("45450 README, ratio 50K", "2.69", d50 / c50)
    for depth, s_claim, p_claim, r_claim in (
        ("1K", "88.67", "110.71", "1.25"),
        ("8K", "52.51", "75.63", "1.44"),
        ("16K", "42.23", "72.13", "1.71"),
    ):
        s = leg_result(f"C{depth}.log")
        p = leg_result(f"D{depth}.log")
        ck(f"45450 README ladder, stock {depth}", s_claim, s)
        ck(f"45450 README ladder, ported {depth}", p_claim, p)
        ck(f"45450 README ladder, ratio {depth}", r_claim, p / s)

    ck("45450 README, vs FlashInfer 30K pct", "2.2", (1 - d30 / fi30["mtp"]) * 100)
    ck("45450 README, vs FlashInfer 50K pct", "8.3", (1 - d50 / fi50["mtp"]) * 100)
    ck("45450 README, cross-VM spread 30K pct", "5.6", (1 - c30 / tf30["mtp"]) * 100)
    ck("45450 README, cross-VM spread 50K pct", "10.5", (1 - c50 / tf50["mtp"]) * 100)

    KDIR = os.path.join(HERE, "..", "cuda-a100", "45450-validation", "logs-ksweep")

    def kleg_result(name):
        text = open(os.path.join(KDIR, name)).read()
        return float(re.search(r"RESULT decode_tok_s=([\d.]+)", text).group(1))

    def kleg_ids(name):
        text = open(os.path.join(KDIR, name)).read()
        return [json.loads(m) for m in re.findall(r"^IDS\d (\[.*\])$", text, re.M)]

    for k, s_claim, p_claim, ratio_claim in (
        (2, "29.02", "56.11", "1.93"),
        (4, "27.54", "53.40", "1.94"),
    ):
        s, p = kleg_result(f"s-k{k}-30k.log"), kleg_result(f"p-k{k}-30k.log")
        ck(f"45450 README k-sweep, stock k={k}", s_claim, s)
        ck(f"45450 README k-sweep, ported k={k}", p_claim, p)
        ck(f"45450 README k-sweep, ratio k={k}", ratio_claim, p / s)
        kid = kleg_ids(f"s-k{k}-ids.log") + kleg_ids(f"p-k{k}-ids.log")
        ck(f"45450 README k-sweep, 4/4 identical k={k}", "1",
           1 if len(kid) == 4 and all(x == kid[0] for x in kid) else 0)

    # ROCm side of the 45450 validation (benchmarks/speculative-decoding/)
    SDIR = os.path.join(HERE, "..", "speculative-decoding")

    def sweep(tag):
        d = json.load(open(os.path.join(SDIR, f"mtp-31b-{tag}.json")))
        return {r["depth"]: r["tok_per_s"] for r in d["rows"]}

    st, pt = sweep("stock45450"), sweep("p45450")
    for depth, s_claim, p_claim in ((1024, "55.84", "74.89"), (8192, "32.76", "63.25"),
                                    (16384, "23.82", "63.09"), (32768, "8.81", "32.57")):
        ck(f"spec-decode doc §5, stock {depth}", s_claim, st[depth])
        ck(f"spec-decode doc §5, ported {depth}", p_claim, pt[depth])
    ck("spec-decode doc §6, 32K ratio", "3.70", pt[32768] / st[32768])
    spec3d = json.load(open(os.path.join(SDIR, "mtp32k-spec3d.json")))
    ck("spec-decode doc §5, rerun vs original pct", "0.5",
       (pt[32768] / spec3d["tok_per_s"] - 1) * 100)
    kc = json.load(open(os.path.join(SDIR, "kcorrect-45450.json")))
    ck("spec-decode doc §5, kcorrect 18 cases", "18", len(kc))
    ck("spec-decode doc §5, kcorrect all deterministic and 3D-written", "1",
       1 if all(r.get("det2") and r.get("det3") and r.get("segm_touched") for r in kc)
       else 0)
    ck("spec-decode doc §5, kcorrect max diff within one bf16 ulp", "1",
       1 if max(r["max_abs_diff"] for r in kc) <= 1e-3 else 0)

    # --- README "Two Radeons against one A100" ------------------------------
    # 3D-path A100-over-Radeons advantage at matched depths, and the other
    # cross-vendor readings the section quotes. The 30K A100 leg is compared
    # against the Radeons' 32K rung, as the prose states.
    ck("README two-vs-one, advantage 1K", "1.48", leg_result("D1K.log") / pt[1024])
    ck("README two-vs-one, advantage 8K", "1.20", leg_result("D8K.log") / pt[8192])
    ck("README two-vs-one, advantage 16K", "1.14", leg_result("D16K.log") / pt[16384])
    ck("README two-vs-one, advantage 30-32K", "1.87", leg_result("D30.log") / pt[32768])
    ck("README two-vs-one, 2D retention Radeons pct", "15.8", st[32768] / st[1024] * 100)
    ck("README two-vs-one, 2D retention A100 pct", "33.6",
       leg_result("C30.log") / leg_result("C1K.log") * 100)
    ck("README two-vs-one, spec gain A100 pct", "39",
       (leg_result("D30.log") / M["30000"]["triton_forced"]["nospec"] - 1) * 100)
    splitkv_stock = json.load(open(os.path.join(SDIR, "splitkv-31b-stock.json")))
    nospec32 = {r["depth"]: r["tok_per_s"] for r in splitkv_stock["rows"]}[32768]
    ck("README two-vs-one, spec gain Radeons pct", "7.5", (pt[32768] / nospec32 - 1) * 100)

    id_lists = sum((leg_ids(f) for f in ("A1.log", "A2.log", "B1.log", "B2.log")), [])
    ck("45450 README, 8/8 generations identical", "1",
       1 if len(id_lists) == 8 and all(x == id_lists[0] for x in id_lists) else 0)
    ck("45450 logs, 3D marker exactly once in every patched leg", "1",
       1 if all(marker_count(f) == 1 for f in ("B1.log", "B2.log", "D30.log", "D50.log"))
       and all(marker_count(f) == 0 for f in ("A1.log", "A2.log", "C30.log", "C50.log"))
       else 0)

    failed = [c for c in checks if not c[0]]
    for ok, where, claim, value, allowed in checks:
        if verbose or not ok:
            print(f"  {'ok  ' if ok else 'FAIL'} {where:<44} prose {claim:>8}   "
                  f"data {value:>9.3f}   allowed +-{allowed:.4g}")
    print(f"\n{len(checks) - len(failed)}/{len(checks)} figures agree with the data files")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
