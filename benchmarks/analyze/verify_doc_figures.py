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
import hashlib, json, os, re, sys

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

    # --- 52684 BLOCK_M on A100 (benchmarks/cuda-a100/52684-blockm/) ---------
    # Chain: the two committed probe JSONLs -> the ratios that README quotes.
    # Pass 1 is the 3-arm matrix, pass 2 the numerics and crossover sweep.
    BDIR = os.path.join(HERE, "..", "cuda-a100", "52684-blockm")
    p1 = [json.loads(l) for l in open(os.path.join(BDIR, "pass1-matrix.jsonl"))]
    p2 = [json.loads(l) for l in open(os.path.join(BDIR, "pass2-numerics-crossover.jsonl"))]
    xover = [r for r in p2 if r["kind"] == "crossover"]
    numer = [r for r in p2 if r["kind"] == "numerics"]

    def median(v):
        v = sorted(v)
        n = len(v)
        return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2

    def arm(r, a, field="speedup_vs_base"):
        return r["arms"][a][field]

    ck("52684 README, 140 pass-1 rows", "140", len(p1))
    ck("52684 README, 117 pass-2 rows", "117", len(p2))
    for depth, m_claim, lo_claim, hi_claim in [
        (256, "1.021", "0.981", "1.046"), (512, "0.948", "0.848", "0.991"),
        (1024, "0.987", "0.911", "1.282"), (2048, "1.281", "1.102", "1.819"),
        (4096, "1.394", "1.181", "2.186"), (8192, "1.529", "1.221", "2.080"),
        (16384, "1.597", "1.238", "2.189"),
    ]:
        # q_len=256 is the control: all three arms select the same launch, so
        # its spread is the noise floor, and both tuned arms count toward it.
        s = ([arm(r, "pr") for r in p1 if r["q_len"] == depth]
             + ([arm(r, "bm64") for r in p1 if r["q_len"] == depth] if depth == 256 else []))
        ck(f"52684 README speed table, median {depth}", m_claim, median(s))
        ck(f"52684 README speed table, min {depth}", lo_claim, min(s))
        ck(f"52684 README speed table, max {depth}", hi_claim, max(s))
    ctl = ([arm(r, "pr") for r in p1 if r["q_len"] == 256]
           + [arm(r, "bm64") for r in p1 if r["q_len"] == 256])
    ck("52684 README, control noise band pct", "4.6",
       100 * max(abs(1 - min(ctl)), abs(max(ctl) - 1)))
    ck("52684 README, all 20 rows at 512 lose", "20",
       sum(1 for r in p1 if r["q_len"] == 512 and arm(r, "pr") < 1.0))
    ck("52684 README, every row at 16384 wins", "20",
       sum(1 for r in p1 if r["q_len"] == 16384 and arm(r, "pr") > 1.0))
    ck("52684 README, warp pin is a wash", "1.001",
       median([arm(r, "pr", "median_ms") / arm(r, "bm64", "median_ms")
               for r in p1 if r["q_len"] >= 512]))
    ck("52684 README, bitwise-equal pass-1 rows", "117",
       sum(1 for r in p1 if arm(r, "pr", "bitwise_equal")))
    ck("52684 README, unequal rows all bf16 head_size 64", "1",
       1 if all(r["head_size"] == 64 and r["dtype"] == "bf16"
                for r in p1 if not arm(r, "pr", "bitwise_equal")) else 0)
    for depth, claim in [(512, "0.982"), (640, "0.996"), (768, "0.991"),
                         (896, "0.982"), (1024, "0.991"), (1280, "1.165"),
                         (1536, "1.244"), (1792, "1.363"), (2048, "1.404"),
                         (3072, "1.536")]:
        ck(f"52684 README crossover, median {depth}", claim,
           median([r["speedup"] for r in xover if r["q_len"] == depth]))
    ck("52684 README, ordering drift", "1.000",
       median([r["drift"] for r in xover]))
    ck("52684 README, max ULP distance", "1.0000",
       max(r["max_ulp"] for r in numer))
    ck("52684 README, elements beyond one ULP", "0",
       sum(r.get("n_gt_1ulp", 0) for r in numer))
    ck("52684 README, differing elements", "507",
       sum(r["n_diff"] for r in numer))
    ck("52684 README, elements compared", "550502400",
       sum(r["n_elems"] for r in numer))
    ck("52684 README, differing element pct", "0.0001",
       100 * sum(r["n_diff"] for r in numer) / sum(r["n_elems"] for r in numer))

    # --- ROCm#6565 contrast cell (benchmarks/rccl-6565/) --------------------
    # Chain: the committed stage logs -> the tallies the README quotes. Parsed
    # back out of the logs rather than trusted from results.json, so a stale
    # JSON cannot agree with prose the logs disagree with.
    RDIR = os.path.join(HERE, "..", "rccl-6565")
    arms = []
    for stage in ("stage1", "stage2a"):
        text = open(os.path.join(RDIR, "logs", f"{stage}.log")).read()
        for m in re.finditer(
            r"=== arm=(\S+) RESULT pass=(\d+) fail=(\d+)(?: error=(\d+))? of (\d+)", text
        ):
            arms.append({"arm": m.group(1), "passed": int(m.group(2)),
                         "failed": int(m.group(3)), "error": int(m.group(4) or 0),
                         "n": int(m.group(5))})
    ck("6565 README, arms measured", "8", len(arms))
    ck("6565 README, cold inits total", "135", sum(a["n"] for a in arms))
    ck("6565 README, cold inits correct", "135", sum(a["passed"] for a in arms))
    ck("6565 README, failures", "0", sum(a["failed"] for a in arms))
    ck("6565 README, init errors", "0", sum(a["error"] for a in arms))
    for name, n in [("default", 20), ("p2pdisable", 20), ("prod", 20),
                    ("ch1", 15), ("ch4", 15), ("ch8", 15), ("ch16", 15),
                    ("shmoff", 15)]:
        got = [a for a in arms if a["arm"] == name]
        ck(f"6565 README table, {name} all correct", str(n),
           got[0]["passed"] if len(got) == 1 and got[0]["n"] == n else -1)
    s1 = open(os.path.join(RDIR, "logs", "stage1.log")).read()
    ck("6565 README, RCCL version is 2.30.4", "1",
       1 if "RCCL version : 2.30.4-HEAD:2b22ab0" in s1 else 0)
    ck("6565 README, default channel count is 2", "1",
       1 if "Channel 00/02" in s1 and "Channel 01/02" in s1 else 0)
    ck("6565 README, the reporter's script is verbatim", "1",
       1 if hashlib.md5(
           open(os.path.join(RDIR, "rccl_allgather_truth.py"), "rb").read()
       ).hexdigest() == "bffbc297cad9f1956c8bb2b7e8a4bb0f" else 0)
    env = open(os.path.join(RDIR, "logs", "environment.txt")).read()
    ck("6565 README, zero PCIE-atomic complaints", "0",
       int(re.search(r"PCIE-atomic complaints: (\d+)", env).group(1)))
    ck("6565 README, both GPUs advertise ReqEn+", "2", env.count("AtomicOpsCtl: ReqEn+"))
    # the rccl-tests limitation the README and our comment both state
    raw = open(os.path.join(RDIR, "logs", "stage2b-raw.log")).read()
    ck("6565 README, rccl-tests fails on both invocations", "2",
       raw.count("invalid device function"))
    ck("6565 README, and at the data-init kernel", "2",
       raw.count("common.cu.cpp:650"))
    ck("6565 README, devices really are gfx1100", "1",
       1 if "['gfx1100', 'gfx1100']" in raw else 0)
    ck("6565 README, the deprecated rccl-tests branch", "1",
       1 if "develop_deprecated:40b1b17" in raw else 0)

    # --- vllm#50603 gfx11 gate (benchmarks/vllm-50603/) ---------------------
    # Chain: the three committed probe JSONLs -> the tables that README quotes.
    GDIR = os.path.join(HERE, "..", "vllm-50603")
    g1 = [json.loads(l) for l in open(os.path.join(GDIR, "stage1-rocm-paths.jsonl"))]
    g1b = [json.loads(l) for l in open(os.path.join(GDIR, "stage1b-tail-control.jsonl"))]
    g2 = [json.loads(l) for l in open(os.path.join(GDIR, "stage2-cuda-control.jsonl"))]

    def med(v):
        v = sorted(v)
        n = len(v)
        return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2

    ck("50603 README, stage1 cells", "30", len(g1))
    ck("50603 README, stage2 cells", "30", len(g2))
    ck("50603 README, stage1b cells", "18", len(g1b))
    ck("50603 README, cells over both arches", "90",
       2 * len(g1) + len(g2))
    # the speed table, Triton/CK per shape and length
    for (h, kv), claims in {
        (8, 8): ("2.06", "2.28", "2.87", "2.51", "4.11", "4.83"),
        (8, 4): ("2.31", "2.42", "3.21", "3.86", "6.53", "7.28"),
        (32, 16): ("2.14", "2.24", "1.84", "1.88", "2.71", "2.86"),
        (12, 4): ("2.40", "2.57", "3.20", "3.64", "6.36", "7.23"),
        (16, 4): ("2.40", "2.35", "3.20", "3.71", "6.42", "7.40"),
    }.items():
        for depth, claim in zip((1024, 2048, 4096, 8192, 16384, 32768), claims):
            row = [r for r in g1 if r["num_heads"] == h and r["num_kv_heads"] == kv
                   and r["ctx_len"] == depth][0]
            ck(f"50603 README speed, {h}/{kv} @{depth}", claim,
               row["triton"]["median_ms"] / row["ck"]["median_ms"])
    # the accuracy table
    for depth, t_claim, k_claim, a_claim in [
        (1024, "2.62e-03", "3.56e-03", "2.86e-03"),
        (2048, "2.66e-03", "3.67e-03", "2.86e-03"),
        (4096, "2.94e-03", "3.38e-03", "2.71e-03"),
        (8192, "2.77e-03", "3.78e-03", "3.25e-03"),
        (16384, "2.89e-03", "3.40e-03", "3.10e-03"),
        (32768, "2.80e-03", "3.42e-03", "2.64e-03"),
    ]:
        for arm, claim, data in (("triton", t_claim, g1), ("ck", k_claim, g1),
                                 ("cuda", a_claim, g2)):
            key = "triton" if arm == "cuda" else arm
            v = med([r[key]["max_rel_err"] for r in data if r["ctx_len"] == depth])
            ck(f"50603 README accuracy, {arm} @{depth}", float(claim), v, tol=5e-3)
    allv = ([r[a]["max_rel_err"] for r in g1 for a in ("triton", "ck")]
            + [r["triton"]["max_rel_err"] for r in g2])
    ck("50603 README, error band low", 2.12e-03, min(allv), tol=5e-3)
    ck("50603 README, error band high", 4.72e-03, max(allv), tol=5e-3)
    ck("50603 README, error band median", 3.02e-03, med(allv), tol=5e-3)
    lo = [r[a]["max_rel_err"] for r in g1 for a in ("triton", "ck") if r["ctx_len"] <= 2048]
    lo += [r["triton"]["max_rel_err"] for r in g2 if r["ctx_len"] <= 2048]
    hi = [r[a]["max_rel_err"] for r in g1 for a in ("triton", "ck") if r["ctx_len"] >= 16384]
    hi += [r["triton"]["max_rel_err"] for r in g2 if r["ctx_len"] >= 16384]
    ck("50603 README, 16x context moves the median by", "1.06", med(hi) / med(lo))
    # the positive control
    nan_rows = [r for r in g1b if r["fill"] == "nan"]
    ck("50603 README, nan rows", "6", len(nan_rows))
    ck("50603 README, poisoned rows", "4",
       sum(1 for r in nan_rows if not r["triton"]["all_finite"]))
    ck("50603 README, poisoned rows are exactly the unaligned ones", "1",
       1 if all(not r["block_aligned"] for r in nan_rows if not r["triton"]["all_finite"])
       and all(r["block_aligned"] for r in nan_rows if r["triton"]["all_finite"]) else 0)
    ck("50603 README, CK is poisoned identically", "1",
       1 if all(r["triton"]["all_finite"] == r["ck"]["all_finite"] for r in nan_rows) else 0)
    ck("50603 README, finite garbage stays correct", "1",
       1 if all(r["triton"]["all_finite"] and r["ck"]["all_finite"]
                for r in g1b if r["fill"] == "garbage") else 0)
    ck("50603 README, gqa 1 and 2 are gated out", "0",
       sum(1 for r in g1 if r["gqa_ratio"] <= 2 and r["gate_as_shipped"]))
    ck("50603 README, gqa 3 and 4 are gated in", "12",
       sum(1 for r in g1 if r["gqa_ratio"] >= 3 and r["gate_as_shipped"]))

    # stage 1c: gfx11 runtime evidence for vllm#53856's V-cache padding fix
    g1c = [json.loads(l) for l in open(os.path.join(GDIR, "stage1c-53856-vcache.jsonl"))]
    ck("50603 README, stage1c cells", "16", len(g1c))
    adm = [r for r in g1c if r["gqa_ratio"] == 4]
    ck("50603 README, gqa4 is admitted by the shipped gate", "8",
       sum(1 for r in adm if r["gate_as_shipped"]))
    ck("50603 README, and really ran the CK kernel", "8",
       sum(1 for r in adm if r["as_shipped"]["used_ck_kernel"]))
    poisoned = [r for r in g1c if not r["as_shipped"]["all_finite"]]
    ck("50603 README, 53856 poisoned rows", "4", len(poisoned))
    ck("50603 README, poison is V-side only", "1",
       1 if all(r["poison"] in ("v_only", "both") for r in poisoned) else 0)
    ck("50603 README, K-only never poisons", "1",
       1 if all(r["as_shipped"]["all_finite"] for r in g1c if r["poison"] == "k_only") else 0)
    ck("50603 README, only a straddling final tile poisons", "1",
       1 if all(not r["block_aligned"] for r in poisoned)
       and all(r["as_shipped"]["all_finite"] for r in g1c if r["block_aligned"]) else 0)
    # stage 3: end-to-end effect of widening the gate, on gemma-3-27b
    g3 = [json.loads(l) for l in open(os.path.join(GDIR, "stage3-endtoend.jsonl"))]
    by3 = {(r["arm"], r["ctx"]): r for r in g3}
    # the headline range, which must be over the EXCLUDED ratios only. Stated
    # as 2.06x once, which was the min over gqa=1 alone and understated the
    # floor; the real floor is the 32/16 shape at 4K.
    exc = [r["triton"]["median_ms"] / r["ck"]["median_ms"]
           for r in g1 if r["gqa_ratio"] <= 2]
    ck("50603 README, excluded-ratio speedup floor", "1.84", min(exc))
    ck("50603 README, excluded-ratio speedup ceiling", "7.28", max(exc))

    ck("50603 README, stage3 cells", "6", len(g3))
    for depth, s_claim, w_claim, r_claim in [
        (1024, "41.44", "42.57", "1.027"),
        (8192, "21.36", "23.88", "1.118"),
        (32768, "8.00", "9.55", "1.194"),
    ]:
        st, wd = by3[("stock", depth)], by3[("widened", depth)]
        ck(f"50603 README stage3, stock {depth}", s_claim, st["decode_tok_s"])
        ck(f"50603 README stage3, widened {depth}", w_claim, wd["decode_tok_s"])
        ck(f"50603 README stage3, ratio {depth}", r_claim,
           wd["decode_tok_s"] / st["decode_tok_s"])
        ck(f"50603 README stage3, gate flipped at {depth}", "1",
           1 if (not st["gate_gqa2"]) and wd["gate_gqa2"] else 0)
    # routing, recorded from inside the TP workers
    rdir = os.path.join(GDIR, "logs", "stage3-routes")
    full_stock = full_wide = slide_any_ck = 0
    for depth in (1024, 8192, 32768):
        for arm in ("stock", "widened"):
            for line in open(os.path.join(rdir, f"route-{arm}-{depth}.txt")):
                if ", 0, " in line:          # full-attention layers
                    if arm == "stock" and "False)" in line:
                        full_stock += 1
                    if arm == "widened" and "True)" in line:
                        full_wide += 1
                elif ", 1023, " in line and "True)" in line:
                    slide_any_ck += 1
    ck("50603 README stage3, full-attn layers off CK in stock", "6", full_stock)
    ck("50603 README stage3, full-attn layers on CK when widened", "6", full_wide)
    ck("50603 README stage3, sliding layers never reach CK", "0", slide_any_ck)

    ck("50603 README, Triton poisons on the same rows", "1",
       1 if all((not r["triton_forced"]["all_finite"]) == (not r["as_shipped"]["all_finite"])
                for r in g1c) else 0)

    # --- w4a16 symmetry on gfx1100 (benchmarks/w4a16-symmetry/) -------------
    # The A/B holds the model fixed and changes only the checkpoint, so the
    # figures that matter are the three ratios; the per-arm rates are checked
    # too because the prose quotes them.
    WDIR = os.path.join(HERE, "..", "w4a16-symmetry")
    ab = [json.loads(l) for l in open(os.path.join(WDIR, "w4a16-ab.jsonl"))]
    ck("w4a16 README, A/B cells", "6", len(ab))
    aby = {(r["arm"], r["ctx"]): r for r in ab}
    for depth, a_claim, s_claim, r_claim in (
        (1024, "11.49", "37.24", "3.241"),
        (8192, "7.58", "13.54", "1.786"),
        (32768, "3.35", "4.24", "1.266"),
    ):
        a, s = aby[("asym", depth)], aby[("sym", depth)]
        ck(f"w4a16 README, asym {depth}", a_claim, a["decode_tok_s"])
        ck(f"w4a16 README, sym {depth}", s_claim, s["decode_tok_s"])
        # both components are checked to their own last quoted place just
        # above; the ratio is derived from them, and its third decimal sits a
        # hair inside the place-based bound at 8K (1.7855 against a quoted
        # 1.786), so a relative tolerance is the honest test here.
        ck(f"w4a16 README, ratio {depth}", r_claim,
           s["decode_tok_s"] / a["decode_tok_s"], tol=1e-3)

    # kernel selection, recorded from inside the TP workers in every cell.
    # Two ranks build layers, so each cell must carry two distinct pids.
    NATIVE_SYM = "('RDNA3W4A16LinearKernel', 'uint4b8', 128, False, True)"
    NATIVE_REJECTED = "('RDNA3W4A16LinearKernel', 'uint4', 32, True, False)"
    TRITON_SEL = "('TritonW4A16LinearKernel', 'uint4', 32, True, True)"
    asym_rows = [r for r in ab if r["arm"] == "asym"]
    sym_rows = [r for r in ab if r["arm"] == "sym"]
    ck("w4a16 README, asym cells fall to Triton", "3",
       sum(1 for r in asym_rows
           if TRITON_SEL in r["kernels"] and NATIVE_REJECTED in r["kernels"]))
    ck("w4a16 README, asym never reaches the native kernel", "0",
       sum(1 for r in asym_rows if "uint4b8" in r["kernels"]))
    ck("w4a16 README, sym cells select the native kernel", "3",
       sum(1 for r in sym_rows if NATIVE_SYM in r["kernels"]))
    ck("w4a16 README, sym never falls to Triton", "0",
       sum(1 for r in sym_rows if "TritonW4A16LinearKernel" in r["kernels"]))
    ck("w4a16 README, both ranks recorded in every cell", "6",
       sum(1 for r in ab if len({t.split()[0] for t in r["kernels"].split("|")
                                 if t.strip().startswith("pid=")}) == 2))

    # the quantized-layer census: the third difference, and its size
    cen = json.load(open(os.path.join(WDIR, "ckpt-layer-census.json")))
    ck("w4a16 README, asym quantized linears", "399",
       cen["asym"]["quantized_linear_layers"])
    ck("w4a16 README, sym quantized linears", "400",
       cen["sym"]["quantized_linear_layers"])
    ck("w4a16 README, the layer sets differ by one", "1",
       len(cen["diff"]["only_asym"]) + len(cen["diff"]["only_sym"]))
    ck("w4a16 README, and the extra layer is on the symmetric side", "1",
       1 if cen["diff"]["only_sym"] == [
           "model.language_model.layers.0.linear_attn.out_proj"] else 0)
    ck("w4a16 README, the two arms differ in symmetry", "1",
       1 if (cen["asym"]["symmetric"] is False
             and cen["sym"]["symmetric"] is True) else 0)

    # the 2x2: selection follows the symmetric flag, not the group size
    x2 = json.load(open(os.path.join(WDIR, "w4a16-selection-2x2.json")))
    corner = {r["corner"]: r["chosen"] for r in x2}
    NATIVE = "RDNA3W4A16LinearKernel"
    ck("w4a16 README, symmetric at group 32 still selects native", "1",
       1 if corner.get("sym g32") == NATIVE else 0)
    ck("w4a16 README, asymmetric at group 128 still does not", "0",
       1 if corner.get("asym g128") == NATIVE else 0)
    ck("w4a16 README, selection tracks symmetry not group size", "1",
       1 if (corner.get("sym g32") == NATIVE and corner.get("sym g128") == NATIVE
             and corner.get("asym g32") != NATIVE
             and corner.get("asym g128") != NATIVE) else 0)

    # the campaign's own checkpoints, which is what retires the group-size
    # confound: the fastest model measured here shares the group size with the
    # slowest, so group size cannot be what separates them.
    camp = json.load(open(os.path.join(WDIR, "w4a16-campaign-selection.json")))
    cby = {r["checkpoint"]: r for r in camp}
    ck("w4a16 README, the fastest campaign model is group 32", "32",
       cby["gemma-4-26B-A4B-AWQ"]["group_size"])
    ck("w4a16 README, and so is the asymmetric one", "32",
       cby["Qwen3.8-27B-AWQ-INT4"]["group_size"])
    ck("w4a16 README, every group-32 checkpoint but the AWQ ones is native", "1",
       1 if all(r["chosen"] == NATIVE for r in camp
                if r["group_size"] == 32 and r["symmetric"]) else 0)
    ck("w4a16 README, only the Qwen3.x AWQ checkpoints are asymmetric", "1",
       1 if sorted(r["checkpoint"] for r in camp if not r["symmetric"]) == [
           "Qwen3.6-27B-AWQ-INT4", "Qwen3.8-27B-AWQ-INT4"] else 0)
    ck("w4a16 README, the AWQ name does not imply asymmetric", "1",
       1 if cby["gemma-4-26B-A4B-AWQ"]["symmetric"] else 0)

    # the replacement comparisons that took over from the confounded 3.6x, in
    # README.md, docs/benchmarks.md and docs/architecture-notes.md. Both use
    # only models that are on their best kernel path.
    ck("README/benchmarks/arch-notes, MoE over 8B dense", "1.355",
       tps(aug, "E-26B-tp2", 500) / tps(aug, "B-8B-tp2", 500))
    ck("README/benchmarks/arch-notes, MoE over the larger 31B dense", "2.513",
       tps(aug, "E-26B-tp2", 500) / tps(aug, "C-31B-tp2", 500))
    # the two 27B models in the 08-24 campaign, parameter count held constant
    ck("w4a16 README, two 27B models a factor apart", "3.64",
       tps(aug, "F-27B-tp2", 500) / tps(aug, "D8-27B-tp2", 500))

    # provenance of the symmetric checkpoint against the Hub's own ETags
    sha = json.load(open(os.path.join(WDIR, "ckpt-sha256-sym.json")))
    ck("w4a16 README, sym checkpoint LFS files verified", "3",
       sha["lfs_files_checked"])
    ck("w4a16 README, and all of them match", "3",
       sha["lfs_files_matching"])

    # the measurement is JIT-free: every Triton compile lands in the warm-up
    # generation, not in either timed call. Checked on both arms so that a
    # compile cannot be inflating one side only.
    jit_ok = 0
    for r in ab:
        path = os.path.join(WDIR, "logs", f"ab-{r['arm']}-{r['ctx']}.log")
        gen = 0
        placements = set()
        for line in open(path, errors="replace"):
            # vLLM's progress bars rewrite with \r, so several markers can share
            # one line; count them rather than counting lines.
            if "Triton kernel JIT compilation during inference" in line:
                placements.add(gen)
            gen += line.count("Rendering prompts:   0%")
        if placements and placements == {1}:
            jit_ok += 1
    ck("w4a16 README, no timed call carries a JIT compile", "6", jit_ok)

    # the only repeat measurement in the experiment: the first attempt's sym arm
    # ran at max_num_seqs 256 (its asym cells died there), this run pinned 128.
    first = [json.loads(l) for l in open(os.path.join(
        WDIR, "logs", "w4a16-ab-firstattempt-symonly.jsonl"))]
    fby = {r["ctx"]: r["decode_tok_s"] for r in first}
    worst = max(abs(fby[c] / aby[("sym", c)]["decode_tok_s"] - 1)
                for c in (1024, 8192, 32768))
    ck("w4a16 README, sym arm repeats to within", "1.31", worst * 100)

    # Internal consistency: decode time per token is additive, so if the
    # penalty really is a per-step GEMM it should be a context-INDEPENDENT
    # number of milliseconds even though the ratio falls as attention grows
    # underneath both arms. This is the check that distinguishes "the
    # checkpoint costs a fixed amount per step" from "the arms differ somehow".
    diffs = {c: 1000.0 / aby[("asym", c)]["decode_tok_s"]
                - 1000.0 / aby[("sym", c)]["decode_tok_s"]
             for c in (1024, 8192, 32768)}
    for c in (1024, 8192, 32768):
        ck(f"w4a16 README, asym penalty ms/token @{c}",
           {1024: "60.17", 8192: "58.03", 32768: "62.79"}[c], diffs[c])
    spread = (max(diffs.values()) - min(diffs.values())) / min(diffs.values())
    ck("w4a16 README, that penalty is flat across a 32x context range to within %",
       "8.2", spread * 100)

    # --- forcing the native kernel onto the asymmetric checkpoint -----------
    # One checkpoint, two kernels. The patched arm is expected to FAIL, and the
    # checks below pin down where: selection succeeds, the kernel call does not.
    fr = [json.loads(l) for l in open(os.path.join(WDIR, "w4a16-forced.jsonl"))]
    ck("w4a16 README, forced-run rows (patched never reached a result)", "1", len(fr))
    stock = fr[0]
    ck("w4a16 README, forced stock tok/s", "11.41", stock["decode_tok_s"])
    ck("w4a16 README, forced stock mean logprob", "-0.1859", stock["mean_logprob"])
    ck("w4a16 README, forced stock logprob sample size", "29", stock["n_logprobs"])
    ck("w4a16 README, forced stock reproduces the A/B asym cell to within %",
       "0.7", abs(stock["decode_tok_s"] / aby[("asym", 1024)]["decode_tok_s"] - 1) * 100)
    ck("w4a16 README, forced stock answers correctly", "1",
       1 if ("Paris" in stock["answer"] and "Seine" in stock["answer"]) else 0)

    # the patch does flip selection: can_implement passes on both ranks, and
    # Triton stops appearing. Recorded inside the workers, as everywhere else.
    fk = {arm: open(os.path.join(WDIR, "logs", f"kernels-forced-{arm}.txt")).read()
          for arm in ("stock", "patched")}
    ck("w4a16 README, patched: native kernel accepted on both ranks", "2",
       sum(1 for l in fk["patched"].splitlines()
           if "('RDNA3W4A16LinearKernel', 'uint4', 32, True, True)" in l))
    ck("w4a16 README, patched: Triton no longer appears", "0",
       fk["patched"].count("TritonW4A16LinearKernel"))
    ck("w4a16 README, stock: native rejected, Triton selected", "1",
       1 if ("not supported by" in fk["stock"]
             and "TritonW4A16LinearKernel" in fk["stock"]) else 0)

    # ...and then dies at the kernel entry check, not before
    plog = open(os.path.join(WDIR, "logs", "forced-patched.log"),
                errors="replace").read()
    ck("w4a16 README, patched fails on the group-count check", "1",
       1 if "b_scales must have same group count as qzeros" in plog else 0)
    ck("w4a16 README, and it fails inside gptq_gemm_rdna3", "1",
       1 if "gptq_gemm_rdna3" in plog else 0)

    # why: the checkpoint ships the transpose of what the kernel expects
    zp = json.load(open(os.path.join(WDIR, "zp-layout.json")))["derived"]
    ck("w4a16 README, zp layout: groups", "544", zp["groups"])
    ck("w4a16 README, zp layout: N/8", "640", zp["n_over_8"])
    ck("w4a16 README, zp layout: group_size", "32", zp["group_size"])
    ck("w4a16 README, the checkpoint zp is the transpose of what is expected", "1",
       1 if zp["is_transpose"] else 0)
    ck("w4a16 README, the two numbers the entry check compares differ", "1",
       0 if zp["groups"] == zp["n_over_8"] else 1)

    # the fix is reachable with the layout tool vLLM already has: the permute
    # moves the packed dim to where the kernel wants it, so no repacking.
    perm = open(os.path.join(WDIR, "logs", "check_permute.log"),
                errors="replace").read()
    ck("w4a16 README, the existing permute produces the kernel layout", "1",
       1 if "PERMUTE_OK=True" in perm else 0)
    ck("w4a16 README, and it lands on (groups, N/8)", "1",
       1 if "after permute      : (544, 640)" in perm else 0)

    # --- the three-line fix, and its control ------------------------------
    fx = {r["arm"]: r for r in (json.loads(l) for l in
                                open(os.path.join(WDIR, "w4a16-fix.jsonl")))}
    ck("w4a16 README, fix arms", "2", len(fx))
    ck("w4a16 README, fixed tok/s", "35.50", fx["fixed"]["decode_tok_s"])
    ck("w4a16 README, fixed mean logprob", "-0.1835", fx["fixed"]["mean_logprob"])
    ck("w4a16 README, fixed over stock Triton", "3.11",
       fx["fixed"]["decode_tok_s"] / stock["decode_tok_s"])
    ck("w4a16 README, fixed reaches this % of the symmetric checkpoint", "95.3",
       fx["fixed"]["decode_tok_s"] / aby[("sym", 1024)]["decode_tok_s"] * 100)
    ck("w4a16 README, fixed answers exactly as the Triton control does", "1",
       1 if fx["fixed"]["answer"].strip() == stock["answer"].strip() else 0)

    # the control: same speed, so the native kernel really ran, but the zero
    # points are read one too high and the output collapses
    ck("w4a16 README, layout_only tok/s", "35.45",
       fx["layout_only"]["decode_tok_s"])
    ck("w4a16 README, layout_only mean logprob", "-4.4321",
       fx["layout_only"]["mean_logprob"])
    ck("w4a16 README, layout_only is as fast as fixed to within %", "0.14",
       abs(fx["layout_only"]["decode_tok_s"] / fx["fixed"]["decode_tok_s"] - 1) * 100)
    ck("w4a16 README, but its logprob is this many times worse", "24.2",
       fx["layout_only"]["mean_logprob"] / fx["fixed"]["mean_logprob"])
    ck("w4a16 README, and its answer is not the control's", "0",
       1 if fx["layout_only"]["answer"].strip() == stock["answer"].strip() else 0)
    for arm in ("fixed", "layout_only"):
        k = open(os.path.join(WDIR, "logs", f"kernels-fix-{arm}.txt")).read()
        ck(f"w4a16 README, {arm} ran the native kernel on both ranks", "2",
           sum(1 for l in k.splitlines()
               if "('RDNA3W4A16LinearKernel', 'uint4', 32, True, True)" in l))

    # why "subtract one" would have been wrong, had use_v2_format not existed
    zpv = open(os.path.join(WDIR, "logs", "zp-values.log"), errors="replace").read()
    ck("w4a16 README, the checkpoint does use a zero point of 0", "1",
       1 if "SUBTRACT_ONE_IS_SAFE=False" in zpv else 0)
    ck("w4a16 README, and there are this many of them in the sample", "22",
       int(zpv.split("count_of_zero=")[1].split()[0]))

    wr = open(os.path.join(WDIR, "logs", "wmma-reach.log"), errors="replace").read()
    ck("w4a16 README, every quantised linear qualifies for the WMMA path", "399",
       int(wr.split("WMMA eligible :")[1].split()[0]))
    ck("w4a16 README, that is all of them", "1",
       1 if "WMMA_FRACTION=1.0000" in wr else 0)

    # the upstream test cases, run on gfx1100 before and after the patch
    tr = open(os.path.join(WDIR, "logs", "tests-run.log"), errors="replace").read()
    before, after = tr.split("applying the three-line patch")
    ck("w4a16 README, new numerical cases fail before the patch", "14",
       max(int(n) for n in re.findall(r"(\d+) failed", before)))
    ck("w4a16 README, selection suite after the patch", "12",
       int(re.findall(r"(\d+) passed", after)[0]))
    ck("w4a16 README, numerical suite after the patch", "38",
       int(re.findall(r"(\d+) passed", after)[1]))
    ck("w4a16 README, nothing fails after the patch", "0", after.count("FAILED"))

    # the version qualifier: what was measured is 0.23.x, where the ROCm
    # registry has no Hybrid kernel. Asserted from the committed registry dump
    # rather than from prose, so the qualifier cannot drift from its evidence.
    ksel = json.load(open(os.path.join(WDIR, "w4a16-selection.json")))
    reg = [v["kernel"] for v in ksel[0]["verdicts"]]
    ck("w4a16 README, the measured registry has no Hybrid kernel", "0",
       sum(1 for k in reg if "Hybrid" in k))
    ck("w4a16 README, and it does have RDNA3 and Triton", "2",
       sum(1 for k in reg if k in ("RDNA3W4A16LinearKernel",
                                   "TritonW4A16LinearKernel")))

    # Two scripts read the same safetensors header independently; an audit on
    # 2026-08-27 found wmma_reach.py had K and N swapped relative to
    # zp_layout.py. Cross-check them against each other so the axes cannot
    # drift apart again. weight_scale is what settles which reading is right:
    # it is (N, groups), so groups must equal K/group_size.
    m = re.search(r"\('down_proj', (\d+), (\d+)\)", wr)
    wr_k, wr_n = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    ck("w4a16, wmma_reach and zp_layout agree on K", str(zp["K"]), wr_k)
    ck("w4a16, wmma_reach and zp_layout agree on N", str(zp["N"]), wr_n)
    ck("w4a16, and that reading is the one weight_scale supports", "1",
       1 if zp["K"] // zp["group_size"] == zp["groups"] else 0)

    # a partial download must not be able to report SHA_OK on a smaller set
    ck("w4a16, no checkpoint file advertised by the Hub is missing", "0",
       sha.get("files_missing", -1))

    # Every cell every probe recorded carries both ranks. Before 2026-08-28 a
    # probe whose instrumentation never reached the workers wrote an empty
    # kernels field and exited 0, so this asserts on the committed data what
    # kernel_record.read now enforces at run time.
    all_recorded = ab + list(fx.values()) + [stock] + first
    ck("w4a16, cells with a kernel record", "12", len(all_recorded))
    ck("w4a16, and every one carries both ranks", str(len(all_recorded)),
       sum(1 for r in all_recorded
           if len({t.split()[0] for t in r["kernels"].split("|")
                   if t.strip().startswith("pid=")}) == 2))
    ck("w4a16, no cell has an empty kernel record", "0",
       sum(1 for r in all_recorded if not r["kernels"].strip()))

    # --- 0.27.1: what main actually does with this checkpoint ---------------
    # The comparison that decides whether the three-line patch is worth
    # proposing. It is not: Hybrid wins, and RDNA3 outranks Hybrid, so the
    # patch would hand the work to the slower kernel.
    a27 = {r["arm"]: r for r in (json.loads(l) for l in
                                 open(os.path.join(WDIR, "w4a16-027.jsonl")))}
    ck("w4a16 README, 0.27 arms", "2", len(a27))
    ck("w4a16 README, 0.27 hybrid tok/s", "37.32", a27["hybrid"]["decode_tok_s"])
    ck("w4a16 README, 0.27 patched rdna3 tok/s", "35.22",
       a27["rdna3"]["decode_tok_s"])
    ck("w4a16 README, Hybrid is faster by", "1.0595",
       a27["hybrid"]["decode_tok_s"] / a27["rdna3"]["decode_tok_s"], tol=1e-3)
    ck("w4a16 README, both 0.27 arms answer identically", "1",
       1 if a27["hybrid"]["answer"].strip() == a27["rdna3"]["answer"].strip() else 0)
    ck("w4a16 README, each 0.27 arm ran the kernel it claims", "2",
       sum(1 for r in a27.values() if r["expected_kernel_selected"]))
    ck("w4a16 README, and both arms used the same max_num_seqs", "1",
       1 if len({r["max_num_seqs"] for r in a27.values()}) == 1 else 0)

    # the image was checked before it was trusted: no _rdna suffix, and
    # q_gemm_rdna3.cu only builds when gfx1100 is in VLLM_GPU_ARCHES
    pre = json.load(open(os.path.join(WDIR, "precheck-027.json")))
    ck("w4a16 README, 0.27 image serves gfx1100", "1", 1 if pre["on_gfx1100"] else 0)
    ck("w4a16 README, and carries both kernels' ops", "2",
       int(pre["op:gptq_gemm_rdna3"]) + int(pre["op:wvSplitK_int4_g"]))
    ck("w4a16 README, on 0.27 an AWQ checkpoint lands on Hybrid", "1",
       1 if pre["awq_selected"] == "RDNAHybridW4A16LinearKernel" else 0)
    ck("w4a16 README, and a symmetric one still lands on RDNA3", "1",
       1 if pre["sym_selected"] == "RDNA3W4A16LinearKernel" else 0)

    # --- stage 1d: runtime validation of vllm#53856's fix on gfx1100 --------
    v53 = []
    for arm in ("stock", "patched"):
        v53 += [json.loads(l) for l in
                open(os.path.join(GDIR, f"53856-027-{arm}.jsonl"))]
    ck("50603 README, stage1d rows", "64", len(v53))
    ck("50603 README, stage1d rows per arm", "32",
       sum(1 for r in v53 if r["arm"] == "stock"))
    poisoned = [r for r in v53 if not r["as_shipped"].get("all_finite", True)]
    ck("50603 README, cells poisoned on the stock arm", "4", len(poisoned))
    ck("50603 README, and none on the patched arm", "1",
       1 if all(r["arm"] == "stock" for r in poisoned) else 0)
    ck("50603 README, the fix breaks nothing new", "0",
       sum(1 for r in v53 if r["arm"] == "patched"
           and not r["as_shipped"].get("all_finite", True)))
    # the pattern, which is what identifies the fix as the right shape
    ck("50603 README, stage1d poison is V-side only", "1",
       1 if all(r["poison"] in ("v_only", "both") for r in poisoned) else 0)
    ck("50603 README, only the straddling length", "1",
       1 if all(r["ctx_len"] == 4090 for r in poisoned) else 0)
    ck("50603 README, only the gate-admitted ratio", "1",
       1 if all(r["gqa_ratio"] == 4 for r in poisoned) else 0)
    ck("50603 README, both dtypes reproduce it", "2",
       len({r["dtype"] for r in poisoned}))
    ck("50603 README, stage1d gqa4 cells really ran CK", "32",
       sum(1 for r in v53 if r["gqa_ratio"] == 4
           and r["as_shipped"].get("used_ck_kernel")))

    # forced past the gate: the same fault at gqa=2, and the same fix. This is
    # the bridge to Stage 3 -- widening the gate admits exactly these rows.
    forced_bad = [r for r in v53 if not r["ck_forced"].get("all_finite", True)]
    ck("50603 README, cells poisoned with CK forced", "8", len(forced_bad))
    ck("50603 README, all of them on the stock arm", "1",
       1 if all(r["arm"] == "stock" for r in forced_bad) else 0)
    ck("50603 README, forced CK reaches gqa=2 as well", "2",
       len({r["gqa_ratio"] for r in forced_bad}))
    ck("50603 README, the gqa=2 rows NaN their whole output", "2048",
       max(r["ck_forced"]["n_nan"] for r in forced_bad if r["gqa_ratio"] == 2))
    ck("50603 README, and the patched arm is clean forced too", "0",
       sum(1 for r in v53 if r["arm"] == "patched"
           and not r["ck_forced"].get("all_finite", True)))
    ck("50603 README, the Triton column is clean in all 64", "0",
       sum(1 for r in v53 if not r["triton_forced"].get("all_finite", True)))

    # --- stage 4: both stages re-asked on 0.27 ------------------------------
    # 0.23.1 is a premise, not a constant. Every figure in that section is
    # recomputed here from the 0.27 rows rather than carried over.
    s1a = [json.loads(l) for l in open(os.path.join(GDIR, "stage1-027-r1.jsonl"))]
    s1b = [json.loads(l) for l in open(os.path.join(GDIR, "stage1-027-r2.jsonl"))]
    ck("50603 README, stage4 kernel cells per round", "30", len(s1a))
    ck("50603 README, and the same grid twice", "30", len(s1b))
    spd = lambda r: r["triton"]["median_ms"] / r["ck"]["median_ms"]
    ck("50603 README, stage4 cells where CK is slower", "0",
       sum(1 for r in s1a + s1b if spd(r) < 1.0))
    for gqa, lo, hi in ((1, "2.03", "4.09"), (2, "1.70", "6.05"),
                        (3, "2.20", "6.05"), (4, "2.35", "6.04")):
        band = [spd(r) for r in s1a + s1b if r["gqa_ratio"] == gqa]
        ck(f"50603 README, stage4 gqa={gqa} floor", lo, min(band))
        ck(f"50603 README, stage4 gqa={gqa} ceiling", hi, max(band))
    kk = lambda r: (r["gqa_ratio"], r["num_heads"], r["ctx_len"])
    A1 = {kk(r): r for r in s1a}
    ck("50603 README, stage4 round-to-round spread at most", "5.8",
       100 * max(abs(spd(A1[kk(r)]) - spd(r)) / spd(A1[kk(r)]) for r in s1b),
       tol=0.02)
    g1 = {kk(r): r for r in
          [json.loads(l) for l in open(os.path.join(GDIR,
                                                    "stage1-rocm-paths.jsonl"))]}
    ck("50603 README, stage4 cells bit-identical across versions", "20",
       sum(1 for r in s1a
           if abs(g1[kk(r)]["ck"]["max_rel_err"] - r["ck"]["max_rel_err"]) < 1e-12))
    ck("50603 README, stage4 Triton 32K gqa=2 on 0.23.1", "2.584",
       g1[(2, 32, 32768)]["triton"]["median_ms"])
    ck("50603 README, and on 0.27", "2.231", A1[(2, 32, 32768)]["triton"]["median_ms"])

    # end to end, two passes with the arms in opposite orders
    e3 = {}
    for tag, fn in (("A", "stage3-027.jsonl"), ("B", "stage3-027b.jsonl")):
        for r in [json.loads(l) for l in open(os.path.join(GDIR, fn))]:
            e3[(tag, r["arm"], r["ctx"])] = r["decode_tok_s"]
    ratio = lambda tag, ctx: e3[(tag, "widened", ctx)] / e3[(tag, "stock", ctx)]
    for ctx, a, b, pooled in ((1024, "1.026", "1.023", "1.024"),
                              (8192, "1.065", "1.103", "1.084"),
                              (32768, "1.168", "1.167", "1.167")):
        ck(f"50603 README, stage4 e2e {ctx} run A", a, ratio("A", ctx))
        ck(f"50603 README, stage4 e2e {ctx} run B", b, ratio("B", ctx))
        ck(f"50603 README, stage4 e2e {ctx} pooled", pooled,
           (ratio("A", ctx) + ratio("B", ctx)) / 2)
    reps = sorted(100 * abs(e3[("A", arm, ctx)] - e3[("B", arm, ctx)])
                  / e3[("A", arm, ctx)]
                  for arm in ("stock", "widened") for ctx in (1024, 8192, 32768))
    ck("50603 README, stage4 five of six cells within", "0.7", reps[4], tol=0.15)
    ck("50603 README, and the sixth at", "2.8", reps[5], tol=0.02)

    # the routing proof, which needed its own run on 0.27
    rt = {r["arm"]: r for r in [json.loads(l) for l in
                                open(os.path.join(GDIR, "route-027.jsonl"))]}
    full = lambda arm, flag: [x for x in rt[arm]["routes"]
                              if "(2, 128, 16, 0, %s)" % flag in x]
    slide = lambda arm, flag: [x for x in rt[arm]["routes"]
                               if "(2, 128, 16, 1023, %s)" % flag in x]
    ck("50603 README, stage4 stock leaves full attention on Triton", "2",
       len(full("stock", "False")))
    ck("50603 README, stage4 widened moves it to CK on both ranks", "2",
       len(full("widened", "True")))
    ck("50603 README, stage4 sliding layers stay put in both arms", "4",
       len(slide("stock", "False")) + len(slide("widened", "False")))
    ck("50603 README, stage4 the widened gate really is widened", "1",
       1 if rt["widened"]["gate_gqa2"] and not rt["stock"]["gate_gqa2"] else 0)

    # --- where the patch actually helps: the coverage sweep -----------------
    # The 0.27 pair measures one configuration and the patch loses there. The
    # sweep is what stops that from being read as "the patch is worthless".
    cg = json.load(open(os.path.join(WDIR, "coverage-gap.json")))
    ck("w4a16 README, asymmetric configs swept", "12", len(cg))
    ck("w4a16 README, configs where the patch displaces Hybrid", "3",
       sum(1 for r in cg if r["region"].startswith("overlap")))
    ck("w4a16 README, configs where it replaces Triton", "1",
       sum(1 for r in cg if r["region"].startswith("GAP:")))
    ck("w4a16 README, configs nothing serves today", "8",
       sum(1 for r in cg if r["region"].startswith("GAP+")))
    ck("w4a16 README, every act-order config is unserved today", "1",
       1 if all(not r["hybrid_accepts"] and not r["triton_accepts"]
                for r in cg if r["has_g_idx"]) else 0)
    ck("w4a16 README, and the patch serves all of them", "1",
       1 if all(r["rdna3_patched_accepts"] for r in cg if r["has_g_idx"]) else 0)
    ck("w4a16 README, the measured checkpoint sits in the losing region", "1",
       1 if next(r["region"] for r in cg
                 if r["group_size"] == 32 and not r["has_g_idx"]
                 ).startswith("overlap") else 0)

    # the act-order case, verified numerically rather than by can_implement
    ao = open(os.path.join(WDIR, "logs", "tests-actorder.log"),
              errors="replace").read()
    ck("w4a16 README, act-order fails before the patch", "4",
       int(re.search(r"(\d+) failed", ao).group(1)))
    ck("w4a16 README, act-order passes after it", "4",
       int(re.findall(r"(\d+) passed", ao)[0]))
    ck("w4a16 README, and the full suite has no regression", "54",
       int(re.findall(r"(\d+) passed", ao)[-1]))

    # --- hybrid-decode-on-rdna §6.6: vllm#45916 on 0.27, two passes ---------
    HDIR = os.path.join(HERE, "..", "hybrid-splitkv-027")
    q38 = {}
    for tag, fn in (("A", "qwen38-027-depth.jsonl"), ("B", "qwen38-027-depth-b.jsonl")):
        for r in [json.loads(l) for l in open(os.path.join(HDIR, fn))]:
            q38[(tag, r["arm"], r["ctx"])] = r
    ck("hybrid-decode 6.6, cells across both passes", "12", len(q38))
    q38tps = lambda t, a, c: q38[(t, a, c)]["decode_tok_s"]
    pooled = lambda a, c: (q38tps("A", a, c) + q38tps("B", a, c)) / 2
    for ctx, sA, sB, pA, pB, ratio in (
            (1024, "37.04", "37.76", "51.81", "51.43", "1.38"),
            (8192, "12.57", "12.62", "47.02", "40.14", "3.46"),
            (32768, "3.83", "3.81", "35.20", "37.05", "9.46")):
        ck(f"hybrid-decode 6.6, {ctx} stock A", sA, q38tps("A", "stock", ctx))
        ck(f"hybrid-decode 6.6, {ctx} stock B", sB, q38tps("B", "stock", ctx))
        ck(f"hybrid-decode 6.6, {ctx} splitkv A", pA, q38tps("A", "splitkv", ctx))
        ck(f"hybrid-decode 6.6, {ctx} splitkv B", pB, q38tps("B", "splitkv", ctx))
        ck(f"hybrid-decode 6.6, {ctx} pooled ratio", ratio,
           pooled("splitkv", ctx) / pooled("stock", ctx))
    ck("hybrid-decode 6.6, pooled 1K ms/tok stock", "26.7", 1000 / pooled("stock", 1024))
    ck("hybrid-decode 6.6, pooled 1K ms/tok with the PR", "19.4",
       1000 / pooled("splitkv", 1024))
    ck("hybrid-decode 6.6, pooled 32K ms/tok stock", "261.9",
       1000 / pooled("stock", 32768))
    ck("hybrid-decode 6.6, pooled 32K ms/tok with the PR", "27.7",
       1000 / pooled("splitkv", 32768))
    for arm, claim in (("stock", "7.408"), ("splitkv", "0.262")):
        lo, hi = 1000 / pooled(arm, 1024), 1000 / pooled(arm, 32768)
        ck(f"hybrid-decode 6.6, pooled {arm} slope per 1K ctx", claim,
           (hi - lo) / ((32768 - 1024) / 1000))
    for arm, claim in (("stock", "10.2"), ("splitkv", "70.0")):
        ck(f"hybrid-decode 6.6, pooled {arm} retained at 32K", claim,
           100 * pooled(arm, 32768) / pooled(arm, 1024))
    # the spread between passes, which is the part that bounds how it may be used
    for arm, claims in (("stock", ("1.9", "0.5", "0.5")),
                        ("splitkv", ("0.7", "14.6", "5.3"))):
        for ctx, claim in zip((1024, 8192, 32768), claims):
            ck(f"hybrid-decode 6.6, {arm} A-vs-B spread at {ctx}", claim,
               100 * abs(q38tps("B", arm, ctx) - q38tps("A", arm, ctx)) / q38tps("A", arm, ctx))
    ck("hybrid-decode 6.6, splitkv really ran, both ranks x3 depths x2 passes", "12",
       sum(1 for t in ("A", "B") for c in (1024, 8192, 32768)
           for r in q38[(t, "splitkv", c)]["routes"] if "(256, 0, True)" in r))
    ck("hybrid-decode 6.6, and never on the stock arm", "0",
       sum(len(q38[(t, "stock", c)]["routes"])
           for t in ("A", "B") for c in (1024, 8192, 32768)))
    warm = json.loads(open(os.path.join(HDIR, "qwen38-warmup-discard.jsonl")).read())
    ck("hybrid-decode 6.6, the discarded warm-up cell", "37.0398",
       warm["decode_tok_s"])
    ck("hybrid-decode 6.6, against run A's first measured cell", "37.0397",
       q38tps("A", "stock", 1024))
    prov = json.load(open(os.path.join(HDIR, "provenance.json")))
    ck("hybrid-decode 6.6, the patched file is 1083 lines", "1083",
       prov["cppd_lines_patched"])
    ck("hybrid-decode 6.6, against 493 stock", "493", prov["cppd_lines_stock"])

    # --- harness calibration: are the two decode harnesses the same thing? ---
    CDIR = os.path.join(HERE, "..", "harness-calibration")
    cal = {}
    for i in (1, 2, 3, 4):
        for line in open(os.path.join(CDIR, f"harness-cal-r{i}.jsonl")):
            r = json.loads(line)
            cal[(i, r["campaign_target"])] = r
    ck("harness-cal README, rounds x depths", "12", len(cal))
    camp = {500: (79.45, 79.49), 8000: (73.43, 73.29), 32000: (61.43, 61.34)}
    aug = [json.loads(l) for l in open(os.path.join(HERE, "..",
                                                    "results-2026-08-24.jsonl"))]
    for tgt, pair in camp.items():
        rows = [r["decode_tps"] for r in aug
                if r["kind"] == "decode" and r["cfg"] == "B-8B-tp2"
                and r["target"] == tgt]
        ck(f"harness-cal README, campaign {tgt} is what it says", "1",
           1 if sorted(rows) == sorted(pair) else 0)
    conv = lambda t, k: (cal[(3, t)][k] + cal[(4, t)][k]) / 2
    for tgt, claim64, claim512 in ((500, "-0.44", "-0.97"),
                                   (8000, "-0.01", "-0.27"),
                                   (32000, "-0.07", "-0.19")):
        c = sum(camp[tgt]) / 2
        ck(f"harness-cal README, {tgt} converged vs campaign", claim64,
           100 * (conv(tgt, "tps_64") / c - 1))
        ck(f"harness-cal README, {tgt} at the campaign's span", claim512,
           100 * (conv(tgt, "tps_512") / c - 1))
    for tgt, r1, r2 in ((500, "-30.7", "-5.1"), (8000, "-17.2", "-6.2"),
                        (32000, "-9.2", "0.5")):
        ck(f"harness-cal README, {tgt} first run reads", r1,
           100 * (cal[(1, tgt)]["tps_64"] / conv(tgt, "tps_64") - 1))
        ck(f"harness-cal README, {tgt} second run still reads", r2,
           100 * (cal[(2, tgt)]["tps_64"] / conv(tgt, "tps_64") - 1))
    spread = [100 * abs(cal[(3, t)]["tps_64"] - cal[(4, t)]["tps_64"])
              / cal[(3, t)]["tps_64"] for t in camp]
    ck("harness-cal README, converged run-to-run floor", "0.07", min(spread))
    ck("harness-cal README, converged run-to-run ceiling", "0.36", max(spread))
    ck("harness-cal README, depths are the campaign's own prompt_tokens", "3",
       sum(1 for t in camp if cal[(3, t)]["prompt_tokens_got"]
           == cal[(3, t)]["prompt_tokens_wanted"]))

    # --- the ledger is a projection, so check it still projects --------------
    sys.path.insert(0, HERE)
    import build_ledger
    led = [json.loads(l) for l in open(os.path.join(HERE, "..", "ledger.jsonl"))]
    ck("benchmarks README, ledger rows", "170", len(led))
    ck("benchmarks README, ledger still matches its sources", "1",
       1 if build_ledger.dump(build_ledger.build())
       == open(os.path.join(HERE, "..", "ledger.jsonl")).read() else 0)
    ck("benchmarks README, points that are not chart grade", "2",
       sum(1 for r in led if not r["chart_grade"]))
    ck("benchmarks README, and both are above the 6% cut", "2",
       sum(1 for r in led if not r["chart_grade"] and r["range_pct"] > 6.0))
    ck("benchmarks README, ledger range median", "0.12",
       med([r["range_pct"] for r in led if r["range_pct"] is not None]), tol=0.05)
    # a row must be recomputable from the file it names, not merely plausible
    spot = next(r for r in led if r["cfg"] == "B-8B-tp2" and r["ctx"] == 32000
                and r["date"] == "2026-08-24")
    raw = sorted(r["decode_tps"] for r in
                 [json.loads(l) for l in
                  open(os.path.join(HERE, "..", "results-2026-08-24.jsonl"))]
                 if r.get("kind") == "decode" and r.get("cfg") == "B-8B-tp2"
                 and r.get("target") == 32000)
    ck("benchmarks README, a ledger row recomputes from its source", "1",
       1 if spot["values"] == raw else 0)

    failed = [c for c in checks if not c[0]]
    for ok, where, claim, value, allowed in checks:
        if verbose or not ok:
            print(f"  {'ok  ' if ok else 'FAIL'} {where:<44} prose {claim:>8}   "
                  f"data {value:>9.3f}   allowed +-{allowed:.4g}")
    print(f"\n{len(checks) - len(failed)}/{len(checks)} figures agree with the data files")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
