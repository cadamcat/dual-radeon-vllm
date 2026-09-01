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
import glob
import hashlib, json, os, re, statistics, sys

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
    ROOT = os.path.join(HERE, "..", "..")
    rm = open(os.path.join(ROOT, "README.md")).read()
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

    # 2026-08-31: the README used to give the peak positions with no campaign
    # and point at S* as the derivation behind them. S* was withdrawn on
    # 2026-08-30, and the positions themselves do not hold still -- the MoE's
    # own configuration peaks at 6 K in July and 4 K in August. These pin the
    # two numbers the README now uses to say so.
    pre_a = {}
    for line in open(AUG):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("kind") == "prefill":
            pre_a.setdefault(r["cfg"], {}).setdefault(r["target"], []).append(
                r["prefill_tps"])
    ba = lambda c, t: max(pre_a[c][t])
    # read out of the sentence that makes the claim, not merely recomputed: the
    # positions are the point of the paragraph, and "4 K" and "6 K" are one
    # transposition apart -- which is what the July and August rows actually are
    _pk = re.search(r"it is 2 K for four of the six,\s*([0-9]+) K for the MoE and "
                    r"([0-9]+) K for the hybrid-SSM; the same MoE configuration "
                    r"measured\s*in July peaks at ([0-9]+) K instead", rm)
    ck("README prefill peaks, the paragraph states three positions", "3",
       len(_pk.groups()) if _pk else 0)
    _pk = _pk or re.match(r"()()()", "")
    ck("README prefill peaks, the MoE in August", (_pk.group(1) or "0") + "000",
       max(pre_a["E-26B-tp2"], key=lambda t: ba("E-26B-tp2", t)))
    ck("README prefill peaks, the hybrid-SSM in August", (_pk.group(2) or "0") + "000",
       max(pre_a["D8-27B-tp2"], key=lambda t: ba("D8-27B-tp2", t)))
    ck("README prefill peaks, and the MoE arm in July", (_pk.group(3) or "0") + "000",
       max(pre_j["E-26B-tp2"], key=lambda t: bj("E-26B-tp2", t)))
    ck("README prefill peaks, and four of the six sit at 2 K", "4",
       sum(1 for c in ("A-12B-tp2", "B-8B-tp2", "C-31B-tp2", "G-30B-tp2")
           if max(pre_a[c], key=lambda t: ba(c, t)) == 2000))
    ck("README prefill, the 8B over the MoE at 500", "2.1",
       ba("B-8B-tp2", 500) / ba("E-26B-tp2", 500))

    # what results.jsonl actually holds, which the README describes
    _kinds = {}
    for line in open(JULY):
        line = line.strip()
        if line:
            k = json.loads(line).get("kind")
            _kinds[k] = _kinds.get(k, 0) + 1
    ck("README, records in results.jsonl", "309", sum(_kinds.values()))
    ck("README, of them prefill", "146", _kinds["prefill"])
    ck("README, and decode", "146", _kinds["decode"])
    ck("README, the rest being metadata, status and notes", "17",
       sum(v for k, v in _kinds.items() if k not in ("prefill", "decode")))

    # July's own ratio for the pair the August number is quoted for, so the
    # table a reader can see and the figure beside it stop disagreeing
    ck("benchmarks.md, MoE over the 31B in July", "2.498",
       tps(jul, "E-26B-tp2", 500) / tps(jul, "C-31B-tp2", 500))

    # 2026-08-31: three grids ran past a 390px viewport -- 6px, 21px and 89px --
    # because nothing scrolled them. Measured in a browser at 390px; this is the
    # static half, so a rule cannot quietly go away again.
    for _css, _sel in (("gqa-extra.css", "fault"),
                       ("measure-extra.css", "agree"),
                       ("rdna3-extra.css", "thr")):
        _t = open(os.path.join(HERE, "..", "..", "site", "src", _css)).read()
        _pat = r"@media \(max-width:560px\)\{[^@]*?\." + _sel + r"\s*\{[^}]*overflow-x:auto"
        ck("site css, %s scrolls .%s on a narrow screen" % (_css, _sel), "1",
           1 if re.search(_pat, _t, re.S) else 0)

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
    # Rewritten 2026-08-31 off the campaign rather than off four single-run
    # probes on a speculative arm. The old text read a U-shaped gap out of those
    # -- 1.48x at 1K, 1.14x at 16K, 1.87x at 32K -- and the campaign does not
    # have that shape, so the checks are rebuilt on the projection instead of
    # being renumbered.
    _XD = [json.loads(l) for l in open(os.path.join(HERE, "..", "decode.jsonl"))]

    def _d31(cfg, date, ctx):
        return next(r for r in _XD if r["cfg"] == cfg and r["date"] == date
                    and r["ctx"] == ctx)

    _PAIR, _PD = "C-31B-tp2", "2026-07-25"      # TP=2, vLLM 0.23, no patches
    _A1, _AD = "G31", "2026-08-30"              # TP=1, vLLM 0.28.0, no patches
    # The table is READ OUT OF THE README, not restated here. A check that
    # hardcodes the expected numbers catches the data drifting from the prose
    # and not the prose drifting from the data, and this section's whole content
    # is the table -- putting the old 1.14x back into it passed, once.
    _sec = rm[rm.index("### Two Radeons against one A100"):rm.index("### Want the raw numbers?")]
    _rows = []
    for _c, _p, _a, _adv in re.findall(
            r"^\|\s*([0-9]+(?:\s*K)?)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*\*{0,2}([\d.]+)×",
            _sec, re.M):
        _rows.append((int(_c.replace("K", "").strip()) * (1000 if "K" in _c else 1),
                      _p, _a, _adv))
    ck("README two-vs-one, rows in the published table", "5", len(_rows))
    # The retraction quotes the numbers it retracts, so those are published
    # figures too and are recomputed from the same probe files the old table
    # read -- pt[] on the Radeon side, the D*.log legs on the A100 side.
    _ret = re.search(r"read a U-shaped gap out of them — ([\d.]+)× at 1 K, "
                     r"narrowing to\s+([\d.]+)× at 16 K, widening to ([\d.]+)× ", _sec)
    ck("README two-vs-one, the retraction still quotes three gaps", "3",
       len(_ret.groups()) if _ret else 0)
    # so a deleted retraction fails as a figure rather than as a traceback
    _ret = _ret or re.match(r"()()()", "")
    ck("README two-vs-one, the retracted 1K gap", _ret.group(1) or "0",
       leg_result("D1K.log") / pt[1024])
    ck("README two-vs-one, the retracted 16K gap", _ret.group(2) or "0",
       leg_result("D16K.log") / pt[16384])
    ck("README two-vs-one, the retracted deep gap", _ret.group(3) or "0",
       d30 / pt[32768])
    # and it was 32K against 30K, which is why the deep end is stated that way
    ck("README two-vs-one, the retracted deep row's depths did not match", "1",
       1 if "32 K against the A100 at 30 K" in rm else 0)

    for _ctx, _p, _a, _adv in _rows:
        _pr, _ar = _d31(_PAIR, _PD, _ctx), _d31(_A1, _AD, _ctx)
        ck("README two-vs-one, pair at %d" % _ctx, _p, _pr["decode_tok_s"])
        ck("README two-vs-one, A100 at %d" % _ctx, _a, _ar["decode_tok_s"])
        ck("README two-vs-one, advantage at %d" % _ctx, _adv,
           _ar["decode_tok_s"] / _pr["decode_tok_s"])
    # both arms stock, and every cell of both graded -- the two properties that
    # make this a comparison rather than the probes it replaced
    _both = [(_d31(_PAIR, _PD, c), _d31(_A1, _AD, c)) for c in
             (500, 1000, 2000, 4000, 6000, 8000, 12000, 16000, 20000, 24000, 32000)]
    ck("README two-vs-one, rungs compared", "11", len(_both))
    ck("README two-vs-one, and every cell of both is chart-grade", "22",
       sum(1 for t in _both for r in t if r["chart_grade"]))
    ck("README two-vs-one, neither arm carries a patch", "0",
       sum(1 for t in _both for r in t if r["patches"]))
    ck("README two-vs-one, nor speculates", "0",
       sum(1 for t in _both for r in t if r["spec"]))
    # the withdrawn claim: no interior minimum, so no U
    _adv = [t[1]["decode_tok_s"] / t[0]["decode_tok_s"] for t in _both]
    ck("README two-vs-one, the narrowest gap is at the shallowest rung", "1",
       1 if min(_adv) == _adv[0] else 0)
    ck("README two-vs-one, and the widest at the deepest", "1",
       1 if max(_adv) == _adv[-1] else 0)
    ck("README two-vs-one, so the spread across the ladder", "0.08",
       max(_adv) - min(_adv), 0.06)
    # speculation inverts it, on arms that are patch-mismatched and said to be
    _sp32 = _d31("G31-mtp-p45450-tp2", "2026-08-29", 32000)["decode_tok_s"]
    _ss32 = _d31("G31-tp2", "2026-08-29", 32000)["decode_tok_s"]
    _ap32 = _d31("A100-G31-mtp-p45450", "2026-08-29", 32000)["decode_tok_s"]
    _as32 = _d31("A100-G31", "2026-08-29", 32000)["decode_tok_s"]
    ck("README two-vs-one, MTP on the pair at 32K pct", "7.9",
       (_sp32 / _ss32 - 1) * 100)
    ck("README two-vs-one, MTP on the A100 at 32K pct", "-20.1",
       (_ap32 / _as32 - 1) * 100)
    ck("README two-vs-one, so the speculative arms are level at 32K", "1.08",
       _ap32 / _sp32)
    ck("README two-vs-one, and at 2K the pair is ahead", "0.99",
       _d31("A100-G31-mtp-p45450", "2026-08-29", 2000)["decode_tok_s"]
       / _d31("G31-mtp-p45450-tp2", "2026-08-29", 2000)["decode_tok_s"])

    # the 2D-retention reading, moved out of the README on 2026-08-31 into
    # spec-decode doc §5 rather than dropped with the section that housed it
    _sd = open(os.path.join(HERE, "..", "..", "docs",
                            "speculative-decoding-on-rdna.md")).read()
    ck("spec-decode doc §5, 2D retention on the pair pct", "15.8",
       st[32768] / st[1024] * 100)
    ck("spec-decode doc §5, 2D retention on the A100 pct", "33.6",
       leg_result("C30.log") / leg_result("C1K.log") * 100)
    ck("spec-decode doc §5, so TP costs this much of the 2D path", "2.13",
       (leg_result("C30.log") / leg_result("C1K.log")) / (st[32768] / st[1024]))
    _splitkv = json.load(open(os.path.join(SDIR, "splitkv-31b-stock.json")))
    _nospec32 = {r["depth"]: r["tok_per_s"] for r in _splitkv["rows"]}[32768]
    ck("spec-decode doc §5, spec gain on the A100 pct", "39.2",
       (leg_result("D30.log") / M["30000"]["triton_forced"]["nospec"] - 1) * 100)
    ck("spec-decode doc §5, spec gain on the pair pct", "7.5",
       (pt[32768] / _nospec32 - 1) * 100)
    ck("spec-decode doc §5, and it says both", "1",
       1 if "+39.2 %" in _sd and "erodes what speculation is worth" in _sd else 0)
    ck("spec-decode doc §5, and it says so there", "1",
       1 if "starved path starve harder" in _sd and "33.6 %" in _sd else 0)

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
    # section 4 said "every collective aborts on rank 1" until 2026-08-29. The
    # committed log holds two invocations and both are all_gather; the script
    # intends six arms and the run stopped after the second.
    _s2b = open(os.path.join(HERE, "..", "..", "benchmarks", "rccl-6565", "logs",
                             "stage2b-raw.log"), encoding="utf-8").read()
    ck("6565 log, all_gather invocations", "2",
       len(re.findall(r"^\+ \./build/all_gather_perf", _s2b, re.M)))
    ck("6565 log, and no other collective ran", "0",
       len(re.findall(r"^\+ \./build/(?!all_gather_perf)\w+_perf", _s2b, re.M)))
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

    # Stage 3, 2026-08-28: the same eight arms through the cross-rank variant,
    # which is what replaced the rank-0 caveat on the 135/135 with a result.
    s3 = open(os.path.join(RDIR, "logs", "stage3-allranks.log")).read()
    a3 = [{"arm": m.group(1), "passed": int(m.group(2)), "failed": int(m.group(3)),
           "error": int(m.group(4) or 0), "n": int(m.group(5))}
          for m in re.finditer(
              r"=== arm=(\S+) RESULT pass=(\d+) fail=(\d+)(?: error=(\d+))? of (\d+)", s3)]
    ck("6565 README, stage 3 arms", "8", len(a3))
    ck("6565 README, stage 3 cold inits", "135", sum(a["n"] for a in a3))
    ck("6565 README, stage 3 correct on every rank", "135", sum(a["passed"] for a in a3))
    ck("6565 README, stage 3 failures", "0", sum(a["failed"] for a in a3))
    ck("6565 README, stage 3 errors", "0", sum(a["error"] for a in a3))
    # same arms, same counts as stages 1 and 2A, or the two sweeps are not
    # comparable and the second does not replace the caveat on the first
    ck("6565 README, stage 3 repeats the same arms at the same counts", "1",
       1 if sorted((a["arm"], a["n"]) for a in a3)
       == sorted((a["arm"], a["n"]) for a in arms) else 0)
    ck("6565 README, stage 3 ran the cross-rank script", "8",
       s3.count("script=/work/rccl_allgather_allranks.py"))

    # The deliberate breakage: corruption only rank 1 detects is invisible to the
    # reporter's script and fatal to the variant. Without this the stage-3 result
    # is just another clean run.
    bs = open(os.path.join(RDIR, "logs", "blindspot-check.log")).read()
    ck("6565 README, blind-spot check passed", "1",
       1 if "BLINDSPOT_CHECK_OK" in bs else 0)
    ck("6565 README, injection applied to both scripts", "2",
       bs.count("injected into "))
    ck("6565 README, the reporter's script reported ALL CORRECT under it", "1",
       1 if re.search(r"==> ALL CORRECT\s*\n\s*exit=0", bs) else 0)
    ck("6565 README, the variant tallied it as one failure", "1",
       1 if "RESULT pass=0 fail=1 error=0 of 1" in bs else 0)
    ck("6565 README, and the arm runner exited non-zero", "1",
       1 if "runner exit=1" in bs else 0)

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

    # Stage 4: MRv2 reaches the builder and the guard vllm#53930 warns about.
    m4 = [json.loads(l) for l in open(os.path.join(GDIR, "mrv2-spec.jsonl"))]
    ran = [r for r in m4 if r.get("engine_error") is None]
    ck("50603 README, stage 4 cells that ran an engine", "2", len(ran))
    by = {r["arm"]: r for r in ran}
    ck("50603 README, stage 4 has both arms", "2", len(by))
    ck("50603 README, V1 arm constructed the V1 runner", "1",
       1 if by.get("v1", {}).get("runner_constructed") == ["RUNNER_V1"] else 0)
    ck("50603 README, V2 arm constructed the V2 runner", "1",
       1 if by.get("v2", {}).get("runner_constructed") == ["RUNNER_V2"] else 0)
    # the load-bearing one: the function #53930 patches runs under MRv2
    ck("50603 README, the builder ran under MRv2", "1",
       1 if by.get("v2", {}).get("builder_init") == ["nospec"] else 0)
    ga = set(by["v1"]["guard_rows"])
    gb = set(by["v2"]["guard_rows"])
    ck("50603 README, stage 4 V1 guard rows", "36", len(ga))
    ck("50603 README, stage 4 V2 guard rows", "37", len(gb))
    ck("50603 README, stage 4 rows shared by both arms", "36", len(ga & gb))
    ck("50603 README, stage 4 rows only V1 has", "0", len(ga - gb))
    # no q>1 row at decode without speculation: the only ones are prefill, and
    # the single unattributed MRv2 capture row
    ck("50603 README, stage 4 decode rows are all max_seqlen_q=1", "0",
       sum(1 for r in ga | gb
           if eval(r)[3] == "q>1" and eval(r)[0] not in (2, 2048)))

    # --- docs/articles/*.html ------------------------------------------------
    # Each article embeds the figures it draws as JSON so a single self-contained
    # file can still be checked against the data. The article is a projection of
    # the repository, exactly like ledger.jsonl: if the data moves and the file
    # does not, this fails rather than letting a published page drift.
    ART = os.path.join(HERE, "..", "..", "docs", "articles")

    # --- the published pages are built, not hand-written -------------------
    # site/build.py --check rebuilds every page into memory and compares. If a
    # published file was edited in place it differs from its source, and the
    # shared head means such an edit reaches one page and not the other four.
    import subprocess
    bp = os.path.join(HERE, "..", "..", "site", "build.py")
    rc = subprocess.run([sys.executable, bp, "--check"], capture_output=True, text=True)
    ck("site, every published page matches its source", "0", rc.returncode)

    # --- the RCCL article: its figures are extracted from root-cause.md, so the
    # check is that the extraction still matches the document rather than that a
    # number matches a data file.
    rc = open(os.path.join(HERE, "..", "..", "docs", "root-cause.md"),
              encoding="utf-8").read()
    rart = open(os.path.join(ART, "rccl-atomics-hostcall.html"), encoding="utf-8").read()
    RF = json.loads(re.search(
        r'<script type="application/json" id="figures">(.*?)</script>', rart, re.S).group(1))

    def md_table(heading):
        body = rc.split(heading, 1)[1]
        rows = []
        for line in body.split("\n"):
            if line.startswith("|") and not re.match(r"^\|[\s|:-]+\|$", line):
                rows.append([c.strip() for c in line.strip().strip("|").split("|")])
            elif rows and not line.startswith("|"):
                break
        return rows[1:]

    ck("rccl article, chain links match the document", str(len(md_table("## 1. The causal chain"))),
       len(RF["chain"]))
    ck("rccl article, shipped-library rows match",
       str(len(md_table("## 2. Why downgrading appears to work"))), len(RF["shipped"]))
    ruled = md_table("## 3. What was ruled out")
    ck("rccl article, hypotheses match the document", str(len(ruled)), len(RF["ruled_out"]))
    ck("rccl article, hypotheses tested", "13", RF["counts"]["hypotheses_total"])
    ck("rccl article, eliminated", "12", RF["counts"]["hypotheses_eliminated"])
    ck("rccl article, exactly one root cause", "1", RF["counts"]["hypotheses_confirmed"])
    # the two shipped builds that carry no hostcall are the two that work
    zero = [r for r in RF["shipped"] if r["hostcall"] == "0"]
    ck("rccl article, the hostcall-free builds are the working ones", str(len(zero)),
       sum(1 for r in zero if "works" in r["behaviour"]))
    ck("rccl article, loads no external asset", "0",
       len(re.findall(r'\ssrc="(https?://[^"]+)"', rart)
           + re.findall(r'<link[^>]+href="(https?://[^"]+)"', rart)))
    # every artifact the article tells the reader to run must exist
    for f in ("diagnose/hipgate3.cpp", "diagnose/check-platform.sh",
              "docs/root-cause.md", "docs/vfio-atomics.md"):
        ck(f"rccl article, {f} exists", "1",
           1 if os.path.exists(os.path.join(HERE, "..", "..", f)) else 0)

    # every host an article may link to. Assets are separate and must be
    # local; the per-page check below keeps the two apart.
    LINK_HOSTS = {"github.com", "bugs.launchpad.net"}
    PAIRS = [["hybrid-ssm-collapse.html", "hybrid-ssm-collapse.zh.html"],
             ["rccl-atomics-hostcall.html", "rccl-atomics-hostcall.zh.html"],
             ["w4a16-two-problems.html", "w4a16-two-problems.zh.html"],
             ["moe-written-off-by-eager.html", "moe-written-off-by-eager.zh.html"],
             ["weight-loading-19x.html", "weight-loading-19x.zh.html"],
             ["speculative-decoding-net-loss.html",
              "speculative-decoding-net-loss.zh.html"],
             ["a100-vs-two-radeons.html", "a100-vs-two-radeons.zh.html"],
             ["gqa-gate-costs-nothing.html", "gqa-gate-costs-nothing.zh.html"],
             ["reporting-a-non-reproduction.html",
              "reporting-a-non-reproduction.zh.html"],
             ["measuring-decode.html", "measuring-decode.zh.html"],
             ["rdna3-second-class.html", "rdna3-second-class.zh.html"]]
    LANGS = [fn for pair in PAIRS for fn in pair]
    pages = {}
    for fn in LANGS:
        pages[fn] = open(os.path.join(ART, fn), encoding="utf-8").read()
        # a published page that pulls a script or a font from elsewhere stops
        # working the day that host does. Hyperlinks are not assets: an article
        # cites the trackers it was reported to, so those are held to a list.
        assets = (re.findall(r'\ssrc="(https?://[^"]+)"', pages[fn])
                  + re.findall(r'<link[^>]+href="(https?://[^"]+)"', pages[fn]))
        ck(f"article {fn}, loads no external asset", "0", len(assets))
        hosts = {u.split("/")[2] for u in
                 re.findall(r'<a [^>]*href="(https?://[^"]+)"', pages[fn])}
        ck(f"article {fn}, links only to known hosts", "0", len(hosts - LINK_HOSTS))

    # A Cyrillic word once survived into a Chinese draft. It reads as CJK at a
    # glance and nothing here would have caught it.
    for fn in LANGS:
        ck(f"article {fn}, no stray Cyrillic", "0",
           len(re.findall(r"[\u0400-\u04ff]", pages[fn])))

    flat = {fn: re.sub(r"\s+", " ", t) for fn, t in pages.items()}
    fl = lambda x: re.sub(r"\s+", " ", x)

    def block(text, ident):
        m = re.search(r'<script type="application/json" id="%s">(.*?)</script>' % ident,
                      text, re.S)
        return m.group(1) if m else None

    # the two language versions are one article: the same measurements drawn by
    # the same code, differing in prose and in the strings table only. If they
    # ever diverge, one of them is quoting numbers nobody checked.
    for pair in PAIRS:
        tag = pair[0].replace(".html", "")
        figs = [block(pages[fn], "figures") for fn in pair]
        ck(f"article {tag}, both versions carry a figures block", "2",
           sum(1 for f in figs if f))
        ck(f"article {tag}, the two versions share one figures block", "1",
           1 if figs[0] == figs[1] else 0)
        scr = [re.search(r"<script>\n\(function \(\).*?\n</script>", pages[fn], re.S)
               for fn in pair]
        ck(f"article {tag}, the two versions share one script", "1",
           1 if all(scr) and scr[0].group(0) == scr[1].group(0) else 0)
        # every string the script prints must exist in both tables, or one
        # version renders "undefined" where a label belongs
        keys = [set(json.loads(block(pages[fn], "strings")).keys()) for fn in pair]
        ck(f"article {tag}, the strings tables have the same keys", "1",
           1 if keys[0] == keys[1] else 0)
        for k in sorted(keys[0]):
            ck(f"article {tag}, script uses string '{k}'", "1",
               1 if ("S." + k) in scr[0].group(0) else 0)

        # each page links to the other and marks itself current, so the switcher
        # cannot end up pointing both buttons at the same file
        for fn in pair:
            nav = re.findall(r'<a class="lang" href="([^"]+)" hreflang="([a-z]+)"'
                             r'( aria-current="page")?>', pages[fn])
            ck(f"article {fn}, both languages in the switcher", "2", len(nav))
            ck(f"article {fn}, targets are the two pages", "1",
               1 if sorted(h for h, _, _ in nav) == sorted(pair) else 0)
            ck(f"article {fn}, exactly one is current", "1",
               1 if sum(1 for _, _, c in nav if c) == 1 else 0)
            ck(f"article {fn}, the current one is this page", "1",
               1 if any(h == fn and c for h, _, c in nav) else 0)

    art = pages[PAIRS[0][0]]
    A = json.loads(block(art, "figures"))

    # every series in fig1 and fig4 must still be exactly what the ledger holds
    led = [json.loads(l) for l in open(os.path.join(HERE, "..", "ledger.jsonl"))]

    def ledger_rows(s):
        return sorted(
            (r for r in led
             if r["model"] == s["model"] and r["quant"] == s["quant"]
             and r["arch"] == s["arch"] and r["tp"] == s["tp"]
             and r["vllm"] == s["vllm"] and r["patches"] == s["patches"]
             and r["harness"] == s["harness"] and r["date"] == s["date"]
             and r.get("spec") == s.get("spec")
             and r.get("attn_backend") == s.get("attn_backend")),
            key=lambda r: r["ctx"])

    for fig in ("fig1", "fig4"):
        for s_ in A[fig]["series"]:
            rows = ledger_rows(s_)
            tag = f'{fig} {s_["model"]} {"+".join(s_["patches"]) or "stock"}'
            ck(f"article, {tag} point count", str(len(rows)), len(s_["points"]))
            same = len(rows) == len(s_["points"]) and all(
                r["ctx"] == p["ctx"] and abs(r["decode_tok_s"] - p["tok_s"]) < 1e-9
                and r["runs"] == p["runs"]
                and abs(r["range_pct"] - p["range_pct"]) < 1e-9
                and r["chart_grade"] == p["graded"]
                for r, p in zip(rows, s_["points"]))
            ck(f"article, {tag} matches the ledger", "1", 1 if same else 0)

    ck("article, fig1 is the five-architecture campaign", "5",
       sum(1 for s_ in A["fig1"]["series"] if not s_.get("campaign")))
    ck("article, fig1 also carries the two 2026-08-29 backend ladders", "2",
       sum(1 for s_ in A["fig1"]["series"] if s_.get("campaign")))
    # the paragraph before section 5: how much of the collapse is the kernel
    HB = A["fig1"]["backends"]
    ck("article, neither backend ladder speculates", "1",
       1 if HB["spec"] is None else 0)
    ck("article, ROCM_ATTN retains", "73.4", HB["retained_pct"]["ROCM_ATTN"], tol=0.05)
    ck("article, TRITON_ATTN retains", "84.3", HB["retained_pct"]["TRITON_ATTN"], tol=0.05)
    ck("article, pinning the kernel is worth this at 32K", "15.0",
       HB["gain_at_deepest_pct"], tol=0.05)
    # a bound, not a value: the worst is 0.114, and "to 0.11%" would be false
    ck("article, and both ladders repeat inside an eighth of a per cent", "1",
       1 if HB["worst_range_pct"] < 0.12 else 0)
    # what is left over is the article's own subject, not the kernel's
    ck("article, the pinned ladder still gives up 15.7% by 32K", "15.7",
       100.0 - HB["retained_pct"]["TRITON_ATTN"], tol=0.05)
    ck("article, fig4 is both arms of the A/B", "2", len(A["fig4"]["series"]))

    # the slope figure is derived, so it is checked against the same helper the
    # rest of this file uses rather than against the ledger's 2-dp rates
    ck("article, fig3 hybrid slope", A["fig3"]["vllm_hybrid_slope_us"],
       slope_us(jul, "D-27B-tp2"))
    ck("article, fig3 hybrid retention", A["fig3"]["vllm_hybrid_retained_pct"],
       retained(jul, "D-27B-tp2"))
    for be in ("rocm", "vulkan"):
        raw = json.load(open(os.path.join(HERE, "..", f"llamacpp-depth-sweep-{be}.json")))
        pts = [(r["n_depth"], r["avg_ts"]) for r in raw]
        got = (1000 / pts[-1][1] - 1000 / pts[0][1]) / (pts[-1][0] - pts[0][0]) * 1000
        ck(f"article, fig3 llama.cpp {be} slope", A["fig3"]["llamacpp"][be]["slope_us"], got)
        ck(f"article, fig3 llama.cpp {be} points", str(len(raw)),
           len(A["fig3"]["llamacpp"][be]["points"]))

    # the abstract compares the hybrid's slope to the band the dense models
    # occupy, and said "four to forty" where the floor is fourteen. Both ends
    # are pinned now, and so is which llama.cpp backend sits inside that band:
    # the sentence claiming both did was true of one.
    _lo, _hi = A["fig3"]["dense_band_us"]
    _h = A["fig3"]["vllm_hybrid_slope_us"]
    ck("article, hybrid slope against the dense band, floor", "14.3", _h / _hi)
    ck("article, hybrid slope against the dense band, ceiling", "41.0", _h / _lo)
    ck("article, llama.cpp on ROCm is inside the dense band", "1",
       1 if _lo <= A["fig3"]["llamacpp"]["rocm"]["slope_us"] <= _hi else 0)
    ck("article, llama.cpp on Vulkan is below it", "1",
       1 if A["fig3"]["llamacpp"]["vulkan"]["slope_us"] < _lo else 0)
    for fn, phrase in (("hybrid-ssm-collapse.html", "fourteen to forty times steeper"),
                       ("hybrid-ssm-collapse.zh.html", "\u9661\u5341\u56db\u5230\u56db\u5341\u500d")):
        ck(f"article {fn}, the abstract says fourteen", "1",
           1 if fl(phrase) in flat[fn] else 0)

    # the one figure that cannot be recomputed says so, in the data and on the page
    ck("article, fig2 declares itself unreproducible", "1",
       0 if A["fig2"].get("reproducible_from_repo", True) else 1)
    ck("article, fig2 carries the marker on the page", "1",
       1 if "raw trace not committed" in flat[LANGS[0]] else 0)
    ck("article, the Chinese version carries it too", "1",
       1 if "原始 trace 未入库" in flat["hybrid-ssm-collapse.zh.html"] else 0)
    ck("rccl article, the Chinese version marks fig2 too", "1",
       1 if "llvm-readelf 输出未入库" in flat["rccl-atomics-hostcall.zh.html"] else 0)
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

    # --- w4a16-two-problems.html: the article is a projection of all of the
    # above, so every figure it embeds has to still equal its source. The page
    # is generated by site/src/genfig-w4a16.py; these checks are what stops the
    # generated copy from being edited, or from going stale when data moves.
    WART = json.loads(block(pages["w4a16-two-problems.html"], "figures"))

    # fig1: the six A/B cells, and the two readings of them the article turns on
    ck("w4a16 article, fig1 cells", "3", len(WART["fig1"]["cells"]))
    for c in WART["fig1"]["cells"]:
        a, s = aby[("asym", c["ctx"])], aby[("sym", c["ctx"])]
        ck(f'w4a16 article, fig1 {c["ctx"]} asym', repr(a["decode_tok_s"]),
           c["asym_tok_s"], tol=0)
        ck(f'w4a16 article, fig1 {c["ctx"]} sym', repr(s["decode_tok_s"]),
           c["sym_tok_s"], tol=0)
        ck(f'w4a16 article, fig1 {c["ctx"]} penalty ms', repr(c["penalty_ms"]),
           1000.0 / a["decode_tok_s"] - 1000.0 / s["decode_tok_s"], tol=1e-12)
        ck(f'w4a16 article, fig1 {c["ctx"]} ratio', repr(c["ratio"]),
           s["decode_tok_s"] / a["decode_tok_s"], tol=1e-12)
        # a single container start per cell must not be dressed up as a range
        ck(f'w4a16 article, fig1 {c["ctx"]} carries its run count', "1", c["runs"])
        ck(f'w4a16 article, fig1 {c["ctx"]} quotes no range it does not have', "1",
           1 if c["range_pct"] is None else 0)
    ck("w4a16 article, fig1 penalty spread", "8.2",
       WART["fig1"]["penalty_spread_pct"])
    ck("w4a16 article, fig1 ratio decline", "60.9",
       WART["fig1"]["ratio_decline_pct"])
    ck("w4a16 article, fig1 sym repeat", "1.31", WART["fig1"]["sym_repeat_worst_pct"])

    # fig2: the four corners and the campaign census, against their own files
    ck("w4a16 article, fig2 corners", "4", len(WART["fig2"]["corners"]))
    ck("w4a16 article, fig2 corners match the 2x2", "4",
       sum(1 for c in WART["fig2"]["corners"]
           if corner.get(c["corner"]) == c["chosen"]
           and c["native"] == (c["chosen"] == NATIVE)))
    ck("w4a16 article, fig2 campaign rows", str(len(camp)),
       len(WART["fig2"]["campaign"]))
    ck("w4a16 article, fig2 campaign rows match the census", str(len(camp)),
       sum(1 for r in WART["fig2"]["campaign"]
           if cby[r["checkpoint"]]["symmetric"] == r["symmetric"]
           and cby[r["checkpoint"]]["group_size"] == r["group_size"]
           and cby[r["checkpoint"]]["chosen"] == r["chosen"]))
    # the point of the figure: both kernels appear among the group-32 rows, so
    # a reader cannot come away thinking group size is what decides
    g32 = [r for r in WART["fig2"]["campaign"] if r["group_size"] == 32]
    ck("w4a16 article, fig2 shows both kernels at one group size", "2",
       len({r["chosen"] for r in g32}))

    # fig3: three stacks. Each arm carries every axis the ledger varies, and
    # each point is the row that identity selects -- keying on (model, ctx)
    # alone would merge the stock and patched arms without saying so.
    AXES = ("model", "quant", "arch", "tp", "vllm", "patches", "harness", "date")
    ck("w4a16 article, fig3 arms", "3", len(WART["fig3"]["arms"]))
    ck("w4a16 article, fig3 every arm carries its full series identity", "3",
       sum(1 for a in WART["fig3"]["arms"] if all(k in a for k in AXES)))
    ck("w4a16 article, fig3 the arms are three distinct series", "3",
       len({tuple(str(a.get(k)) for k in AXES) for a in WART["fig3"]["arms"]}))
    for a in WART["fig3"]["arms"]:
        ck(f'w4a16 article, fig3 {a["id"]} points', "3", len(a["points"]))
        if a["id"] == "023":
            same = all(abs(p["tok_s"] - aby[("asym", p["ctx"])]["decode_tok_s"]) < 1e-12
                       for p in a["points"])
        else:
            # .get, not [k]: an arm that has lost an identity axis must fail
            # the check above and this one, rather than crash the whole file
            rows = {r["ctx"]: r for r in led
                    if all(r[k] == a.get(k) for k in AXES)}
            same = len(rows) == 3 and all(
                abs(p["tok_s"] - rows[p["ctx"]]["decode_tok_s"]) < 1e-12
                and p["runs"] == rows[p["ctx"]]["runs"]
                and abs(p["range_pct"] - rows[p["ctx"]]["range_pct"]) < 1e-12
                and p["graded"] == rows[p["ctx"]]["chart_grade"]
                for p in a["points"])
        ck(f'w4a16 article, fig3 {a["id"]} matches its source', "1", 1 if same else 0)
    ck("w4a16 article, fig3 marks the ungraded 8K cell", "1",
       sum(1 for a in WART["fig3"]["arms"] for p in a["points"] if not p["graded"]))
    gain = WART["fig3"]["gain"]
    arms = {a["id"]: a for a in WART["fig3"]["arms"]}
    for i, c in enumerate(("1024", "8192", "32768")):
        ck(f"w4a16 article, fig3 gain {c} short", repr(gain[c]["short_fix"]),
           arms["027"]["points"][i]["tok_s"] / arms["023"]["points"][i]["tok_s"],
           tol=1e-12)
        ck(f"w4a16 article, fig3 gain {c} long", repr(gain[c]["long_fix"]),
           arms["027p"]["points"][i]["tok_s"] / arms["027"]["points"][i]["tok_s"],
           tol=1e-12)
    # the claim the figure is built on: the two fixes cross over between 1K and
    # 32K rather than both being worth more at one end
    ck("w4a16 article, fig3 the linear-kernel fix is worth most at 1K", "1",
       1 if gain["1024"]["short_fix"] > gain["32768"]["short_fix"] else 0)
    ck("w4a16 article, fig3 and the attention fix worth most at 32K", "1",
       1 if gain["32768"]["long_fix"] > gain["1024"]["long_fix"] else 0)
    ck("w4a16 article, fig3 they cross", "1",
       1 if (gain["1024"]["short_fix"] > gain["1024"]["long_fix"]
             and gain["32768"]["long_fix"] > gain["32768"]["short_fix"]) else 0)
    # the cross-campaign control, which is what makes the 0.27 arm credible on
    # its own: two campaigns, weeks apart, different scripts, same cell
    ctl = WART["fig3"]["control"]
    ck("w4a16 article, fig3 control is the 0.27 kernel pair's hybrid arm",
       repr(ctl["w4a16_hybrid_1k"]), a27["hybrid"]["decode_tok_s"], tol=0)
    ck("w4a16 article, fig3 control against the ledger", "0.21", ctl["apart_pct"])
    ck("w4a16 article, fig3 control names the Triton baseline",
       repr(ctl["forced_stock_1k"]), stock["decode_tok_s"], tol=0)

    # fig4: the coverage sweep, against the sweep's own output
    ck("w4a16 article, fig4 configs", str(len(cg)), len(WART["fig4"]["configs"]))
    REG = {"overlap": "overlap", "GAP:": "triton", "GAP+": "none"}
    cgby = {(r["group_size"], r["has_g_idx"]): r for r in cg}
    ck("w4a16 article, fig4 every cell matches the sweep", str(len(cg)),
       sum(1 for c in WART["fig4"]["configs"]
           if (lambda r: r is not None
               and c["region"] == next(v for k, v in REG.items()
                                       if r["region"].startswith(k))
               and (c["served_by"] is None) == (not r["hybrid_accepts"]
                                                and not r["triton_accepts"]))
              (cgby.get((c["group_size"], c["act_order"])))))
    ck("w4a16 article, fig4 unserved count", "8", WART["fig4"]["counts"]["none"])
    ck("w4a16 article, fig4 overlap count", "3", WART["fig4"]["counts"]["overlap"])
    ck("w4a16 article, fig4 triton-only count", "1", WART["fig4"]["counts"]["triton"])
    ck("w4a16 article, fig4 every act-order config is unserved", "1",
       1 if WART["fig4"]["act_order_unserved"] == WART["fig4"]["act_order_total"]
       else 0)

    # the two sections the house style makes mandatory, in both languages, and
    # the honesty markers that go with a single-run cell
    for fn, heads, marker in (
            ("w4a16-two-problems.html",
             ("What is not established", "What has changed since"),
             "One container start per cell"),
            ("w4a16-two-problems.zh.html",
             ("没有被确立的部分", "此后发生的变化"),
             "每格只起了一次容器")):
        for h in heads:
            ck(f"w4a16 article {fn}, carries '{h[:22]}'", "1",
               1 if fl(h) in flat[fn] else 0)
        ck(f"w4a16 article {fn}, says the A/B cells have no repeat", "1",
           1 if fl(marker) in flat[fn] else 0)
    # figure 3 crosses images, and the page has to say so where the figure is
    for fn, phrase in (("w4a16-two-problems.html", "not equivalent in kind"),
                       ("w4a16-two-problems.zh.html", "两组纵向对比的性质并不相同")):
        ck(f"w4a16 article {fn}, fig3 discloses the cross-stack step", "1",
           1 if fl(phrase) in flat[fn] else 0)

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
    extra8k = sorted(json.loads(l)["decode_tok_s"] for l in
                     open(os.path.join(HDIR, "qwen38-8k-r3r4.jsonl")))
    ck("hybrid-decode 6.6, two further passes at 8K", "2", len(extra8k))
    ck("hybrid-decode 6.6, the third 8K pass", "42.41", extra8k[0])
    ck("hybrid-decode 6.6, the fourth", "47.58", extra8k[1])
    all8k = sorted([q38[("A", "splitkv", 8192)]["decode_tok_s"],
                    q38[("B", "splitkv", 8192)]["decode_tok_s"]] + extra8k)
    ck("hybrid-decode 6.6, the low mode", "41.3", (all8k[0] + all8k[1]) / 2)
    ck("hybrid-decode 6.6, the high mode", "47.3", (all8k[2] + all8k[3]) / 2)
    ck("hybrid-decode 6.6, the modes are apart by", "15",
       100 * ((all8k[2] + all8k[3]) / (all8k[0] + all8k[1]) - 1), tol=0.05)
    ck("hybrid-decode 6.6, averaging the four gives", "44.29", sum(all8k) / 4)
    ck("hybrid-decode 6.6, cells across both passes", "12", len(q38))
    q38tps = lambda t, a, c: q38[(t, a, c)]["decode_tok_s"]
    pooled = lambda a, c: (q38tps("A", a, c) + q38tps("B", a, c)) / 2
    for ctx, sA, sB, pA, pB, ratio in (
            (1024, "37.04", "37.76", "51.81", "51.43", "1.38"),
            (8192, "12.57", "12.62", "47.02", "40.14", "3.52"),
            (32768, "3.83", "3.81", "35.20", "37.05", "9.46")):
        ck(f"hybrid-decode 6.6, {ctx} stock A", sA, q38tps("A", "stock", ctx))
        ck(f"hybrid-decode 6.6, {ctx} stock B", sB, q38tps("B", "stock", ctx))
        ck(f"hybrid-decode 6.6, {ctx} splitkv A", pA, q38tps("A", "splitkv", ctx))
        ck(f"hybrid-decode 6.6, {ctx} splitkv B", pB, q38tps("B", "splitkv", ctx))
        ck(f"hybrid-decode 6.6, {ctx} pooled ratio", ratio,
           (sum(all8k) / 4 if ctx == 8192 else pooled("splitkv", ctx))
           / pooled("stock", ctx))
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
    # Structural, not numeric. "What is here" carried two rows that opened a
    # table cell and never closed it; the blank line after each ended the table,
    # so 23K characters rendered as loose prose between four broken tables and
    # two stray "|" showed as text. Fixed 2026-08-31 by moving every long body
    # into a section of its own. This is the check that would have caught it.
    _br = open(os.path.join(HERE, "..", "README.md")).read()
    _wh = _br[_br.index("## What is here"):_br.index("## Reproducing the analysis")]
    _open = [l for l in _wh.split("\n")
             if l.startswith("|") and not l.rstrip().endswith("|")]
    ck("benchmarks README, table rows that open a cell and never close it", "0",
       len(_open))
    _idx = [l for l in _wh.split("\n")
            if l.startswith("| `") and l.rstrip().endswith("|")]
    ck("benchmarks README, paths in the index", "27", len(_idx))
    ck("benchmarks README, and a section for each that needs one", "23",
       len(re.findall(r"^### `", _wh, re.M)))
    ck("benchmarks README, no index line long enough to be a wall", "0",
       sum(1 for l in _idx if len(l) > 220))

    ck("benchmarks README, ledger rows", "265", len(led))
    ck("benchmarks README, ledger still matches its sources", "1",
       1 if build_ledger.dump(build_ledger.build())
       == open(os.path.join(HERE, "..", "ledger.jsonl")).read() else 0)
    ck("benchmarks README, points that are not chart grade", "29",
       sum(1 for r in led if not r["chart_grade"]))
    ck("benchmarks README, and it is above the cut", "29",
       sum(1 for r in led if not r["chart_grade"]
           and r["range_pct"] > build_ledger.RANGE_CUT))
    # --- the 2026-08-29 campaign's backend column, against its own logs ------
    # attn_backend is not decoration: it is what says why two rows differing
    # only in `patches` measure the same thing. A column carrying that has to
    # be checked against the logs rather than against itself.
    c29prov = json.load(open(os.path.join(
        HERE, "..", "campaign-2026-08-29", "provenance.json")))["arms"]
    c29 = [r for r in led if r["date"] == "2026-08-29"]
    ck("campaign 0829, every arm has a serve log naming its backend", "0",
       sum(1 for r in c29 if r["cfg"] not in c29prov
           or not c29prov[r["cfg"]]["attn_backend_evidence"]))
    ck("campaign 0829, and the ledger's backend is that log's", "1",
       1 if all(r["attn_backend"] == c29prov[r["cfg"]]["attn_backend"]
                for r in c29 if r["cfg"] in c29prov) else 0)
    # The probe prints once per process when the 3D path serves a step wider
    # than one token, so TP=2 prints twice. It needs three things at once: the
    # patch installed, the Triton kernel on the path, and speculation on.
    # Every arm missing any one of them must print zero -- that is what makes
    # the two that print a measurement rather than a coincidence.
    _fired = {k: v["probe_3d_spec_active"] for k, v in c29prov.items()}
    ck("campaign 0829, the probe fired twice on gemma-4's patched Triton arm",
       "2", _fired["G31-mtp-p45450-tp2"])
    ck("campaign 0829, and twice on Qwen3.8's",
       "2", _fired["Q38-mtp-triton-p45450-tp2"])
    ck("campaign 0829, and zero everywhere one of its three conditions is missing",
       "0", sum(v for k, v in _fired.items()
                if k not in ("G31-mtp-p45450-tp2", "Q38-mtp-triton-p45450-tp2")))
    ck("benchmarks README, points a later session supersedes", "2",
       sum(1 for r in led if r.get("superseded_values")))
    ck("benchmarks README, ledger range median", "0.17",
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

    # --- the best-of charts: drawn from the ledger, and from nothing else ----
    import gen_best_charts as gbc
    chosen = gbc.pick(led)
    ck("best charts, models drawn", "6", len(chosen))
    want = {
        "gemma-4-26B-A4B": ("2026-07-25", ()),
        "Qwen3-8B": ("2026-07-25", ()),
        "gemma-4-12B-it": ("2026-07-25", ()),
        "gemma-4-31B-it": ("2026-07-25", ()),
        "Muse-Glimmer-30B": ("2026-08-24", ("vllm#45916 split-KV", "window block-skip")),
        "Qwen3.8-27B": ("2026-08-28", ("vllm#45916 split-KV",)),
    }
    ck("best charts, each model comes from the stack the rule picks", "6",
       sum(1 for m, (d, p) in want.items()
           if m in chosen and chosen[m][2][1] == d and chosen[m][2][4] == p))
    ck("best charts, lines that need an unmerged patch", "2",
       sum(1 for m in chosen if chosen[m][2][4]))
    plotted = sum(1 for m in chosen for r in chosen[m][3] if r["chart_grade"])
    ck("best charts, points the ledger grades", "57", plotted)
    svgd = open(os.path.join(HERE, "..", "..", "docs", "assets",
                             "decode-vs-context-best.svg")).read()
    svgm = open(os.path.join(HERE, "..", "..", "docs", "assets",
                             "decode-ms-per-token-best.svg")).read()
    # 57 graded plus the bimodal 8K point plotted at its upper cluster
    ck("best charts, decode chart carries every point it draws", "58",
       svgd.count("<circle "))
    ck("best charts, and so does the cost chart", "58", svgm.count("<circle "))
    ck("best charts, the Qwen3.8 line is unbroken on both", "6",
       len(re.findall(r'<line [^>]*stroke="#e05c48"[^>]*/>', svgd))
       + len(re.findall(r'<line [^>]*stroke="#e05c48"[^>]*/>', svgm)))
    ck("best charts, the note says which mode was plotted", "1",
       1 if "two modes, 41 and 47 tok/s; plotted at the upper one" in
       " ".join(re.findall(r">([^<]*)</text>", svgd)) else 0)
    q8 = next(r for r in led if r["model"] == "Qwen3.8-27B" and r["ctx"] == 8192
              and r["patches"])
    hi = sorted(q8["values"])[2:]
    ck("best charts, the plotted 8K value", "47.30", sum(hi) / len(hi))
    ck("best charts, and the ledger still refuses to grade it", "0",
       1 if q8["chart_grade"] else 0)

    # the collapse has its own figure now: one model, two arms, one patch apart
    svgc = open(os.path.join(HERE, "..", "..", "docs", "assets",
                             "hybrid-ssm-collapse.svg")).read()
    ck("collapse chart, three depths on each arm", "6", svgc.count("<circle "))
    flatc = " ".join(re.findall(r">([^<]*)</text>", svgc))
    ck("collapse chart, the legend names both arms", "2",
       sum(1 for lab in ("released vLLM 0.27", "with vllm#45916") if lab in flatc))
    ck("collapse chart, and it is the only figure carrying the unpatched arm", "0",
       svgm.count("unpatched"))
    # the cost chart must not quietly get the collapse back: its ceiling is what
    # keeps the other six readable
    ck("cost chart ceiling stays at 40", "1",
       1 if re.search(r'text-anchor="end">40</text>', svgm) else 0)
    ck("collapse chart ceiling reaches 275", "1",
       1 if re.search(r'text-anchor="end">275</text>', svgc) else 0)

    # --- the front pages: what they embed, and the numbers they quote --------
    zh = open(os.path.join(ROOT, "README.zh.md")).read()
    bm = open(os.path.join(ROOT, "docs", "benchmarks.md")).read()
    for name, txt in (("README.md", rm), ("README.zh.md", zh)):
        ck(f"{name} embeds the best-of decode chart", "1",
           txt.count("decode-vs-context-best.svg"))
        # the complaint that started the refactor: one figure, not two of it
        ck(f"{name} no longer embeds a dated decode chart", "0",
           txt.count("decode-vs-context-2026-08-24.svg")
           + txt.count("assets/decode-vs-context.svg"))
        ck(f"{name} no longer embeds two ms-per-token charts", "0",
           txt.count("decode-ms-per-token-2026-08-24.svg")
           + txt.count("assets/decode-ms-per-token.svg"))
    ck("README.md embeds the best-of ms chart", "1",
       rm.count("decode-ms-per-token-best.svg"))
    for name, txt in (("README.md", rm), ("README.zh.md", zh), ("benchmarks.md", bm)):
        ck(f"{name} embeds the collapse chart", "1",
           txt.count("hybrid-ssm-collapse.svg"))
    ck("benchmarks.md keeps the per-campaign charts as the record", "2",
       bm.count("assets/decode-vs-context.svg")
       + bm.count("decode-vs-context-2026-08-24.svg"))
    ck("benchmarks.md also carries both best-of charts", "2",
       bm.count("decode-vs-context-best.svg") + bm.count("decode-ms-per-token-best.svg"))
    # the two numbers the front page deliberately puts side by side
    q32p = next(r for r in led if r["model"] == "Qwen3.8-27B" and r["ctx"] == 32768
                and r["patches"])
    q32s = next(r for r in led if r["model"] == "Qwen3.8-27B" and r["ctx"] == 32768
                and not r["patches"])
    ck("README, the chart's Qwen3.8 at 32K", "36.1", q32p["decode_tok_s"])
    ck("README, the campaign table's Qwen3.8 at 32K", "10.68",
       next(r["decode_tok_s"] for r in led if r["cfg"] == "D8-27B-tp2"
            and r["ctx"] == 32000 and r["date"] == "2026-08-24"))
    ck("README.zh, unpatched at 32K", "3.8", q32s["decode_tok_s"])
    ck("README.zh, and the ratio between them", "9.5",
       q32p["decode_tok_s"] / q32s["decode_tok_s"], tol=0.01)
    ck("README, the solid ms line at 32K", "261.9", 1000 / q32s["decode_tok_s"])
    ck("README, the dashed one", "27.7", 1000 / q32p["decode_tok_s"])

    # --- the ROCm and CUDA probes must keep sharing their reference ---------
    # These two files are 61% identical and an audit called for merging them.
    # They are deliberately not merged: each was verified on different hardware
    # and the halves that differ (run_arm, timed, main) are the point of having
    # both. What must not drift is the half that makes them comparable -- the
    # input construction, the fp32 reference and the scorer. A guard is the
    # protection merging would have given, without touching code verified on
    # two machines.
    def _top_level(path):
        src = open(path).read()
        out, cur, name = {}, [], None
        for line in src.split("\n"):
            m = re.match(r"^def (\w+)\(", line)
            if m or (line and not line[0].isspace() and cur):
                if name:
                    out[name] = "\n".join(cur).rstrip()
                name, cur = (m.group(1) if m else None), ([line] if m else [])
            elif name is not None:
                cur.append(line)
            k = re.match(r"^([A-Z_]+(?:, ?[A-Z_]+)*) = (.+)$", line)
            if k:
                # the value, not the trailing comment: the two files annotate
                # the same constants differently and that is not divergence
                val = k.group(2).split("#")[0].strip() if '"' not in k.group(2) \
                    else k.group(2)
                out["const:" + k.group(1).replace(" ", "")] = val
        if name:
            out[name] = "\n".join(cur).rstrip()
        return out

    rocm = _top_level(os.path.join(GDIR, "probe_50603.py"))
    cuda = _top_level(os.path.join(GDIR, "probe_50603_cuda.py"))
    shared = ["build", "reference", "score",
              "const:HEAD_SIZE", "const:BLOCK_SIZE", "const:DTYPE",
              "const:SHAPES", "const:CTX_LENS", "const:WARMUP,ITERS"]
    ck("50603 probes, shared pieces present in both files", str(len(shared)),
       sum(1 for k in shared if k in rocm and k in cuda))
    ck("50603 probes, and identical in both", str(len(shared)),
       sum(1 for k in shared if rocm.get(k) is not None and rocm.get(k) == cuda.get(k)))
    # the halves that are supposed to differ still do, so the guard is not
    # passing because someone made the two files the same
    ck("50603 probes, the arm-specific halves still differ", "3",
       sum(1 for k in ("run_arm", "timed", "main")
           if rocm.get(k) != cuda.get(k)))

    # --- invariants the audit found stated in three places and checked in none
    ROOTD = os.path.join(HERE, "..", "..")
    read = lambda *r: open(os.path.join(ROOTD, *r)).read()

    # the prompt ladder. cut_prompts.py generates it; analyze.py and
    # gen_charts.py restate it, and a target added to the generator and not
    # copied into both would be silently dropped from every table and chart
    ladders = {}
    for f in ("benchmarks/prompts/cut_prompts.py", "benchmarks/analyze/analyze.py",
              "benchmarks/analyze/gen_charts.py"):
        m = re.search(r"^TARGETS = (\[[\d, ]+\])$", read(f), re.M)
        ladders[f] = eval(m.group(1)) if m else None
    ck("prompt ladder, restated in three files", "3",
       sum(1 for v in ladders.values() if v))
    ck("prompt ladder, and all three agree", "1",
       1 if len({tuple(v or ()) for v in ladders.values()}) == 1 else 0)
    manifest_targets = set()
    for mf in sorted(glob.glob(os.path.join(ROOTD, "benchmarks/prompts/manifest-*.json"))):
        manifest_targets |= {e["target"] for e in json.load(open(mf))}
    ck("prompt ladder, and matches the committed manifests", "1",
       1 if set(ladders["benchmarks/prompts/cut_prompts.py"] or []) == manifest_targets
       else 0)

    # every configuration the runner can produce must be known to the ledger,
    # which asserts on an unknown cfg rather than skipping it
    runner_ids = set(re.findall(r'dict\(id="([\w-]+)"', read("benchmarks/bench_runner.py")))
    ledger_ids = set(re.findall(r'^\s+"([\w.-]+)":\s+\("',
                                read("benchmarks/analyze/build_ledger.py"), re.M))
    ck("every runner configuration is known to the ledger", "0",
       len(runner_ids - ledger_ids))

    # a deliberate divergence, kept deliberate. probe_w4a16_fix.py is the script
    # that produced w4a16-fix.jsonl and keys the zero-point convention on
    # c.zero_points; the unit tests later showed that is the wrong key for GPTQ,
    # so apply_patch.py and the upstream diff use has_bias(). Reconciling them
    # by editing apply_patch.py would make the patch wrong and nothing would say so
    ck("the W4A16 installer keys on has_bias", "1",
       1 if "not c.weight_type.has_bias()" in
       read("benchmarks/w4a16-symmetry/apply_patch.py") else 0)
    ck("and the as-run probe still keys on zero_points", "1",
       1 if "c.zero_points)" in
       read("benchmarks/w4a16-symmetry/probe_w4a16_fix.py") else 0)

    # both chart generators carry their own palette, keyed differently -- one by
    # cfg id, one by model name. A model drawn in two colours across two figures
    # on one page reads as two models, so the overlap has to agree
    gc = read("benchmarks/analyze/gen_charts.py")
    gb = read("benchmarks/analyze/gen_best_charts.py")
    by_cfg = dict(re.findall(r'"([\w.-]+)":\s+\("(#[0-9a-f]{6})"', gc))
    by_model = dict(re.findall(r'"([\w.-]+)": \("(#[0-9a-f]{6})"', gb))
    same_model = {"E-26B-tp2": "gemma-4-26B-A4B", "B-8B-tp2": "Qwen3-8B",
                  "A-12B-tp2": "gemma-4-12B-it", "C-31B-tp2": "gemma-4-31B-it",
                  "D8-27B-tp2": "Qwen3.8-27B", "G-30B-tp2": "Muse-Glimmer-30B"}
    ck("both chart generators agree on every shared model colour", "6",
       sum(1 for cfg, model in same_model.items()
           if by_cfg.get(cfg) and by_cfg.get(cfg) == by_model.get(model)))

    # the 53856 pair, same reasoning as the 50603 pair above
    r8 = _top_level(os.path.join(GDIR, "probe_53856.py"))
    r8b = _top_level(os.path.join(GDIR, "probe_53856_027.py"))
    ck("53856 probes, shared reference and dispatch identical", "3",
       sum(1 for k in ("reference", "run", "_counting_ck")
           if r8.get(k) is not None and r8.get(k) == r8b.get(k)))

    # --- moe-written-off-by-eager.html --------------------------------------
    # The compiled side is recomputed from results.jsonl. The eager side has no
    # committed raw output and is extracted from docs/benchmarks.md, so what is
    # checked there is that the extraction still matches the document.
    MART = json.loads(block(pages["moe-written-off-by-eager.html"], "figures"))
    jrows = [json.loads(l) for l in open(JULY)]
    jmeta = [r for r in jrows if r.get("kind") == "model_meta"]
    jdec = [r for r in jrows if r.get("kind") == "decode"]

    # fig1: the ranking, and the one bar that is not recomputable
    ck("moe article, fig1 bars", "5", len(MART["fig1"]["bars"]))
    ck("moe article, fig1 bars match the campaign", "5",
       sum(1 for b in MART["fig1"]["bars"]
           if abs(b["tok_s"] - tps(jul, b["cfg"], 500)) < 1e-9
           and b["runs"] == len(jul[b["cfg"]][500]["tps"])))
    ck("moe article, fig1 the MoE leads", "1",
       1 if MART["fig1"]["bars"][0]["cfg"] == "E-26B-tp2" else 0)
    ck("moe article, fig1 the eager bar declares itself unreproducible", "1",
       0 if MART["fig1"]["eager"].get("reproducible_from_repo", True) else 1)
    # ...and it still says what the document says, rather than a number of ours
    bmt = open(os.path.join(HERE, "..", "..", "docs", "benchmarks.md"),
               encoding="utf-8").read()
    moe_sec = bmt.split("## 1. The MoE was written off because nobody waited", 1)[1]
    ck("moe article, fig1 eager rate matches the document",
       re.search(r"~([\d.]+) tok/s", moe_sec).group(1), MART["fig1"]["eager"]["tok_s"])
    ck("moe article, fig1 12B eager rate matches the document",
       re.search(r"a flat ([\d.]+) tok/s", moe_sec).group(1), MART["fig1"]["eager_12b"])
    ck("moe article, fig1 eager is this far below compiled", "7.2",
       MART["fig1"]["ratio"])
    ck("moe article, fig1 and the 12B's is", "3.8", MART["fig1"]["ratio_12b"])

    # fig2: every engine start the campaign recorded, in order
    ck("moe article, fig2 starts", str(len(jmeta)), len(MART["fig2"]["starts"]))
    ck("moe article, fig2 starts match model_meta", str(len(jmeta)),
       sum(1 for s, m in zip(MART["fig2"]["starts"], jmeta)
           if s["cfg"] == m["cfg"]
           and abs(s["init_engine_s"] - float(m["init_engine_s"])) < 1e-9
           and abs(s["model_load_s"] - float(m["model_load_s"])) < 1e-9))
    ck("moe article, fig2 starts over 20 minutes", "2", len(MART["fig2"]["over_20min"]))
    # until 2026-08-29 the caption said "the other six are under two". Four are;
    # the 31B and the 27B sit between two and twenty minutes, and the figure
    # prints both of them, directly above the sentence that denied them.
    _mins = [(s["cfg"], s["init_engine_s"] / 60.0) for s in MART["fig2"]["starts"]]
    ck("moe article, fig2 starts under two minutes", "4",
       sum(1 for _, m in _mins if m < 2))
    ck("moe article, fig2 starts between two and twenty", "2",
       sum(1 for _, m in _mins if 2 <= m <= 20))
    ck("moe article, fig2 those two are the 31B and the 27B", "1",
       1 if sorted(c for c, m in _mins if 2 <= m <= 20) == ["C-31B-tp2", "D-27B-tp2"] else 0)
    ck("moe article, fig2 and neither was ever restarted", "2",
       len(MART["fig2"]["no_warm_start_for"]))
    ck("moe article, fig2 the slowest start", "1569.01",
       MART["fig2"]["slowest"]["init_engine_s"])
    ck("moe article, fig2 and it is the MoE", "1",
       1 if MART["fig2"]["slowest"]["cfg"] == "E-26B-tp2" else 0)
    ck("moe article, fig2 the only repeat is cold 59.67", "59.67",
       MART["fig2"]["repeat"]["cold"])
    ck("moe article, fig2 against warm 33.36", "33.36", MART["fig2"]["repeat"]["warm"])
    ck("moe article, fig2 the repeat is not one of the slow ones", "1",
       1 if MART["fig2"]["repeat"]["cfg"] not in MART["fig2"]["over_20min"] else 0)
    # what rules the loader out: on both slow starts the weights are a rounding
    # error, which is the figure's whole claim about where the time goes
    share = {s["cfg"]: s["load_share_pct"] for s in MART["fig2"]["starts"]}
    for cfg, claim in (("A-12B-tp2", "0.36"), ("E-26B-tp2", "3.43")):
        ck(f"moe article, fig2 weight-loading share {cfg}", claim, share.get(cfg, -1))

    # fig3: the two phenomena eager produced, against 22 committed rows
    moe_dec = [r for r in jdec if r["cfg"] == "E-26B-tp2"]
    ck("moe article, fig3 rows", str(len(moe_dec)), MART["fig3"]["rows"])
    ck("moe article, fig3 depths", "11", len(MART["fig3"]["points"]))
    asym = [abs(r["p1_max"] - r["p2_max"]) / max(r["p1_max"], r["p2_max"]) * 100.0
            for r in moe_dec]
    ck("moe article, fig3 worst card-to-card gap", "2.26", MART["fig3"]["worst_asym_pct"])
    ck("moe article, fig3 recomputes that gap", repr(max(asym)),
       MART["fig3"]["worst_asym_pct"], tol=1e-12)
    ck("moe article, fig3 median gap", "0.75", MART["fig3"]["median_asym_pct"])
    ck("moe article, fig3 VRAM is equal on both cards everywhere", "1",
       1 if MART["fig3"]["vram_equal_everywhere"]
       and all(r["v1_g"] == r["v2_g"] for r in moe_dec) else 0)
    ck("moe article, fig3 retention at 32K", repr(retained(jul, "E-26B-tp2")),
       MART["fig3"]["retained_pct"], tol=1e-12)
    ck("moe article, fig3 the run is not context-independent", "1",
       1 if MART["fig3"]["retained_pct"] < 90 else 0)
    ck("moe article, fig3 eager power matches the document", "1",
       1 if all(str(w) + " W" in moe_sec for w in MART["fig3"]["eager_power"]["w"])
       else 0)
    ck("moe article, fig3 the eager row declares itself unreproducible", "1",
       0 if MART["fig3"]["eager_power"].get("reproducible_from_repo", True) else 1)
    ck("moe article, fig3 eager gap", "50.4", MART["fig3"]["eager_power"]["asym_pct"])
    # the comparison only means anything if the two are far apart in kind
    ck("moe article, fig3 the eager gap is an order above the measured one", "1",
       1 if MART["fig3"]["eager_power"]["asym_pct"]
       > 10 * MART["fig3"]["worst_asym_pct"] else 0)

    # the compile figures the documents quote must be the ones results.jsonl
    # holds. Three of them were wrong until 2026-08-29, in three files, and
    # nothing checked them, which is why they are pinned here.
    meta_by = {}
    for m in jmeta:
        meta_by.setdefault(m["cfg"], []).append(float(m["init_engine_s"]))
    an = open(os.path.join(HERE, "..", "..", "docs", "architecture-notes.md"),
              encoding="utf-8").read()
    rmd = open(os.path.join(HERE, "..", "..", "README.md"), encoding="utf-8").read()
    ck("arch-notes, the MoE engine init it quotes", "1",
       1 if f'{meta_by["E-26B-tp2"][0]:.2f}' in an else 0)
    ck("arch-notes, the 12B TP=2 engine init it quotes", "1",
       1 if f'{meta_by["A-12B-tp2"][0]:.2f}' in an else 0)
    ck("arch-notes, and the TP=1 starts it contrasts them with", "2",
       sum(1 for v in meta_by["A-12B-tp1"] if f"{v:.2f}" in an))
    ck("README, the MoE engine init it quotes", "1",
       1 if f'{round(meta_by["E-26B-tp2"][0]):.0f} s' in rmd else 0)
    ck("README, the 12B one, and at the right TP", "1",
       1 if f'{round(meta_by["A-12B-tp2"][0]):.0f} s' in rmd and "at **TP=2**" in rmd
       else 0)
    # A withdrawn claim stays in the document -- that is the point of dating a
    # correction -- but only ever quoted. If one of these turns up asserted
    # again, it is outside every pair of quotation marks and this fails.
    def unquoted(txt, phrase):
        spans = [m.span() for m in re.finditer(r'"[^"]{0,400}"', txt, re.S)]
        return sum(1 for m in re.finditer(re.escape(phrase), txt)
                   if not any(a < m.start() and m.end() <= b for a, b in spans))
    for name, txt, phrase in (
            ("arch-notes", an, "a warm start is seconds"),
            ("benchmarks.md", bmt, "cold, 33 s warm")):
        ck(f"{name}, the unmeasured warm-start claim appears only quoted", "0",
           unquoted(txt, phrase))

    # both language versions carry the two sections the house style requires
    for fn, heads in (("moe-written-off-by-eager.html",
                       ("What is not established", "What has changed since")),
                      ("moe-written-off-by-eager.zh.html",
                       ("没有被确立的部分", "此后发生的变化"))):
        for h in heads:
            ck(f"moe article {fn}, carries '{h[:22]}'", "1",
               1 if fl(h) in flat[fn] else 0)
    # and both say that the number the verdict came from is the uncommitted one
    for fn, phrase in (("moe-written-off-by-eager.html",
                        "The eager raw output is not in this repository"),
                       ("moe-written-off-by-eager.zh.html",
                        "eager 的原始输出不在这个仓库里")):
        ck(f"moe article {fn}, discloses the eager provenance", "1",
           1 if fl(phrase) in flat[fn] else 0)

    # --- weight-loading-19x.html ---------------------------------------------
    # Two committed data files behind three figures, plus one opening pair that
    # has no data file at all and is extracted from the document that carries it.
    LART = json.loads(block(pages["weight-loading-19x.html"], "figures"))
    hmm = json.load(open(os.path.join(HERE, "..", "hmm-kernel-three-states.json")))
    lflag = json.load(open(os.path.join(HERE, "..", "loader-flag-kernel-30.json")))
    oq = open(os.path.join(HERE, "..", "..", "docs", "open-questions.md"),
              encoding="utf-8").read()

    # the opening question: extracted, not measured, and it has to say so
    ck("loader article, opening declares itself unreproducible", "1",
       0 if LART["opening"].get("reproducible_from_repo", True) else 1)
    hist = oq.split("**Measured on the verified configuration (2026-07-24):**", 1)[1]
    ck("loader article, opening disk rate matches the document",
       re.search(r"read \(`dd iflag=direct`\) \| ([\d.]+) GB/s", hist).group(1),
       LART["opening"]["disk_gb_s"])
    doc_rates = [int(m) for m in re.findall(r"\*\*(\d+) MB/s\*\*", hist)]
    ck("loader article, opening rows match the document", "2",
       sum(1 for r in LART["opening"]["rows"] if r["mb_s"] in doc_rates))
    for r in LART["opening"]["rows"]:
        ck(f'loader article, opening {r["model"]} ratio',
           repr(LART["opening"]["disk_gb_s"] * 1000.0 / r["mb_s"]),
           r["times_slower"], tol=1e-12)
    # the section heading says 19-48x; the table's own figures give 19-52x, and
    # the document now carries the unit note that says so. If either moves,
    # this is what notices.
    ck("loader article, and the band the numbers actually give", "52",
       max(r["times_slower"] for r in LART["opening"]["rows"]), tol=0.01)
    ck("open-questions, the unit note is present", "1",
       1 if "its own two rows give **19\u00d7 and 52\u00d7**" in oq else 0)

    # fig1: three kernel states, three mapping cases, one file
    ck("loader article, fig1 states", "3", len(LART["fig1"]["states"]))
    # a scalar or the middle of a list of repeats; NOT the med() helper above,
    # which is a real median and would average the two middles
    mid = lambda v: sorted(v)[len(v) // 2] if isinstance(v, list) else float(v)
    for st, src in zip(LART["fig1"]["states"], hmm["states"]):
        ck(f'loader article, fig1 {src["kernel"]} {src["date"]} cases', "3",
           len(st["cases"]))
        same = all(abs(c["ms"] - mid(src[c["key"]])) < 1e-9 for c in st["cases"])
        ck(f'loader article, fig1 {src["kernel"]} {src["date"]} matches its source',
           "1", 1 if same and st["kernel"] == src["kernel"] else 0)
    ck("loader article, fig1 stock -28", "16019.6", LART["fig1"]["stock28"])
    ck("loader article, fig1 with the missing commit", "17.0", LART["fig1"]["reverted"])
    ck("loader article, fig1 as Canonical ships it", "15.3", LART["fig1"]["shipped30"])
    ck("loader article, fig1 factor between the ends", "1047",
       LART["fig1"]["fix_factor"], tol=5e-4)
    # the arithmetic that identified the mechanism: whole timeout windows
    ck("loader article, fig1 timeout", "1000", LART["fig1"]["timeout_ms"])
    ck("loader article, fig1 windows", "16.0196", LART["fig1"]["windows"])
    ck("loader article, fig1 residual work", "19.6", LART["fig1"]["residual_ms"])
    ck("loader article, fig1 the residual is smaller than one window", "1",
       1 if LART["fig1"]["residual_ms"] < LART["fig1"]["timeout_ms"] else 0)
    # the two control cases must NOT move, or the figure proves nothing
    ro = [c["ms"] for st in LART["fig1"]["states"] for c in st["cases"]
          if c["key"] == "r_p_resident"]
    nr = [c["ms"] for st in LART["fig1"]["states"] for c in st["cases"]
          if c["key"] == "rw_p_not_resident"]
    ck("loader article, fig1 the read-only control stays put", "1",
       1 if max(ro) / min(ro) < 1.2 else 0)
    ck("loader article, fig1 and so does the not-resident one", "1",
       1 if max(nr) / min(nr) < 1.2 else 0)
    ck("loader article, fig1 the writable penalty that survives", "4.8",
       LART["fig1"]["writable_penalty_30"])

    # fig2: what the permanent half costs end to end
    med_by = {(r["model"], r["cache"], r["mode"]): r for r in lflag["medians_seconds"]}
    cells = [(g, c) for g in LART["fig2"]["groups"] for c in g["cells"]]
    ck("loader article, fig2 cells", str(len(lflag["medians_seconds"])), len(cells))
    # the caption said "89 cells" until 2026-08-29. The file holds 89; the figure
    # draws only the cache-controlled ones, and the other 16 are the `asis`
    # ordering controls that deliberately do not control page cache.
    _cache = [c["cache"] for c in lflag["cells"]]
    ck("loader article, fig2 the file holds 89 cells", "89", len(_cache))
    ck("loader article, fig2 draws the cache-controlled ones", "73",
       _cache.count("warm") + _cache.count("cold"))
    ck("loader article, fig2 the rest are asis ordering controls", "16", _cache.count("asis"))
    ck("loader article, fig2 cells match their medians", str(len(cells)),
       sum(1 for g, c in cells
           if (g["model"], g["cache"], c["mode"]) in med_by
           and med_by[(g["model"], g["cache"], c["mode"])]["median_s"] == c["median_s"]
           and med_by[(g["model"], g["cache"], c["mode"])]["n"] == c["n"]))
    ck("loader article, fig2 every ratio is against its own baseline", str(len(cells)),
       sum(1 for g, c in cells
           if abs(c["vs_baseline"]
                  - med_by[(g["model"], g["cache"], "baseline")]["median_s"]
                  / c["median_s"]) < 1e-9))
    ck("loader article, fig2 the flag at its best", "7.53", LART["fig2"]["best_flag"])
    ck("loader article, fig2 and at its worst", "0.96", LART["fig2"]["worst_flag"])
    ck("loader article, fig2 the worst case is the MoE", "1",
       1 if abs(LART["fig2"]["moe_flag"] - LART["fig2"]["worst_flag"]) < 1e-9 else 0)
    ck("loader article, fig2 and it is a loss", "1",
       1 if LART["fig2"]["moe_flag"] < 1 else 0)
    # the band the withdrawn 3.9-5.6x was replaced by, for the cells that fit
    fits = [c["vs_baseline"] for g, c in cells if c["mode"] == "flag"
            and "31B" not in g["model"] and "MoE" not in g["model"]]
    ck("loader article, fig2 the in-RAM band, low", "1.51", min(fits))
    ck("loader article, fig2 the in-RAM band, high", "1.98", max(fits))
    ck("loader article, fig2 no cell reaches the withdrawn 3.9x", "0",
       sum(1 for v in fits if v >= 3.9))
    # the end-to-end control: a real server start, not the isolated harness
    ck("loader article, fig2 end-to-end pairs", "4", len(LART["fig2"]["end_to_end"]))
    e2e_raw = {}
    for r in lflag["end_to_end_loading_weights_took_seconds"]:
        e2e_raw.setdefault((r["cache"], r["mode"]), []).append(r["loading_weights_took_s"])
    e2e = {(r["cache"], r["mode"]): r["mean_s"] for r in LART["fig2"]["end_to_end"]}
    ck("loader article, fig2 end-to-end means match the file", "4",
       sum(1 for k, v in e2e.items() if abs(v - sum(e2e_raw[k]) / len(e2e_raw[k])) < 1e-9))
    for row, claim in zip(LART["fig2"]["e2e_vs_harness"], ("1.79", "1.52")):
        cache = row["cache"]
        ck(f"loader article, fig2 the {cache} server start", claim,
           e2e[(cache, "baseline")] / e2e[(cache, "flag")])
        ck(f"loader article, fig2 and the {cache} figure the article quotes", claim,
           row["served"])
        harness = [c["vs_baseline"] for g, c in cells
                   if g["model"] == "gemma-4-12B-w4a16" and g["cache"] == cache
                   and c["mode"] == "flag"][0]
        ck(f"loader article, fig2 the {cache} harness it is compared with",
           repr(harness), row["harness"], tol=1e-12)
    # the direction matters more than the size: the isolated harness must not be
    # pessimistic, or it would be flattering the flag by understating the baseline
    ck("loader article, fig2 the harness is the optimistic one, both caches", "2",
       sum(1 for r in LART["fig2"]["e2e_vs_harness"]
           if 0 < r["harness_optimistic_pct"] < 20))

    # fig3: the mechanism, in the resident set
    rss_by = {(r["model"], r["cache"], r["mode"]): r
              for r in lflag["resident_set_split_mib"]}
    ck("loader article, fig3 rows", str(len(lflag["resident_set_split_mib"])),
       len(LART["fig3"]["rows"]))
    ck("loader article, fig3 rows match their source", str(len(LART["fig3"]["rows"])),
       sum(1 for r in LART["fig3"]["rows"]
           if (r["model"], r["cache"], r["mode"]) in rss_by
           and rss_by[(r["model"], r["cache"], r["mode"])]["peak_rss_anon_mib"] == r["anon"]
           and rss_by[(r["model"], r["cache"], r["mode"])]["peak_rss_file_mib"] == r["file"]))
    sw = LART["fig3"]["swap"]
    ck("loader article, fig3 the 31B default is anonymous", "21390", sw["baseline"][0])
    ck("loader article, fig3 against file-backed", "782", sw["baseline"][1])
    ck("loader article, fig3 and the clone swaps them", "1",
       1 if sw["flag"][0] < sw["baseline"][1] * 1.2
       and sw["flag"][1] > sw["baseline"][0] else 0)
    ck("loader article, fig3 sharding bounds the anonymous peak", "4535",
       LART["fig3"]["sharded"])
    ck("loader article, fig3 and that is about one shard of five", "1",
       1 if LART["fig3"]["sharded"] < sw["baseline"][0] / 4 else 0)
    npread = sum(1 for r in LART["fig3"]["rows"] if r["mode"] == "pread")
    ck("loader article, fig3 pread never maps the file", str(npread),
       sum(1 for r in LART["fig3"]["rows"] if r["mode"] == "pread" and r["file"] < 500))
    ck("loader article, fig3 and there are that many pread rows", "6", npread)

    # the house-style sections, and the disclosure that the opening pair is not
    # recomputable, in both languages
    for fn, heads, phrase in (
            ("weight-loading-19x.html",
             ("What is not established", "What has changed since"),
             "Where that pair of numbers comes from"),
            ("weight-loading-19x.zh.html",
             ("没有被确立的部分", "此后发生的变化"),
             "这两个数字是从哪来的")):
        for h in heads:
            ck(f"loader article {fn}, carries '{h[:22]}'", "1",
               1 if fl(h) in flat[fn] else 0)
        ck(f"loader article {fn}, discloses the opening's provenance", "1",
           1 if fl(phrase) in flat[fn] else 0)
    # the reproducers the article tells the reader to run must exist
    for f in ("benchmarks/repro-mmap-prot.py", "benchmarks/repro-mmap-prot.hip.cpp",
              "benchmarks/hmm-kernel-three-states.json",
              "benchmarks/loader-flag-kernel-30.json"):
        ck(f"loader article, {f} exists", "1",
           1 if os.path.exists(os.path.join(HERE, "..", "..", f)) else 0)

    # --- speculative-decoding-net-loss.html ----------------------------------
    # Four committed sources behind four figures, plus a profiler summary whose
    # traces are not in the repository and which has to say so.
    PART = json.loads(block(pages["speculative-decoding-net-loss.html"], "figures"))
    SP = os.path.join(HERE, "..", "speculative-decoding")
    lad = lambda fn: {r["depth"]: r["tok_per_s"] for r in
                      json.load(open(os.path.join(SP, fn)))["rows"]}
    ns, mt, mt2 = (lad("splitkv-31b-stock.json"), lad("mtp-31b-mtp.json"),
                   lad("mtp-31b-stock45450.json"))

    # fig1: the ladder, its sign change, and the replicate it turns out to have
    ck("spec article, fig1 rows", "4", len(PART["fig1"]["rows"]))
    ck("spec article, fig1 rows match their files", "4",
       sum(1 for r in PART["fig1"]["rows"]
           if abs(r["nospec"] - ns[r["ctx"]]) < 1e-9
           and abs(r["mtp"] - mt[r["ctx"]]) < 1e-9
           and abs(r["mtp_repeat"] - mt2[r["ctx"]]) < 1e-9))
    for r in PART["fig1"]["rows"]:
        ck(f'spec article, fig1 {r["ctx"]} delta',
           repr((mt[r["ctx"]] / ns[r["ctx"]] - 1) * 100.0), r["delta_pct"], tol=1e-12)
    ck("spec article, fig1 best", "36.9", PART["fig1"]["best"])
    ck("spec article, fig1 worst", "-70.8", PART["fig1"]["worst"])
    ck("spec article, fig1 is monotonic", "1", 1 if PART["fig1"]["monotonic"] else 0)
    ck("spec article, fig1 changes sign", "1",
       1 if PART["fig1"]["best"] > 0 > PART["fig1"]["worst"] else 0)
    # the claim the article makes about which end of the ladder reproduces
    ck("spec article, fig1 the long depths repeat to", "0.45",
       PART["fig1"]["repeat_worst_long_pct"])
    ck("spec article, fig1 and 1K does not", "6.0", PART["fig1"]["repeat_1k_pct"])
    ck("spec article, fig1 the repeat is worst where the finding is not", "1",
       1 if PART["fig1"]["repeat_1k_pct"] > 5 * PART["fig1"]["repeat_worst_long_pct"]
       else 0)
    sd = open(os.path.join(HERE, "..", "..", "docs",
                           "speculative-decoding-on-rdna.md"), encoding="utf-8").read()
    ck("spec doc, the replicate correction is present", "1",
       1 if "Corrected 2026-08-29" in sd and "mtp-31b-stock45450.json" in sd else 0)
    ck("spec doc, and it no longer claims there are none", "0",
       len(re.findall(r"there are no process replicates", sd)))

    # fig2: two independent constructions of the same sweep
    kb = {t_: {(r["kv_len"], r["q_len"]): r["us"]
               for r in json.load(open(os.path.join(SP, fn)))}
          for t_, fn in (("A", "kbench-0.json"), ("B", "kbench2-0.json"))}
    ck("spec article, fig2 sweeps", "2", len(PART["fig2"]["sweeps"]))
    ck("spec article, fig2 sweeps match their files", "2",
       sum(1 for s in PART["fig2"]["sweeps"]
           if all(abs(row["us"][j] - kb[s["id"]][(row["kv"], q)]) < 1e-9
                  for row in s["rows"] for j, q in enumerate(PART["fig2"]["q_lens"]))))
    for i, tag in enumerate(("A", "B")):
        ck(f"spec article, fig2 32K one row to two, construction {tag}",
           repr((kb[tag][(32768, 2)] / kb[tag][(32768, 1)] - 1) * 100.0),
           PART["fig2"]["q1_to_q2_32k_pct"][i], tol=1e-12)
    ck("spec article, fig2 that is under a per cent in both", "2",
       sum(1 for v in PART["fig2"]["q1_to_q2_32k_pct"] if abs(v) < 1.0))
    ck("spec article, fig2 the launch grids", "8", PART["fig2"]["workgroups"]["2d"])
    ck("spec article, fig2 against", "128", PART["fig2"]["workgroups"]["3d"])
    ck("spec article, fig2 which is what the clause gives up", "16",
       PART["fig2"]["workgroups"]["3d"] // PART["fig2"]["workgroups"]["2d"])
    # where the two constructions disagree, which the caption states rather than
    # averaging away
    bc = {r["kv"]: r for r in PART["fig2"]["between_constructions_pct"]}
    ck("spec article, fig2 the constructions agree at 32K", "0.18", bc[32768]["q1"])
    ck("spec article, fig2 and disagree most at 16K", "10.1",
       PART["fig2"]["worst_within_disagreement_pct"])
    ck("spec article, fig2 the 16K disagreement is in the one-row cell", "1",
       1 if bc[16384]["q1"] > 4 * bc[16384]["q2plus_max"] else 0)
    ck("spec article, fig2 the 16K two-row-and-up cells agree to", "2.23",
       bc[16384]["q2plus_max"])

    # fig3: the other vendor
    mat = json.load(open(os.path.join(HERE, "..", "cuda-a100",
                                      "gemma4-mtp-backend-matrix.json")))
    ck("spec article, fig3 cells", "5", len(PART["fig3"]["cells"]))
    ck("spec article, fig3 cells match the matrix", "5",
       sum(1 for c in PART["fig3"]["cells"]
           if mat["decode_tok_s"][str(c["ctx"])][c["backend"]]["mtp"] == c["mtp"]
           and mat["decode_tok_s"][str(c["ctx"])][c["backend"]]["nospec"] == c["nospec"]
           and abs(c["delta_pct"] - (c["mtp"] / c["nospec"] - 1) * 100.0) < 1e-9))
    tri = {c["ctx"]: c["delta_pct"] for c in PART["fig3"]["cells"]
           if c["backend"] == "triton_forced"}
    ck("spec article, fig3 forced default at 30K", "-28.2", tri[30000])
    ck("spec article, fig3 and at 50K", "-61.1", tri[50000])
    ck("spec article, fig3 it deepens with depth", "1",
       1 if tri[50000] < tri[30000] < 0 else 0)
    heal = [c["delta_pct"] for c in PART["fig3"]["cells"]
            if c["backend"] != "triton_forced" and c["ctx"] == 30000]
    ck("spec article, fig3 both healthy backends are positive at 30K", "2",
       sum(1 for v in heal if v > 0))
    ck("spec article, fig3 the ROCm reference it is drawn against", "-70.8",
       PART["fig3"]["rocm_reference_pct"])
    ck("spec article, fig3 says how many runs a cell is", "1",
       PART["fig3"]["runs_per_cell"])

    # fig4: what the admission is worth, and that it is the same thing
    stk, prt = lad("mtp-31b-stock45450.json"), lad("mtp-31b-p45450.json")
    ck("spec article, fig4 rows", "4", len(PART["fig4"]["rows"]))
    ck("spec article, fig4 rows match their files", "4",
       sum(1 for r in PART["fig4"]["rows"]
           if abs(r["stock"] - stk[r["ctx"]]) < 1e-9
           and abs(r["ported"] - prt[r["ctx"]]) < 1e-9
           and abs(r["ratio"] - prt[r["ctx"]] / stk[r["ctx"]]) < 1e-9))
    ck("spec article, fig4 at 32K", "3.70",
       [r["ratio"] for r in PART["fig4"]["rows"] if r["ctx"] == 32768][0])
    ck("spec article, fig4 the hand-widened experiment", "32.42",
       PART["fig4"]["hand_forced_32k"])
    ck("spec article, fig4 lands this close to the PR", "0.5",
       PART["fig4"]["hand_vs_ported_pct"])
    ck("spec article, fig4 net positive at every depth tried", "1",
       1 if PART["fig4"]["net_positive_everywhere"] else 0)
    kc2 = json.load(open(os.path.join(SP, "kcorrect-45450.json")))
    ck("spec article, fig4 correctness cases", str(len(kc2)),
       PART["fig4"]["correctness"]["cases"])
    ck("spec article, fig4 all deterministic", str(len(kc2)),
       PART["fig4"]["correctness"]["deterministic"])
    ck("spec article, fig4 and within one bf16 ulp", "1",
       1 if PART["fig4"]["correctness"]["max_abs_diff"]
       <= PART["fig4"]["correctness"]["bf16_ulp_at_1"] else 0)

    # --- the same A/B five times, and the probe that predicts which is which --
    SC = PART["fig4"]["campaign"]
    ck("spec article, fig4 the campaign ran the A/B five times", "5", len(SC["pairs"]))
    ck("spec article, and every one of them is eleven rungs", "1",
       1 if all(len(p["rows"]) == 11 for p in SC["pairs"]) else 0)
    _by = {(p["model"], p["attn_backend"]): p for p in SC["pairs"]}
    ck("spec article, gemma-4-31B on Triton gains", "45.6",
       _by[("gemma-4-31B-it", "TRITON_ATTN")]["mean_delta_pct"], tol=0.05)
    ck("spec article, and 1.95x at its deepest", "1.95",
       _by[("gemma-4-31B-it", "TRITON_ATTN")]["ratio_at_deepest"])
    ck("spec article, gemma-4-26B-A4B on Triton gains", "48.4",
       _by[("gemma-4-26B-A4B", "TRITON_ATTN")]["mean_delta_pct"], tol=0.05)
    ck("spec article, and 1.99x at its deepest", "1.99",
       _by[("gemma-4-26B-A4B", "TRITON_ATTN")]["ratio_at_deepest"])
    ck("spec article, Qwen3.8 pinned to Triton gains", "83.9",
       _by[("Qwen3.8-27B", "TRITON_ATTN")]["mean_delta_pct"], tol=0.05)
    ck("spec article, and 2.87x at its deepest", "2.87",
       _by[("Qwen3.8-27B", "TRITON_ATTN")]["ratio_at_deepest"])
    ck("spec article, Qwen3.8 on ROCM_ATTN moves", "-1.9",
       _by[("Qwen3.8-27B", "ROCM_ATTN")]["mean_delta_pct"], tol=0.05)
    ck("spec article, Qwen3.8 on FLASH_ATTN moves", "-0.08",
       _by[("Qwen3.8-27B", "FLASH_ATTN")]["mean_delta_pct"], tol=0.05)
    # the probe's count is the worker count, and its silence is the prediction
    ck("spec article, the probe fired once per worker at TP=1", "1",
       _by[("gemma-4-31B-it", "TRITON_ATTN")]["probe"])
    ck("spec article, and twice at TP=2", "2",
       _by[("Qwen3.8-27B", "TRITON_ATTN")]["probe"])
    ck("spec article, and stayed silent wherever the patch could not act", "0",
       sum(p["probe"] for p in SC["pairs"] if not p["acted"]))
    ck("spec article, the probe and the outcome agree in all five", "1",
       1 if SC["probe_predicts"] else 0)
    ck("spec article, and no rung moves past 8.8% where it is silent", "8.8",
       SC["inert_worst_pct"], tol=0.05)

    # the profiler block: derived from traces that are not here, and it says so
    ck("spec article, the profile declares itself unreproducible", "1",
       0 if PART["profile"].get("reproducible_from_repo", True) else 1)
    trj = json.load(open(os.path.join(SP, "trace-unified-attention.json")))
    ck("spec article, the profile matches its summary file", "2",
       sum(1 for k in ("no-speculation", "mtp")
           if all(PART["profile"]["runs"][k][s] == trj["runs"][k][s]
                  for s in ("calls", "median_us", "p75_us", "max_us", "mean_us"))))
    ck("spec article, the profile's median ratio", "1.2",
       PART["profile"]["ratios"]["median_us"])
    ck("spec article, and its max ratio", "15.4", PART["profile"]["ratios"]["max_us"])
    ck("spec article, the mean is the misleading one", "1",
       1 if PART["profile"]["ratios"]["mean_us"]
       > 5 * PART["profile"]["ratios"]["median_us"] else 0)
    for fn, marker in (("speculative-decoding-net-loss.html",
                        "raw traces not committed"),
                       ("speculative-decoding-net-loss.zh.html",
                        "\u539f\u59cb trace \u672a\u5165\u5e93")):
        ck(f"spec article {fn}, marks the profile block", "1",
           1 if fl(marker) in flat[fn] else 0)

    for fn, heads in (("speculative-decoding-net-loss.html",
                       ("What is not established", "What has changed since")),
                      ("speculative-decoding-net-loss.zh.html",
                       ("没有被确立的部分", "此后发生的变化"))):
        for h in heads:
            ck(f"spec article {fn}, carries '{h[:22]}'", "1",
               1 if fl(h) in flat[fn] else 0)

    # --- a100-vs-two-radeons.html --------------------------------------------
    # Both columns are already checked against their sources above; what this
    # adds is that the article's own figures block still equals them, and the
    # bandwidth utilisation it quotes is recomputed rather than transcribed.
    AART = json.loads(block(pages["a100-vs-two-radeons.html"], "figures"))
    # load the ladders here rather than reuse names bound a thousand lines up,
    # which is how this block first picked up a different `st`
    ADIR = os.path.join(HERE, "..", "speculative-decoding")
    aladder = lambda fn: {r["depth"]: r["tok_per_s"] for r in
                          json.load(open(os.path.join(ADIR, fn)))["rows"]}
    p45 = aladder("mtp-31b-p45450.json")
    s45 = aladder("mtp-31b-stock45450.json")
    ns45 = aladder("splitkv-31b-stock.json")

    ck("a100 article, fig1 rows", "4", len(AART["fig1"]["rows"]))
    A100_LEGS = {1024: "D1K.log", 8192: "D8K.log", 16384: "D16K.log", 32768: "D30.log"}
    ck("a100 article, fig1 rows match both sources", "4",
       sum(1 for r in AART["fig1"]["rows"]
           if abs(r["radeons"] - p45[r["ctx"]]) < 1e-9
           and abs(r["a100"] - leg_result(A100_LEGS[r["ctx"]])) < 1e-9
           and abs(r["advantage"] - r["a100"] / r["radeons"]) < 1e-9))
    ck("a100 article, fig1 the nominal ratio", "1.274",
       AART["fig1"]["nominal_ratio"], tol=1e-3)
    ck("a100 article, fig1 and the ceilings it comes from", "1",
       1 if AART["fig1"]["nominal"]["radeons_gb_s"] == 1600.0
       and AART["fig1"]["nominal"]["a100_gb_s"] == 2039.0 else 0)
    ck("a100 article, fig1 the gap is U-shaped", "1",
       1 if AART["fig1"]["u_shaped"] else 0)
    ck("a100 article, fig1 its minimum", "1.14", AART["fig1"]["min_advantage"])

    # --- the stock ladder, which is what the article's title actually asks ----
    # The 2026-08-29 pair below it is the only session with both speculation
    # arms, and its Radeon side carries three patches the A100 side does not.
    # That flatters the pair on exactly the number the title is about, so the
    # headline is this instead: both arms stock, no patch on either side. Same
    # comparison the repository README publishes, recomputed here from the
    # projection rather than from the figure block it is checking.
    ASK = AART["fig1"]["stock"]
    _sp = {r["ctx"]: r for r in _XD if r["cfg"] == "C-31B-tp2"
           and r["date"] == "2026-07-25" and r["spec"] is None}
    _sa = {r["ctx"]: r for r in _XD if r["cfg"] == "G31"
           and r["date"] == "2026-08-30" and r["spec"] is None}
    ck("a100 article, stock ladder rungs", "11", len(ASK["rows"]))
    ck("a100 article, stock ladder recomputes from decode.jsonl", "11",
       sum(1 for r in ASK["rows"]
           if abs(r["radeons"] - _sp[r["ctx"]]["decode_tok_s"]) < 1e-9
           and abs(r["a100"] - _sa[r["ctx"]]["decode_tok_s"]) < 1e-9
           and abs(r["advantage"] - r["a100"] / r["radeons"]) < 1e-9))
    ck("a100 article, and neither of its arms carries a patch", "0",
       sum(len(_sp[c]["patches"]) + len(_sa[c]["patches"]) for c in _sp if c in _sa))
    ck("a100 article, every cell of it chart-grade", "22",
       sum(1 for c in _sp if c in _sa
           for r in (_sp[c], _sa[c]) if r["chart_grade"]))
    ck("a100 article, stock gap at its narrowest", "1.36", ASK["min"])
    ck("a100 article, and at its widest", "1.44", ASK["max"])
    ck("a100 article, the narrowest is the shallowest rung", "500", ASK["min_at"])
    ck("a100 article, and the widest the deepest", "32000", ASK["max_at"])
    ck("a100 article, so the stock ladder has no interior minimum", "0",
       1 if ASK["u_shaped"] else 0)
    ck("a100 article, and the A100 leads at every rung of it", "1",
       1 if ASK["always_ahead"] else 0)

    # the asymmetry the page went a month without disclosing
    _pm = AART["fig1"]["campaign"]["patch_mismatch"]
    ck("a100 article, patches on the 2026-08-29 pair, no speculation", "3",
       _pm["nospec"][0])
    ck("a100 article, and on the A100 beside it", "0", _pm["nospec"][1])
    ck("a100 article, patches on the speculative pair", "3", _pm["mtp"][0])
    ck("a100 article, and on the speculative A100", "1", _pm["mtp"][1])
    # but the speculation itself is like for like, asserted from the serve logs
    ck("a100 article, both sides speculated the same way", "1",
       1 if AART["fig1"]["campaign"]["speculation"]["same"] else 0)
    ck("a100 article, and it was mtp at k=3", "3",
       AART["fig1"]["campaign"]["speculation"]["a100"]["k"])

    # The numbers are pulled OUT OF THE SENTENCE that makes the claim, not
    # merely looked for somewhere on the page: 1.36× also appears in the figure
    # note and the changelog entry, so a presence check passes while the claim
    # itself says something else. Editing the sentence alone passed, once.
    for _lang, _fn, _re, _pw in (
            ("EN", "a100-vs-two-radeons.html",
             r"the A100 leads ([\d.]+)× at 500 tokens and ([\d.]+)× at 32K",
             "three patches the A100 side does not"),
            ("ZH", "a100-vs-two-radeons.zh.html",
             r"A100 在 500 档领先\s*([\d.]+)×，到 32K 是 ([\d.]+)×",
             "三个 A100 一侧没有的补丁")):
        _m = re.search(_re, flat[_fn])
        ck("a100 article %s, states the stock gap in words" % _lang, "1",
           1 if _m else 0)
        ck("a100 article %s, quotes the stock gap floor" % _lang,
           _m.group(1) if _m else "0", ASK["min"])
        ck("a100 article %s, and its ceiling" % _lang,
           _m.group(2) if _m else "0", ASK["max"])
        ck("a100 article %s, and discloses the patch asymmetry" % _lang, "1",
           1 if fl(_pw) in flat[_fn] else 0)

    # --- the 2026-08-29 ladder drawn beside it, and the prose that reads it ---
    # Off by default in the page, so nothing here is about the default view; it
    # is about the second measurement being what the paragraph says it is.
    AC = AART["fig1"]["campaign"]
    ck("a100 article, the 2026-08-29 ladder is eleven matched rungs", "11",
       len(AC["rows"]))
    ck("a100 article, and every cell of it has two rounds", "1",
       1 if all(r[k]["runs"] == 2 for r in AC["rows"]
                for k in ("radeons_nospec", "radeons_mtp",
                          "a100_nospec", "a100_mtp")) else 0)
    ck("a100 article, without speculation the A100 leads at every depth", "1",
       1 if AC["nospec_always_ahead"] else 0)
    ck("a100 article, and by 1.30x at least", "1.30", AC["nospec_min"])
    ck("a100 article, and 1.46x at most", "1.46", AC["nospec_max"])
    ck("a100 article, with MTP it is behind at 500", "0.88", AC["mtp_at_500"])
    ck("a100 article, and only ahead by 1.08x at 32K", "1.08", AC["mtp_at_32k"])
    # the paragraph's claim is that acceptance is not the explanation
    AE = AART["fig1"]["economics"]
    ck("a100 article, the A100 accepts more than the Radeons do", "1",
       1 if AE["a100_accepts_more"] else 0)
    ck("a100 article, Radeon acceptance floor", "1.46", AE["radeons_acceptance_range"][0])
    ck("a100 article, A100 acceptance ceiling", "2.07", AE["a100_acceptance_range"][1])
    # and that the cost is
    ck("a100 article, a speculative step costs the Radeons 1.35x at least",
       "1.35", AE["radeons_step_cost_range"][0])
    ck("a100 article, and 1.39x at most", "1.39", AE["radeons_step_cost_range"][1])
    ck("a100 article, it costs the A100 2.40x at least", "2.40", AE["a100_step_cost_range"][0])
    ck("a100 article, and 2.57x at most", "2.57", AE["a100_step_cost_range"][1])
    ck("a100 article, fig1 and where it sits", "16384", AART["fig1"]["min_at"])
    ck("a100 article, fig1 rungs below the nominal ratio", "2",
       len(AART["fig1"]["below_nominal"]))
    # the one pair that is not matched has to be marked as such
    ck("a100 article, fig1 marks the unmatched pair", "1",
       sum(1 for r in AART["fig1"]["rows"] if not r["matched"]))
    unmatched = [r["ctx"] for r in AART["fig1"]["rows"] if not r["matched"]]
    ck("a100 article, fig1 and it is the longest one", "1",
       1 if unmatched == [32768] else 0)

    # fig2: the 2D retention pair, and the arithmetic behind it
    r2 = {r["who"]: r for r in AART["fig2"]["retention"]}
    ck("a100 article, fig2 Radeons retain", "15.8", r2["radeons"]["pct"])
    ck("a100 article, fig2 the A100 retains", "33.6", r2["a100"]["pct"])
    ck("a100 article, fig2 Radeons recompute from the stock ladder",
       repr(s45[32768] / s45[1024] * 100.0), r2["radeons"]["pct"], tol=1e-12)
    ck("a100 article, fig2 and the A100 from its legs",
       repr(leg_result("C30.log") / leg_result("C1K.log") * 100.0),
       r2["a100"]["pct"], tol=1e-12)
    ck("a100 article, fig2 twice as hard on two cards", "2.1", AART["fig2"]["ratio"])
    ck("a100 article, fig2 the KV heads split in half", "1",
       1 if AART["fig2"]["kv_heads"]["radeons_per_rank"] * 2
       == AART["fig2"]["kv_heads"]["model_total"]
       == AART["fig2"]["kv_heads"]["a100_per_rank"] else 0)

    # fig3: speculation's economics
    e3 = {c["who"]: c for c in AART["fig3"]["cases"]}
    ck("a100 article, fig3 gain here", "7.5", e3["radeons"]["gain_pct"])
    ck("a100 article, fig3 gain there", "39.2", e3["a100"]["gain_pct"])
    ck("a100 article, fig3 the Radeon baseline is the no-speculation ladder",
       repr(ns45[32768]), e3["radeons"]["nospec"], tol=0)
    amat = json.load(open(os.path.join(HERE, "..", "cuda-a100",
                                      "gemma4-mtp-backend-matrix.json")))
    ck("a100 article, fig3 the A100 baseline is the forced-default column",
       repr(amat["decode_tok_s"]["30000"]["triton_forced"]["nospec"]),
       e3["a100"]["nospec"], tol=0)
    ck("a100 article, fig3 the ratio between them", "5.25",
       AART["fig3"]["ratio"])

    # fig4: realized bandwidth, recomputed from the campaign rather than quoted
    GIBGB = 1024 ** 3 / 1e9
    WANT = {"B-8B-tp2": 7.01, "C-31B-tp2": 10.84, "A-12B-tp2": 4.78}
    ck("a100 article, fig4 rows", "3", len(AART["bandwidth"]["rows"]))
    ck("a100 article, fig4 rows recompute from the campaign", "3",
       sum(1 for r in AART["bandwidth"]["rows"]
           if abs(r["tok_s"] - tps(jul, r["cfg"], 500)) < 1e-9
           and r["gib_per_token"] == WANT[r["cfg"]]
           and abs(r["pct"] - WANT[r["cfg"]] * GIBGB * r["tok_s"] / 8.0) < 1e-9))
    ck("a100 article, fig4 the 8B", "74.9",
       next(r["pct"] for r in AART["bandwidth"]["rows"] if r["cfg"] == "B-8B-tp2"))
    ck("a100 article, fig4 the 31B, which is this comparison's model", "62.8",
       AART["bandwidth"]["subject_pct"])
    ck("a100 article, fig4 the 12B", "38.4",
       next(r["pct"] for r in AART["bandwidth"]["rows"] if r["cfg"] == "A-12B-tp2"))

    # --- the three steps -----------------------------------------------------
    # Everything the rewritten article claims about one card, the second card
    # and the two terms of prefill, recomputed from the projections rather than
    # read back out of the figure it feeds.
    ASP = AART["split"]
    ADEC = [json.loads(l) for l in
            open(os.path.join(HERE, "..", "decode.jsonl"), encoding="utf-8")]
    APRE = [json.loads(l) for l in
            open(os.path.join(HERE, "..", "prefill.jsonl"), encoding="utf-8")]
    import build_prefill as _abp
    AFIT = {(f["machine"], f["cfg"], f["date"]): f for f in _abp.fits(APRE)}

    # The rule the article states in its own §1: decode and prefill for a
    # machine and a model come from one session. If that ever stops being true
    # the article is comparing across a session boundary while saying it is not.
    for k, (cfg, date) in ASP["arms"].items():
        mid, model = k.split("|")
        mach = {"one": "RX 7900 XT", "two": "RX 7900 XT",
                "a100": "A100-SXM4-80GB", "l4": "L4"}[mid]
        ck("a100 article, %s %s decode and prefill are one session" % (mid, model), "1",
           1 if any(r["machine"] == mach and r["cfg"] == cfg and r["date"] == date
                    and r["spec"] is None for r in ADEC)
                and (mach, cfg, date) in AFIT else 0)

    for l in ASP["ladders"]:
        want = sorted([r for r in ADEC if r["machine"] == l["machine_name"]
                       and r["cfg"] == l["cfg"] and r["date"] == l["date"]
                       and r["spec"] is None], key=lambda r: r["ctx"])
        tag = "%s %s" % (l["machine"], l["model"])
        ck("a100 article, %s ladder recomputes" % tag, "1",
           1 if len(want) == len(l["points"]) and all(
               a["ctx"] == b["ctx"] and abs(a["decode_tok_s"] - b["tok_s"]) < 1e-9
               for a, b in zip(want, l["points"])) else 0)

    # The headline: what the second card is worth, and that it is worth much
    # more at prefill than at decode. Both halves are pinned, and so is the fact
    # that the decode gain is not monotone -- quoting either end alone would
    # hide that it rises to 1.23x at 6 000 before falling back.
    A12 = next(x for x in ASP["second"] if x["model"] == "gemma-4-12B-it")
    A26 = next(x for x in ASP["second"] if x["model"] == "gemma-4-26B-A4B")
    ck("a100 article, the second card at decode, dense floor", "1.13", A12["decode_min"])
    ck("a100 article, and its ceiling", "1.23", A12["decode_max"])
    ck("a100 article, the second card at decode, MoE floor", "1.11", A26["decode_min"])
    ck("a100 article, and its ceiling", "1.12", A26["decode_max"])
    ck("a100 article, the decode gain is not monotone", "1",
       1 if A12["decode_max"] > A12["decode_at_shortest"]
            and A12["decode_max"] > A12["decode_at_deepest"] else 0)
    ck("a100 article, prefill wall time on the dense model, one card", "40.1",
       A12["wall"]["one_s"])
    ck("a100 article, and on two", "21.5", A12["wall"]["two_s"])
    ck("a100 article, so the second card buys", "1.87", A12["wall"]["gain"])
    ck("a100 article, prefill wall time on the MoE, one card", "6.21", A26["wall"]["one_s"])
    ck("a100 article, and on two", "3.99", A26["wall"]["two_s"])
    ck("a100 article, so the second card buys", "1.56", A26["wall"]["gain"])
    ck("a100 article, the second card is worth more at prefill than at decode", "2",
       sum(1 for x in ASP["second"] if x["wall"]["gain"] > x["decode_max"]))

    # prefill's two terms, and the contrast the article is built on: they come
    # apart by two on the dense model and not at all on the mixture-of-experts
    for t in ASP["terms"]:
        f = AFIT[({"one": "RX 7900 XT", "two": "RX 7900 XT",
                   "a100": "A100-SXM4-80GB", "l4": "L4"}[t["machine"]],) +
                 tuple(ASP["arms"]["%s|%s" % (t["machine"], t["model"])])]
        ref = AFIT[("RX 7900 XT",) + tuple(ASP["arms"]["one|%s" % t["model"]])]
        ck("a100 article, %s %s b against one card" % (t["machine"], t["model"]), "1",
           1 if abs(t["b"] - ref["b_us_tok"] / f["b_us_tok"]) < 1e-9 else 0)
        ck("a100 article, %s %s c against one card" % (t["machine"], t["model"]), "1",
           1 if abs(t["c"] - ref["c_ns_tok2"] / f["c_ns_tok2"]) < 1e-9 else 0)
    _apc = {(r["model"], r["machine"]): r for r in ASP["percard"]}
    ck("a100 article, A100 over one card on b, dense", "3.29",
       _apc[("gemma-4-12B-it", "a100")]["b_ratio"])
    ck("a100 article, and on c", "6.67", _apc[("gemma-4-12B-it", "a100")]["c_ratio"])
    ck("a100 article, so attention is twice compute there", "2.03",
       _apc[("gemma-4-12B-it", "a100")]["terms_separate"])
    ck("a100 article, A100 over one card on b, MoE", "5.75",
       _apc[("gemma-4-26B-A4B", "a100")]["b_ratio"])
    ck("a100 article, and on c", "5.71", _apc[("gemma-4-26B-A4B", "a100")]["c_ratio"])
    ck("a100 article, so the two terms do not separate there", "0.99",
       _apc[("gemma-4-26B-A4B", "a100")]["terms_separate"])
    ck("a100 article, the L4 is slower than one consumer card on b", "0.90",
       _apc[("gemma-4-12B-it", "l4")]["b_ratio"])
    ck("a100 article, and three times better on c", "3.01",
       _apc[("gemma-4-12B-it", "l4")]["c_ratio"])

    # and the claim §2 leads with: a datacentre card that loses at this job
    _al4 = [p["vs_one"] for l in ASP["ladders"] if l["machine"] == "l4"
            for p in l["points"] if p["vs_one"] is not None]
    ck("a100 article, the L4 never reaches one 7900 XT at decode", "0",
       sum(1 for v in _al4 if v >= 1.0))
    ck("a100 article, its floor", "0.54", min(_al4))
    ck("a100 article, and its ceiling", "0.68", max(_al4))
    _aret = {(r["machine"], r["model"]): r["pct"] for r in ASP["retention"]}
    ck("a100 article, what the L4 retains to 32K", "88.8",
       _aret[("l4", "gemma-4-12B-it")])
    ck("a100 article, and what the A100 retains", "62.0",
       _aret[("a100", "gemma-4-12B-it")])
    # b and c point opposite ways on the L4, so putting them back together at a
    # long enough prompt reverses the card that is behind on the linear term
    ck("a100 article, the L4 over one card on a 32K prompt", "1.58",
       ASP["crossover"]["l4_gain"])
    ck("a100 article, and it is behind that card on b", "1",
       1 if _apc[("gemma-4-12B-it", "l4")]["b_ratio"] < 1.0 else 0)
    ck("a100 article, the second card on b, floor", "1.44",
       min(x["b_gain"] for x in ASP["second"]))
    ck("a100 article, and ceiling", "1.48", max(x["b_gain"] for x in ASP["second"]))
    ck("a100 article, the second card on c, floor", "1.91",
       min(x["c_gain"] for x in ASP["second"]))

    # --- the pair at prefill, and the kernel it splits by --------------------
    # Every arm of Figure 4 against the fit build_prefill reports, and the
    # reading the figure exists for: b does not split by which attention kernel
    # the A100 used and c does, cleanly enough that the two groups' ranges do
    # not touch.
    for x in ASP["pair"]:
        r = AFIT[("RX 7900 XT", x["radeon_cfg"], x["radeon_date"])]
        a = AFIT[("A100-SXM4-80GB", x["a100_cfg"], x["a100_date"])]
        ck("a100 article, pair %s recomputes" % x["model"], "1",
           1 if abs(x["b"] - r["b_us_tok"] / a["b_us_tok"]) < 1e-9
                and abs(x["c"] - r["c_ns_tok2"] / a["c_ns_tok2"]) < 1e-9 else 0)
        # the grouping is by a backend read off a serve log, not by a guess
        ck("a100 article, pair %s names the A100's backend" % x["model"], "1",
           1 if x["a100_backend"] in ("TRITON_ATTN", "FLASH_ATTN") else 0)
    APS = ASP["pair_split"]
    ck("a100 article, models where the A100 is on Triton", "3", APS["triton"]["n"])
    ck("a100 article, and where it is on FlashAttention", "2", APS["flash"]["n"])
    ck("a100 article, c on Triton, floor", "2.99", APS["triton"]["c_min"])
    ck("a100 article, and ceiling", "3.18", APS["triton"]["c_max"])
    ck("a100 article, c on FlashAttention, floor", "12.75", APS["flash"]["c_min"])
    ck("a100 article, and ceiling", "19.01", APS["flash"]["c_max"])
    ck("a100 article, the two c groups do not touch", "1",
       1 if APS["c_separates"] else 0)
    ck("a100 article, and the b groups do", "1", 1 if APS["b_overlaps"] else 0)
    ck("a100 article, b across all five, floor", "2.08",
       min(x["b"] for x in ASP["pair"]))
    ck("a100 article, and ceiling", "4.00", max(x["b"] for x in ASP["pair"]))

    # one machine, one day, one model, one flag -- which is what turns the five
    # models above from a correlation into a demonstration
    AFG = ASP["flag"]
    ck("a100 article, Qwen3.8 on its default backend, c", "3.43", AFG["default_c"])
    ck("a100 article, and the A100 ahead by", "2.37", AFG["default_ratio"])
    ck("a100 article, pinned to Triton, c", "18.44", AFG["pinned_c"])
    ck("a100 article, and the A100 ahead by", "12.75", AFG["pinned_ratio"])
    ck("a100 article, so one flag moves the answer by", "5.38", AFG["swing"])
    ck("a100 article, and the two arms are the same day", "1",
       1 if AFG["date"] == next(x["radeon_date"] for x in ASP["pair"]
                                if x["model"] == AFG["model"]) else 0)

    # what section 1 owes the reader for pairing two A100 sessions
    ASS = ASP["sessions"]
    ck("a100 article, the two A100 sessions past 2K", "0.4", ASS["deep_worst"])
    ck("a100 article, and at the two shallowest rungs", "5.5", ASS["shallow_worst"])
    ck("a100 article, and ceiling", "2.22", max(x["c_gain"] for x in ASP["second"]))
    # the README quoted the 12B's number for the 31B's comparison until 2026-08-29
    ck("README, the utilisation it cites is this comparison's model", "1",
       1 if "reach 63 % of their 800 GB/s" in rm else 0)
    ck("README, and the correction says which number it replaced", "1",
       1 if "this cited 38 %, which is the 12B" in rm else 0)

    for fn, heads in (("a100-vs-two-radeons.html",
                       ("What is not established", "What has changed since")),
                      ("a100-vs-two-radeons.zh.html",
                       ("没有被确立的部分", "此后发生的变化"))):
        for h in heads:
            ck(f"a100 article {fn}, carries '{h[:22]}'", "1",
               1 if fl(h) in flat[fn] else 0)
    # the figure whose counterpart was never measured has to say so
    for fn, phrase in (("a100-vs-two-radeons.html",
                        "No equivalent figure was measured on the\n    A100"),
                       ("a100-vs-two-radeons.zh.html",
                        "A100 那边没有测过对应的数字")):
        ck(f"a100 article {fn}, says the A100 side of fig4 is missing", "1",
           1 if phrase.replace("\n    ", " ") in " ".join(pages[fn].split()) else 0)

    # --- gqa-gate-costs-nothing.html -----------------------------------------
    # The argument is that no cell is an exception, so the check is per cell
    # rather than against the summary ranges the prose quotes.
    QART = json.loads(block(pages["gqa-gate-costs-nothing.html"], "figures"))
    QDIR = os.path.join(HERE, "..", "vllm-50603")
    qjl = lambda fn: [json.loads(l) for l in open(os.path.join(QDIR, fn))]
    QSRC = {"023": qjl("stage1-rocm-paths.jsonl"), "027": qjl("stage1-027-r1.jsonl")}

    ck("gqa article, fig1 versions", "2", len(QART["fig1"]["versions"]))
    ck("gqa article, fig1 cells drawn", "60", QART["fig1"]["cells"])
    matched = 0
    for v in QART["fig1"]["versions"]:
        by = {(r["num_heads"], r["num_kv_heads"], r["ctx_len"]): r for r in QSRC[v["id"]]}
        for row in v["rows"]:
            h, kv = (int(x) for x in row["shape"].split("/"))
            for c in row["cells"]:
                s = by[(h, kv, c["ctx"])]
                if (abs(c["ratio"] - s["triton"]["median_ms"] / s["ck"]["median_ms"]) < 1e-12
                        and c["triton_ms"] == s["triton"]["median_ms"]
                        and c["ck_ms"] == s["ck"]["median_ms"]
                        and row["admitted"] == s["gate_as_shipped"]
                        and row["gqa"] == s["gqa_ratio"]):
                    matched += 1
    ck("gqa article, fig1 every cell matches its source", "60", matched)
    ck("gqa article, fig1 no cell has the custom kernel slower", "60",
       QART["fig1"]["never_slower"])
    ck("gqa article, fig1 the floor across both versions", "1.70", QART["fig1"]["min"])
    ck("gqa article, fig1 and the ceiling", "7.40", QART["fig1"]["max"])
    for v in QART["fig1"]["versions"]:
        ck(f'gqa article, fig1 {v["id"]} bands overlap', "1",
           1 if v["bands_overlap"] else 0)
        # the bands are quoted over every round measured on that version
        ck(f'gqa article, fig1 {v["id"]} rounds pooled', "1" if v["id"] == "023" else "2",
           v["rounds"])
    b23 = [v for v in QART["fig1"]["versions"] if v["id"] == "023"][0]
    b27 = [v for v in QART["fig1"]["versions"] if v["id"] == "027"][0]
    ck("gqa article, fig1 0.23.1 excluded floor", "1.84", b23["excluded_band"][0])
    ck("gqa article, fig1 0.23.1 excluded ceiling", "7.28", b23["excluded_band"][1])
    ck("gqa article, fig1 0.23.1 admitted floor", "2.35", b23["admitted_band"][0])
    ck("gqa article, fig1 0.27 excluded floor", "1.70", b27["excluded_band"][0])
    ck("gqa article, fig1 0.27 admitted floor", "2.20", b27["admitted_band"][0])
    ck("gqa article, fig1 round-to-round spread, worst", "5.8",
       QART["fig1"]["round_spread_pct"]["max"])
    ck("gqa article, fig1 and median", "2.3", QART["fig1"]["round_spread_pct"]["median"])
    # the shape set has to include the one that is a real model's
    ck("gqa article, fig1 includes gemma-3-27b's own shape", "1",
       sum(1 for r in QART["fig1"]["versions"][0]["rows"] if r["shape"] == "32/16"))

    # fig2: the flat band, its full extent, and the control under it
    s1q = QSRC["023"]
    cuq = qjl("stage2-cuda-control.jsonl")
    allq = ([r["triton"]["max_rel_err"] for r in s1q]
            + [r["ck"]["max_rel_err"] for r in s1q]
            + [r["triton"]["max_rel_err"] for r in cuq])
    ck("gqa article, fig2 cells", str(len(allq)), QART["fig2"]["cells"])
    ck("gqa article, fig2 band floor", repr(min(allq)), QART["fig2"]["band"][0], tol=1e-12)
    ck("gqa article, fig2 band ceiling", repr(max(allq)), QART["fig2"]["band"][1], tol=1e-12)
    ck("gqa article, fig2 median", repr(med(allq)), QART["fig2"]["median"], tol=1e-12)
    ck("gqa article, fig2 the bucketed span the source quotes", "16",
       QART["fig2"]["bucket_span"], tol=1e-3)
    ck("gqa article, fig2 and what it moves the median by", "1.06",
       QART["fig2"]["bucket_move"])
    # recomputed here rather than reusing `lo`/`hi` from the 50603 block: both
    # names are rebound to floats further down this function
    qbucket = lambda cs: med(
        [r[a]["max_rel_err"] for r in s1q for a in ("triton", "ck") if r["ctx_len"] in cs]
        + [r["triton"]["max_rel_err"] for r in cuq if r["ctx_len"] in cs])
    ck("gqa article, fig2 that is the same number the directory quotes",
       repr(qbucket([16384, 32768]) / qbucket([1024, 2048])),
       QART["fig2"]["bucket_move"], tol=1e-12)
    ck("gqa article, fig2 the context span it is quoted over", "32",
       QART["fig2"]["context_span"])
    ck("gqa article, fig2 what that span moves the median by", "1.03",
       QART["fig2"]["context_move"])
    ck("gqa article, fig2 and the widest gap between any two depths", "1.09",
       QART["fig2"]["context_worst_ratio"])
    ck("gqa article, fig2 the band is flat by any reading", "1",
       1 if QART["fig2"]["context_worst_ratio"] < 1.2 else 0)
    ck("gqa article, fig2 three series", "3", len(QART["fig2"]["series"]))
    ck("gqa article, fig2 the CUDA control is one of them", "1",
       sum(1 for s in QART["fig2"]["series"] if s["id"] == "a100_triton"))
    # the positive control is what makes a flat band mean anything
    C = QART["fig2"]["control"]
    ctrlq = qjl("stage1b-tail-control.jsonl")
    ck("gqa article, fig2 control rows", str(len(ctrlq)), C["rows"])
    ck("gqa article, fig2 the harness does see corruption", "1",
       1 if C["poisoned"] > 0 else 0)
    ck("gqa article, fig2 and only where the tile straddles", "1",
       1 if C["all_poisoned_straddle"] and C["aligned_all_clean"] else 0)
    ck("gqa article, fig2 both paths poison alike on this version", "1",
       1 if C["ck_matches_triton"] else 0)
    ck("gqa article, fig2 finite garbage does not poison", "1",
       1 if C["garbage_clean"] else 0)
    ck("gqa article, fig2 control rows match their file", str(len(ctrlq)),
       sum(1 for a, b in zip(C["cases"], ctrlq)
           if a["ctx"] == b["ctx_len"] and a["fill"] == b["fill"]
           and a["triton_finite"] == b["triton"]["all_finite"]
           and a["ck_finite"] == b["ck"]["all_finite"]))

    # fig3: end to end, three passes
    E2E = {"023": "stage3-endtoend.jsonl", "027A": "stage3-027.jsonl",
           "027B": "stage3-027b.jsonl"}
    ck("gqa article, fig3 runs", "3", len(QART["fig3"]["runs"]))
    ok3 = 0
    for k, fn in E2E.items():
        src = {(r["arm"], r["ctx"]): r for r in qjl(fn)}
        for c in QART["fig3"]["depths"]:
            r = QART["fig3"]["runs"][k][str(c)] if str(c) in QART["fig3"]["runs"][k] \
                else QART["fig3"]["runs"][k][c]
            if (abs(r["stock"] - src[("stock", c)]["decode_tok_s"]) < 1e-9
                    and abs(r["widened"] - src[("widened", c)]["decode_tok_s"]) < 1e-9
                    and abs(r["ratio"] - r["widened"] / r["stock"]) < 1e-12):
                ok3 += 1
    ck("gqa article, fig3 every cell matches its file", "9", ok3)
    ck("gqa article, fig3 the gate flipped in every cell", "1",
       1 if QART["fig3"]["gate_flipped_everywhere"] else 0)
    ck("gqa article, fig3 the gain grows with context", "1",
       1 if QART["fig3"]["grows_with_context"] else 0)
    ck("gqa article, fig3 worst spread between the two 0.27 passes", "2.8",
       QART["fig3"]["worst_pass_spread_pct"])
    pooled = QART["fig3"]["pooled_027"]
    ck("gqa article, fig3 pooled 32K", "1.167", pooled.get("32768", pooled.get(32768)))
    ck("gqa article, fig3 pooled 1K", "1.024", pooled.get("1024", pooled.get(1024)))
    # the effect at 1K is small; it has to be larger than the repeatability
    ck("gqa article, fig3 and 1K is still bigger than the pass spread", "1",
       1 if (pooled.get("1024", pooled.get(1024)) - 1) * 100
       > QART["fig3"]["worst_pass_spread_pct"] / 3 else 0)

    # fig4: the fault widening the gate would admit
    stq, paq = qjl("53856-027-stock.jsonl"), qjl("53856-027-patched.jsonl")
    ck("gqa article, fig4 rows per arm", str(len(stq)), QART["fig4"]["rows_per_arm"])
    ck("gqa article, fig4 poisoned on the stock arm", "4",
       QART["fig4"]["poisoned_stock"])
    ck("gqa article, fig4 recomputes that count",
       str(sum(1 for r in stq if not r["as_shipped"]["all_finite"])),
       QART["fig4"]["poisoned_stock"])
    ck("gqa article, fig4 all of them fixed", "4", QART["fig4"]["fixed"])
    ck("gqa article, fig4 none newly broken", "0", QART["fig4"]["newly_broken"])
    ck("gqa article, fig4 forced past the gate it is eight", "8",
       QART["fig4"]["poisoned_forced"])
    ck("gqa article, fig4 and all eight are fixed", "8", QART["fig4"]["fixed_forced"])
    ck("gqa article, fig4 K alone never poisons", "0",
       QART["fig4"]["k_only_ever_poisons"])
    ck("gqa article, fig4 only straddling lengths poison", "1",
       1 if QART["fig4"]["all_poisoned_straddle"] else 0)
    ck("gqa article, fig4 the custom kernel really ran", str(QART["fig4"]["ck_expected"]),
       QART["fig4"]["ck_ran"])
    ck("gqa article, fig4 Triton is clean throughout", str(len(stq)),
       QART["fig4"]["triton_clean"])

    # the source directory's two corrections must stay stated
    gq = open(os.path.join(QDIR, "README.md"), encoding="utf-8").read()
    ck("50603 README, the speedup-floor correction is stated", "1",
       1 if "first said 2.06x, which was the minimum" in gq else 0)
    ck("50603 README, and the derivation of the 1.06x", "1",
       1 if "whose centres are" in gq and "about 16x apart" in gq else 0)
    ck("50603 README, which states both readings", "1",
       1 if "pooled median moves 1.03x" in " ".join(gq.split()) else 0)

    for fn, heads in (("gqa-gate-costs-nothing.html",
                       ("What is not established", "What has changed since")),
                      ("gqa-gate-costs-nothing.zh.html",
                       ("没有被确立的部分", "此后发生的变化"))):
        for h in heads:
            ck(f"gqa article {fn}, carries '{h[:22]}'", "1",
               1 if fl(h) in flat[fn] else 0)

    # --- reporting-a-non-reproduction.html -----------------------------------
    # The tallies are already checked against the logs above; what this adds is
    # that the article's figures block still equals them, that the machine
    # contrast still matches the table it was extracted from, and that the
    # honesty the article is about is present on both language pages.
    NART = json.loads(block(pages["reporting-a-non-reproduction.html"], "figures"))
    NDIR = os.path.join(HERE, "..", "rccl-6565")
    nread = lambda *q: open(os.path.join(NDIR, *q), errors="replace").read()
    NRE = r"=== arm=(\S+) RESULT pass=(\d+) fail=(\d+)(?: error=(\d+))? of (\d+)"
    ntally = lambda txt: [{"arm": m.group(1), "passed": int(m.group(2)),
                           "failed": int(m.group(3)), "error": int(m.group(4) or 0),
                           "n": int(m.group(5))} for m in re.finditer(NRE, txt)]
    nfirst = ntally(nread("logs", "stage1.log")) + ntally(nread("logs", "stage2a.log"))
    nthird = ntally(nread("logs", "stage3-allranks.log"))
    nby3 = {a["arm"]: a for a in nthird}

    ck("6565 article, fig1 arms", "8", len(NART["fig1"]["arms"]))
    ck("6565 article, fig1 arms match the logs", "8",
       sum(1 for a, s in zip(NART["fig1"]["arms"], nfirst)
           if a["arm"] == s["arm"] and a["passed"] == s["passed"]
           and a["n"] == s["n"] and a["failed"] == s["failed"]
           and a["cross"]["passed"] == nby3[a["arm"]]["passed"]
           and a["cross"]["n"] == nby3[a["arm"]]["n"]))
    ck("6565 article, fig1 every arm names its environment", "8",
       sum(1 for a in NART["fig1"]["arms"] if a["env"]))
    ck("6565 article, fig1 the channel sweep is present", "4",
       sum(1 for a in NART["fig1"]["arms"] if a["arm"].startswith("ch")))
    ck("6565 article, fig1 and so is the transport arm", "1",
       sum(1 for a in NART["fig1"]["arms"] if a["arm"] == "shmoff"))
    for s in NART["fig1"]["sweeps"]:
        src = nfirst if s["id"] == "rank0" else nthird
        ck(f'6565 article, fig1 {s["id"]} total', "135", s["total"])
        ck(f'6565 article, fig1 {s["id"]} recomputes from the logs',
           str(sum(a["n"] for a in src)), s["total"])
        ck(f'6565 article, fig1 {s["id"]} all correct', str(s["total"]), s["passed"])
        ck(f'6565 article, fig1 {s["id"]} no failures', "0", s["failed"])
    ck("6565 article, fig1 the second sweep repeats the first's arms and counts", "1",
       1 if NART["fig1"]["same_arms_same_counts"] else 0)
    ck("6565 article, fig1 the box builds two channels", "2",
       NART["fig1"]["default_channels"])
    ck("6565 article, fig1 the reporter's script is verbatim", "1",
       1 if NART["fig1"]["verbatim"] else 0)
    ck("6565 article, fig1 and quotes the md5 the directory does", "1",
       1 if NART["fig1"]["reporter_md5"] ==
       hashlib.md5(open(os.path.join(NDIR, "rccl_allgather_truth.py"), "rb").read()
                   ).hexdigest() else 0)

    # fig2: extracted from the directory's own table, so it cannot drift
    nmd = nread("README.md")
    body = nmd.split("## The machine, against theirs", 1)[1]
    nrows = []
    for line in body.split("\n"):
        if line.startswith("|") and not re.match(r"^\|[\s|:-]+\|$", line):
            nrows.append([c.strip() for c in line.strip().strip("|").split("|")])
        elif nrows and not line.startswith("|"):
            break
    nstrip = lambda s: re.sub(r"\s+", " ", re.sub(r"[*`]", "", s)).strip()
    ck("6565 article, fig2 rows", str(len(nrows) - 1), len(NART["fig2"]["rows"]))
    ck("6565 article, fig2 rows match the table", str(len(nrows) - 1),
       sum(1 for a, s in zip(NART["fig2"]["rows"], nrows[1:])
           if a["axis"] == nstrip(s[0]) and a["theirs"] == nstrip(s[1])
           and a["ours"] == nstrip(s[2])))
    # the rows that AGREE are what stop the negative being explained away
    ck("6565 article, fig2 marks the axes that agree", "2", NART["fig2"]["same_rows"])
    ck("6565 article, fig2 and that count is the rows themselves",
       str(sum(1 for r in NART["fig2"]["rows"] if r["same"])),
       NART["fig2"]["same_rows"])
    ck("6565 article, fig2 atomics is one of them", "1",
       sum(1 for r in NART["fig2"]["rows"] if "atomic" in r["axis"].lower() and r["same"]))
    nenv = nread("logs", "environment.txt")
    ck("6565 article, fig2 zero atomic complaints", "0",
       NART["fig2"]["atomic_complaints"])
    ck("6565 article, fig2 recomputes that from the fingerprint",
       re.search(r"PCIE-atomic complaints: (\d+)", nenv).group(1),
       NART["fig2"]["atomic_complaints"])
    ck("6565 article, fig2 both GPUs advertise ReqEn+", "2", NART["fig2"]["reqen"])

    # fig3: the injected one-sided fault, which is the article's centre
    nbs = nread("logs", "blindspot-check.log")
    ck("6565 article, fig3 the injection check passed", "1",
       1 if NART["fig3"]["ok"] and "BLINDSPOT_CHECK_OK" in nbs else 0)
    ck("6565 article, fig3 both scripts were injected", "2", NART["fig3"]["injections"])
    ck("6565 article, fig3 the reporter's script called it correct", "1",
       1 if NART["fig3"]["reporter_said_all_correct"] else 0)
    ck("6565 article, fig3 the variant counted it", "1",
       1 if NART["fig3"]["variant_counted_it"] else 0)
    ck("6565 article, fig3 and the runner exited non-zero", "1",
       NART["fig3"]["runner_exit"])
    ck("6565 article, fig3 exactly one of the three is blind", "1",
       sum(1 for r in NART["fig3"]["rows"] if not r["saw_it"]))
    blind = [r["who"] for r in NART["fig3"]["rows"] if not r["saw_it"]]
    ck("6565 article, fig3 and it is the reporter's", "1",
       1 if blind == ["reporter"] else 0)
    # the injection must not have touched the original
    ck("6565 article, fig3 the reporter's script still hashes the same", "1",
       1 if NART["fig3"]["md5_after"] == NART["fig1"]["reporter_md5"] else 0)
    # the miscounting check, kept because it is the same class of mistake
    ck("6565 article, fig3 records the runner's own counting defect", "1",
       1 if NART["fig3"]["grep_defect"]["reported_one_sided"] == 6
       and NART["fig3"]["grep_defect"]["of"] == 20 else 0)

    for fn, heads, phrase in (
            ("reporting-a-non-reproduction.html",
             ("What is not established", "What has changed since"),
             "no corruption observed on rank 0"),
            ("reporting-a-non-reproduction.zh.html",
             ("没有被确立的部分", "此后发生的变化"),
             "在 rank 0 上没有观察到损坏")):
        for h in heads:
            ck(f"6565 article {fn}, carries '{h[:22]}'", "1",
               1 if fl(h) in flat[fn] else 0)
        ck(f"6565 article {fn}, states what the first sweep really showed", "1",
           1 if phrase in " ".join(pages[fn].split()) else 0)

    # --- measuring-decode.html -----------------------------------------------
    # The methodology article. Its figures are the rules the rest of the
    # repository is measured under, so they are pinned to the files that set
    # them rather than to the prose that describes them.
    XART = json.loads(block(pages["measuring-decode.html"], "figures"))
    XDIR = os.path.join(HERE, "..", "harness-calibration")
    xcal = {}
    for i in (1, 2, 3, 4):
        for line in open(os.path.join(XDIR, f"harness-cal-r{i}.jsonl")):
            r = json.loads(line)
            xcal[(i, r["campaign_target"])] = r
    xaug = [json.loads(l) for l in
            open(os.path.join(HERE, "..", "results-2026-08-24.jsonl"))]
    xcamp = {t_: [r["decode_tps"] for r in xaug if r.get("kind") == "decode"
                  and r.get("cfg") == "B-8B-tp2" and r.get("target") == t_]
             for t_ in (500, 8000, 32000)}
    xconv = lambda t_, k: (xcal[(3, t_)][k] + xcal[(4, t_)][k]) / 2

    ck("measure article, fig1 rows", "3", len(XART["fig1"]["rows"]))
    # section 2 said "the campaigns already run two rounds per cell which agree
    # to 0.2 %" until 2026-08-29. True of the calibration's own model, and not of
    # the campaigns at large, so the sentence is scoped and the bound is pinned.
    _q8 = [r["range_pct"] for r in led
           if "Qwen3-8B" in r.get("model", "") and r.get("range_pct") is not None]
    ck("measure article, the calibration model's worst spread", "0.2", max(_q8))
    ck("measure article, and the ledger at large is wider than that", "1",
       1 if sum(1 for r in led if (r.get("range_pct") or 0) > 1.0) > 0 else 0)
    ck("measure article, fig1 rows match the calibration", "3",
       sum(1 for r in XART["fig1"]["rows"]
           if abs(r["campaign"] - sum(xcamp[r["ctx"]]) / len(xcamp[r["ctx"]])) < 1e-9
           and abs(r["probe_64"] - xconv(r["ctx"], "tps_64")) < 1e-9
           and abs(r["probe_512"] - xconv(r["ctx"], "tps_512")) < 1e-9))
    ck("measure article, fig1 the two harnesses agree to", "0.44",
       XART["fig1"]["worst_delta_pct"])
    ck("measure article, fig1 and matching the span does not change that", "0.97",
       XART["fig1"]["worst_delta_512_pct"])
    # the depths have to be the campaign's own reported prompt lengths, or the
    # two sides are not at the same context
    ck("measure article, fig1 every depth is matched", "1",
       1 if XART["fig1"]["all_depths_matched"] else 0)
    ck("measure article, fig1 recomputes that from the probe", "3",
       sum(1 for r in XART["fig1"]["rows"]
           if xcal[(3, r["ctx"])]["prompt_tokens_got"]
           == xcal[(3, r["ctx"])]["prompt_tokens_wanted"]))
    ck("measure article, fig1 states why this model", "4",
       len(XART["fig1"]["why_this_model"]))

    ck("measure article, fig2 rows", "3", len(XART["fig2"]["rows"]))
    ck("measure article, fig2 four runs each", "3",
       sum(1 for r in XART["fig2"]["rows"] if len(r["runs"]) == 4))
    ck("measure article, fig2 runs match the files", "12",
       sum(1 for r in XART["fig2"]["rows"] for i, v in enumerate(r["runs"])
           if abs(v - xcal[(i + 1, r["ctx"])]["tps_64"]) < 1e-9))
    ck("measure article, fig2 the first run's worst", "-30.7",
       XART["fig2"]["first_run_worst_pct"])
    ck("measure article, fig2 the second is still", "-6.2",
       XART["fig2"]["second_run_worst_pct"])
    ck("measure article, fig2 the converged pair agrees to at least", "0.07",
       XART["fig2"]["converged_spread_pct"]["min"])
    ck("measure article, fig2 and at most", "0.36",
       XART["fig2"]["converged_spread_pct"]["max"])
    # the shape is the claim: the deficit is worst where the machine is fastest
    ck("measure article, fig2 the deficit is worst at the shortest depth", "500",
       XART["fig2"]["worst_at"])

    # fig3: the cross-campaign control
    ck("measure article, fig3 controls", "6", len(XART["fig3"]["controls"]))
    # `aug` is rebound to a list of raw rows further up this function, so the
    # two decode projections are rebuilt here rather than reused
    xjul, xaugd = decode(JULY), decode(AUG)
    ck("measure article, fig3 offsets recompute", "6",
       sum(1 for c in XART["fig3"]["controls"]
           if abs(c["offset_pct"] - offset(xjul, xaugd, c["cfg"])) < 1e-9))
    ck("measure article, fig3 inside the band", "4", XART["fig3"]["within_band"])
    ck("measure article, fig3 and the band is", "0.25", XART["fig3"]["band_pct"])
    ck("measure article, fig3 two are outside", "2", len(XART["fig3"]["outside"]))

    # fig4: the ledger's own spread distribution and where it cuts
    xled = [json.loads(l) for l in open(os.path.join(HERE, "..", "ledger.jsonl"))]
    xspread = sorted(r["range_pct"] for r in xled if r["range_pct"] is not None)
    ck("measure article, fig4 ledger rows", str(len(xled)), XART["fig4"]["rows"])
    ck("measure article, fig4 points with a range", str(len(xspread)),
       XART["fig4"]["with_range"])
    ck("measure article, fig4 the spread list is the ledger's", "1",
       1 if XART["fig4"]["spreads"] == xspread else 0)
    ck("measure article, fig4 the cut is build_ledger's", str(build_ledger.RANGE_CUT),
       XART["fig4"]["cut"])
    ck("measure article, fig4 median", "0.17", XART["fig4"]["median"])
    ck("measure article, fig4 p95", "13.93", XART["fig4"]["p95"])
    ck("measure article, fig4 points above the cut", "29", len(XART["fig4"]["above_cut"]))
    ck("measure article, fig4 which is the ungraded count", "29", XART["fig4"]["ungraded"])
    ck("measure article, fig4 no ledger row is a single run", "0",
       XART["fig4"]["single_run"])

    # --- the 2026-08-29 campaign as this article's own case study ------------
    MC = XART["fig4"]["campaign"]
    ck("measure article, the campaign is eight arms", "8", len(MC["arms"]))
    ck("measure article, and every arm is eleven rungs", "1",
       1 if all(a["rungs"] == 11 for a in MC["arms"]) else 0)
    ck("measure article, three of them do not speculate", "3", MC["nospec_arms"])
    ck("measure article, and none of their 33 rungs is above the cut", "0",
       MC["nospec_ungraded"])
    ck("measure article, their worst range", "0.59", MC["nospec_worst_range_pct"],
       tol=0.005)
    ck("measure article, five of them do", "5", MC["spec_arms"])
    ck("measure article, and 28 of their 55 rungs are above it", "28",
       MC["spec_ungraded"])
    # the sentence that stops this being a claim about speculation as such
    ck("measure article, one speculative arm is graded throughout", "1",
       len(MC["spec_arms_fully_graded"]))
    ck("measure article, and every ungraded rung is one model's", "1",
       1 if MC["ungraded_all_one_model"] else 0)
    # the tail is what justifies a cut at 6 rather than a convention, and
    # build_ledger's comment now describes it; both have to stay true
    ck("measure article, fig4 the tail above 2.5%", "43", len(XART["fig4"]["tail"]))
    # The cut has to sit inside the widest break in the tail, not merely above
    # the quiet part: that is what "chosen from the distribution rather than
    # picked" means, and it is the property that went stale when this
    # campaign's speculative arms landed. Stated as the gap itself -- nothing
    # within a point below the cut, nothing within a point above it -- so it
    # survives the next campaign moving the numbers without surviving the cut
    # drifting out of the gap again.
    _below = [v for v in XART["fig4"]["tail"] if v < XART["fig4"]["cut"]]
    _above = XART["fig4"]["above_cut"]
    ck("measure article, fig4 the gap the cut sits in", "1",
       1 if _below and _above
       and XART["fig4"]["cut"] - max(_below) > 1.0
       and min(_above) - XART["fig4"]["cut"] > 1.0 else 0)
    bl = open(os.path.join(HERE, "build_ledger.py"), encoding="utf-8").read()
    ck("build_ledger, its comment describes the current distribution", "1",
       1 if "5.30, 5.33, 5.97, 6.02, 6.10, and then jump to 9.50" in bl else 0)
    ck("build_ledger, and says which gap the cut sits in", "1",
       1 if "6.10 -> 9.50" in bl else 0)

    # fig5: why the range rather than a standard deviation
    B5 = XART["fig4"]["bimodal"]
    ck("measure article, fig5 the cell has four runs", "4", B5["runs"])
    ck("measure article, fig5 range at two", "15.79", B5["range_at_two"])
    ck("measure article, fig5 range at four", "16.79", B5["range_at_four"])
    ck("measure article, fig5 the range widened", "1",
       1 if B5["widened_by_adding_runs"] else 0)
    ck("measure article, fig5 the standard deviation narrowed", "1",
       1 if B5["stdev_at_four"] < B5["stdev_at_two"] else 0)
    ck("measure article, fig5 stdev at two", "4.87", B5["stdev_at_two"])
    ck("measure article, fig5 stdev at four", "3.61", B5["stdev_at_four"])
    # the run order is not recoverable from the ledger, so it has to come from
    # the files that produced the four runs
    HD = os.path.join(HERE, "..", "hybrid-splitkv-027")
    cell = []
    for fn in ("qwen38-027-depth.jsonl", "qwen38-027-depth-b.jsonl",
               "qwen38-8k-r3r4.jsonl"):
        cell += [r["decode_tok_s"] for r in
                 (json.loads(l) for l in open(os.path.join(HD, fn)))
                 if r.get("ctx") == 8192 and r.get("arm") == "splitkv"]
    ck("measure article, fig5 the values are in run order", "1",
       1 if B5["in_run_order"] == cell else 0)
    ck("measure article, fig5 and sorted they are the ledger's", "1",
       1 if sorted(B5["in_run_order"]) == B5["values"] else 0)

    # fig6: why token comparison is not the correctness test here
    xnd = json.load(open(os.path.join(HERE, "..",
                                      "gfx1100-greedy-nondeterminism.json")))
    N6 = XART["fig5"]
    ck("measure article, fig6 cells", str(xnd["result"]["cells"]), N6["cells"])
    ck("measure article, fig6 varying",
       str(xnd["result"]["cells_with_more_than_one_output"]), N6["varying"])
    ck("measure article, fig6 symmetric between kernel states", "1",
       1 if N6["by_state"]["before"] == N6["by_state"]["after"] else 0)
    ck("measure article, fig6 within-process cells",
       str(len(xnd["within_process"]["cells"])), N6["within_process"])
    ck("measure article, fig6 and most of them vary too", "12",
       N6["within_process_varying"])
    ck("measure article, fig6 the worst cell", "8", N6["worst_distinct_of_8"])

    # --- the KV-depth run for vllm#52684, 2026-08-31 ------------------------
    # The claim is a sign, not a magnitude, so the sign is what is gated: every
    # short-q cell at a fixed KV depth is below 1, and the crossover the author
    # sees on gfx1100 does not appear on this card. Each median is recomputed
    # from the rows rather than read off the summary.
    XKV = json.load(open(os.path.join(HERE, "..", "cuda-a100", "52684-kv-depth",
                                      "kv-depth-summary.json")))
    _kvp = {}
    for _f, _tag in (("kv_depth.jsonl", 1), ("kv_depth2.jsonl", 2)):
        for _l in open(os.path.join(HERE, "..", "cuda-a100", "52684-kv-depth", _f)):
            _r = json.loads(_l)
            if "bm64_speedup" in _r:
                _kvp.setdefault((_tag, _r["kv_mode"], _r["q_len"]), []).append(
                    _r["bm64_speedup"])
    ck("52684 kv-depth, pass 1 rows", "48",
       sum(1 for _l in open(os.path.join(HERE, "..", "cuda-a100", "52684-kv-depth",
                                         "kv_depth.jsonl"))))
    ck("52684 kv-depth, pass 2 rows", "28",
       sum(1 for _l in open(os.path.join(HERE, "..", "cuda-a100", "52684-kv-depth",
                                         "kv_depth2.jsonl"))))
    for _g in XKV["grid_median_over_head_patterns"]:
        for _mode in ("eq", "4096", "16384"):
            _k = "kv_" + _mode
            if _k not in _g:
                continue
            _v = _kvp[(_g[_k + "_pass"], _mode, _g["q_len"])]
            ck("52684 kv-depth, kv=%s q=%d recomputes" % (_mode, _g["q_len"]),
               "%.3f" % _g[_k], statistics.median(_v))
    # the finding: at a fixed KV depth every short-q cell has BLOCK_M=64 slower
    _short = [g for g in XKV["grid_median_over_head_patterns"] if g["q_len"] <= 128]
    ck("52684 kv-depth, short-q cells at a fixed depth", "10",
       sum(1 for g in _short for m in ("4096", "16384") if "kv_" + m in g))
    ck("52684 kv-depth, and how many of them favour BLOCK_M=64", "0",
       sum(1 for g in _short for m in ("4096", "16384")
           if g.get("kv_" + m, 0) > 1.0))
    ck("52684 kv-depth, the worst of them", "0.537",
       min(g["kv_" + m] for g in _short for m in ("4096", "16384") if "kv_" + m in g))
    # and the cross-check that lets pass 2's short rows be believed at all
    ck("52684 kv-depth, cells both passes measured", "4",
       len(XKV["timing"]["overlap_rows"]))
    ck("52684 kv-depth, and the worst disagreement between them", "0.88",
       max(o["diff_pct"] for o in XKV["timing"]["overlap_rows"]), 0.02)
    # Pass 3 exists because passes 1 and 2 licensed an inference this file then
    # published: that the ~0.2 ms at small shapes was the Python wrapper, and so
    # a cost both arms pay. It is not. These are the numbers that withdrew it,
    # and they are gated so the withdrawal cannot quietly un-withdraw.
    _kv3 = [json.loads(l) for l in
            open(os.path.join(HERE, "..", "cuda-a100", "52684-kv-depth",
                              "kv_depth3.jsonl"))]
    _host = [r[f"{a}_host_ms"] for r in _kv3 for a in ("production", "bm64")]
    ck("52684 kv-depth, arm-cells timed both ways", "56", len(_host))
    ck("52684 kv-depth, the host-side cost is a rounding error", "1",
       1 if max(abs(h) for h in _host) < 0.1 else 0)
    ck("52684 kv-depth, its median in ms", "0.006", statistics.median(_host))
    _dw = sorted(abs(r["dev_speedup"] - r["wall_speedup"]) / r["wall_speedup"] * 100
                 for r in _kv3)
    ck("52684 kv-depth, median device-vs-wall disagreement", "0.32",
       statistics.median(_dw))
    # not uniform: the host cost is asymmetric between arms at the four smallest
    # cells and moves the ratio there by up to a fifth, in BOTH directions
    # depending on the cell. Stated rather than smoothed, because "they agree"
    # would be the second wrong claim about this constant.
    ck("52684 kv-depth, and its worst cell", "20.79", _dw[-1], 0.005)
    ck("52684 kv-depth, cells disagreeing by more than 5%", "3",
       sum(1 for x in _dw if x > 5.0))
    # the sign survives being measured at the kernel rather than at the wrapper
    ck("52684 kv-depth, short-q device ratios favouring BLOCK_M=64", "0",
       sum(1 for r in _kv3 if r["q_len"] <= 128 and r["dev_speedup"] > 1.0))
    ck("52684 kv-depth, and the claim that was withdrawn says so", "1",
       1 if XKV["timing"]["dilution"].startswith("WITHDRAWN") else 0)

    # --- a repository that was renamed under our links ----------------------
    # 2026-08-30: `ROCm/ROCm` became `ROCm/legacy-rocm-build`. GitHub redirects
    # the repository root and **not** the deep issue links, so every
    # `github.com/ROCm/ROCm/issues/N` in this repository 404'd for a reader the
    # day it happened -- 51 of them, across the README, four docs and four
    # published article pages. They were rewritten; this is what stops one
    # coming back, and it is static, so it costs nothing and needs no network.
    _renamed = []
    for _root, _dirs, _files in os.walk(os.path.join(HERE, "..", "..")):
        _dirs[:] = [d for d in _dirs if d not in (".git", "__pycache__", "node_modules")]
        for _fn in _files:
            if not _fn.endswith((".md", ".html", ".json", ".py")):
                continue
            _p = os.path.join(_root, _fn)
            # this file names the old path in order to look for it, so it is the
            # one file that must not count itself -- the same shape as a pgrep
            # pattern matching the shell that runs pgrep
            if os.path.realpath(_p) == os.path.realpath(__file__):
                continue
            try:
                if "github.com/ROCm/ROCm/" in open(_p, encoding="utf-8").read():
                    _renamed.append(os.path.relpath(_p, os.path.join(HERE, "..", "..")))
            except Exception:
                pass
    ck("links, none point at the repository GitHub renamed on 2026-08-30", "0",
       len(_renamed))

    # --- the enforce_eager A/B, 2026-08-30 ---------------------------------
    # The claim the directory exists to make is a binary one, so it is gated as
    # a binary: no cell that varied with graphs on became stable with eager.
    # Every count is recomputed from the token sequences rather than read off
    # the summary the run wrote.
    XEA = json.load(open(os.path.join(HERE, "..", "gfx1100-greedy-eager-ab",
                                      "eager-ab.json")))
    ck("eager A/B, cells", "8", len(XEA["cells"]))
    ck("eager A/B, pairs", "4", len(XEA["pairs"]))
    for _c in XEA["cells"]:
        _f = os.path.join(HERE, "..", "gfx1100-greedy-eager-ab",
                          "nondet-eager-%s-e%d-p1.json"
                          % (_c["model"], int(_c["enforce_eager"])))
        _r = next(r for r in json.load(open(_f))["rows"] if r["depth"] == _c["depth"])
        ck("eager A/B, %s e%d ctx%d recomputes from its sequences"
           % (_c["model"], int(_c["enforce_eager"]), _c["depth"]),
           str(_c["distinct"]), len({tuple(x) for x in _r["seqs"]}))
        ck("eager A/B, and %s e%d ctx%d is eight repeats"
           % (_c["model"], int(_c["enforce_eager"]), _c["depth"]),
           "8", len(_r["seqs"]))
    ck("eager A/B, cells that varied with graphs on", "2",
       sum(1 for p_ in XEA["pairs"] if p_["varied_with_graphs"]))
    ck("eager A/B, and how many of those eager made stable", "0",
       sum(1 for p_ in XEA["pairs"]
           if p_["varied_with_graphs"] and not p_["varied_with_eager"]))
    ck("eager A/B, cells eager made unstable that were not", "1",
       sum(1 for p_ in XEA["pairs"]
           if not p_["varied_with_graphs"] and p_["varied_with_eager"]))
    # the control has to reproduce the published cells or it is measuring
    # something else: muse at 512 with graphs on, inside the recorded 5-to-8 band
    _ctl = next(p_ for p_ in XEA["pairs"]
                if p_["model"] == "muse" and p_["depth"] == 512)
    ck("eager A/B, the control reproduces the published band", "1",
       1 if 5 <= _ctl["graphs"] <= 8 else 0)
    ck("eager A/B, and it is the band the earlier campaign recorded", "8",
       xnd["within_process"]["cells"][0]["generations"])

    for fn, heads, phrase in (
            ("measuring-decode.html",
             ("What is not established", "What has changed since"),
             "a single run is not a measurement"),
            ("measuring-decode.zh.html",
             ("没有被确立的部分", "此后发生的变化"),
             "一次运行不构成一次测量")):
        for h in heads:
            ck(f"measure article {fn}, carries '{h[:22]}'", "1",
               1 if fl(h) in flat[fn] else 0)
        ck(f"measure article {fn}, states the rule it exists for", "1",
           1 if phrase in " ".join(pages[fn].split()) else 0)

    # --- rdna3-second-class.html ---------------------------------------------
    # The synthesis. Its classification is an argument, so what is checked is
    # that every magnitude beside it is the number its own article's file
    # produces, that every finding links to a page that exists, and that the
    # count the headline turns on is the count in the data.
    ZART = json.loads(block(pages["rdna3-second-class.html"], "figures"))
    zjul = decode(JULY)
    zab = {(r["arm"], r["ctx"]): r for r in
           (json.loads(l) for l in
            open(os.path.join(HERE, "..", "w4a16-symmetry", "w4a16-ab.jsonl")))}
    zg1 = [json.loads(l) for l in
           open(os.path.join(HERE, "..", "vllm-50603", "stage1-rocm-paths.jsonl"))]
    zgr = [r["triton"]["median_ms"] / r["ck"]["median_ms"] for r in zg1]
    zlf = json.load(open(os.path.join(HERE, "..", "loader-flag-kernel-30.json")))
    zmm = {(r["model"], r["cache"], r["mode"]): r["median_s"]
           for r in zlf["medians_seconds"]}
    zlad = lambda fn: {r["depth"]: r["tok_per_s"] for r in
                       json.load(open(os.path.join(HERE, "..", "speculative-decoding",
                                                   fn)))["rows"]}
    zmtp, zns = zlad("mtp-31b-mtp.json"), zlad("splitkv-31b-stock.json")
    zstk = zlad("mtp-31b-stock45450.json")
    ZVD = os.path.join(HERE, "..", "cuda-a100", "45450-validation", "logs")
    zleg = lambda fn: float(re.search(r"RESULT decode_tok_s=([\d.]+)",
                                      open(os.path.join(ZVD, fn)).read()).group(1))

    ck("rdna3 article, findings", "8", ZART["fig1"]["total"])
    ck("rdna3 article, findings listed", "8", len(ZART["fig1"]["findings"]))
    ck("rdna3 article, the headline count", "3", ZART["fig1"]["counts"]["rdna3"])
    ck("rdna3 article, and it is the count in the rows", "3",
       sum(1 for f in ZART["fig1"]["findings"] if f["axis"] == "rdna3"))
    ck("rdna3 article, four are not this", "4",
       sum(1 for f in ZART["fig1"]["findings"]
           if f["axis"] in ("amd", "neutral", "platform")))
    ck("rdna3 article, every axis it uses is named", "8",
       sum(1 for f in ZART["fig1"]["findings"] if f["axis"] in ZART["fig1"]["axes"]))
    # the magnitudes, each against the file its own article draws
    ZMAG = {
        "hybrid-ssm-collapse": tps(zjul, "D-27B-tp2", 500) / tps(zjul, "D-27B-tp2", 32000),
        "w4a16-two-problems": (zab[("sym", 1024)]["decode_tok_s"]
                               / zab[("asym", 1024)]["decode_tok_s"]),
        "gqa-gate-costs-nothing": max(zgr),
        "weight-loading-19x": (zmm[("gemma-4-31B-w4a16", "cold", "baseline")]
                               / zmm[("gemma-4-31B-w4a16", "cold", "flag")]),
        "moe-written-off-by-eager": tps(zjul, "E-26B-tp2", 500) / 15.0,
        "speculative-decoding-net-loss": zns[32768] / zmtp[32768],
        "a100-vs-two-radeons": ((zleg("C30.log") / zleg("C1K.log"))
                                / (zstk[32768] / zstk[1024])),
        "rccl-atomics-hostcall": None,
    }
    # .get, not [slug]: a renamed finding must fail this check rather than
    # crash the file before anything is reported
    ck("rdna3 article, every magnitude recomputes", "8",
       sum(1 for f in ZART["fig1"]["findings"]
           if f["slug"] in ZMAG
           and ((f["magnitude"] is None and ZMAG[f["slug"]] is None)
                or (f["magnitude"] is not None and ZMAG[f["slug"]] is not None
                    and abs(f["magnitude"] - ZMAG[f["slug"]]) < 1e-9))))
    ck("rdna3 article, and every slug is one this file knows", "8",
       sum(1 for f in ZART["fig1"]["findings"] if f["slug"] in ZMAG))
    # every finding must link to a page that exists in both languages
    ck("rdna3 article, every finding links to a published page", "8",
       sum(1 for f in ZART["fig1"]["findings"]
           if os.path.exists(os.path.join(ART, f["slug"] + ".html"))
           and os.path.exists(os.path.join(ART, f["slug"] + ".zh.html"))))
    ck("rdna3 article, and the English page links to each of them", "8",
       sum(1 for f in ZART["fig1"]["findings"]
           if f'href="{f["slug"]}.html"' in pages["rdna3-second-class.html"]))
    ck("rdna3 article, the Chinese page links to the Chinese ones", "8",
       sum(1 for f in ZART["fig1"]["findings"]
           if f'href="{f["slug"]}.zh.html"' in pages["rdna3-second-class.zh.html"]))

    # fig2: the gate, and the ecosystem list extracted from the notes
    zan = open(os.path.join(HERE, "..", "..", "docs", "architecture-notes.md"),
               encoding="utf-8").read()
    ck("rdna3 article, fig2 gaps", "4", len(ZART["fig2"]["gaps"]))
    ck("rdna3 article, fig2 gaps are in the notes", "4",
       sum(1 for g in ZART["fig2"]["gaps"] if g["what"] in zan))
    ck("rdna3 article, fig2 the notes still extend it to RDNA4", "1",
       1 if ZART["fig2"]["extends_to_rdna4"] and "extends to RDNA4." in zan else 0)
    zm = ZART["fig2"]["measured"]
    ck("rdna3 article, fig2 excluded floor", "1.84", zm["excluded_low"])
    ck("rdna3 article, fig2 excluded ceiling", "7.28", zm["excluded_high"])
    ck("rdna3 article, fig2 admitted floor", "2.35", zm["admitted_low"])
    ck("rdna3 article, fig2 recomputes the excluded floor",
       repr(min(r["triton"]["median_ms"] / r["ck"]["median_ms"]
                for r in zg1 if not r["gate_as_shipped"])),
       zm["excluded_low"], tol=1e-12)
    ck("rdna3 article, fig2 the bands overlap", "1",
       1 if zm["excluded_high"] > zm["admitted_low"] else 0)
    ck("rdna3 article, fig2 end to end at 32K", "1.19", zm["end_to_end_32k"])
    ck("rdna3 article, fig2 names both branches", "1",
       1 if ZART["fig2"]["gate"]["gfx11"] == "gqa_ratio >= 3"
       and ZART["fig2"]["gate"]["cdna"] == "gqa_ratio >= 1" else 0)

    # fig3: the upstream tally, and that it is dated
    ck("rdna3 article, fig3 threads", "16", len(ZART["fig3"]["threads"]))
    ck("rdna3 article, fig3 half are ours", "8", ZART["fig3"]["ours"])
    ck("rdna3 article, fig3 and half are not", "8", ZART["fig3"]["others"])
    ck("rdna3 article, fig3 the counts add up", "16",
       ZART["fig3"]["ours"] + ZART["fig3"]["others"])
    ck("rdna3 article, fig3 one is merged", "1", ZART["fig3"]["merged"])
    ck("rdna3 article, fig3 says when it was read", "1",
       1 if ZART["fig3"]["checked"] == "2026-08-29" else 0)
    ck("rdna3 article, fig3 every thread names an author", "16",
       sum(1 for s in ZART["fig3"]["threads"] if s["author"]))
    # every finding's upstream references must appear in the thread table
    zids = {s["id"] for s in ZART["fig3"]["threads"]}
    ck("rdna3 article, every cited thread is in the table", "1",
       1 if all(u in zids for f in ZART["fig1"]["findings"] for u in f["upstream"])
       else 0)

    for fn, heads in (("rdna3-second-class.html",
                       ("What is not established", "What has changed since")),
                      ("rdna3-second-class.zh.html",
                       ("没有被确立的部分", "此后发生的变化"))):
        for h in heads:
            ck(f"rdna3 article {fn}, carries '{h[:22]}'", "1",
               1 if fl(h) in flat[fn] else 0)
    # the article's own point: it says which findings are NOT this
    for fn, phrase in (("rdna3-second-class.html", "are <em>not</em> RDNA3 problems"),
                       ("rdna3-second-class.zh.html", "不是</em> RDNA3 的问题")):
        ck(f"rdna3 article {fn}, says which are not", "1",
           1 if fl(phrase) in flat[fn] else 0)


    # --- the typed chips, the index and the timeline -------------------------
    # The masthead chips became data: every machine string -- a card name, a
    # version, an issue id, a date -- is written once in site/src/chips.json and
    # rendered into both language versions by site/build.py. What follows is
    # what stops that from being a longer way to write the same duplication.
    # Everything this block needs is loaded here rather than reused from above:
    # main() is one namespace and names get rebound between blocks.
    XSRC = os.path.join(HERE, "..", "..", "site", "src")
    XDOCS = os.path.join(HERE, "..", "..", "docs")
    xld = lambda p: json.load(open(p, encoding="utf-8"))
    CH = xld(os.path.join(XSRC, "chips.json"))
    CW = {lg: xld(os.path.join(XSRC, "chipwords-%s.json" % lg)) for lg in ("en", "zh")}
    AJ = xld(os.path.join(XSRC, "articles.json"))["articles"]
    ZFIG = xld(os.path.join(XSRC, "figures-rdna3.json"))["fig1"]["findings"]

    def xblock(text, ident):
        m = re.search(r'<script type="application/json" id="%s">(.*?)</script>' % ident,
                      text, re.S)
        return m.group(1) if m else None

    xslots = lambda t: sorted(re.findall(r"\{(\d+)\}", t))
    ck("chips, the two word tables have the same keys", "1",
       1 if set(CW["en"]) == set(CW["zh"]) else 0)
    ck("chips, every word takes the same slots in both languages", "0",
       sum(1 for k in CW["en"]
           if k not in CW["zh"] or xslots(CW["en"][k]) != xslots(CW["zh"][k])))

    XKINDS = ["hardware", "model", "stack", "platform", "scale", "upstream", "date", "link"]
    XSLUGS = [s for s in CH if not s.startswith("_")]
    xall = [c for s in XSLUGS for c in CH[s]]
    ck("chips, every chip has a known kind", "0",
       sum(1 for c in xall if c.get("kind") not in XKINDS))
    ck("chips, every chip names a word that exists", "0",
       sum(1 for c in xall if c.get("w") not in CW["en"]))
    # a template with an unfilled slot renders a literal "{0}" on the page, and
    # a value with no slot is a machine string nobody ever sees
    ck("chips, every value has a slot and every slot a value", "0",
       sum(1 for c in xall
           if c.get("w") not in CW["en"]
           or xslots(CW["en"][c["w"]]) != [str(i) for i in range(len(c.get("v", [])))]))
    xused = {c.get("w") for c in xall} | {"k" + k[0].upper() + k[1:] for k in XKINDS}
    ck("chips, no word is left unused", "0", len(set(CW["en"]) - xused))

    # the three verbs are not the same claim, so the timeline has to be told
    # which one each date is
    XTLK = ["measured", "reported", "reviewed"]
    XDC = {s: [c for c in CH[s] if c.get("kind") == "date"] for s in XSLUGS if s != "index"}
    ck("chips, every article carries exactly one date chip", "11",
       sum(1 for v in XDC.values() if len(v) == 1))
    ck("chips, every date chip says what kind of claim it is", "11",
       sum(1 for v in XDC.values() if len(v) == 1 and v[0].get("tl") in XTLK))
    ck("chips, every date is an ISO date", "0",
       sum(1 for v in XDC.values() for c in v for d in c.get("v", [])
           if not re.match(r"^\d{4}-\d{2}-\d{2}$", str(d))))
    ck("chips, the one reported date is the RCCL report", "1",
       1 if sorted(s for s, v in XDC.items() if v and v[0].get("tl") == "reported")
       == ["rccl-atomics-hostcall"] else 0)
    ck("chips, the one reviewed date is the synthesis", "1",
       1 if sorted(s for s, v in XDC.items() if v and v[0].get("tl") == "reviewed")
       == ["rdna3-second-class"] else 0)

    ck("index, one record per article", "11", len(AJ))

    ck("index, every record's dates come from its date chip", "0",
       sum(1 for a in AJ if a.get("dates") != (XDC.get(a.get("slug")) or [{}])[0].get("v")))
    ck("index, every record's kind comes from its date chip", "0",
       sum(1 for a in AJ if a.get("kind") != (XDC.get(a.get("slug")) or [{}])[0].get("tl")))
    ck("index, the date a record sorts by is its last one", "0",
       sum(1 for a in AJ if a.get("date") != max(a.get("dates") or [""])))
    ck("index, every record carries both languages", "0",
       sum(1 for a in AJ for f in ("title", "blurb", "establishes", "href", "sub")
           if sorted(a.get(f) or {}) != ["en", "zh"]
           or not all((a.get(f) or {}).values())))

    # the one-line summary is the synthesis article's own classification of that
    # finding, not a second description written for the index
    ZF = {f["slug"]: f for f in ZFIG}
    ck("index, the eight classified summaries are the synthesis's own", "8",
       sum(1 for a in AJ if a.get("slug") in ZF
           and (a.get("establishes") or {}).get("en") == ZF[a["slug"]].get("mechanism")
           and (a.get("establishes") or {}).get("zh") == ZF[a["slug"]].get("mechanism_zh")))
    ck("index, and the other three are written for it", "3",
       sum(1 for a in AJ if a.get("slug") not in ZF
           and (a.get("establishes") or {}).get("en")
           and (a.get("establishes") or {}).get("zh")))
    # a title lives in the page, in build.py and in the synthesis's figure; this
    # is what makes retitling an article fail loudly rather than drift
    ck("rdna3 article, every finding's title is that article's own title", "16",
       sum(1 for a in AJ if a.get("slug") in ZF
           for lg, fk in (("en", "title"), ("zh", "title_zh"))
           if (a.get("title") or {}).get(lg) == ZF[a["slug"]].get(fk)))

    # the point of the typed chips: a machine string is written once, so it has
    # to come out identical on both language versions of the page
    XPAIRS = [[a["href"]["en"].split("/")[-1], a["href"]["zh"].split("/")[-1]] for a in AJ]
    XPAGES = {fn: open(os.path.join(XDOCS, "articles", fn), encoding="utf-8").read()
              for pr in XPAIRS for fn in pr}
    def xdupes(t):
        ids = re.findall(r'\sid="([^"]+)"', t)
        return sorted({i for i in ids if ids.count(i) > 1})

    for fn, t in XPAGES.items():
        ck("article %s, no id is declared twice" % fn, "0", len(xdupes(t)))

    def xmeta(text):
        """the rendered masthead chips, as (kind, contents) in page order"""
        m = re.search(r'<div class="meta">(.*?)</div>', text, re.S)
        return re.findall(r'<span class="chip k-([a-z]+)"[^>]*>(.*?)</span>',
                          m.group(1), re.S) if m else []

    for a in AJ:
        pr = [a["href"]["en"].split("/")[-1], a["href"]["zh"].split("/")[-1]]
        want = CH[a["slug"]]
        got = [xmeta(XPAGES[fn]) for fn in pr]
        ck("article %s, both versions carry the same chips" % a["slug"], "1",
           1 if [k for k, _ in got[0]] == [k for k, _ in got[1]]
           == [c["kind"] for c in want] else 0)
        # inside the chip, not merely somewhere on the page: the prose mentions
        # these model names too, so a chip could lose one and still look clean
        ck("article %s, every chip value reaches both versions" % a["slug"], "0",
           sum(1 for g in got for i, c in enumerate(want) for v in c.get("v", [])
               if i >= len(g) or v not in g[i][1]))
        # the title now says what was found and the old one is the subtitle, so
        # the subtitle is prose the index quotes off the page rather than a
        # second copy of it
        ck("article %s, the subtitle on the page is the one the index prints" % a["slug"], "2",
           sum(1 for lg, fn in zip(("en", "zh"), pr)
               if '<p class="sub">%s</p>' % (a.get("sub") or {}).get(lg) in XPAGES[fn]))

    # figure 1's caption said the newer image is uniformly lower because the
    # fallback got faster and not because the custom kernel got slower. Both
    # halves are false in the data it draws, so both counts are pinned.
    QV = json.loads(xblock(XPAGES["gqa-gate-costs-nothing.html"], "figures"))["fig1"]["versions"]
    q0 = {(r["shape"], c["ctx"]): c for r in QV[0]["rows"] for c in r["cells"]}
    q1 = {(r["shape"], c["ctx"]): c for r in QV[1]["rows"] for c in r["cells"]}
    ck("gqa article, paired cells across the two images", "30", len(q0))
    ck("gqa article, cells where the newer ratio is higher", "2",
       sum(1 for k in q0 if q1[k]["ratio"] > q0[k]["ratio"]))
    ck("gqa article, cells where the Triton fallback got faster", "21",
       sum(1 for k in q0 if q1[k]["triton_ms"] < q0[k]["triton_ms"]))
    ck("gqa article, cells where the custom kernel got slower", "25",
       sum(1 for k in q0 if q1[k]["ck_ms"] > q0[k]["ck_ms"]))
    for fn, phrase in (("gqa-gate-costs-nothing.html",
                        "Both arms moved between the images and this comparison separates neither"),
                       ("gqa-gate-costs-nothing.zh.html",
                        "\u4e24\u6761\u81c2\u5728\u4e24\u4e2a\u955c\u50cf\u4e4b\u95f4\u90fd\u52a8\u4e86")):
        ck(f"gqa article {fn}, the caption no longer attributes it", "1",
           1 if fl(phrase) in flat[fn] else 0)

    # --- a shared script reaches for ids that must exist in both bodies ------
    # The Chinese page takes the English script byte for byte, so a container
    # renamed on one side and not the other leaves getElementById returning null
    # and the script throwing on the first use. Neither --check nor any figure
    # check executes anything, so this is the only thing that sees it: four
    # Chinese pages shipped broken this way before it existed.
    for a in AJ:
        pr = [a["href"]["en"].split("/")[-1], a["href"]["zh"].split("/")[-1]]
        scr = re.search(r"<script>\n\(function \(\).*?\n</script>", XPAGES[pr[0]], re.S)
        want = set(re.findall(r'getElementById\("([^"]+)"\)', scr.group(0) if scr else ""))
        for fn in pr:
            have = set(re.findall(r'\sid="([^"]+)"', XPAGES[fn]))
            ck("article %s, %s has every id its script asks for" % (a["slug"], fn), "0",
               len(want - have))

    # --- the index pages are a language pair like any other ------------------
    XIP = ["index.html", "index.zh.html"]
    XHOSTS = {"github.com", "bugs.launchpad.net"}
    XI = {fn: open(os.path.join(XDOCS, fn), encoding="utf-8").read() for fn in XIP}
    for fn in XIP:
        ck("index %s, loads no external asset" % fn, "0",
           len(re.findall(r'\ssrc="(https?://[^"]+)"', XI[fn])
               + re.findall(r'<link[^>]+href="(https?://[^"]+)"', XI[fn])))
        ck("index %s, links only to known hosts" % fn, "0",
           len({u.split("/")[2] for u in
                re.findall(r'<a [^>]*href="(https?://[^"]+)"', XI[fn])} - XHOSTS))
        ck("index %s, no stray Cyrillic" % fn, "0",
           len(re.findall(r"[Ѐ-ӿ]", XI[fn])))
        xnav = re.findall(r'<a class="lang" href="([^"]+)" hreflang="([a-z]+)"'
                          r'( aria-current="page")?>', XI[fn])
        ck("index %s, both languages in the switcher" % fn, "2", len(xnav))
        ck("index %s, targets are the two pages" % fn, "1",
           1 if sorted(h for h, _, _ in xnav) == sorted(XIP) else 0)
        ck("index %s, exactly one is current, and it is this page" % fn, "1",
           1 if sum(1 for h, _, cur in xnav if cur and h == fn) == 1
           and sum(1 for _, _, cur in xnav if cur) == 1 else 0)
        # every card on the page is drawn by the script, so the eleven titles
        # only reach a reader through this block
        ck("index %s, carries the article data" % fn, "1",
           1 if xblock(XI[fn], "articles") else 0)
    # the index prose names the sittings and the gap between them, which is a
    # claim about the timeline drawn directly below it
    XSIT = sorted({d for a in AJ for d in (a.get("dates") or [])})
    xday = lambda d: (int(d[:4]), int(d[5:7]), int(d[8:10]))
    ck("index, distinct sitting dates on the timeline", "10", len(XSIT))
    ck("index, the sittings before the long gap", "4",
       sum(1 for d in XSIT if d <= "2026-08-01"))
    ck("index, the long gap is three weeks", "22",
       xday(XSIT[4])[2] - xday(XSIT[3])[2])
    for fn, phrase in (("index.html",
                        "on 25, 26 and 28 July and on 1 August, then not again for three weeks"),
                       ("index.zh.html",
                        "\u5728 7 \u6708 25\u300126\u300128 \u65e5\u548c 8 \u6708 1 \u65e5\u8dd1\u7684")):
        ck("index %s, the prose names the sittings" % fn, "1",
           1 if fl(phrase) in re.sub(r"\s+", " ", XI[fn]) else 0)

    # --- the index's best-measured-today figure -------------------------------
    # Every line is recomputed from the file it claims, and the claim the whole
    # figure rests on -- that nothing faster is left undrawn -- is re-derived
    # here rather than taken from the script that emitted it.
    XFIG = json.loads(xblock(XI["index.html"], "bestdata"))
    XB = XFIG["best"]
    # --- the small figure each card carries --------------------------------
    # Every card's numbers are read out of that article's own figures-*.json by
    # genfig-index.py, so the card and the page it links to cannot disagree.
    # These re-derive the same values from the same files and compare, exactly
    # as the article figures are compared to the ledger.
    XCARD = json.loads(xblock(XI["index.html"], "bestdata"))["cards"]
    _sd = os.path.join(HERE, "..", "..", "site", "src")
    _af = lambda n: json.load(open(os.path.join(_sd, n), encoding="utf-8"))
    ck("index cards, one per article", "11", len(XCARD))
    ck("index cards, every article has one", "0",
       sum(1 for a in AJ if a["slug"] not in XCARD))

    # a missing card must fail the count check above rather than throw here
    _card = lambda s: XCARD.get(s) or {"series": [], "bars": [], "rows": [], "rule": -1}
    _hy = _af("figures.json")["fig1"]["series"]
    _hyb = [s for s in _hy if s["arch"] == "hybrid SSM"][0]
    _dense = [s for s in _hy if s["arch"] == "dense"
              and len(s["points"]) == len(_hyb["points"])][0]
    _a1 = _af("figures-a100.json")["fig1"]["rows"]
    _sp = _af("figures-spec.json")["fig1"]["rows"]
    _w4 = _af("figures-w4a16.json")["fig1"]["cells"]
    _ms = _af("figures-measure.json")["fig2"]["rows"][0]
    _gqa = [r for r in _af("figures-gqa.json")["fig1"]["versions"][0]["rows"]
            if not r["admitted"]]

    # where the article's finding is a comparison the card draws both sides, so
    # what is checked is both sides, in order
    WANT = {
      # drawn as retention, because the article's claim is about slope
      "hybrid-ssm-collapse": [
        [[p["ctx"], p["tok_s"] / _hyb["points"][0]["tok_s"] * 100.0] for p in _hyb["points"]],
        [[p["ctx"], p["tok_s"] / _dense["points"][0]["tok_s"] * 100.0]
         for p in _dense["points"]]],
      "a100-vs-two-radeons": [[[r["ctx"], r["radeons"]] for r in _a1],
                              [[r["a100_ctx"], r["a100"]] for r in _a1]],
      "speculative-decoding-net-loss": [[[r["ctx"], r["nospec"]] for r in _sp],
                                        [[r["ctx"], r["mtp"]] for r in _sp]],
      "w4a16-two-problems": [[[c["ctx"], c["asym_ms"]] for c in _w4],
                             [[c["ctx"], c["sym_ms"]] for c in _w4]],
      "measuring-decode": [[[i + 1, v] for i, v in enumerate(_ms["runs"])]],
    }
    for slug, want in WANT.items():
        got = [s["pts"] for s in _card(slug).get("series", [])]
        ck("index card %s, lines" % slug, str(len(want)), len(got))
        ck("index card %s, points" % slug, str(sum(len(w) for w in want)),
           sum(len(g) for g in got))
        ck("index card %s, matches its article's data" % slug, "1",
           1 if len(want) == len(got) and all(
               len(w) == len(g) and all(abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9
                                        for a, b in zip(w, g))
               for w, g in zip(want, got)) else 0)
    ck("index card measure, the rule is the converged rate", repr(_ms["converged"]),
       _card("measuring-decode")["rule"])

    ck("index card gqa, one line per excluded shape", str(len(_gqa)),
       len(_card("gqa-gate-costs-nothing")["series"]))
    ck("index card gqa, every ratio is the article's", "1",
       1 if all(abs(c["ratio"] - p[1]) < 1e-9 and c["ctx"] == p[0]
                for r, s in zip(_gqa, _card("gqa-gate-costs-nothing")["series"])
                for c, p in zip(r["cells"], s["pts"])) else 0)

    _moe = _af("figures-moe.json")["fig1"]
    ck("index card moe, bars", str(len(_moe["bars"]) + 1),
       len(_card("moe-written-off-by-eager")["bars"]))
    ck("index card moe, the compiled bars are the article's", "1",
       1 if all(abs(b["tok_s"] - c["v"]) < 1e-9
                for b, c in zip(_moe["bars"],
                                _card("moe-written-off-by-eager")["bars"])) else 0)
    ck("index card moe, and the eager bar is the one marked", "1",
       1 if _card("moe-written-off-by-eager")["bars"][-1:] and
       _card("moe-written-off-by-eager")["bars"][-1]["v"] == _moe["eager"]["tok_s"]
       and _card("moe-written-off-by-eager")["bars"][-1]["kind"] == "bad" else 0)

    _ld = _af("figures-loader.json")["fig1"]["states"]
    ck("index card loader, one bar per kernel state", str(len(_ld)),
       len(_card("weight-loading-19x")["bars"]))
    ck("index card loader, each is that state's rw-p resident case", "1",
       1 if all(abs([c for c in s["cases"] if c["key"] == "rw_p_resident"][0]["ms"] - b["v"])
                < 1e-9 for s, b in zip(_ld, _card("weight-loading-19x")["bars"])) else 0)

    _rd = _af("figures-rdna3.json")["fig1"]
    ck("index card rdna3, the RDNA3 count", str(_rd["counts"]["rdna3"]),
       (_card("rdna3-second-class")["bars"][0:1] or [{"v": -1}])[0]["v"])
    ck("index card rdna3, and the rest of the eight",
       str(_rd["total"] - _rd["counts"]["rdna3"]),
       (_card("rdna3-second-class")["bars"][1:2] or [{"v": -1}])[0]["v"])

    _65 = _af("figures-6565.json")["fig1"]["arms"]
    ck("index card 6565, one bar per arm", str(len(_65)),
       len(_card("reporting-a-non-reproduction")["bars"]))
    ck("index card 6565, and they sum to the sweep", "135",
       sum(b["v"] for b in _card("reporting-a-non-reproduction")["bars"]))

    _rc = _af("figures-rccl.json")["shipped"]
    ck("index card rccl, one row per shipped library", str(len(_rc)),
       len(_card("rccl-atomics-hostcall")["rows"]))
    ck("index card rccl, and each keeps its behaviour", "1",
       1 if all((s["behaviour"] == "works") == r["ok"]
                for s, r in zip(_rc, _card("rccl-atomics-hostcall")["rows"])) else 0)
    XLED = [json.loads(l) for l in open(os.path.join(HERE, "..", "ledger.jsonl"))]
    ck("index figure, series", "27", len(XB["series"]))
    ck("index figure, points", "277", sum(len(x["points"]) for x in XB["series"]))
    # Every line is one session at the campaign ladder -- not a three-point probe
    # beside an eleven-rung campaign -- unless the card could not hold the KV for
    # the next rung, which two of the single-card lines could not. A short line
    # has to say why it is short, or it reads as an abandoned run.
    ck("index figure, and every line is eleven rungs unless the card capped it", "1",
       1 if all(len(x["points"]) == 11 or x.get("rungs_capped")
                for x in XB["series"]) else 0)
    XCAP = [x for x in XB["series"] if x.get("rungs_capped")]
    ck("index figure, lines the card capped", "4", len(XCAP))
    ck("index figure, and each says how many tokens its KV held", "4",
       sum(1 for x in XCAP if x["rungs_capped"].get("kv_tokens")))
    # the cap is arithmetic, and the arithmetic is in the serve logs: the last
    # rung drawn plus what the harness generates has to fit what the KV held,
    # and the next rung up must not
    for x in XCAP:
        cap = x["rungs_capped"]["kv_tokens"]
        deepest = x["points"][-1]["ctx"]
        ck("index figure, %s's deepest rung fits its KV" % x["cfg"], "1",
           1 if deepest + 512 <= cap else 0)
        nxt = {500: 1000, 1000: 2000, 2000: 4000, 4000: 6000, 6000: 8000,
               8000: 12000, 12000: 16000, 16000: 20000, 20000: 24000,
               24000: 32000}[deepest]
        ck("index figure, and %s's next rung up does not" % x["cfg"], "1",
           1 if nxt + 512 > cap else 0)
    ck("index figure, speculative lines", "6",
       sum(1 for x in XB["series"] if x["spec"]))
    ck("index figure, and none of them is lit without being asked", "0",
       sum(1 for x in XB["series"] if x["spec"] and x["lit"]))
    ck("index figure, lines on the Radeons", "10",
       sum(1 for x in XB["series"] if x["machine"] == "rdna3"))
    # The alternative arm: same model, same machine, same day, same stack, one
    # flag. It is not a pick competing for "fastest" -- it is the other half of
    # a trade the front page would otherwise state only one side of.
    XALT = [x for x in XB["series"] if x.get("alt")]
    ck("index figure, alternative arms", "1", len(XALT))
    ck("index figure, and no alternative arm is lit without being asked", "0",
       sum(1 for x in XALT if x["lit"]))
    ck("index figure, and each is named for the backend it ran", "1",
       sum(1 for x in XALT if x["alt_label"] == x["attn_backend"]))
    _ap = XB["alt_pairs"]
    ck("index figure, alternative pairs", "1", len(_ap))
    ck("index figure, and the pair differs in its backend", "1",
       sum(1 for a in _ap if a["base_backend"] != a["alt_backend"]))
    ck("index figure, and in nothing else", "1",
       1 if all(a["date"] == "2026-08-29" for a in _ap) else 0)
    # what the flag is worth at decode, which is the published finding inverted
    ck("index figure, the backend ROCm picks at 500", "-0.21",
       _ap[0]["delta_pct"][0]["pct"], 0.01)
    ck("index figure, and the backend ROCm picks at 32K", "-13.1",
       _ap[0]["delta_pct"][-1]["pct"], 0.01)
    ck("index figure, lines on the A100", "9",
       sum(1 for x in XB["series"] if x["machine"] == "a100"))
    ck("index figure, lines on one Radeon", "3",
       sum(1 for x in XB["series"] if x["machine"] == "rdna3-1"))
    ck("index figure, lines on the L4", "4",
       sum(1 for x in XB["series"] if x["machine"] == "l4"))
    ck("index figure, machines offered", "5", len(XB["machines"]))
    ck("index figure, and one of them is on by default", "1",
       sum(1 for m in XB["machines"] if m["default"]))
    # every machine the figure draws has to have a name in both languages, or
    # the row renders "undefined" in one of them and nothing catches it
    for _lang, _fn in (("en", "index-body.html"), ("zh", "index-body-zh.html")):
        _sb = json.loads(block(open(os.path.join(
            HERE, "..", "..", "site", "src", _fn), encoding="utf-8").read(), "strings"))
        ck("index figure, every machine is named in %s" % _lang,
           str(len(XB["machines"])),
           sum(1 for m in XB["machines"] if _sb["bestMach"].get(m["id"])))
        # A caveated prefill line renders its caveat from this key. Without it
        # the tooltip prints "undefined" in one language and nothing catches it,
        # which is the same failure the machine-name check above exists for.
        ck("prefill figure, the caveat has a string in %s" % _lang, "1",
           1 if _sb.get("preCaveat") else 0)

    # --- the fifth machine ---------------------------------------------------
    # The T4 exists in this repository at all because of vllm#39018, and that is
    # not decoration: it is the only patch in either figure that changes an
    # attention kernel, so it is the only line whose prefill coefficients mean
    # something different from every other line's. Decode is untouched by it.
    _t4 = [x for x in XB["series"] if x["machine"] == "t4"]
    ck("index figure, lines on the T4", "1", len(_t4))
    ck("index figure, and the T4 line carries the patch it needs to exist", "1",
       1 if _t4 and _t4[0]["patches"] == ["vllm#39018"] else 0)
    ck("index figure, no other line in it carries that patch", "0",
       sum(1 for x in XB["series"]
           if x["machine"] != "t4" and "vllm#39018" in x["patches"]))
    # Two series this repository measured and this figure does not draw. A note
    # about a series that does not exist would be worse than no note, so each
    # one is looked up in the projection it claims to be in.
    _ND = XB["not_drawn"]
    ck("index figure, series measured and deliberately not drawn", "2", len(_ND))
    ck("index figure, and each of them says why", "2",
       sum(1 for n in _ND if n.get("why")))
    _DECROWS = [json.loads(l) for l in
                open(os.path.join(HERE, "..", "decode.jsonl"))]
    ck("index figure, and each is a real series in decode.jsonl", "2",
       sum(1 for n in _ND if any(r["cfg"] == n["cfg"] for r in _DECROWS)))
    ck("index figure, and none of them is also drawn", "0",
       sum(1 for n in _ND if any(x["cfg"] == n["cfg"] for x in XB["series"])))

    # Prefix caching was on for the 2026-08-29 A100 campaign and off for the
    # 2026-08-30 one. Its prefill cannot be used; its decode can, and this is
    # the measurement that says so rather than the assertion -- the same two
    # models, eleven rungs each, measured both ways.
    XCC = XB["cache_control"]
    ck("index figure, models measured with the cache both ways", "2", len(XCC))
    ck("index figure, and each is eleven rungs", "2",
       sum(1 for c in XCC if c["rungs"] == 11))
    ck("index figure, decode agrees across the two, worst", "2.34",
       max(c["worst_pct"] for c in XCC), 0.01)
    ck("index figure, and inside the chart-grade cut", "1",
       1 if max(c["worst_pct"] for c in XCC) <= 8.0 else 0)
    ck("index figure, lit without being asked", "4",
       sum(1 for x in XB["series"] if x["lit"] and x["machine"] == "rdna3"))
    ck("index figure, and the same models on the other machine", "4",
       sum(1 for x in XB["series"] if x["lit"] and x["machine"] == "a100"))

    # One colour per model, seven of them defined, and adding a machine must not
    # quietly add an eighth: colour[m] is var(--m{i%7+1}), so the eighth model
    # would be drawn in the first one's colour and two lines would claim to be
    # the same thing. Every A100 model is already a Radeon model, which is why
    # nine new lines cost no colours -- this is the check that says so.
    xmodels = sorted({x["model"] for x in XB["series"]})
    ck("index figure, models drawn", "7", len(xmodels))
    ck("index figure, and a colour for each without wrapping", "1",
       1 if len(xmodels) <= len(set(re.findall(r"--m(\d):",
          open(os.path.join(HERE, "..", "..", "site", "src", "index-extra.css"),
               encoding="utf-8").read()))) else 0)

    xsid = lambda r: (r["model"], r["tp"], r["vllm"], tuple(r["patches"]),
                      r["harness"], r["date"])
    for x in XB["series"]:
        if x["source"] != "benchmarks/ledger.jsonl":
            continue
        want = [r for r in XLED
                if xsid(r) == (x["model"], x["tp"], x["vllm"], tuple(x["patches"]),
                               x["harness"], x["date"])
                and (x.get("cfg") is None or r["cfg"] == x["cfg"])]
        want.sort(key=lambda r: r["ctx"])
        tag = x["model"] + (" MTP" if x["spec"] else "")
        ck("index figure, %s point count" % tag, str(len(want)), len(x["points"]))
        ck("index figure, %s matches the ledger" % tag, "1",
           1 if len(want) == len(x["points"]) and all(
               r["ctx"] == p["ctx"] and abs(r["decode_tok_s"] - p["tok_s"]) < 1e-9
               and r["chart_grade"] == p["graded"]
               for r, p in zip(want, x["points"])) else 0)

    # the two lines the campaign does not represent, against the campaign
    xrep0 = XB["repro"]
    xcamp = {}
    for r in XLED:
        if r["date"] == XB["campaign"]["date"] and r["tp"] == 2:
            xcamp.setdefault(r["model"], {})[r["ctx"]] = r["decode_tok_s"]
    for o in XB["overrides"]:
        mine = {p["ctx"]: p["tok_s"] for x in XB["series"]
                if x["model"] == o["model"] and x["machine"] == "rdna3"
                for p in x["points"]}
        c = xcamp[o["model"]]
        ratios = [mine[d] / c[min(c, key=lambda k: abs(k - d))]
                  for d in mine
                  if abs(min(c, key=lambda k: abs(k - d)) - d) / d < 0.06]
        ck("index figure, %s says why it replaces the campaign" % o["model"], "1",
           1 if o["why"] in ("faster", "reproduces") else 0)
        ck("index figure, %s is that" % o["model"], "1",
           1 if (min(ratios) > 1.0 if o["why"] == "faster"
                 else (1 - min(ratios)) * 100.0 <= xrep0["worst_pct"]) else 0)
        ck("index figure, %s deepest gain" % o["model"],
           "%.2f" % (o["picked_deepest"] / o["campaign_deepest"]),
           o["picked_deepest"] / o["campaign_deepest"])

    # the claim the figure is named for: nothing in the ledger beats a drawn
    # line by more than this machine repeats a whole campaign
    xrep = XB["repro"]
    xc1 = {(r["model"], r["tp"], r["ctx"]): r["decode_tok_s"]
           for r in XLED if r["date"] == xrep["campaigns"][0]}
    xc2 = {(r["model"], r["tp"], r["ctx"]): r["decode_tok_s"]
           for r in XLED if r["date"] == xrep["campaigns"][1]}
    xsh = sorted(set(xc1) & set(xc2))
    ck("index figure, cells the two campaigns share", str(xrep["cells"]), len(xsh))
    ck("index figure, worst campaign-to-campaign disagreement",
       "%.3f" % xrep["worst_pct"],
       max(abs(xc1[k] - xc2[k]) / max(xc1[k], xc2[k]) * 100 for k in xsh))
    xundrawn = 0
    for x in XB["series"]:
        if x["machine"] != "rdna3" or x["spec"]:
            continue          # a speculative row cannot beat a stock line
        mine = {p["ctx"]: p["tok_s"] for p in x["points"]}
        for r in XLED:
            if (r["model"] != x["model"] or r["tp"] != x["tp"] or r["ctx"] not in mine
                    or r["spec"] is not None
                    or xsid(r) == (x["model"], x["tp"], x["vllm"], tuple(x["patches"]),
                                   x["harness"], x["date"])):
                continue
            slack = max(r["range_pct"] or 0.0, xrep["worst_pct"]) / 100.0 * r["decode_tok_s"]
            if r["decode_tok_s"] - mine[r["ctx"]] > slack:
                xundrawn += 1
    ck("index figure, faster measurements left undrawn", "0", xundrawn)
    ck("index figure, the omitted model is in the ledger", "1",
       1 if all(any(r["model"] == m for r in XLED) for m in XB["omitted"]) else 0)
    ck("index figure, and is not drawn", "0",
       sum(1 for x in XB["series"] if x["model"] in XB["omitted"]))
    # the A100 pair, recomputed from the campaign file rather than quoted
    import collections as _co
    XA = _co.defaultdict(lambda: _co.defaultdict(list))
    for line in open(os.path.join(HERE, "..", "cuda-a100", "campaign-2026-08-29",
                                  "results.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if r.get("kind") == "decode" and r.get("decode_tps"):
            XA[r["cfg"]][r["target"]].append(r["decode_tps"])
    for xa in [x for x in XB["series"] if x["machine"] == "a100"]:
        src = XA[xa["cfg"]]
        tag = xa["cfg"]
        ck("index figure, %s point count" % tag, str(len(src)), len(xa["points"]))
        ck("index figure, %s recomputes" % tag, "1",
           1 if all(abs(sum(src[p["ctx"]]) / len(src[p["ctx"]]) - p["tok_s"]) < 1e-9
                    for p in xa["points"]) else 0)

    # The A100 side is the whole campaign now, not one model of it, so the claim
    # to check is that nothing eleven-rung was left out. Twelve configurations
    # were measured; nine are drawn, and the three that are not are exactly the
    # unpatched twins of the three patched speculative arms that are.
    XA11 = sorted(c for c in XA if len(XA[c]) == 11)
    XADRAWN = {x["cfg"] for x in XB["series"] if x["machine"] == "a100"}
    ck("index figure, A100 configurations at eleven rungs", "12", len(XA11))
    ck("index figure, and every one is drawn or is a drawn arm's control", "0",
       sum(1 for c in XA11
           if c not in XADRAWN and c + "-p45450" not in XADRAWN))
    ck("index figure, A100 models drawn", "5",
       len({x["model"] for x in XB["series"] if x["machine"] == "a100"}))
    # The arm chosen is the patched one throughout, which is a choice and not an
    # inheritance: every A100 speculative line that has an unpatched twin in the
    # campaign carries #45450.
    ck("index figure, and every A100 arm with a twin is the patched one", "0",
       sum(1 for x in XB["series"]
           if x["machine"] == "a100" and x["spec"]
           and x["cfg"].endswith("-p45450")
           and "vllm#45450 3D admission" not in x["patches"]))
    ck("index figure, and the arms without a twin are the ones that had none", "1",
       1 if {x["cfg"] for x in XB["series"]
             if x["machine"] == "a100" and x["spec"]
             and not x["cfg"].endswith("-p45450")} == {"A100-MG30-dflash"} else 0)
    # It is defensible on Qwen3.8 because there the two arms are one
    # measurement: #45450 patches the Triton files and vLLM serves that model
    # through FLASH_ATTN here, so the probe never fires. Recomputed, not quoted.
    _q38a = {c: sum(v) / len(v) for c, v in XA["A100-Q38-mtp"].items()}
    _q38b = {c: sum(v) / len(v) for c, v in XA["A100-Q38-mtp-p45450"].items()}
    ck("index figure, A100 Qwen3.8 patched against unpatched, mean pct", "-0.08",
       sum(_q38b[c] / _q38a[c] - 1 for c in _q38a) / len(_q38a) * 100.0)
    ck("index figure, and at 32K", "0.04",
       (_q38b[32000] / _q38a[32000] - 1) * 100.0)
    ck("index figure, and no rung of it moves by half a per cent", "0",
       sum(1 for c in _q38a if abs(_q38b[c] / _q38a[c] - 1) * 100.0 >= 0.5))
    # Which way speculation goes is not one answer, and the caption says so, so
    # both halves are checked. On the A100 every arm loses at every depth it was
    # measured at. On the Radeons neither does: gemma-4-31B is ahead the whole
    # way and Qwen3.8 starts ahead and crosses. A caption claiming one rule for
    # all six would be wrong about two of them.
    ck("index figure, A100 arms that lose at every depth", "4",
       sum(1 for pr in XB["mtp_pairs"]
           if pr["machine"] == "a100" and all(d["pct"] < 0 for d in pr["delta_pct"])))
    ck("index figure, and A100 arms drawn", "4",
       sum(1 for pr in XB["mtp_pairs"] if pr["machine"] == "a100"))
    ck("index figure, Radeon arms that are ahead at the shortest depth", "2",
       sum(1 for pr in XB["mtp_pairs"]
           if pr["machine"] == "rdna3" and pr["at_shortest_pct"] > 0))
    ck("index figure, and Radeon arms still ahead at the deepest", "1",
       sum(1 for pr in XB["mtp_pairs"]
           if pr["machine"] == "rdna3" and pr["at_deepest_pct"] > 0))
    ck("index figure, arms that cross zero", "1",
       sum(1 for pr in XB["mtp_pairs"] if pr["crosses_zero"]))
    # the switch's promise: each speculative line is its own line's arm, on its
    # own machine, same day and same kernel -- not the fastest speculative run
    # anywhere. This is what stops the chart drawing a fast speculative arm
    # beside a control that is not on the page.
    for pr in XB["mtp_pairs"]:
        tag = "%s %s %s" % (pr["machine"], pr["model"], pr["label"])
        ck("index figure, %s is paired to its own line" % tag, "1",
           1 if any(x["cfg"] == pr["base_cfg"] and not x["spec"]
                    and x["machine"] == pr["machine"] for x in XB["series"]) else 0)
        ck("index figure, %s pairs on the same day" % tag, "1",
           1 if all(x["date"] == pr["date"] for x in XB["series"]
                    if x["cfg"] in (pr["base_cfg"], pr["mtp_cfg"])) else 0)
    ck("index figure, one pair per speculative line", "6", len(XB["mtp_pairs"]))
    ck("index figure, and each is a switch on its model", "1",
       1 if all(XB["labels"][pr["model"]]["spec_label"] == pr["label"]
                for pr in XB["mtp_pairs"]) else 0)

    # --- the labels -----------------------------------------------------------
    # The label is drawn, so it is checked. `quant` is the ledger's own string
    # for every line including the A100 ones, which read it by model name rather
    # than carrying a copy; `quant_label` is that string with its first token
    # upper-cased and nothing else changed, which is the whole rule.
    XQ = {}
    for r in XLED:
        XQ.setdefault(r["model"], set()).add(r["quant"])
    for x in XB["series"]:
        ck("index figure, %s %s quant is the ledger's" % (x["machine"], x["cfg"]), "1",
           1 if XQ.get(x["model"]) == {x["quant"]} else 0)
        ck("index figure, %s %s quant label" % (x["machine"], x["cfg"]), "1",
           1 if x["quant_label"] == x["quant"].split(" ")[0].upper()
                                    + x["quant"][len(x["quant"].split(" ")[0]):] else 0)
    ck("index figure, a label for every model drawn", "0",
       len({x["model"] for x in XB["series"]} - set(XB["labels"])))
    ck("index figure, and no line disagrees with its model's label", "0",
       sum(1 for x in XB["series"]
           if XB["labels"][x["model"]]["quant_label"] != x["quant_label"]))
    # the switch is named for what the arm resolved to, not for the button it
    # replaced: three arms are mtp and Muse-Glimmer's is a block-diffusion
    # drafter at k=8, which a switch marked MTP would misname
    XSL = {"mtp": "MTP", "dflash": "DFlash"}
    ck("index figure, every switch is named for its own method", "0",
       sum(1 for x in XB["series"] if x["spec"]
           and x["spec_label"] != XSL.get(x["spec_desc"]["method"])))
    ck("index figure, and Muse-Glimmer's is not called MTP", "1",
       1 if XB["labels"]["Muse-Glimmer-30B"]["spec_label"] == "DFlash" else 0)
    ck("index figure, models with a switch", "4",
       sum(1 for m in XB["labels"] if XB["labels"][m]["spec_label"]))
    ck("index figure, and only those with an arm drawn", "1",
       1 if {m for m in XB["labels"] if XB["labels"][m]["spec_label"]}
            == {x["model"] for x in XB["series"] if x["spec"]} else 0)

    # --- the x axis -----------------------------------------------------------
    # Every tick is a depth that was actually measured, both ends of the range
    # are labelled, and nothing is labelled outside it. The list used to be typed
    # and ended at 50 000 -- past the deepest rung, so it printed itself beyond
    # the right edge of the frame -- while the left edge carried no label at all,
    # which made an axis starting at 500 read as though it started at zero.
    xrungs = {p["ctx"] for x in XB["series"] for p in x["points"]}
    ck("index figure, x ticks", "7", len(XB["ctx_ticks"]))
    ck("index figure, and every one is a measured depth", "0",
       len(set(XB["ctx_ticks"]) - xrungs))
    ck("index figure, and none is outside the axis", "0",
       sum(1 for t in XB["ctx_ticks"] if t < XB["ctx_min"] or t > XB["ctx_max"]))
    ck("index figure, and both ends are labelled", "1",
       1 if XB["ctx_ticks"][0] == XB["ctx_min"]
            and XB["ctx_ticks"][-1] == XB["ctx_max"] else 0)
    ck("index figure, and they are the depths that double", "1",
       1 if XB["ctx_ticks"] == sorted(c for c in xrungs
                                      if c in {500 * 2 ** i for i in range(8)}) else 0)

    # A series carrying `points` is positioned by value, which only means
    # anything on an axis that declares scale "log" or "linear". Drawn on an
    # ordinal axis the value is read as a category index instead, and the whole
    # series lands thousands of frame-widths off the side -- silently, because
    # nothing is left on screen to look wrong. The A100 article's rate view
    # shipped that way and was empty. Every lineChart call is checked here.
    xsrc = os.path.join(HERE, "..", "..", "site", "src")
    xbad = []
    for xfn in sorted(os.listdir(xsrc)):
        if not xfn.endswith("-body.html"):
            continue
        xt = open(os.path.join(xsrc, xfn), encoding="utf-8").read()
        for xm in re.finditer(r"lineChart\(\{", xt):
            xi = xm.end() - 1
            xd = 0
            for xj in range(xi, len(xt)):
                if xt[xj] == "{":
                    xd += 1
                elif xt[xj] == "}":
                    xd -= 1
                    if xd == 0:
                        break
            xblk = xt[xi:xj + 1]
            if "points:" not in xblk:
                continue
            # the axis is either written inline or comes out of a helper; a
            # helper is resolved by looking for a scale in the file at large
            xax = re.search(r"\n\s*x:\s*([^\n]*)", xblk)
            xax = xax.group(1) if xax else ""
            if "scale:" in xax:
                continue
            xh = re.match(r"\s*(\w+)\(\)", xax)
            if xh and re.search(r"function %s\s*\(\)[\s\S]{0,600}?scale:" % xh.group(1), xt):
                continue
            xbad.append(xfn + ":" + xax[:40])
    ck("site, charts drawing points on an axis that cannot place them", "0", len(xbad))

    # The rccl figure's data records what the library does in the words the
    # investigation used, and the page looks each one up so the Chinese version
    # does not print "fails" in English. A value with no entry falls back to the
    # recorded string, which is a silent leak, so every value has to have one --
    # in both tables, or one language renders the other's.
    xrccl = json.loads(open(os.path.join(xsrc, "figures-rccl.json"), encoding="utf-8").read())
    xrb = {r["behaviour"] for r in xrccl["shipped"]}
    xrq = {m.group(1) for r in xrccl["shipped"]
           for m in [re.match(r"^\S+\s+\((.*)\)$", r["hostcall"])] if m}
    for xfn in ("rccl-body.html", "rccl-body-zh.html"):
        xst = json.loads(xblock(open(os.path.join(xsrc, xfn), encoding="utf-8").read(),
                                "strings"))
        ck("rccl article, %s says every recorded behaviour" % xfn, "0",
           len(xrb - set(xst.get("shipBehaviour", {}))))
        ck("rccl article, %s says every hostcall qualifier" % xfn, "0",
           len(xrq - set(xst.get("shipNotes", {}))))

    xis = re.search(r"<script>\n\(function \(\).*?\n</script>", XI[XIP[0]], re.S)
    xiwant = set(re.findall(r'getElementById\("([^"]+)"\)', xis.group(0) if xis else ""))
    for fn in XIP:
        ck("index %s, no id is declared twice" % fn, "0", len(xdupes(XI[fn])))
        ck("index %s, has every id its script asks for" % fn, "0",
           len(xiwant - set(re.findall(r'\sid="([^"]+)"', XI[fn]))))

    # --- the index's two long captions --------------------------------------
    # They were 612 and 570 words of unbroken prose, which is a wall rather than
    # a caption, and were restructured into a lede plus labelled blocks. The two
    # languages have to carry the same blocks or one of them is saying less than
    # the other, which is the drift the language-pair rule exists to catch.
    _caps = {}
    for _lang, _fn in (("en", "index.html"), ("zh", "index.zh.html")):
        _t = XI[_fn]
        _caps[_lang] = [len(re.findall(r"<dt>", _c))
                        for _c in re.findall(r"<figcaption>(.*?)</figcaption>", _t, re.S)
                        if "capnotes" in _c]
    ck("index captions, structured ones in en", "2", len(_caps["en"]))
    ck("index captions, and the same number in zh", str(len(_caps["en"])),
       len(_caps["zh"]))
    ck("index captions, the two languages carry the same blocks", "1",
       1 if _caps["en"] == _caps["zh"] else 0)
    ck("index captions, and every structured one opens with a lede", "4",
       sum(1 for _f in XIP
           for _c in re.findall(r"<figcaption>(.*?)</figcaption>", XI[_f], re.S)
           if "capnotes" in _c and 'class="caplede"' in _c))
    # a caption that grew back into a wall is the thing this replaced
    for _lang, _fn in (("en", "index.html"), ("zh", "index.zh.html")):
        _worst = max(len(re.sub(r"<[^>]+>", " ", _c).split())
                     for _c in re.findall(r"<figcaption>(.*?)</figcaption>",
                                          XI[_fn], re.S))
        ck("index captions, longest one in %s stays under 500 words" % _lang, "1",
           1 if _worst < 500 else 0)

    ck("index, the two versions share one data block", "1",
       1 if xblock(XI[XIP[0]], "articles") == xblock(XI[XIP[1]], "articles") else 0)
    ck("index, the data block is the file on disk", "1",
       1 if json.loads(xblock(XI[XIP[0]], "articles") or "{}") == {"articles": AJ} else 0)
    ck("index, each version carries its own chip words", "2",
       sum(1 for fn, lg in zip(XIP, ("en", "zh"))
           if json.loads(xblock(XI[fn], "chipwords") or "{}") == CW[lg]))
    xscr = [re.search(r"<script>\n\(function \(\).*?\n</script>", XI[fn], re.S) for fn in XIP]
    ck("index, the two versions share one script", "1",
       1 if all(xscr) and xscr[0].group(0) == xscr[1].group(0) else 0)
    xkeys = [set(json.loads(xblock(XI[fn], "strings") or "{}").keys()) for fn in XIP]
    # --- the navigation furniture -------------------------------------------
    # Every article carries a link back to the index in its own language, and
    # the index carries none, because a link to the page you are on is
    # furniture. The href is what would fail silently: an English article
    # pointing at index.zh.html still renders, still clicks, and drops the
    # reader into the other language.
    xhome = 0
    for a in AJ:
        for lg, fn in (("en", a["href"]["en"].split("/")[-1]),
                       ("zh", a["href"]["zh"].split("/")[-1])):
            want = '../index.html' if lg == "en" else '../index.zh.html'
            m = re.search(r'<a class="home" href="([^"]+)"', XPAGES[fn])
            ck("article %s, %s has a way back to the index" % (a["slug"], lg), "1",
               1 if m and m.group(1) == want else 0)
            xhome += 1 if m else 0
    ck("site, articles carrying a back link", str(xhome), xhome)
    for fn in XIP:
        ck("index %s, does not link back to itself" % fn, "0",
           len(re.findall(r'<a class="home"', XI[fn])))
        # the rail is built from the page's own headings and draws nothing at
        # all below three of them, which would be a silent disappearance
        ck("index %s, headings for the rail to list" % fn, "4",
           len(re.findall(r"<h2><span class=\"n\">", XI[fn])))

    # The rail lives in the shared head now, so every page has one -- and every
    # page therefore needs its label in its own language and no leftover slot.
    # A slot that survived would render the placeholder as the accessible name.
    XLBL = {"en": "sections of this page", "zh": "本页目录"}
    for a in AJ:
        for lg, fn in (("en", a["href"]["en"].split("/")[-1]),
                       ("zh", a["href"]["zh"].split("/")[-1])):
            ck("article %s, %s labels its rail" % (a["slug"], lg), "1",
               XPAGES[fn].count('nav.setAttribute("aria-label", "%s")' % XLBL[lg]))
            # three sections is the floor the rail draws at all
            ck("article %s, %s sections for the rail" % (a["slug"], lg), "1",
               1 if len(re.findall(r"<h2><span class=\"n\">", XPAGES[fn])) >= 3 else 0)
    for fn, lg in zip(XIP, ("en", "zh")):
        ck("index %s, labels its rail" % fn, "1",
           XI[fn].count('nav.setAttribute("aria-label", "%s")' % XLBL[lg]))
    ck("site, pages with an unfilled slot", "0",
       sum(1 for t in list(XPAGES.values()) + list(XI.values())
           if re.search(r"__[A-Z][A-Z_]*__", t)))
    # the index numbered two different figures "Figure 2" until the rail listed
    # them one under the other and made it obvious
    for fn, pat in zip(XIP, (r"Figure (\d+) &middot;|Figure (\d+) ·", r"图 (\d+) ·")):
        xnums = [int(m) for m in re.findall(r'figtitle">(?:Figure|图) (\d+)', XI[fn])]
        ck("index %s, figure numbers are distinct" % fn, str(len(xnums)), len(set(xnums)))

    ck("index, the strings tables have the same keys", "1",
       1 if xkeys[0] == xkeys[1] and xkeys[0] else 0)
    for k in sorted(xkeys[0]):
        ck("index, script uses string '%s'" % k, "1",
           1 if xscr[0] and ("S." + k) in xscr[0].group(0) else 0)

    # --- the single-card decode table in benchmarks/README.md ----------------
    # Stock arms only, grouped by cfg. The commit message of 73fa06e reports
    # this table with the A100's MoE figure taken from a speculative arm,
    # because the query behind it keyed on (machine, model, date) and that name
    # covers three configurations there. The README carries the correction and
    # these recompute it, so the corrected numbers cannot themselves rot.
    XDEC = [json.loads(l) for l in open(os.path.join(HERE, "..", "decode.jsonl"))]

    def _sc(cfg, ctx):
        return next(r["decode_tok_s"] for r in XDEC
                    if r["cfg"] == cfg and r["ctx"] == ctx and r["tp"] == 1
                    and r["spec"] is None and r["chart_grade"])

    # cfg is not a key on its own here: A-12B-tp1 was measured in two campaigns
    # and G12/G26A4B name a line on two machines, so the machine and the date
    # are part of what the README's table cites. It cites the runs the two
    # index figures draw.
    RETAINED = {}
    for cfg, mach, date, ctx_deep, want_s, want_d in (
            ("A100-G12",    "A100-SXM4-80GB", "2026-08-29", 32000, "115.0", "71.3"),
            ("A-12B-tp1",   "RX 7900 XT",     "2026-08-24", 32000, "50.6",  "36.7"),
            ("G12",         "L4",             "2026-08-30", 32000, "28.2",  "25.1"),
            ("A100-G26A4B", "A100-SXM4-80GB", "2026-08-29", 32000, "161.0", "105.0"),
            ("E26-tp1-u95", "RX 7900 XT",     "2026-08-30", 12000, "96.9",  "79.1"),
            ("G26A4B",      "L4",             "2026-08-30", 32000, "52.4",  "44.1"),
            # the 2026-08-30 four-machine round
            ("G12",         "T4",             "2026-08-30", 32000, "20.3",  "9.0"),
            ("B8-tp1-u95",  "RX 7900 XT",     "2026-08-30",  6000, "46.6",  "44.1"),
            ("B8",          "L4",             "2026-08-30", 24000, "16.6",  "13.5"),
            ("Q38S",        "L4",             "2026-08-30",  8000, "15.9",  "15.4"),
            ("G31",         "A100-SXM4-80GB", "2026-08-30", 32000, "58.5",  "42.4"),
            ("G31-eager",   "L4",             "2026-08-30",  1000, "11.1",  "11.1")):
        pick = lambda c, _cfg=cfg, _m=mach, _d=date: next(
            r["decode_tok_s"] for r in XDEC
            if r["cfg"] == _cfg and r["ctx"] == c and r["tp"] == 1
            and r["spec"] is None and r["chart_grade"]
            and r["machine"] == _m and r["date"] == _d)
        shallow = 500
        ck("benchmarks README, single-card decode %s at 500" % cfg, want_s, pick(shallow))
        ck("benchmarks README, and %s at its deepest" % cfg, want_d, pick(ctx_deep))
        RETAINED[(cfg, mach)] = pick(ctx_deep) / pick(shallow) * 100.0
    # The retained column, recomputed rather than trusted. It is a ratio of two
    # numbers this file already checks, which is exactly why it was left ungated
    # and exactly why it should not have been: a ratio can be right about its
    # ends and wrong about which run they came from.
    for (cfg, mach), want in (
            (("A100-G12", "A100-SXM4-80GB"), "61.9"),
            (("A-12B-tp1", "RX 7900 XT"),    "72.6"),
            (("G12", "L4"),                  "88.8"),
            (("G12", "T4"),                  "44.3"),
            (("A100-G26A4B", "A100-SXM4-80GB"), "65.2"),
            (("E26-tp1-u95", "RX 7900 XT"),  "81.6"),
            (("G26A4B", "L4"),               "84.1"),
            (("B8-tp1-u95", "RX 7900 XT"),   "94.7"),
            (("B8", "L4"),                   "81.3"),
            (("Q38S", "L4"),                 "96.6"),
            (("G31", "A100-SXM4-80GB"),      "72.5"),
            (("G31-eager", "L4"),            "99.7")):
        ck("benchmarks README, %s on %s retains" % (cfg, mach), want, RETAINED[(cfg, mach)])
    # The T4's is the claim the table bolds: the only card here whose decode
    # more than halves across the ladder, and the ratios against the L4 that
    # say how it gets there.
    ck("benchmarks README, and the T4 is the only one to lose more than half", "1",
       1 if sum(1 for v in RETAINED.values() if v < 50.0) == 1
            and RETAINED[("G12", "T4")] < 50.0 else 0)
    ck("benchmarks README, the T4 against the L4 at 500", "0.72",
       RETAINED and next(r["decode_tok_s"] for r in XDEC if r["cfg"] == "G12"
                         and r["machine"] == "T4" and r["ctx"] == 500)
       / next(r["decode_tok_s"] for r in XDEC if r["cfg"] == "G12"
              and r["machine"] == "L4" and r["ctx"] == 500))
    ck("benchmarks README, and at 32K", "0.36",
       next(r["decode_tok_s"] for r in XDEC if r["cfg"] == "G12"
            and r["machine"] == "T4" and r["ctx"] == 32000)
       / next(r["decode_tok_s"] for r in XDEC if r["cfg"] == "G12"
              and r["machine"] == "L4" and r["ctx"] == 32000))

    # --- README.md, the five-machine chart and the table under it ------------
    # gemma-4-12B is the only model measured on all five, and the figure's whole
    # argument is that the five do not order the same way at 500 as at 32 K. Both
    # ends of every line, the retention, and each ratio the prose states.
    _M5 = {"a100":  ("A100-SXM4-80GB", "A100-G12",  "2026-08-29"),
           "pair":  ("RX 7900 XT",     "A-12B-tp2", "2026-08-24"),
           "one":   ("RX 7900 XT",     "A-12B-tp1", "2026-08-24"),
           "l4":    ("L4",             "G12",       "2026-08-30"),
           "t4":    ("T4",             "G12",       "2026-08-30")}

    def _m5(k, ctx):
        mach, cfg, date = _M5[k]
        return next(r["decode_tok_s"] for r in XDEC
                    if r["machine"] == mach and r["cfg"] == cfg and r["date"] == date
                    and r["ctx"] == ctx and r["chart_grade"])

    for k, w500, w32k, wret in (("a100", "115.0", "71.3", "61.9"),
                                ("pair", "59.9",  "41.4", "69.2"),
                                ("one",  "50.6",  "36.7", "72.6"),
                                ("l4",   "28.2",  "25.1", "88.8"),
                                ("t4",   "20.3",  "9.0",  "44.3")):
        ck("README five machines, %s at 500" % k, w500, _m5(k, 500))
        ck("README five machines, %s at 32K" % k, w32k, _m5(k, 32000))
        ck("README five machines, %s retains" % k, wret,
           _m5(k, 32000) / _m5(k, 500) * 100.0)
    # every one of the five is eleven rungs and every rung chart-grade, which is
    # what lets the figure draw them without a single gap
    for k, (mach, cfg, date) in _M5.items():
        got = [r for r in XDEC if r["machine"] == mach and r["cfg"] == cfg
               and r["date"] == date]
        ck("README five machines, %s is eleven rungs" % k, "11", len(got))
        ck("README five machines, and all of %s is chart-grade" % k, "11",
           sum(1 for r in got if r["chart_grade"]))
    # the three readings the prose leads with
    ck("README five machines, the A100 over the pair at 500", "1.92",
       _m5("a100", 500) / _m5("pair", 500), 0.005)
    ck("README five machines, and at 32K", "1.72",
       _m5("a100", 32000) / _m5("pair", 32000), 0.005)
    ck("README five machines, the A100's lead narrows with depth", "1",
       1 if (_m5("a100", 500) / _m5("pair", 500)
             > _m5("a100", 32000) / _m5("pair", 32000)) else 0)
    ck("README five machines, the second card at 500", "1.18",
       _m5("pair", 500) / _m5("one", 500), 0.005)
    ck("README five machines, and at 32K", "1.13",
       _m5("pair", 32000) / _m5("one", 32000), 0.005)
    ck("README five machines, the T4 against the L4 at 500", "0.72",
       _m5("t4", 500) / _m5("l4", 500), 0.01)
    ck("README five machines, and at 32K", "0.36",
       _m5("t4", 32000) / _m5("l4", 32000), 0.01)
    # "the T4 is last on both, and last on retention by a different mechanism",
    # and "the L4 is slowest but flattest" -- the two orderings the chart exists
    # to show disagree with each other
    _by500 = sorted(_M5, key=lambda k: _m5(k, 500))
    _byret = sorted(_M5, key=lambda k: _m5(k, 32000) / _m5(k, 500))
    ck("README five machines, the T4 is last on throughput at 500", "1",
       1 if _by500[0] == "t4" else 0)
    ck("README five machines, and last on retention too", "1",
       1 if _byret[0] == "t4" else 0)
    ck("README five machines, the L4 is slowest but one and flattest", "1",
       1 if _by500[1] == "l4" and _byret[-1] == "l4" else 0)
    ck("README five machines, so the two orderings are not the same", "1",
       1 if _by500 != _byret else 0)
    ck("README five machines, only the T4 loses more than half", "1",
       sum(1 for k in _M5 if _m5(k, 32000) / _m5(k, 500) < 0.5))
    # and the claim that lets the decode numbers be quoted at all: the patch the
    # T4 needs touches prefill only, so it travels on the prefill rows too
    ck("README five machines, the T4's rows carry the patch", "11",
       sum(1 for r in XDEC if r["machine"] == "T4"
           and r["patches"] == ["vllm#39018"]))
    ck("README five machines, and no other machine's decode row does", "0",
       sum(1 for r in XDEC if r["machine"] != "T4" and "vllm#39018" in r["patches"]))
    # The chart itself: it is generated, so what it draws has to be the rows.
    # Each circle's y is inverted through the chart's own axis and compared with
    # the value it claims to plot -- a chart with the right number of wrong
    # circles passes a count, and counting is all a count does.
    _svg = open(os.path.join(HERE, "..", "..", "docs", "assets",
                             "decode-five-machines-gemma4-12b.svg"),
                encoding="utf-8").read()
    _cols = {"a100": "#2ea36a", "pair": "#e05c48", "one": "#d99a24",
             "l4": "#8b6ee0", "t4": "#3f8fd4"}
    ck("README five machines, the chart draws five lines", "5",
       sum(1 for c in _cols.values() if ('stroke="%s"' % c) in _svg))
    ck("README five machines, and 55 points", "55",
       sum(_svg.count('fill="%s"/>' % c) for c in _cols.values()))
    # gen_best_charts.build(): T=76, PLOT_H is the band height, one band 0..vmax
    _T, _H, _VMAX = 76.0, 400.0, 130.0
    _unmap = lambda cy: (1.0 - (cy - _T) / _H) * _VMAX
    for k, c in _cols.items():
        _pts = re.findall(
            r'<circle cx="([\d.]+)" cy="([\d.]+)" r="3" fill="%s"/>' % re.escape(c),
            _svg)
        ck("README five machines, %s has eleven points in the chart" % k, "11",
           len(_pts))
        mach, cfg, date = _M5[k]
        _want = sorted(r["decode_tok_s"] for r in XDEC
                       if r["machine"] == mach and r["cfg"] == cfg
                       and r["date"] == date and r["chart_grade"])
        _got = sorted(_unmap(float(cy)) for _, cy in _pts)
        # cy carries one decimal, so a value is recoverable to 130/400/10 tok/s
        ck("README five machines, and every one of %s's is its measured value" % k,
           "11", sum(1 for a, b in zip(_want, _got) if abs(a - b) <= 0.04))
    # the dashed lines are exactly the patched ones, which is what the header
    # line promises a reader
    ck("README five machines, machines needing a patch", "3",
       sum(1 for k, (mach, cfg, date) in _M5.items()
           if any(r["patches"] for r in XDEC if r["machine"] == mach
                  and r["cfg"] == cfg and r["date"] == date)))
    ck("README five machines, and that many dashed lines in the chart", "3",
       sum(1 for c in _cols.values()
           if ('stroke="%s" stroke-width="2.4" stroke-linecap="round" '
               'stroke-dasharray="7 4"' % c) in _svg))

    # the two statements the table is there to make
    _a100_moe = next(r["decode_tok_s"] for r in XDEC if r["cfg"] == "A100-G26A4B"
                     and r["ctx"] == 500 and r["chart_grade"])
    _rdna_moe = next(r["decode_tok_s"] for r in XDEC if r["cfg"] == "E26-tp1-u95"
                     and r["ctx"] == 500 and r["chart_grade"])
    ck("benchmarks README, the A100 leads the MoE at 500", "1.66",
       _a100_moe / _rdna_moe, 0.005)
    ck("benchmarks README, and it is not the other way round", "1",
       1 if _a100_moe > _rdna_moe else 0)

    # --- Figure 2: prefill on one card of each kind --------------------------
    # The figure states two coefficients per line and four ratios between them,
    # and every one of those is recomputed here from prefill.jsonl through the
    # same fitter the projection uses -- so the figure and `--fits` cannot
    # disagree, and neither can drift from the rows.
    import build_prefill as _bpf
    XP = XFIG["prefill"]
    XPFROWS = [json.loads(l) for l in open(os.path.join(HERE, "..", "prefill.jsonl"))]
    ck("prefill figure, lines", "19", len(XP["series"]))
    ck("prefill figure, machines", "5", len({x["machine"] for x in XP["series"]}))
    ck("prefill figure, models", "7", len({x["model"] for x in XP["series"]}))
    ck("prefill figure, single-card lines", "12",
       sum(1 for x in XP["series"] if x["tp"] == 1))
    ck("prefill figure, and lines on the pair", "7",
       sum(1 for x in XP["series"] if x["tp"] == 2))
    # eight lit: the two models every machine ran, on all four machines
    ck("prefill figure, lit to start", "8",
       sum(1 for x in XP["series"] if x["lit"]))
    ck("prefill figure, and they are the models every machine ran", "2",
       len({x["model"] for x in XP["series"] if x["lit"]}))
    # every model this repository has a chart-grade prefill ladder for is drawn
    ck("prefill figure, models with a fittable ladder left undrawn", "0",
       len({f["model"] for f in _bpf.fits(XPFROWS)
            if f.get("rungs", 0) >= 4 and "b_us_tok" in f
            and f["machine"] == "RX 7900 XT" and f["spec"] is None}
           - {x["model"] for x in XP["series"]} - {"Qwen3.6-27B"}))
    # The whole reason the A100 was re-run. Four of the six lines are the CUDA
    # campaigns and had the cache off; one is the Radeon MoE, whose 0.23
    # container has it ON and produced no hits anyway -- its rounds agree to
    # 1.46 % at 500 and 0.03 % at 12 K -- and one is the 2026-08-24 Radeon
    # campaign, whose serve logs were not kept and which therefore records
    # neither. The flag is not the discriminator; repeatability is, and no
    # ungraded rung is drawn.
    ck("prefill figure, lines measured with prefix caching off", "10",
       sum(1 for x in XP["series"] if x["prefix_caching"] is False))
    ck("prefill figure, lines that had it on and show no hits", "2",
       sum(1 for x in XP["series"] if x["prefix_caching"] is True))
    ck("prefill figure, lines with no log to say", "7",
       sum(1 for x in XP["series"] if x["prefix_caching"] is None))
    # this column is for what was read: five logs survive and all five say
    # TRITON_ATTN, and none of the six records anything else
    ck("prefill figure, lines recording TRITON_ATTN", "9",
       sum(1 for x in XP["series"] if x["attn_backend"] == "TRITON_ATTN"))
    # Two lines do record a different one, and it is not an anomaly: vLLM sends
    # Qwen3.8 and Muse-Glimmer to FLASH_ATTN on the A100 and to TRITON_ATTN (or
    # an unrecorded backend) on the Radeons. That is why `backend_mixed` exists
    # -- a c ratio across those lines is a kernel difference as well as a card
    # one, and the figure has to say so rather than let it be assumed.
    ck("prefill figure, lines recording FLASH_ATTN", "3",
       sum(1 for x in XP["series"] if x["attn_backend"] == "FLASH_ATTN"))
    # --- the one line drawn on different terms -------------------------------
    # Its kernel was patched, so its c is not this card against the others, and
    # its b is not determined by its own ladder. Both facts have to travel with
    # the line rather than living only in the caption, and the line must be off
    # by default and out of the card-against-card ratios.
    _CAV = [x for x in XP["series"] if x.get("caveat")]
    ck("prefill figure, lines drawn on different terms", "1", len(_CAV))
    ck("prefill figure, and it is the T4's", "1",
       1 if _CAV and _CAV[0]["machine"] == "t4" else 0)
    ck("prefill figure, and it names the patch that makes it different", "1",
       1 if _CAV and _CAV[0]["caveat"]["patch"] == "vllm#39018" else 0)
    ck("prefill figure, and it is the only line carrying that patch", "1",
       sum(1 for x in XP["series"] if "vllm#39018" in
           [pp for r in XPFROWS if r["cfg"] == x["cfg"] and r["machine"] == x["machine_name"]
            for pp in r["patches"]]))
    ck("prefill figure, and a caveated line is never lit without being asked", "0",
       sum(1 for x in _CAV if x["lit"]))
    ck("prefill figure, and it is left out of the card-against-card ratios", "0",
       sum(1 for c in XP["compare"] if c["machine"] == "t4"))
    # The ratios are still every uncaveated single-card line for those models,
    # so excluding one is not quietly excluding others.
    ck("prefill figure, ratios drawn", "4", len(XP["compare"]))

    ck("prefill figure, and no line records anything else", "0",
       sum(1 for x in XP["series"]
           if x["attn_backend"] not in (None, "TRITON_ATTN", "FLASH_ATTN")))
    # Which rung each line had to drop, and it is not one story. On the four
    # CUDA lines it is the shallowest -- the cold engine's first request, which
    # the runner did not discard until 2026-08-30. On both Radeon lines it is
    # the 4 000 rung, on two different models in two different campaigns, round
    # 1 slower than round 2 each time, and this repository does not know why.
    _drop = {(x["machine"], x["model"]): [d["ctx"] for d in x["dropped"]]
             for x in XP["series"]}
    ck("prefill figure, lines dropping their shallowest rung", "10",
       sum(1 for d in _drop.values() if d == [500]))
    ck("prefill figure, lines dropping the 4000 rung", "4",
       sum(1 for d in _drop.values() if d == [4000]))
    ck("prefill figure, lines dropping nothing", "5",
       sum(1 for d in _drop.values() if not d))

    _xfits = {(f["machine"], f["cfg"], f["date"]): f for f in _bpf.fits(XPFROWS)}
    _bad_fit = _bad_pts = _ungraded = 0
    for x in XP["series"]:
        f = _xfits.get((x["machine_name"], x["cfg"], x["date"]))
        if not f or abs(f["b_us_tok"] - x["fit"]["b_us_tok"]) > 1e-9 \
                or abs(f["c_ns_tok2"] - x["fit"]["c_ns_tok2"]) > 1e-9:
            _bad_fit += 1
        rows = {r["ctx"]: r for r in XPFROWS
                if r["machine"] == x["machine_name"] and r["cfg"] == x["cfg"]
                and r["date"] == x["date"]}
        for p in x["points"]:
            r = rows.get(p["ctx"])
            if not r or abs(r["prefill_tok_s"] - p["tok_s"]) > 1e-9:
                _bad_pts += 1
            elif not r["chart_grade"]:
                _ungraded += 1
    ck("prefill figure, lines whose fit is not the projection's", "0", _bad_fit)
    ck("prefill figure, points that do not match prefill.jsonl", "0", _bad_pts)
    # a rung whose two rounds disagree is not a measurement; none may be drawn
    ck("prefill figure, ungraded rungs drawn", "0", _ungraded)

    # Which models can be compared across machines without also comparing
    # kernels. Three states, because "no contradiction recorded" is not the same
    # claim as "known to be the same kernel", and only the first is true where a
    # serve log did not survive a reclaim.
    _bm = {m["model"]: m for m in XP["backend_mixed"]}
    ck("prefill figure, models drawn on more than one machine", "6", len(_bm))
    ck("prefill figure, and one is known to differ in kernel", "1",
       sum(1 for m in _bm.values() if m["kernel"] == "different"))
    ck("prefill figure, and that one is Qwen3.8", "1",
       1 if _bm["Qwen3.8-27B"]["kernel"] == "different" else 0)
    ck("prefill figure, models known to share a kernel", "1",
       sum(1 for m in _bm.values() if m["kernel"] == "same"))
    # the cleanest cross-machine comparison in the repository, and it is the
    # model a100-vs-two-radeons is about
    ck("prefill figure, and that one is gemma-4-31B", "1",
       1 if _bm["gemma-4-31B-it"]["kernel"] == "same" else 0)
    ck("prefill figure, the rest have a backend nobody recorded", "4",
       sum(1 for m in _bm.values() if m["kernel"] == "unknown"))

    # The argument. Against one 7900 XT the A100 leads on both terms and leads
    # by more on the quadratic; the L4 is BEHIND on the linear term and ahead on
    # the quadratic, which is the crossing that a single tok/s number hides.
    _cmp = {(c["model"], c["machine"]): c for c in XP["compare"]}
    ck("prefill figure, ratios stated", "4", len(XP["compare"]))
    ck("prefill figure, A100 b on the 12B", "3.29",
       _cmp[("gemma-4-12B-it", "a100")]["b_ratio"], 0.01)
    ck("prefill figure, A100 c on the 12B", "6.67",
       _cmp[("gemma-4-12B-it", "a100")]["c_ratio"], 0.01)
    ck("prefill figure, so attention is the wider gap", "1",
       1 if _cmp[("gemma-4-12B-it", "a100")]["c_ratio"]
       > 2 * _cmp[("gemma-4-12B-it", "a100")]["b_ratio"] - 0.5 else 0)
    ck("prefill figure, L4 b on the 12B", "0.90",
       _cmp[("gemma-4-12B-it", "l4")]["b_ratio"], 0.01)
    ck("prefill figure, L4 c on the 12B", "3.01",
       _cmp[("gemma-4-12B-it", "l4")]["c_ratio"], 0.01)
    ck("prefill figure, the L4 loses on b and wins on c", "1",
       1 if _cmp[("gemma-4-12B-it", "l4")]["b_ratio"] < 1.0
       < _cmp[("gemma-4-12B-it", "l4")]["c_ratio"] else 0)
    ck("prefill figure, A100 b on the MoE", "5.75",
       _cmp[("gemma-4-26B-A4B", "a100")]["b_ratio"], 0.01)
    ck("prefill figure, A100 c on the MoE", "5.71",
       _cmp[("gemma-4-26B-A4B", "a100")]["c_ratio"], 0.01)
    # Two models on one chart have to be two colours a reader can tell apart at
    # stroke width, and "there are seven colours" does not establish that: the
    # palette's closest pair by CIE76 is m1 against m7 at 17.8, where every
    # other pair is 40 or more, and the first draft of this figure drew both of
    # its models in exactly that pair. The order in genfig-index.py pulls
    # Figure 2's models forward to avoid it; this is the check that says the
    # avoidance worked rather than that it was intended.
    _pal = re.findall(r"--m(\d):\s*(#[0-9a-fA-F]{6})",
                      open(os.path.join(HERE, "..", "..", "site", "src",
                                        "index-extra.css"), encoding="utf-8").read())
    _pal = {int(i): h for i, h in _pal}

    def _lab(h):
        r, g, b = (int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))
        f = lambda c: c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        r, g, b = f(r), f(g), f(b)
        X, Y, Z = (0.4124 * r + 0.3576 * g + 0.1805 * b,
                   0.2126 * r + 0.7152 * g + 0.0722 * b,
                   0.0193 * r + 0.1192 * g + 0.9505 * b)
        n = lambda t: t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
        fx, fy, fz = n(X / 0.95047), n(Y / 1.0), n(Z / 1.08883)
        return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))

    def _de(a, b):
        return sum((x - y) ** 2 for x, y in zip(_lab(a), _lab(b))) ** 0.5

    _slot = {m: i % 7 + 1 for i, m in enumerate(XB["model_order"])}
    _pf_models = sorted({x["model"] for x in XP["series"]})
    _pf_lit = sorted({x["model"] for x in XP["series"] if x["lit"]})
    # Colour carries the model and stroke carries the machine, so a (machine,
    # model) pair drawn twice would be two lines a reader cannot tell apart at
    # all. Fourteen lines, fourteen distinct pairs -- verified in the page as
    # fourteen distinct (stroke-dasharray, stroke) combinations.
    ck("prefill figure, lines drawn identically to another", "0",
       len(XP["series"]) - len({(x["machine"], x["model"]) for x in XP["series"]}))

    ck("prefill figure, its models are one palette slot each", str(len(_pf_models)),
       len({_slot[m] for m in _pf_models}))
    # The default view is what has to be legible without a click, so the lit
    # models are the ones held to a distance. m1 against m7 is 17.8 and every
    # other pair is 40 or more; the first draft drew both lit models in exactly
    # that pair.
    _worst = min((_de(_pal[_slot[a]], _pal[_slot[b]])
                  for i, a in enumerate(_pf_lit) for b in _pf_lit[i + 1:]),
                 default=None)
    ck("prefill figure, the lit models are far enough apart to tell", "78.7",
       _worst, 0.01)
    ck("prefill figure, which the palette's closest pair is not", "1",
       1 if _worst >= 40 else 0)

    # What the second card buys, on the coefficients that reproduce. This is the
    # replacement for the claim docs/benchmarks.md section 4 withdrew, and the
    # shape is the point: c gains more than b on every model, because attention
    # needs no communication and the GEMMs do.
    _tg = {g["model"]: g for g in XP["tp_gain"]}
    ck("prefill figure, models with both topologies", "3", len(_tg))
    ck("prefill figure, and c gains more than b on every one", "3",
       sum(1 for g in _tg.values() if g["c_gain"] > g["b_gain"]))
    ck("prefill figure, second card on the 12B, b", "1.48",
       _tg["gemma-4-12B-it"]["b_gain"], 0.01)
    ck("prefill figure, and its c", "2.22", _tg["gemma-4-12B-it"]["c_gain"], 0.01)
    ck("prefill figure, second card on the MoE, b", "1.44",
       _tg["gemma-4-26B-A4B"]["b_gain"], 0.01)
    ck("prefill figure, and the MoE's c", "1.91",
       _tg["gemma-4-26B-A4B"]["c_gain"], 0.01)
    ck("prefill figure, second card on Qwen3-8B, b", "1.23",
       _tg["Qwen3-8B"]["b_gain"], 0.01)
    ck("prefill figure, and Qwen3-8B's c", "2.08", _tg["Qwen3-8B"]["c_gain"], 0.01)

    # The flag this repository called free. Same model, same day, same stack,
    # one flag: it buys decode at depth and sells prefill at depth, and the two
    # directions are what makes it a trade rather than a win.
    _bt = XP["backend_tradeoff"]
    ck("prefill figure, the Triton flag buys decode at 32K", "1.15",
       _bt["decode_gain"], 0.01)
    ck("prefill figure, and sells prefill at 32K", "1.40",
       _bt["prefill_gain"], 0.01)
    ck("prefill figure, and the two go opposite ways", "1",
       1 if _bt["prefill_gain"] > 1 and _bt["decode_gain"] > 1 else 0)
    ck("prefill figure, the backend's quadratic term, ROCm", "3.43",
       _bt["c_rocm"], 0.01)
    ck("prefill figure, and Triton's", "18.44", _bt["c_triton"], 0.01)
    # both arms are the same model on the same day, or it is not one flag
    ck("prefill figure, and the pair differs only in the flag", "1",
       1 if _bt["date"] == "2026-08-29" and _bt["model"] == "Qwen3.8-27B" else 0)

    # the machine ids have to be Figure 1's, or a reader carrying a stroke or a
    # colour between the two figures is carrying it to the wrong line
    ck("prefill figure, its machines are Figure 1's", "1",
       1 if {x["machine"] for x in XP["series"]} <=
       {m["id"] for m in XB["machines"]} else 0)

    # --- what section 4 of benchmarks.md withdrew, and what it kept ----------
    # The claims retired on 2026-08-30 were retired because nothing recomputed
    # them. Their replacements are recomputed here, from prefill.jsonl, so the
    # same thing cannot happen twice.
    XPF = [json.loads(l) for l in open(os.path.join(HERE, "..", "prefill.jsonl"))]

    def xfit(cfg, date):
        rs = sorted([r for r in XPF if r["cfg"] == cfg and r["date"] == date
                     and r["chart_grade"]], key=lambda r: r["ctx"])
        S = [r["prompt_tokens"] for r in rs]
        T = [min(r["values"]) for r in rs]
        n = len(S)
        P = [[sum(x ** (i + j) for x in S) for j in range(3)] for i in range(3)]
        q = [sum(T[k] * S[k] ** i for k in range(n)) for i in range(3)]
        m = [row[:] + [q[i]] for i, row in enumerate(P)]
        for col in range(3):
            pv = max(range(col, 3), key=lambda r: abs(m[r][col]))
            m[col], m[pv] = m[pv], m[col]
            for r in range(3):
                if r != col and m[col][col]:
                    f = m[r][col] / m[col][col]
                    for k in range(col, 4):
                        m[r][k] -= f * m[col][k]
        return [m[i][3] / m[i][i] for i in range(3)]

    def xcell(cfg, date, ctx):
        return next(r for r in XPF if r["cfg"] == cfg and r["date"] == date
                    and r["ctx"] == ctx)

    # a does not reproduce, and one fit returns it below zero
    for cfg, date, want in (("B-8B-tp2", "2026-07-25", 79.3),
                            ("B-8B-tp2", "2026-08-24", 30.5),
                            ("A-12B-tp2", "2026-07-25", 9.8),
                            ("A-12B-tp2", "2026-08-24", 99.8),
                            ("A-12B-tp1", "2026-07-25", 70.1),
                            ("A-12B-tp1", "2026-08-24", -22.1)):
        ck(f"benchmarks.md s4, a for {cfg} {date}", f"{want}",
           xfit(cfg, date)[0] * 1000, 0.01)
    ck("benchmarks.md s4, and one of them is below zero", "1",
       1 if xfit("A-12B-tp1", "2026-08-24")[0] < 0 else 0)

    # b and c do reproduce, and the two ratios the subsection now rests on
    for date, wb, wc in (("2026-07-25", 1.29, 1.87), ("2026-08-24", 1.23, 2.08)):
        one, two = xfit("B-8B-tp1", date), xfit("B-8B-tp2", date)
        ck(f"benchmarks.md s4, b improves {date}", f"{wb}", one[1] / two[1], 0.01)
        ck(f"benchmarks.md s4, c improves {date}", f"{wc}", one[2] / two[2], 0.01)

    # the crossover was a first-request cost, and it moved between arms
    for cfg, date, want_tps, want_rng in (
            ("B-8B-tp1", "2026-07-25", 3444, 0.87),
            ("B-8B-tp2", "2026-07-25", 2019, 22.13),
            ("B-8B-tp1", "2026-08-24", 3265, 18.24),
            ("B-8B-tp2", "2026-08-24", 3690, 1.72)):
        c500 = xcell(cfg, date, 500)
        ck(f"benchmarks.md s4, {cfg} {date} at 500", f"{want_tps}",
           c500["prefill_tok_s"], 0.01)
        ck(f"benchmarks.md s4, and its rounds differ by {want_rng}",
           f"{want_rng}", c500["range_pct"], 0.01)
    ck("benchmarks.md s4, the ungraded arm at 500 moved between campaigns", "1",
       1 if (not xcell("B-8B-tp2", "2026-07-25", 500)["chart_grade"]
             and xcell("B-8B-tp1", "2026-07-25", 500)["chart_grade"]
             and not xcell("B-8B-tp1", "2026-08-24", 500)["chart_grade"]
             and xcell("B-8B-tp2", "2026-08-24", 500)["chart_grade"]) else 0)
    ck("benchmarks.md s4, and where it happens round 1 is the slow one", "2",
       sum(1 for cfg, date in (("B-8B-tp2", "2026-07-25"), ("B-8B-tp1", "2026-08-24"))
           if xcell(cfg, date, 500)["values"][-1]
           == max(xcell(cfg, date, 500)["values"])))
    # and the two campaigns disagree about which arm is faster there
    ck("benchmarks.md s4, July says one card and August says two", "1",
       1 if (xcell("B-8B-tp1", "2026-07-25", 500)["prefill_tok_s"]
             > xcell("B-8B-tp2", "2026-07-25", 500)["prefill_tok_s"]
             and xcell("B-8B-tp2", "2026-08-24", 500)["prefill_tok_s"]
             > xcell("B-8B-tp1", "2026-08-24", 500)["prefill_tok_s"]) else 0)

    # --- the table that replaced what section 4 withdrew ---------------------
    # Twelve numbers of prose, each recomputed from prefill.jsonl through the
    # projection's own fitter. The claims this replaced went unchecked for a
    # month and were wrong; these are checked.
    _pf_fit = {(f["machine"], f["cfg"], f["date"]): f for f in _bpf.fits(XPFROWS)}
    for mach, cfg, date, wb, wc in (
            ("A100-SXM4-80GB", "G12",         "2026-08-30", "145.7", "3.62"),
            ("RX 7900 XT",     "A-12B-tp1",   "2026-08-24", "479.0", "24.16"),
            ("L4",             "G12",         "2026-08-30", "534.7", "8.03"),
            ("A100-SXM4-80GB", "G26A4B",      "2026-08-30", "62.6",  "2.30"),
            ("RX 7900 XT",     "E26-tp1-u95", "2026-08-30", "360.0", "13.13"),
            ("L4",             "G26A4B",      "2026-08-30", "204.4", "5.53"),
            # the fifth machine, whose row the table marks as not comparable
            ("T4",             "G12",         "2026-08-30", "3033.2", "218.89"),
            # and Qwen3-8B, the second model with more than one single card
            ("RX 7900 XT",     "B8-tp1-u95",  "2026-08-30", "206.7", "8.87"),
            ("L4",             "B8",          "2026-08-30", "288.3", "5.38")):
        f = _pf_fit[(mach, cfg, date)]
        ck("benchmarks.md s4, b for %s on %s" % (cfg, mach), wb, f["b_us_tok"], 0.001)
        ck("benchmarks.md s4, c for %s on %s" % (cfg, mach), wc, f["c_ns_tok2"], 0.001)
    # and the two ratios the prose leads with
    _r = lambda a, b: _pf_fit[a]["b_us_tok"] / _pf_fit[b]["b_us_tok"]
    _rc = lambda a, b: _pf_fit[a]["c_ns_tok2"] / _pf_fit[b]["c_ns_tok2"]
    _rad12 = ("RX 7900 XT", "A-12B-tp1", "2026-08-24")
    _a100_12 = ("A100-SXM4-80GB", "G12", "2026-08-30")
    _l4_12 = ("L4", "G12", "2026-08-30")
    ck("benchmarks.md s4, the A100 on b against one Radeon", "3.3",
       _r(_rad12, _a100_12), 0.01)
    ck("benchmarks.md s4, and on c", "6.7", _rc(_rad12, _a100_12), 0.01)
    ck("benchmarks.md s4, the L4 is slower on b", "0.90", _r(_rad12, _l4_12), 0.01)
    ck("benchmarks.md s4, and better on c", "3.0", _rc(_rad12, _l4_12), 0.01)
    ck("benchmarks.md s4, so the L4 loses b and wins c", "1",
       1 if _r(_rad12, _l4_12) < 1.0 < _rc(_rad12, _l4_12) else 0)
    # Qwen3-8B splits the same way and by different amounts, which is what makes
    # it worth stating twice rather than generalising from the 12B alone
    _rad8 = ("RX 7900 XT", "B8-tp1-u95", "2026-08-30")
    _l4_8 = ("L4", "B8", "2026-08-30")
    ck("benchmarks README, Qwen3-8B: the Radeon wins b", "1.39", _r(_l4_8, _rad8), 0.01)
    ck("benchmarks README, and loses c", "1.65", _rc(_rad8, _l4_8), 0.01)
    ck("benchmarks README, so the split has the same sign as the 12B's", "1",
       1 if (_r(_rad12, _l4_12) < 1.0 < _rc(_rad12, _l4_12))
            == (_r(_l4_8, _rad8) > 1.0 > 1 / _rc(_rad8, _l4_8)) else 0)
    # The T4's row carries a footnote rather than a comparison, and the two
    # numbers that footnote states are recomputed here so it cannot drift: what
    # changing which VM supplies the 32 000 rung does to b and to c.
    _t4rows = [r for r in XPFROWS if r["machine"] == "T4" and r["chart_grade"]]
    _base = [(r["prompt_tokens"], min(r["values"])) for r in _t4rows if r["ctx"] != 32000]
    def _fit3(pairs):
        S = [x for x, _ in pairs]; T = [t for _, t in pairs]; n = len(S)
        A = [[float(n), sum(S), sum(x * x for x in S)],
             [sum(S), sum(x * x for x in S), sum(x ** 3 for x in S)],
             [sum(x * x for x in S), sum(x ** 3 for x in S), sum(x ** 4 for x in S)]]
        y = [sum(T), sum(a * b for a, b in zip(S, T)), sum(a * a * b for a, b in zip(S, T))]
        return _bpf.solve(A, y)
    _t4d = _fit3(_base + [(32013, 316.8056)])
    _t4c = _fit3(_base + [(32013, 331.4023)])
    ck("benchmarks README, which VM supplied the T4's deepest rung moves b by", "29.9",
       abs(_t4d[1] - _t4c[1]) / _t4c[1] * 100.0, 0.005)
    ck("benchmarks README, and c by", "12.8",
       abs(_t4d[2] - _t4c[2]) / _t4c[2] * 100.0, 0.005)
    ck("benchmarks README, the two VMs disagree at that rung by", "4.61",
       abs(331.4023 - 316.8056) / 316.8056 * 100.0, 0.005)
    # and why the linear term is the one that absorbs it: this curve is
    # quadratic-dominated where no other line here is
    _bS = _t4d[1] * 32013
    _cS2 = _t4d[2] * 32013 ** 2
    ck("benchmarks README, the T4's quadratic term at 32K, seconds", "224", _cS2, 0.01)
    ck("benchmarks README, against its linear term", "97", _bS, 0.01)
    ck("benchmarks README, so the quadratic dominates here and nowhere else", "1",
       1 if _cS2 > _bS else 0)
    # the second-card table, which is the claim that replaced the 76 ms one
    _tg2 = {g["model"]: g for g in XP["tp_gain"]}
    for model, wb, wc in (("gemma-4-12B-it", "1.48", "2.22"),
                          ("gemma-4-26B-A4B", "1.44", "1.91"),
                          ("Qwen3-8B", "1.23", "2.08")):
        ck("benchmarks.md s4, second card on %s, b" % model, wb,
           _tg2[model]["b_gain"], 0.01)
        ck("benchmarks.md s4, second card on %s, c" % model, wc,
           _tg2[model]["c_gain"], 0.01)
    ck("benchmarks.md s4, and c gains more on all three", "3",
       sum(1 for g in _tg2.values() if g["c_gain"] > g["b_gain"]))

    # --- the A100's second pass kept its serve logs, so check them ----------
    # The first pass lost its logs to a reclaim and its backends survive only in
    # model_meta. These three have both, so the projection's backend column can
    # be checked against the log it claims to come from rather than trusted.
    _A30 = os.path.join(HERE, "..", "cuda-a100", "campaign-2026-08-30", "logs")
    _BR = re.compile(r"Using (?:AttentionBackendEnum\.)?([A-Z0-9_]+)(?: attention)? backend")
    _VIT = re.compile(r"vit attention|MMEncoderAttention")

    def _logbackend(name):
        for line in open(os.path.join(_A30, name), errors="ignore"):
            if _VIT.search(line):
                continue
            m = _BR.search(line)
            if m:
                return m.group(1)
        return None

    _prow = {r["cfg"]: r for r in XPFROWS
             if r["machine"] == "A100-SXM4-80GB" and r["date"] == "2026-08-30"}
    for cfg, want in (("G31", "TRITON_ATTN"), ("Q38", "FLASH_ATTN"),
                      ("MG30", "FLASH_ATTN")):
        ck("A100 2026-08-30, %s's backend is what its log says" % cfg, "1",
           1 if _logbackend("serve-%s.log" % cfg) == want
           and _prow[cfg]["attn_backend"] == want else 0)
    # and the two forms are both present across these three, which is the bug
    # that left the 2026-08-29 campaign with no backend at all
    ck("A100 2026-08-30, logs using the enum form", "3",
       sum(1 for n in ("serve-G31.log", "serve-Q38.log", "serve-MG30.log")
           if "Using AttentionBackendEnum." in open(os.path.join(_A30, n),
                                                    errors="ignore").read()))
    ck("A100 2026-08-30, and logs using the other form", "2",
       sum(1 for n in ("serve-G31.log", "serve-Q38.log", "serve-MG30.log")
           if "attention backend out of potential backends"
           in open(os.path.join(_A30, n), errors="ignore").read()))

    # --- the T4 pre-flight, which is evidence for an upstream report --------
    # Its README states what three backends did on sm75. Every number is read
    # back out of preflight.jsonl and the serve logs it cites, because the
    # report will quote them and a number nothing recomputes is how this
    # repository has been wrong before.
    _T4 = os.path.join(HERE, "..", "cuda-t4", "preflight-2026-08-30")
    _t4rows = [json.loads(l) for l in open(os.path.join(_T4, "preflight.jsonl"))
               if l.strip()]
    ck("T4 preflight, engine starts recorded", "3", len(_t4rows))
    ck("T4 preflight, and every one crashed", "3",
       sum(1 for r in _t4rows if r.get("status") == "crash"))
    # W4A16 is not the wall: it loads, and it takes Marlin
    ck("T4 preflight, and every one loaded W4A16 on Marlin", "3",
       sum(1 for r in _t4rows if r.get("wna16_kernel") == "MarlinLinearKernel"))
    # memory is not the wall either
    _u90 = [r for r in _t4rows if r.get("util") is None]
    _u95 = [r for r in _t4rows if r.get("util") == "0.95"]
    ck("T4 preflight, KV at the default utilisation", "0.65", float(_u90[0]["kv_gib"]))
    ck("T4 preflight, and at util 0.95 with one sequence", "3.5",
       float(_u95[0]["kv_gib"]))
    ck("T4 preflight, which is this many tokens", "55809",
       float(_u95[0]["kv_tokens"]))
    ck("T4 preflight, more than the deepest rung needs", "1",
       1 if float(_u95[0]["kv_tokens"]) > 33000 else 0)
    ck("T4 preflight, weights resident", "8.28", float(_t4rows[0]["weights_gib"]))

    def _t4log(name):
        return open(os.path.join(_T4, "logs", name), errors="ignore").read()

    _tri, _flex, _fa = (_t4log("serve-T4-G12-triton.log"),
                        _t4log("serve-T4-G12-flex.log"),
                        _t4log("serve-T4-G12-flash.log"))
    # the shared-memory ceiling and what each backend asked for
    ck("T4 preflight, Turing's shared-memory ceiling", "65536",
       float(re.search(r"Hardware limit:?\s*(\d+)", _tri).group(1)))
    ck("T4 preflight, what TRITON_ATTN asked for", "98304",
       float(re.search(r"Required: (\d+)", _tri).group(1)))
    ck("T4 preflight, what FLEX_ATTENTION asked for", "163840",
       float(re.search(r"Required: (\d+)", _flex).group(1)))
    ck("T4 preflight, both are over the ceiling", "2",
       sum(1 for t in (_tri, _flex)
           if float(re.search(r"Required: (\d+)", t).group(1))
           > float(re.search(r"Hardware limit:?\s*(\d+)", t).group(1))))
    # the selector accepted the two that cannot run and rejected the one that says so
    ck("T4 preflight, backends the selector accepted", "2",
       sum(1 for t in (_tri, _flex) if "Using AttentionBackendEnum." in t))
    ck("T4 preflight, and FLASH_ATTN is the one it rejected", "1",
       1 if "compute capability not supported" in _fa
       and "Using AttentionBackendEnum." not in _fa else 0)
    # the sampler, a different subsystem, is honest about sm75
    ck("T4 preflight, the sampler names the capability it lacks", "1",
       1 if "unsupported compute capability 7.5" in _tri else 0)

    # --- hybrid fig5: section 6's prefill claim, now with data under it ------
    # The prose said "throughput improves with length, 805 -> 880 tok/s" from
    # two hand-typed numbers, one model, one machine, one stack. Both numbers
    # are right; the generalisation to the architecture is what these check.
    _F = json.load(open(os.path.join(HERE, "..", "..", "site", "src", "figures.json")))
    _f5 = _F["fig5"]["series"]
    ck("hybrid fig5, lines", "6", len(_f5))
    ck("hybrid fig5, hybrid lines", "3", _F["fig5"]["hybrid_lines"])
    ck("hybrid fig5, and only one of them rises", "1",
       sum(1 for x in _f5 if x["arch"] == "hybrid SSM" and x["change_pct"] > 0))
    ck("hybrid fig5, and it is Qwen3.6", "1",
       1 if _F["fig5"]["hybrid_that_rises"] == "Qwen3.6-27B" else 0)
    _q36 = next(x for x in _f5 if x["model"] == "Qwen3.6-27B")
    ck("hybrid fig5, the rise the prose quotes", "9.9", _q36["change_pct"], 0.01)

    # 2026-08-31: fig5 was built to put data under section 6's prefill claim and
    # the claim was never changed to match what it found. The paragraph now says
    # the rise is this configuration's, so these read the paragraph.
    _hy = {}
    for _r in XPF:
        if (_r.get("model") in ("Qwen3.6-27B", "Qwen3.8-27B") and _r["chart_grade"]
                and _r.get("spec") is None):
            _hy.setdefault((_r["cfg"], _r["machine"], _r["date"]), []).append(_r)
    _lad = []
    for _k, _rs in _hy.items():
        _rs.sort(key=lambda r: r["ctx"])
        if len(_rs) > 1:
            _lad.append((_rs[-1]["prefill_tok_s"] / _rs[0]["prefill_tok_s"] - 1) * 100)
    # the half of the sentence that does hold: the dense arms measured beside it
    _dn = []
    for _cfg in ("B-8B-tp2", "C-31B-tp2", "A-12B-tp2"):
        _rs = sorted([r for r in XPF if r["cfg"] == _cfg and r["date"] == "2026-07-25"
                      and r["chart_grade"]], key=lambda r: r["ctx"])
        _dn.append((_rs[-1]["prefill_tok_s"] / _rs[0]["prefill_tok_s"] - 1) * 100)
    ck("hybrid section 6, the dense arms beside it, best case", "-36",
       max(_dn), 0.5)
    ck("hybrid section 6, and worst", "-44", min(_dn), 0.5)
    # why it used to read 8: the 8B's 500 rung fails the repeatability cut and
    # the old number took the better of its two rounds
    _b8 = [r for r in XPF if r["cfg"] == "B-8B-tp2" and r["date"] == "2026-07-25"
           and r["ctx"] == 500]
    ck("hybrid section 6, the 8B's 500 rung is not chart-grade", "0",
       1 if _b8[0]["chart_grade"] else 0)
    ck("hybrid section 6, and its two rounds are this far apart", "22",
       _b8[0]["range_pct"], 0.5)
    ck("hybrid section 6, stock hybrid-SSM prefill ladders", "6", len(_lad))
    ck("hybrid section 6, and only one of them rises", "1",
       sum(1 for x in _lad if x > 0))
    ck("hybrid section 6, the other five fall", "5", sum(1 for x in _lad if x < 0))

    def _pf(cfg, machine, date):
        rs = sorted([r for r in XPF if r["cfg"] == cfg and r["machine"] == machine
                     and r["date"] == date and r["chart_grade"]], key=lambda r: r["ctx"])
        return (rs[-1]["prefill_tok_s"] / rs[0]["prefill_tok_s"] - 1) * 100

    for _lang, _fn in (("EN", "hybrid-ssm-collapse.html"),
                       ("ZH", "hybrid-ssm-collapse.zh.html")):
        _t = flat[_fn]
        ck("hybrid section 6 %s, the sibling falls on 0.27" % _lang, "7.5",
           -_pf("Q38-tp2", "RX 7900 XT", "2026-08-29"))
        ck("hybrid section 6 %s, and on 0.23" % _lang, "11.8",
           -_pf("D8-27B-tp2", "RX 7900 XT", "2026-08-24"))
        ck("hybrid section 6 %s, and on the A100" % _lang, "8.1",
           -_pf("Q38", "A100-SXM4-80GB", "2026-08-30"))
        ck("hybrid section 6 %s, the MoE rises further, July" % _lang, "23.8",
           _pf("E-26B-tp2", "RX 7900 XT", "2026-07-25"))
        ck("hybrid section 6 %s, and August" % _lang, "21.9",
           _pf("E-26B-tp2", "RX 7900 XT", "2026-08-24"))
        # the dense range, read out of the sentence rather than only recomputed:
        # it was published as 8-44 and the 8 came from a rung that fails the cut
        _dr = re.search(r"(?:loses|都掉了)\s*([0-9]+)[-–]([0-9]+)\s*%", _t)
        ck("hybrid section 6 %s, states the dense range" % _lang, "2",
           len(_dr.groups()) if _dr else 0)
        _dr = _dr or re.match(r"()()", "")
        ck("hybrid section 6 %s, dense best case" % _lang,
           "-" + (_dr.group(1) or "0"), max(_dn), 0.5)
        ck("hybrid section 6 %s, dense worst case" % _lang,
           "-" + (_dr.group(2) or "0"), min(_dn), 0.5)
        ck("hybrid section 6 %s, quotes all five" % _lang, "5",
           sum(1 for _v in ("7.5 %", "11.8 %", "8.1 %", "23.8 %", "21.9 %")
               if fl(_v) in _t))
        # the retracted generalisation, in both languages
        ck("hybrid section 6 %s, no longer credits the architecture" % _lang, "0",
           1 if (fl("behaves as the architecture promises") in _t
                 or fl("正如架构所承诺") in _t) else 0)
    ck("hybrid fig5, from this rate", "802", _q36["shallow_tok_s"], 0.01)
    ck("hybrid fig5, to this one", "881", _q36["deep_tok_s"], 0.01)
    # its sibling, same architecture in the ledger's own column, on two machines
    for mach, want in (("RX 7900 XT", "-7.5"), ("A100-SXM4-80GB", "-8.1")):
        x = next(x for x in _f5 if x["model"] == "Qwen3.8-27B" and x["machine"] == mach)
        ck("hybrid fig5, Qwen3.8 on %s" % mach, want, x["change_pct"], 0.01)
    ck("hybrid fig5, so the sibling falls on both machines", "2",
       sum(1 for x in _f5 if x["model"] == "Qwen3.8-27B" and x["change_pct"] < 0))
    # the half of the contrast that does hold, and now on two machines
    ck("hybrid fig5, dense lines", "3", sum(1 for x in _f5 if x["arch"] == "dense"))
    ck("hybrid fig5, and every one loses more than 30%", "3",
       sum(1 for x in _f5 if x["arch"] == "dense" and x["change_pct"] < -30))
    ck("hybrid fig5, the worst of them", "-42.8",
       min(x["change_pct"] for x in _f5 if x["arch"] == "dense"), 0.01)
    # every line recomputes from the projection
    _pfix = {(r["machine"], r["cfg"], r["date"], r["ctx"]): r for r in XPFROWS}
    ck("hybrid fig5, lines that do not match prefill.jsonl", "0",
       sum(1 for x in _f5
           if abs(_pfix[(x["machine"], x["cfg"], x["date"],
                         x["deep_ctx"])]["prefill_tok_s"] - x["deep_tok_s"]) > 1e-9))

    failed = [c for c in checks if not c[0]]
    for ok, where, claim, value, allowed in checks:
        if verbose or not ok:
            print(f"  {'ok  ' if ok else 'FAIL'} {where:<44} prose {claim:>8}   "
                  f"data {value:>9.3f}   allowed +-{allowed:.4g}")
    print(f"\n{len(checks) - len(failed)}/{len(checks)} figures agree with the data files")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
