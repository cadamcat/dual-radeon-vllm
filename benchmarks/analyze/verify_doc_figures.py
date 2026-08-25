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
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
JULY = os.path.join(HERE, "..", "results.jsonl")
AUG = os.path.join(HERE, "..", "results-2026-08-25.jsonl")


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
        if tol is None:
            # half a unit in the last place the prose actually quotes
            text = repr(float(claim))
            decimals = len(text.split(".")[1].rstrip("0")) if "." in text else 0
            allowed = 0.5 * 10 ** -decimals
        else:
            allowed = abs(value) * tol
        checks.append((abs(claim - value) <= allowed + 1e-12, where, claim, value, allowed))

    # --- README.md, decode table, 2026-07-25 ---------------------------------
    ck("README decode table, 26B MoE 500", 107.8, tps(jul, "E-26B-tp2", 500))
    ck("README decode table, 26B MoE 32K", 72.8, tps(jul, "E-26B-tp2", 32000))
    ck("README decode table, 8B 500", 79.6, tps(jul, "B-8B-tp2", 500))
    ck("README decode table, 31B 32K", 29.5, tps(jul, "C-31B-tp2", 32000))
    ck("README decode table, 27B SSM 32K", 4.2, tps(jul, "D-27B-tp2", 32000))

    # --- README.md, patched table, 2026-08-25 -------------------------------
    ck("README patched table, Muse 500", 43.7, tps(aug, "G-30B-tp2", 500))
    ck("README patched table, Muse 8K", 37.8, tps(aug, "G-30B-tp2", 8000))
    ck("README patched table, Muse 32K", 37.4, tps(aug, "G-30B-tp2", 32000))
    ck("README patched table, Qwen3.8 500", 12.3, tps(aug, "D8-27B-tp2", 500))
    ck("README patched table, Qwen3.8 8K", 11.7, tps(aug, "D8-27B-tp2", 8000))
    ck("README patched table, Qwen3.8 32K", 10.7, tps(aug, "D8-27B-tp2", 32000))
    ck("README patched table, gemma-3 500", 44.8, tps(aug, "F-27B-tp2", 500))
    ck("README patched table, gemma-3 8K", 34.6, tps(aug, "F-27B-tp2", 8000))
    ck("README patched table, gemma-3 32K", 22.1, tps(aug, "F-27B-tp2", 32000))

    # --- benchmarks.md §2, slopes on stock vLLM -----------------------------
    ck("benchmarks.md §2 slope, 8B", 0.118, slope_us(jul, "B-8B-tp2"))
    ck("benchmarks.md §2 slope, 26B", 0.142, slope_us(jul, "E-26B-tp2"))
    ck("benchmarks.md §2 slope, 12B", 0.228, slope_us(jul, "A-12B-tp2"))
    ck("benchmarks.md §2 slope, 31B", 0.339, slope_us(jul, "C-31B-tp2"))
    ck("benchmarks.md §2 slope, 27B SSM", 4.840, slope_us(jul, "D-27B-tp2"))

    # --- benchmarks.md §6, the patched campaign -----------------------------
    ck("benchmarks.md §6 Qwen3.6 retained %", 35.1, retained(jul, "D-27B-tp2"))
    ck("benchmarks.md §6 Qwen3.8 retained %", 86.8, retained(aug, "D8-27B-tp2"))
    ck("benchmarks.md §6 Qwen3.8 slope", 0.390, slope_us(aug, "D8-27B-tp2"))
    ck("benchmarks.md §6 32K speedup",
       2.51, tps(aug, "D8-27B-tp2", 32000) / tps(jul, "D-27B-tp2", 32000))
    ck("benchmarks.md §6 slope ratio 12.4x",
       12.4, slope_us(jul, "D-27B-tp2") / slope_us(aug, "D8-27B-tp2"))
    ck("benchmarks.md §6 Muse slope", 0.122, slope_us(aug, "G-30B-tp2"))
    ck("benchmarks.md §6 gemma-3 slope", 0.730, slope_us(aug, "F-27B-tp2"))
    ck("benchmarks.md §6 Muse 500->32K %", -14.4,
       (tps(aug, "G-30B-tp2", 32000) / tps(aug, "G-30B-tp2", 500) - 1) * 100)
    ck("benchmarks.md §6 gemma-3 500->32K %", -50.7,
       (tps(aug, "F-27B-tp2", 32000) / tps(aug, "F-27B-tp2", 500) - 1) * 100)
    ck("benchmarks.md §6 gemma-3 32K ms/token",
       45.34, 1000 / tps(aug, "F-27B-tp2", 32000))
    ck("benchmarks.md §6 Muse flat at 2000", 37.99, tps(aug, "G-30B-tp2", 2000))

    # --- benchmarks.md §6, the control offsets ------------------------------
    ck("benchmarks.md §6 control, 8B", -0.10, offset(jul, aug, "B-8B-tp2"))
    ck("benchmarks.md §6 control, 12B", -0.02, offset(jul, aug, "A-12B-tp2"))
    ck("benchmarks.md §6 control, 26B", -0.23, offset(jul, aug, "E-26B-tp2"))
    ck("benchmarks.md §6 control, 31B", -0.85, offset(jul, aug, "C-31B-tp2"))
    ck("benchmarks.md §6 control, 8B TP=1", 0.01, offset(jul, aug, "B-8B-tp1"))
    ck("benchmarks.md §6 control, 12B TP=1", -0.82, offset(jul, aug, "A-12B-tp1"))
    # the prose says six of the nine August configurations are July reruns
    ck("benchmarks.md §6 control count", 6, len(set(jul) & set(aug)))
    ck("benchmarks.md §6 new configuration count", 3, len(set(aug) - set(jul)))

    # --- benchmarks.md §6, the gemma-4-31B offset investigation --------------
    off_file = os.path.join(HERE, "..", "gemma-4-31b-campaign-offset.json")
    if os.path.exists(off_file):
        off = json.load(open(off_file))
        rep = off["reproducibility"]
        ck("§6 offset, run1", -0.85, rep["offset_against_july_pct"]["run1_in_campaign"])
        ck("§6 offset, run2 cold", -0.90, rep["offset_against_july_pct"]["run2_cold"])
        ck("§6 offset, run3 warm", -0.79, rep["offset_against_july_pct"]["run3_warm"])
        ck("§6 offset, RSD mean", 0.077, rep["relative_stddev_across_the_three_pct"]["mean"])
        ck("§6 offset, RSD max", 0.146, rep["relative_stddev_across_the_three_pct"]["max"])
        ck("§6 offset, warm minus cold",
           0.11, off["temperature_is_not_the_cause"]["warm_minus_cold_pct"]["mean"])

    failed = [c for c in checks if not c[0]]
    for ok, where, claim, value, allowed in checks:
        if verbose or not ok:
            print(f"  {'ok  ' if ok else 'FAIL'} {where:<44} prose {claim:>8}   "
                  f"data {value:>9.3f}   allowed +-{allowed:.4g}")
    print(f"\n{len(checks) - len(failed)}/{len(checks)} figures agree with the data files")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
