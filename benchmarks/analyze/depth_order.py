#!/usr/bin/env python3
"""Compare retention and absolute depth-cost rankings from committed raw rows.

This reanalysis reproduces the depth article's nominal-rung OLS calculation,
then varies the axis, repeat and cost definition. It writes no projection.
Run with --json to inspect every fit, pair and source, or without it for tables.
"""
import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import statistics as stats

from build_prefill import CFG, CFG_CUDA, RANGE_CUT

ROOT = Path(__file__).resolve().parents[2]
RUNGS = (500, 1000, 2000, 4000, 6000, 8000, 12000, 16000, 20000, 24000, 32000)
FIVE = ("B8", "G12", "G26A4B", "G31", "Q38")
# One source per (machine, cfg). H100's mml/mns control arms are deliberately
# absent; merging them would count one checkpoint more than once.
SOURCES = {
    "RX 7900 XT TP2": {
        "campaign-2026-09-03/results.jsonl": (
            "B-8B-tp2-long", "A-12B-tp2-long", "E-26B-tp2-long",
            "C-31B-tp2-long", "D8-27B-tp2-long", "G-30B-tp2-long")},
    "H100 TP1": {
        "cuda-h100/campaign-2026-09-03/results.jsonl": FIVE[:-1],
        "cuda-h100/campaign-2026-09-03/results-q38.jsonl": ("Q38",),
        "cuda-h100/campaign-2026-09-03b/results.jsonl": ("MG30",)},
    "H200 TP1": {"cuda-h200/campaign-2026-09-03/results.jsonl": FIVE + ("MG30",)},
    "B300 TP1": {"cuda-b300/campaign-2026-09-03/results.jsonl": FIVE + ("MG30",)},
    "RTX PRO 6000 TP1": {"cuda-pro6000/campaign-2026-09-03/results.jsonl": FIVE + ("MG30",)},
    "H100 TP2": {"cuda-h100/campaign-2026-09-03-tp2/results.jsonl": FIVE},
    "RTX PRO 6000 TP2": {"cuda-pro6000/campaign-2026-09-03-tp2/results.jsonl": FIVE},
}


def fit(rows, actual_tokens=False, repeat=None):
    cells = defaultdict(list)
    for row in rows:
        if repeat is None or row["round"] == repeat:
            cells[row["target"]].append(row)
    rates = [stats.median(r["decode_tps"] for r in cells[t]) for t in RUNGS]
    xs = [stats.mean(r["prompt_tokens"] for r in cells[t]) if actual_tokens else t
          for t in RUNGS]
    # Microseconds per generated token, fitted against context length.
    ys = [1e6 / rate for rate in rates]
    slope, intercept = stats.linear_regression(xs, ys)
    return dict(retention=rates[-1] / rates[0], slope_us=slope,
                intercept_ms=intercept / 1000, r2=stats.correlation(xs, ys) ** 2,
                endpoint_us=(ys[-1] - ys[0]) / (xs[-1] - xs[0]),
                points=[dict(target=t, tokens=x, decode_tps=v)
                        for t, x, v in zip(RUNGS, xs, rates)])


def compare(fits, metric="slope_us"):
    pairs = []
    for a, b in combinations(fits, 2):
        dr = fits[a]["retention"] - fits[b]["retention"]
        dc = fits[a][metric] - fits[b][metric]
        # Higher retention and lower cost are the two desired orderings.
        pairs.append(dict(a=a, b=b, discordant=dr * dc > 0, tied=dr * dc == 0))
    n = sum(p["discordant"] for p in pairs)
    return dict(discordant=n, total=len(pairs), pct=100 * n / len(pairs),
                ties=sum(p["tied"] for p in pairs), pairs=pairs)


def analyze(root=ROOT):
    result = []
    models = dict(CFG, **CFG_CUDA)
    for machine, sources in SOURCES.items():
        by_cfg = {}
        provenance = {}
        for source, configs in sources.items():
            with open(Path(root) / "benchmarks" / source, encoding="utf-8") as f:
                raw = [json.loads(line) for line in f if line.strip()]
            for cfg in configs:
                selected = [r for r in raw if r.get("cfg") == cfg
                            and r.get("kind") == "decode" and r["target"] in RUNGS]
                keys = [(r["target"], r["round"]) for r in selected]
                expected = {(t, repeat) for t in RUNGS for repeat in (1, 2)}
                if len(keys) != len(expected) or set(keys) != expected:
                    raise ValueError(f"{machine} {cfg}: incomplete or duplicate repeat ladder")
                if any(r.get("err") or r["decode_tps"] <= 0 for r in selected):
                    raise ValueError(f"{machine} {cfg}: invalid decode row")
                by_cfg[cfg] = selected
                provenance[cfg] = dict(source=source, model=models[cfg][0])
        fits = {cfg: dict(fit(rows), **provenance[cfg]) for cfg, rows in by_cfg.items()}
        actual = {cfg: fit(rows, actual_tokens=True) for cfg, rows in by_cfg.items()}
        without_window = {cfg: v for cfg, v in fits.items() if not v["model"].startswith("Muse-Glimmer")}
        repeats = [{cfg: fit(rows, repeat=r) for cfg, rows in by_cfg.items()} for r in (1, 2)]
        spreads = [(max(v) - min(v)) / stats.median(v) * 100
                   for rows in by_cfg.values() for t in RUNGS
                   for v in [[r["decode_tps"] for r in rows if r["target"] == t]]]
        result.append(dict(machine=machine, models=len(fits), fits=fits,
                           primary=compare(fits), actual_tokens=compare(actual),
                           no_window=compare(without_window), endpoint=compare(fits, "endpoint_us"),
                           repeats=[compare(f) for f in repeats],
                           baseline_spread=max(f["intercept_ms"] for f in fits.values()) /
                                           min(f["intercept_ms"] for f in fits.values()),
                           worst_repeat_spread_pct=max(spreads),
                           ungraded_cells=sum(s > RANGE_CUT for s in spreads),
                           min_r2=min(f["r2"] for f in fits.values())))
    return dict(configurations=result,
                correlation=stats.correlation([r["primary"]["pct"] for r in result],
                                              [r["baseline_spread"] for r in result]),
                correlation_without_pair=stats.correlation(
                    [r["primary"]["pct"] for r in result[1:]],
                    [r["baseline_spread"] for r in result[1:]]))


def tables(data):
    lines = ["| Configuration | Models | Discordant / pairs | Percent | Baseline spread | Min r² |",
             "|---|--:|--:|--:|--:|--:|"]
    for r in data["configurations"]:
        p = r["primary"]
        lines.append(f"| {r['machine']} | {r['models']} | {p['discordant']}/{p['total']} | "
                     f"{p['pct']:.1f} | {r['baseline_spread']:.3f} | {r['min_r2']:.4f} |")
    lines += ["", "| Configuration | Actual tokens | Round 1 | Round 2 | Endpoint cost | Without Muse-Glimmer |",
              "|---|--:|--:|--:|--:|--:|"]
    for r in data["configurations"]:
        variants = [r["actual_tokens"], *r["repeats"], r["endpoint"], r["no_window"]]
        lines.append("| " + r["machine"] + " | " + " | ".join(
            f"{v['discordant']}/{v['total']}" for v in variants) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="include every fit, pair and source")
    args = parser.parse_args()
    data = analyze()
    print(json.dumps(data, indent=2) if args.json else tables(data))
