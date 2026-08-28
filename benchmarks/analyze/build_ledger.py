#!/usr/bin/env python3
"""build_ledger.py — project every decode measurement onto one row format.

Charts in this repository have been per-campaign: one source file, one stamp
line, one set of SVGs. That splits a question a reader actually has ("what does
this machine do?") across several figures, and the front page ended up carrying
two versions of the same chart.

This builds the projection a single chart can be drawn from. The raw files stay
the source of truth; every row here names the file it came from and can be
recomputed from it, which is what `--check` and `verify_doc_figures.py` do.

What a row is: one (model, context) point, aggregated over the rounds that
measured it, carrying the stack it was measured on rather than inheriting one
from the chart it lands in.

    python3 build_ledger.py            # write ../ledger.jsonl
    python3 build_ledger.py --check    # fail if the committed file is stale
    python3 build_ledger.py --spread   # the spread distribution, for choosing
                                       # where chart_grade should cut

Two rules are deliberately encoded rather than left to the chart:

  * `runs` and `range_pct` travel with the point. A single run is not a
    measurement on this box -- see benchmarks/harness-calibration -- and the
    split-KV arm at 8K moves 14.6% between passes while its stock arm moves
    0.5%, so "how many times" is not a footnote.
  * `patches` is a list, not a boolean. A point that needs an unmerged PR is
    not the same claim as one you get by installing a release, and a chart that
    hides the difference is making a promise the repository cannot keep.
"""
import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
B = lambda *p: os.path.join(HERE, "..", *p)
LEDGER = B("ledger.jsonl")

# cfg id -> what the model is. Ids are the campaign's; the probe sources name
# their model directly and map through MODELS by path.
CFG = {
    "E-26B-tp2":  ("gemma-4-26B-A4B", "int4", "MoE, 128 experts", 2),
    "B-8B-tp2":   ("Qwen3-8B", "bf16", "dense", 2),
    "B-8B-tp1":   ("Qwen3-8B", "bf16", "dense", 1),
    "A-12B-tp2":  ("gemma-4-12B-it", "w4a16 QAT", "dense", 2),
    "A-12B-tp1":  ("gemma-4-12B-it", "w4a16 QAT", "dense", 1),
    "C-31B-tp2":  ("gemma-4-31B-it", "w4a16 QAT", "dense", 2),
    "D-27B-tp2":  ("Qwen3.6-27B", "AWQ int4", "hybrid SSM", 2),
    "D8-27B-tp2": ("Qwen3.8-27B", "AWQ int4", "hybrid SSM", 2),
    "F-27B-tp2":  ("gemma-3-27b-it", "w4a16", "sliding window", 2),
    "G-30B-tp2":  ("Muse-Glimmer-30B", "int4", "sliding window 2048", 2),
}
MODELS = {"/data/incoming/Qwen3.8-27B-AWQ-INT4": "D8-27B-tp2"}

# Where chart_grade cuts. Chosen from the distribution rather than picked: the
# ranges run 0.00-2.36% up to p95, then 3.4, 4.1, 5.1, and then jump to 15.4 and
# 15.8. The cut sits in that gap, so it excludes the two points that are known
# to be unstable and nothing else.
RANGE_CUT = 6.0

CAMPAIGNS = [
    dict(file="results.jsonl", date="2026-07-25", vllm="0.23", rocm="7.14",
         kernel="7.0.0-28", patches=[]),
    dict(file="results-2026-08-24.jsonl", date="2026-08-24",
         vllm="0.23.1.dev1+g9ddef7117", rocm="7.14", kernel="7.0.0-30",
         patches=["vllm#45916 split-KV", "window block-skip"]),
]
# The probe sources are arm-structured: the same cells measured with and
# without a patch, so the arm decides `patches` rather than the file does.
PROBES = [
    dict(files=["hybrid-splitkv-027/qwen38-027-depth.jsonl",
                "hybrid-splitkv-027/qwen38-027-depth-b.jsonl",
                "hybrid-splitkv-027/qwen38-8k-r3r4.jsonl"],
         date="2026-08-28", vllm="0.27.1.dev5+gf46a9dfe2", rocm="10.0",
         kernel=None,
         arms={"stock": [], "splitkv": ["vllm#45916 split-KV"]}),
]


SESSION_GAP_S = 3600  # two measurements ten hours apart are two sessions


def latest_session(pairs):
    """(ts, value) -> the last cluster of them, plus whatever it supersedes.

    The July campaign re-ran two of gemma-4-12B's cells 10.4 hours later. Those
    are a second measurement, not rounds three and four of the first: averaging
    across them turned one bad reading at 2000 tokens into a 15.4% range that
    described neither session. The later session's own two rounds agree to
    0.85%.
    """
    pairs = sorted(pairs)
    cut = 0
    for i in range(1, len(pairs)):
        if pairs[i][0] - pairs[i - 1][0] > SESSION_GAP_S:
            cut = i
    return [v for _, v in pairs[cut:]], [v for _, v in pairs[:cut]]


def aggregate(values, **row):
    v = sorted(values)
    row["values"] = v
    row["runs"] = len(v)
    row["decode_tok_s"] = statistics.mean(v)
    # (max - min) / mean, which generalises past two runs and does not depend
    # on which run is called first. Documents that quote an A-against-B figure
    # are quoting a different quantity; this one is deliberately named for its
    # own definition rather than borrowing theirs.
    row["range_pct"] = (v[-1] - v[0]) / statistics.mean(v) * 100 if len(v) > 1 else None
    row["chart_grade"] = row["runs"] >= 2 and row["range_pct"] <= RANGE_CUT
    if not row["chart_grade"]:
        row["chart_grade_note"] = (
            f"{row['runs']} run(s), range {row['range_pct']:.2f}%"
            if row["runs"] > 1 else "one run")
    return row


def build():
    rows, seen = [], set()

    for c in CAMPAIGNS:
        by = {}
        for line in open(B(c["file"])):
            r = json.loads(line)
            if r.get("kind") != "decode" or not r.get("decode_tps"):
                continue
            by.setdefault((r["cfg"], r["target"]), []).append((r["ts"], r["decode_tps"]))
        for (cfg, target), pairs in sorted(by.items()):
            assert cfg in CFG, f"{c['file']}: unknown cfg {cfg}"
            name, quant, arch, tp = CFG[cfg]
            vals, superseded = latest_session(pairs)
            extra = {"superseded_values": sorted(superseded)} if superseded else {}
            rows.append(aggregate(
                vals, **extra, model=name, quant=quant, arch=arch, tp=tp, ctx=target,
                date=c["date"], vllm=c["vllm"], rocm=c["rocm"],
                kernel=c["kernel"], patches=c["patches"],
                harness="campaign-server", source=c["file"], cfg=cfg))
            seen.add((c["file"], cfg, target))

    for p in PROBES:
        by = {}
        for f in p["files"]:
            for line in open(B(f)):
                r = json.loads(line)
                cfg = MODELS[r["model"]]
                by.setdefault((cfg, r["arm"], r["ctx"]), []).append(r["decode_tok_s"])
        for (cfg, arm, ctx), vals in sorted(by.items()):
            name, quant, arch, tp = CFG[cfg]
            rows.append(aggregate(
                vals, model=name, quant=quant, arch=arch, tp=tp, ctx=ctx,
                date=p["date"], vllm=p["vllm"], rocm=p["rocm"],
                kernel=p["kernel"], patches=p["arms"][arm],
                harness="probe-t8t64", source=", ".join(p["files"]), cfg=cfg))

    rows.sort(key=lambda r: (r["model"], r["tp"], r["date"],
                             ",".join(r["patches"]), r["ctx"]))
    return rows


def dump(rows):
    return "".join(json.dumps(r) + "\n" for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--spread", action="store_true")
    a = ap.parse_args()
    rows = build()

    if a.spread:
        sp = sorted(r["range_pct"] for r in rows if r["range_pct"] is not None)
        print(f"{len(sp)} points with more than one run")
        if not sp:
            print("  no point has a second run, so there is no spread to show")
            return 0
        for q in (50, 75, 90, 95, 99, 100):
            print(f"  p{q:<3} {sp[min(len(sp) - 1, int(len(sp) * q / 100))]:6.2f}%")
        print("  worst ten:")
        for r in sorted(rows, key=lambda r: -(r["range_pct"] or 0))[:10]:
            print(f"    {r['range_pct']:6.2f}%  {r['model']:<18} tp{r['tp']} "
                  f"ctx={r['ctx']:<6} {r['date']} {','.join(r['patches']) or 'stock'}")
        return 0

    text = dump(rows)
    if a.check:
        have = open(LEDGER).read() if os.path.exists(LEDGER) else ""
        if have != text:
            print("ledger.jsonl is stale; re-run build_ledger.py", file=sys.stderr)
            return 1
        print(f"ledger.jsonl matches its sources: {len(rows)} rows")
        return 0
    open(LEDGER, "w").write(text)
    print(f"wrote {LEDGER}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
