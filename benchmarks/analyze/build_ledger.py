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
#
# `quant` has one grammar: the numeric format first, then how the checkpoint was
# produced, where that is known -- "int4 AWQ", not "AWQ int4". The format is the
# axis every model can be compared on and the method is a qualifier only some
# carry, so the format leads and the qualifier trails, the way "w4a16 QAT"
# already did. Six of the eight strings already read that way; the two Qwen
# rows were the exception and were turned round on 2026-08-30, when the field
# started being drawn on the index chart's labels rather than only recorded.
#
# gemma-4-26B-A4B was "int4" until the same day, which understated it: the
# checkpoint is cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit on both machines --
# campaign-2026-08-29's setup.log downloads it by that name and bench_runner
# serves /models/gemma-4-26B-A4B-AWQ -- so it is AWQ for exactly the reason
# Qwen3.8 is, and only this row failed to say so.
CFG = {
    "E-26B-tp2":  ("gemma-4-26B-A4B", "int4 AWQ", "MoE, 128 experts", 2),
    "B-8B-tp2":   ("Qwen3-8B", "bf16", "dense", 2),
    "B-8B-tp1":   ("Qwen3-8B", "bf16", "dense", 1),
    "A-12B-tp2":  ("gemma-4-12B-it", "w4a16 QAT", "dense", 2),
    "A-12B-tp1":  ("gemma-4-12B-it", "w4a16 QAT", "dense", 1),
    "C-31B-tp2":  ("gemma-4-31B-it", "w4a16 QAT", "dense", 2),
    "D-27B-tp2":  ("Qwen3.6-27B", "int4 AWQ", "hybrid SSM", 2),
    "D8-27B-tp2": ("Qwen3.8-27B", "int4 AWQ", "hybrid SSM", 2),
    "F-27B-tp2":  ("gemma-3-27b-it", "w4a16", "sliding window", 2),
    "G-30B-tp2":  ("Muse-Glimmer-30B", "int4", "sliding window 2048", 2),
    # 2026-08-29. The campaign names its own arms, so a model appears under
    # several ids that differ in speculation and in which kernel served them.
    "Q38-tp2":                   ("Qwen3.8-27B", "int4 AWQ", "hybrid SSM", 2),
    "Q38-mtp-tp2":               ("Qwen3.8-27B", "int4 AWQ", "hybrid SSM", 2),
    "Q38-mtp-p45450-tp2":        ("Qwen3.8-27B", "int4 AWQ", "hybrid SSM", 2),
    "Q38-triton-tp2":            ("Qwen3.8-27B", "int4 AWQ", "hybrid SSM", 2),
    "Q38-mtp-triton-tp2":        ("Qwen3.8-27B", "int4 AWQ", "hybrid SSM", 2),
    "Q38-mtp-triton-p45450-tp2": ("Qwen3.8-27B", "int4 AWQ", "hybrid SSM", 2),
    "G31-tp2":                   ("gemma-4-31B-it", "w4a16 QAT", "dense", 2),
    "G31-mtp-p45450-tp2":        ("gemma-4-31B-it", "w4a16 QAT", "dense", 2),
    # 2026-08-30. The MoE on a single card. The id carries its utilisation
    # because it is the only row here not measured at this campaign's own
    # 0.92: 16.96 GiB of weights resident on a 19.98 GiB card leaves no KV at
    # 0.92 (1536 tokens, recorded as a config_failed in the same file), and
    # 0.95 buys 13 149 -- seven rungs, not eleven.
    "E26-tp1-u95":               ("gemma-4-26B-A4B", "int4 AWQ", "MoE, 128 experts", 1),
    # 2026-08-30, on the 0.27 image and fully stock. Same reason for carrying
    # the utilisation in the id: it is the only Qwen3-8B row not measured at
    # 0.90. Raising it did not lift the July ceiling -- 8 236 tokens against
    # 8 442 -- because the weights are 15.27 GiB on 0.27 rather than the 14.02
    # that 0.23 reported, and this model's activation overhead is 2.58 GiB.
    "B8-tp1-u95":                ("Qwen3-8B", "bf16", "dense", 1),
    # 2026-09-02. `G31-tp2` re-run with byte-identical serve arguments on the
    # link the 2026-09-02 22:44 reboot restored to x16. A separate id because
    # the link is part of the configuration, not a second round of the x8 one.
    "G31-tp2-x16":               ("gemma-4-31B-it", "w4a16 QAT", "dense", 2),
    # 2026-09-02b. Five rounds at the 500-token rung, both arms of the 8B --
    # the cell whose two rounds disagreed by 22.13% in July and 18.24% in
    # August. Five rounds at one rung is not the two-round eleven-rung ladder,
    # so these must not be picked up as a third sitting of `B-8B-tp*`.
    "B8-tp2-r5":                 ("Qwen3-8B", "bf16", "dense", 2),
    "B8-tp1-r5":                 ("Qwen3-8B", "bf16", "dense", 1),
    # 2026-09-02c. The other two lines Figure 2 draws from the narrowed link,
    # re-measured on the restored one. Same reason for the `-x16` suffix.
    "Q38-tp2-x16":               ("Qwen3.8-27B", "int4 AWQ", "hybrid SSM", 2),
    "Q38-triton-tp2-x16":        ("Qwen3.8-27B", "int4 AWQ", "hybrid SSM", 2),
    # 2026-09-02d. The two models whose second-card gain differs most, both
    # topologies, with telemetry. `-p45450` because the container carries that
    # patch and August's sitting of these arms did not.
    "A12-tp1-p45450":            ("gemma-4-12B-it", "w4a16 QAT", "dense", 1),
    "A12-tp2-p45450":            ("gemma-4-12B-it", "w4a16 QAT", "dense", 2),
    "B8-tp1-p45450":             ("Qwen3-8B", "bf16", "dense", 1),
    "B8-tp2-p45450":             ("Qwen3-8B", "bf16", "dense", 2),
    # 2026-09-03. The same six models as the rented sweep of that day, on the
    # pair, with the ladder carried to 128 000 -- the first Radeon rows past
    # 32 000. `-long` because the ladder is a new sixteen-rung cut and not the
    # eleven every earlier arm measured, so these are not a third sitting of
    # `A-12B-tp2` and must not be picked up as one.
    "A-12B-tp2-long":            ("gemma-4-12B-it", "w4a16 QAT", "dense", 2),
    "B-8B-tp2-long":             ("Qwen3-8B", "bf16", "dense", 2),
    "E-26B-tp2-long":            ("gemma-4-26B-A4B", "int4 AWQ", "MoE, 128 experts", 2),
    "G-30B-tp2-long":            ("Muse-Glimmer-30B", "int4", "sliding window 2048", 2),
    "D8-27B-tp2-long":           ("Qwen3.8-27B", "int4 AWQ", "hybrid SSM", 2),
    "C-31B-tp2-long":            ("gemma-4-31B-it", "w4a16 QAT", "dense", 2),
}

# What each 2026-08-29 arm ran with, beyond the model.
#
#   spec          handoff 6's descriptor, recording what the *engine* resolved
#                 rather than what the flag asked for: the gemma arm requests
#                 method "draft_model" and vLLM reports
#                 SpeculativeConfig(method='mtp', ...). `drafter` is what keeps
#                 the two shapes apart -- Qwen3.8 carries its mtp head in its
#                 own weights, gemma-4 loads a separate assistant checkpoint.
#                 k is 3 throughout this campaign. Every gemma-4 speculation
#                 number already published here is k=1, so a point-for-point
#                 comparison with those is not available.
#   attn_backend  which kernel actually served the arm, read off the serve log
#                 named beside it in campaign-2026-08-29/serve-logs/.
#
# The backend is a column because without it two rows differing only in
# `patches` would claim a difference the kernel path makes impossible.
# vllm#45450 patches triton_unified_attention.py and triton_attn.py; Qwen3.8 on
# the 0.27 ROCm image is routed to ROCM_ATTN, whose backend file imports
# chunked_prefill_paged_decode and neither of those. The two Qwen3.8 mtp arms,
# one patched and one not, agree to a mean of -1.93% -- inside their own
# 2.0-21.2% run-to-run spread -- and the patch's own probe never prints.
# `patches` still lists #45450, because it is installed; `attn_backend` is what
# says it could not act.
MTP3 = {"method": "mtp", "k": 3}
DRAFT3 = {"method": "mtp", "drafter": "gemma-4-31B-it-assistant", "k": 3}
ARMS = {
    "Q38-tp2":                   (None,   "ROCM_ATTN"),
    "Q38-mtp-tp2":               (MTP3,   "ROCM_ATTN"),
    "Q38-mtp-p45450-tp2":        (MTP3,   "ROCM_ATTN"),
    "Q38-triton-tp2":            (None,   "TRITON_ATTN"),
    "Q38-mtp-triton-tp2":        (MTP3,   "TRITON_ATTN"),
    "Q38-mtp-triton-p45450-tp2": (MTP3,   "TRITON_ATTN"),
    "G31-tp2":                   (None,   "TRITON_ATTN"),
    "G31-mtp-p45450-tp2":        (DRAFT3, "TRITON_ATTN"),
    "E26-tp1-u95":               (None,   "TRITON_ATTN"),
    # ROCM_ATTN, and it is the case vllm#54438 deliberately leaves alone:
    # Qwen3-8B is head_dim 128 with gqa_ratio 4, so it satisfies
    # `use_rocm_custom_paged_attention` on RDNA and gets the actual HIP kernel
    # rather than a second Triton one. Read from the serve log's
    # `Overriding with ROCM_ATTN out of potential backends` line.
    "B8-tp1-u95":                (None,   "ROCM_ATTN"),
    "G31-tp2-x16":               (None,   "TRITON_ATTN"),
    "Q38-tp2-x16":               (None,   "ROCM_ATTN"),
    "Q38-triton-tp2-x16":        (None,   "TRITON_ATTN"),
    "A12-tp1-p45450":            (None,   "TRITON_ATTN"),
    "A12-tp2-p45450":            (None,   "TRITON_ATTN"),
    "B8-tp1-p45450":             (None,   "ROCM_ATTN"),
    "B8-tp2-p45450":             (None,   "ROCM_ATTN"),
    "B8-tp2-r5":                 (None,   "ROCM_ATTN"),
    "B8-tp1-r5":                 (None,   "ROCM_ATTN"),
    # 2026-09-03, read from each arm's serve log in campaign-2026-09-03/logs/.
    # gemma-4 is forced onto TRITON_ATTN by the engine ("selected via
    # --attention-backend" in 0.23's wording, though no flag was passed).
    "A-12B-tp2-long":            (None,   "TRITON_ATTN"),
    "E-26B-tp2-long":            (None,   "TRITON_ATTN"),
    "C-31B-tp2-long":            (None,   "TRITON_ATTN"),
    # the bf16 8B: no flag, vLLM's own `Overriding with ROCM_ATTN` line, the
    # same choice as B8-tp1-u95 and every Q38 arm above (logs/B-8B-tp2-long.log)
    "B-8B-tp2-long":             (None,   "ROCM_ATTN"),
    # Muse-Glimmer int4: the same `Overriding with ROCM_ATTN` line, twice, in
    # logs/G-30B-tp2-long.log; no backend flag anywhere in this campaign
    "G-30B-tp2-long":            (None,   "ROCM_ATTN"),
    # Qwen3.8-27B int4 AWQ: the override line again (logs/D8-27B-tp2-long.log),
    # as Q38-tp2 / Q38-tp2-x16 above; served at max_num_seqs 161
    "D8-27B-tp2-long":           (None,   "ROCM_ATTN"),
}
MODELS = {"/data/incoming/Qwen3.8-27B-AWQ-INT4": "D8-27B-tp2"}

# Where chart_grade cuts. Chosen from the distribution rather than picked, and
# restated on 2026-08-29 when the ladder campaign's speculative arms landed and
# moved the tail properly. The distribution is no longer one quiet stretch with
# a single outlier: of 258 points with more than one run, the ranges climb
# 0.00 ... 5.30, 5.33, 5.97, 6.02, 6.10, and then jump to 9.50 and stay high,
# 29 of them above the cut. Twenty-eight of those 29 are speculative rungs.
# The twenty-ninth is not: it is the bimodal 8 K cell from 2026-08-28, at
# 16.79 %, worse than most of the speculative ones. So the reading is that
# speculation is where the instability concentrates, not that an arm without
# it cannot produce the same thing.
#
# The widest gap in that region is 6.10 -> 9.50, 3.40 percentage points, and
# the cut sits in the middle of it. The previous 6.0 was chosen from the
# distribution as it stood before this campaign and no longer sits in a gap:
# it stranded 6.02 and 6.10 between itself and the real break. Moving it grades
# those two rungs and nothing else -- both are 2026-08-29 rows, so no figure
# published before today changes. verify_doc_figures.py pins the tail and the
# gap, which is what forced this restatement rather than letting it drift.
RANGE_CUT = 8.0

CAMPAIGNS = [
    dict(file="results.jsonl", date="2026-07-25", vllm="0.23", rocm="7.14",
         kernel="7.0.0-28", patches=[]),
    dict(file="results-2026-08-24.jsonl", date="2026-08-24",
         vllm="0.23.1.dev1+g9ddef7117", rocm="7.14", kernel="7.0.0-30",
         patches=["vllm#45916 split-KV", "window block-skip"]),
    # One file, two stacks. gemma-4 cannot be served on the 0.27 image at all --
    # its Quark plugin reads head_dim off a heterogeneous config and dies before
    # loading -- so its arms ran on the 0.23 container the 08-24 campaign used,
    # and `per_cfg` carries what differs per arm rather than per file.
    dict(file="campaign-2026-08-29/results.jsonl", date="2026-08-29",
         vllm="0.27.1.dev5+gf46a9dfe2", rocm="10.0", kernel="7.0.0-30",
         patches=["vllm#45916 split-KV"],
         per_cfg={
             "Q38-mtp-p45450-tp2": dict(
                 patches=["vllm#45916 split-KV", "vllm#45450 3D admission"]),
             "Q38-mtp-triton-p45450-tp2": dict(
                 patches=["vllm#45916 split-KV", "vllm#45450 3D admission"]),
             "G31-tp2": dict(
                 vllm="0.23.1.dev1+g9ddef7117", rocm="7.14",
                 patches=["vllm#45916 split-KV", "window block-skip",
                          "vllm#45450 3D admission"]),
             "G31-mtp-p45450-tp2": dict(
                 vllm="0.23.1.dev1+g9ddef7117", rocm="7.14",
                 patches=["vllm#45916 split-KV", "window block-skip",
                          "vllm#45450 3D admission"]),
         }),
    dict(file="campaign-2026-08-30/results.jsonl", date="2026-08-30",
         vllm="0.23.1.dev1+g9ddef7117.d20260715", rocm="7.14",
         kernel="7.0.0-30", patches=[]),
    # `campaign-2026-08-30b` (Qwen3-8B on one card, 0.27 image) is DELIBERATELY
    # not here. This file is the Radeon box's decode projection as it stood,
    # and its 265 rows and the distribution statistics published from them --
    # the median range, the tail, and the gap `RANGE_CUT` sits in -- are quoted
    # in the front page and in the measure article. New work goes into
    # `prefill.jsonl` and `decode.jsonl`, which carry a `machine` column and
    # take their sources from `build_prefill.SOURCES`; that arm reaches
    # `decode.jsonl` from there. `CFG` and `ARMS` above still name it, because
    # `build_prefill.meta_for` and `arm_for` look here first.
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
            over = c.get("per_cfg", {}).get(cfg, {})
            spec, backend = ARMS.get(cfg, (None, None))
            rows.append(aggregate(
                vals, **extra, model=name, quant=quant, arch=arch, tp=tp, ctx=target,
                date=c["date"], vllm=over.get("vllm", c["vllm"]),
                rocm=over.get("rocm", c["rocm"]),
                kernel=over.get("kernel", c["kernel"]),
                patches=over.get("patches", c["patches"]),
                harness="campaign-server", source=c["file"], cfg=cfg,
                spec=spec, attn_backend=backend))
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
                harness="probe-t8t64", source=", ".join(p["files"]), cfg=cfg,
                spec=None, attn_backend=None))

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
