#!/usr/bin/env python3
"""gen_machines_chart.py — one model, five machines, the chart the ladder earned.

Every other figure in this directory holds the machine fixed and varies the
model, because for a year there was one machine. There are five now, and the
question a reader arriving at a repository called dual-radeon-vllm actually has
-- what do two consumer Radeons do against the cards people rent? -- has never
had a figure that answers it from campaign-grade data. The one table that tried
is single-run probes on speculative arms and says so.

`gemma-4-12B-it` is the model to draw it with, and it is the only one: it is the
sole model in this repository measured on all five machines, at eleven rungs and
two rounds a cell on each. Every point here is chart-grade.

    python3 gen_machines_chart.py

Reads `decode.jsonl` -- the cross-machine projection, which `build_decode.py
--check` recomputes from its sources and reconciles against `ledger.jsonl`. Not
the raw campaign files: a third aggregation of rows that already have one is how
a repository ends up with two answers.
"""
import json
import os

from gen_best_charts import build

HERE = os.path.dirname(os.path.abspath(__file__))
DECODE = os.path.join(HERE, "..", "decode.jsonl")

# Colour is the machine here, not the model, and the pair gets the warm accent
# because it is what this repository is about. The two NVIDIA single cards that
# bracket it -- the L4 above at depth and the T4 below everywhere -- are cool,
# so the eye reads "rented silicon" as one family without the legend.
MACHINES = {
    "rdna3-2":  ("#e05c48", "2&#215; RX 7900 XT &#183; TP=2"),
    "rdna3-1":  ("#d99a24", "one RX 7900 XT &#183; TP=1"),
    "a100":     ("#2ea36a", "one A100 80G &#183; TP=1"),
    "l4":       ("#8b6ee0", "one L4 24G &#183; TP=1"),
    "t4":       ("#3f8fd4", "one Tesla T4 16G &#183; TP=1"),
}

# id, machine string, cfg, date. These are the runs the front page's Figure 1
# draws, machine for machine and day for day, so a reader moving between the
# two is looking at the same measurements rather than two picks of one box.
LINES = [
    ("rdna3-2", "RX 7900 XT",     "A-12B-tp2", "2026-08-24"),
    ("rdna3-1", "RX 7900 XT",     "A-12B-tp1", "2026-08-24"),
    ("a100",    "A100-SXM4-80GB", "A100-G12",  "2026-08-29"),
    ("l4",      "L4",             "G12",       "2026-08-30"),
    ("t4",      "T4",             "G12",       "2026-08-30"),
]

MODEL = "gemma-4-12B-it"


def main():
    rows = [json.loads(l) for l in open(DECODE)]
    series, notes = [], []
    for mid, machine, cfg, date in LINES:
        got = sorted([r for r in rows if r["model"] == MODEL and r["cfg"] == cfg
                      and r["machine"] == machine and r["date"] == date],
                     key=lambda r: r["ctx"])
        assert got, (machine, cfg, date)
        graded = [r for r in got if r["chart_grade"]]
        # Nothing here is ungraded, and the assertion is the point: a figure
        # that quietly dropped a rung would read as a machine that stopped
        # early. If one ever fails this, draw the gap rather than deleting it.
        assert len(graded) == len(got) == 11, (cfg, len(graded), len(got))
        patched = bool(got[0]["patches"])
        series.append((mid, [(r["ctx"], r["decode_tok_s"], False) for r in graded],
                       patched))
        stack = f"vLLM {got[0]['vllm'].split('+')[0]}"
        if got[0]["rocm"]:
            stack += f" &#183; ROCm {got[0]['rocm']}"
        if got[0]["cuda"]:
            stack += f" &#183; CUDA {got[0]['cuda']}"
        note = (f"{MACHINES[mid][1].split(' &#183;')[0]} &#183; {date} &#183; {stack}"
                f" &#183; {got[0]['attn_backend'] or 'backend not recorded'}")
        if patched:
            note += " &#183; needs " + ", ".join(
                x.split(" ")[0] for x in got[0]["patches"])
        first, last = graded[0], graded[-1]
        note += (f" &#183; {first['decode_tok_s']:.1f} at 500, "
                 f"{last['decode_tok_s']:.1f} at 32K, "
                 f"retains {last['decode_tok_s'] / first['decode_tok_s'] * 100:.0f}%")
        notes.append(note)

    out = build(
        "decode-five-machines-gemma4-12b.svg",
        "One model, five machines, batch-1 decode",
        "gemma-4-12B-it w4a16 QAT &#183; the only model this repository has "
        "measured on every machine it has",
        series, 130, "decode tok/s", notes, step=10,
        head="eleven rungs, two rounds a cell, every point chart-grade &#183; "
             "solid: released vLLM, no patch &#183; dashed: needs a patch, named below",
        colours=MACHINES)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
