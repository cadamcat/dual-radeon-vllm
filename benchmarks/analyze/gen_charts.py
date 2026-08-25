#!/usr/bin/env python3
"""gen_charts.py — standalone SVG charts for the GitHub repo.
Colours are baked in (no CSS vars) and chosen to read on both light and dark
GitHub themes; axes/text use neutral grey. Legend and title live inside the SVG
so the file works as a plain <img> in markdown.

Every chart carries a stamp line naming the date, the software and the kernel it
was measured on. Two campaigns are not comparable unless that line matches, so it
is required rather than optional:

    # the 2026-07-25 campaign, stock vLLM
    python3 gen_charts.py

    # the 2026-08-25 re-sweep, patched container
    python3 gen_charts.py --source ../results-2026-08-25.jsonl --suffix -2026-08-25 \\
        --stamp "2026-08-25 - vLLM 0.23.1.dev1+g9ddef7117 + ROCm 7.14 - kernel 7.0.0-30 - ..." \\
        --series E-26B-tp2,G-30B-tp2,B-8B-tp2,A-12B-tp2,C-31B-tp2,D8-27B-tp2

gemma-3 (F-27B-tp2) is measured but deliberately not plotted: between 500 and
4000 it runs within two tok/s of both Muse-Glimmer and gemma-4-31B and the three
lines read as one. benchmarks.md quotes its numbers in prose instead.
"""
import argparse, json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "docs", "assets")

JULY_STAMP = ("2026-07-25 &#183; 2x RX 7900 XT &#183; vLLM 0.23 &#183; ROCm 7.14 &#183; "
              "kernel 7.0.0-28 &#183; stock, no patches")

C = {
    "E-26B-tp2":  ("#8b6ee0", "gemma-4-26B-A4B &#183; MoE &#183; TP2"),
    "B-8B-tp2":   ("#3f8fd4", "Qwen3-8B &#183; BF16 &#183; TP2"),
    "A-12B-tp2":  ("#2ea36a", "gemma-4-12B &#183; w4a16 &#183; TP2"),
    "C-31B-tp2":  ("#d99a24", "gemma-4-31B &#183; w4a16 &#183; TP2"),
    "D-27B-tp2":  ("#e05c48", "Qwen3.6-27B &#183; hybrid SSM &#183; TP2"),
    "B-8B-tp1":   ("#3f8fd4", "Qwen3-8B &#183; BF16 &#183; TP1 (single card)"),
    "A-12B-tp1":  ("#2ea36a", "gemma-4-12B &#183; w4a16 &#183; TP1 (single card)"),
    # added 2026-08-25. Qwen3.8 keeps Qwen3.6's colour: same architecture, same
    # slot in the chart, so the two campaigns read as one line moving.
    "D8-27B-tp2": ("#e05c48", "Qwen3.8-27B &#183; hybrid SSM &#183; TP2"),
    "F-27B-tp2":  ("#21a0a0", "gemma-3-27b &#183; w4a16, sliding window &#183; TP2"),
    "G-30B-tp2":  ("#d1519a", "Muse-Glimmer-30B &#183; int4, sliding window &#183; TP2"),
}
GREY, GRID = "#8a8a8a", "#8a8a8a"

ap = argparse.ArgumentParser()
ap.add_argument("--source", default=os.path.join(HERE, "..", "results.jsonl"))
ap.add_argument("--stamp", default=JULY_STAMP,
                help="date / software / kernel line, drawn under the subtitle on every chart")
ap.add_argument("--suffix", default="", help="appended to each output filename")
ap.add_argument("--series", default="E-26B-tp2,B-8B-tp2,A-12B-tp2,C-31B-tp2,D-27B-tp2",
                help="comma-separated TP2 configs to plot, in legend order")
ap.add_argument("--tp1-series", default="B-8B-tp2,B-8B-tp1,A-12B-tp2,A-12B-tp1",
                help="comma-separated configs for the single-vs-dual chart; empty to skip it")
ap.add_argument("--vmax-decode", type=float, default=115)
ap.add_argument("--vmax-prefill", type=float, default=4600,
                help="4600 because the 2026-08-25 Qwen3-8B 2K point reaches 4445 tok/s "
                     "and both campaigns' prefill charts share an axis")
ap.add_argument("--vmax-tp1", type=float, default=90)
ap.add_argument("--vmax-ms", type=float, default=250)
a = ap.parse_args()

os.makedirs(OUT, exist_ok=True)
rows = [json.loads(l) for l in open(a.source) if l.strip()]

dec, pre = {}, {}
for r in rows:
    if r["kind"] == "decode" and r.get("decode_tps"):
        dec.setdefault(r["cfg"], {}).setdefault(r["target"], []).append(r["decode_tps"])
    elif r["kind"] == "prefill":
        pre.setdefault(r["cfg"], {}).setdefault(r["target"], []).append(r["prefill_tps"])
D = lambda c, t: sum(dec[c][t]) / len(dec[c][t]) if c in dec and t in dec[c] else None
P = lambda c, t: max(pre[c][t]) if c in pre and t in pre[c] else None
TARGETS = [500, 1000, 2000, 4000, 6000, 8000, 12000, 16000, 20000, 24000, 32000]


def nice_ticks(vmax):
    """0..vmax in 4-6 steps of a round size. Reproduces the hand-picked ticks the
    2026-07-25 charts used, and keeps working when a ceiling is changed."""
    for step in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000):
        if 4 <= vmax / step <= 6:
            return [i * step for i in range(int(vmax // step) + 1)]
    return [0, vmax]


def build(fn, title, sub, series, vmax, ylab, ncol=2):
    ticks = nice_ticks(vmax)
    rowsn = math.ceil(len(series) / ncol)
    W, H = 780, 344 + rowsn * 19
    L, R, T, B = 62, 762, 76, 282
    xm = lambda s: L + (math.log10(s) - math.log10(450)) / (math.log10(34000) - math.log10(450)) * (R - L)
    ym = lambda v: T + (1 - v / vmax) * (B - T)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif">',
         f'<text x="{L}" y="24" font-size="16" font-weight="700" fill="{GREY}">{title}</text>',
         f'<text x="{L}" y="42" font-size="11.5" fill="{GREY}" opacity=".85">{sub}</text>',
         f'<text x="{L}" y="58" font-size="10.5" fill="{GREY}" opacity=".7">{a.stamp}</text>']
    for tv in ticks:
        y = ym(tv)
        o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{R}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1" opacity=".28"/>')
        o.append(f'<text x="{L-8}" y="{y+4:.1f}" font-size="10.5" fill="{GREY}" text-anchor="end">{tv:g}</text>')
    for s in [500, 1000, 2000, 4000, 8000, 16000, 32000]:
        x = xm(s)
        o.append(f'<line x1="{x:.1f}" y1="{B}" x2="{x:.1f}" y2="{B+5}" stroke="{GRID}" stroke-width="1" opacity=".5"/>')
        o.append(f'<text x="{x:.1f}" y="{B+19}" font-size="10.5" fill="{GREY}" text-anchor="middle">'
                 f'{f"{s//1000}K" if s >= 1000 else s}</text>')
    o.append(f'<text x="18" y="{(T+B)/2:.0f}" font-size="11" fill="{GREY}" text-anchor="middle" '
             f'transform="rotate(-90 18 {(T+B)/2:.0f})">{ylab}</text>')
    o.append(f'<text x="{(L+R)/2:.0f}" y="{B+37}" font-size="11" fill="{GREY}" text-anchor="middle">'
             f'context length (tokens, log scale)</text>')
    for key, pts in series:
        col = C[key][0]
        dash = ' stroke-dasharray="7 4"' if key.endswith("tp1") else ""
        o.append('<polyline points="' + " ".join(f"{xm(s):.1f},{ym(v):.1f}" for s, v in pts) +
                 f'" fill="none" stroke="{col}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"{dash}/>')
        for s, v in pts:
            o.append(f'<circle cx="{xm(s):.1f}" cy="{ym(v):.1f}" r="3" fill="{col}"/>')
    for i, (key, _) in enumerate(series):
        col, lab = C[key]
        cx = L + (i % ncol) * ((R - L) / ncol)
        cy = B + 58 + (i // ncol) * 19
        dash = ' stroke-dasharray="7 4"' if key.endswith("tp1") else ""
        o.append(f'<line x1="{cx}" y1="{cy-4}" x2="{cx+22}" y2="{cy-4}" stroke="{col}" stroke-width="3"{dash}/>')
        o.append(f'<text x="{cx+29}" y="{cy}" font-size="11.5" fill="{GREY}">{lab}</text>')
    o.append("</svg>")
    open(f"{OUT}/{fn}", "w").write("\n".join(o))
    return fn


def name(base):
    return f"{base}{a.suffix}.svg"


TP2 = [k for k in a.series.split(",") if k.strip()]
missing = [k for k in TP2 if k not in dec]
if missing:
    print(f"note: no decode data for {missing} in {os.path.basename(a.source)}")
    TP2 = [k for k in TP2 if k not in missing]

sd = [(k, [(t, D(k, t)) for t in TARGETS if D(k, t)]) for k in TP2]
sp = [(k, [(t, P(k, t)) for t in TARGETS if P(k, t)]) for k in TP2]
sm = [(k, [(t, 1000 / D(k, t)) for t in TARGETS if D(k, t)]) for k in TP2]

charts = [
    build(name("decode-vs-context"), "Decode throughput vs context length",
          "TP=2, CUDA graph, 512-token outputs, mean of 2 runs",
          sd, a.vmax_decode, "decode tok/s"),
    build(name("prefill-vs-context"), "Prefill throughput vs context length",
          "max_tokens=1, throughput = prompt tokens / TTFT, best of 2 runs",
          sp, a.vmax_prefill, "prefill tok/s"),
    build(name("decode-ms-per-token"), "Cost of one context token at decode time",
          "slope = ms added per token of context; a linear-attention model should be flat",
          sm, a.vmax_ms, "ms per generated token"),
]

tp1_keys = [k for k in a.tp1_series.split(",") if k.strip() and k in dec]
if len(tp1_keys) >= 2:
    st = [(k, [(t, D(k, t)) for t in TARGETS if D(k, t)]) for k in tp1_keys]
    charts.append(
        build(name("tp1-vs-tp2"), "Single card vs dual card (TP=1 dashed, TP=2 solid)",
              "BF16 scales 1.70x; w4a16 only 1.19x - see 'why' in benchmarks.md",
              st, a.vmax_tp1, "decode tok/s"))
else:
    print("note: not enough TP=1 data for tp1-vs-tp2, skipped")

for f in charts:
    print("wrote", f)
