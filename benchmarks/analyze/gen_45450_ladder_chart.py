#!/usr/bin/env python3
"""gen_45450_ladder_chart.py — the dual-vendor spec-decode ladder.

Four series from committed data: the stock MTP collapse and vllm#45450's
admission, on the Radeons (mtp-31b-{stock45450,p45450}.json, four depths)
and on the A100 (the validation logs, five depths).
Same conventions as the other generators: baked colours, grey text,
legend and stamp inside the SVG, byte-identical output. ROCm keeps the
house orange, the A100 the palette blue; dashed = stock 2D, solid = the
admission's 3D.

    python3 gen_45450_ladder_chart.py
"""
import json, math, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "docs", "assets")
SDIR = os.path.join(HERE, "..", "speculative-decoding")
VDIR = os.path.join(HERE, "..", "cuda-a100", "45450-validation", "logs")

STAMP = ("2026-08-26 &#183; 2x RX 7900 XT (TP2, ROCm 7.14) &#183; "
         "A100 80G (CUDA, vLLM 0.28.0) &#183; gemma-4-31B w4a16 + MTP k=1 "
         "&#183; single-run probes")

ROCM, A100 = "#d99a24", "#3f8fd4"
GREY, GRID = "#8a8a8a", "#8a8a8a"


def sweep(tag):
    d = json.load(open(os.path.join(SDIR, f"mtp-31b-{tag}.json")))
    return [(r["depth"], r["tok_per_s"]) for r in d["rows"]]


def leg(name, depth):
    text = open(os.path.join(VDIR, name)).read()
    return (depth, float(re.search(r"RESULT decode_tok_s=([\d.]+)", text).group(1)))


SERIES = [
    (ROCM, True,  "2x RX 7900 XT, stock (2D under speculation)", sweep("stock45450")),
    (ROCM, False, "2x RX 7900 XT, with #45450 (3D admitted)", sweep("p45450")),
    (A100, True,  "A100 80G, stock",
     [leg("C1K.log", 1000), leg("C8K.log", 8000), leg("C16K.log", 16000),
      leg("C30.log", 30000), leg("C50.log", 50000)]),
    (A100, False, "A100 80G, with #45450",
     [leg("D1K.log", 1000), leg("D8K.log", 8000), leg("D16K.log", 16000),
      leg("D30.log", 30000), leg("D50.log", 50000)]),
]

VMAX = 115.0
TICKS = [0, 25, 50, 75, 100]
W, H = 780, 396
L, R, T, B = 62, 762, 76, 300
xm = lambda s: L + (math.log10(s) - math.log10(900)) / (math.log10(55000) - math.log10(900)) * (R - L)
ym = lambda v: T + (1 - v / VMAX) * (B - T)

o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
     f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif">',
     f'<text x="{L}" y="24" font-size="16" font-weight="700" fill="{GREY}">'
     f'The MTP collapse, fixed on both vendors (vllm#45450)</text>',
     f'<text x="{L}" y="42" font-size="11.5" fill="{GREY}" opacity=".85">'
     f'decode tok/s vs context, speculation on; dashed = stock routing '
     f'(2D path), solid = with the 3D admission</text>',
     f'<text x="{L}" y="58" font-size="10.5" fill="{GREY}" opacity=".7">{STAMP}</text>']

for tv in TICKS:
    y = ym(tv)
    o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{R}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1" opacity=".28"/>')
    o.append(f'<text x="{L-8}" y="{y+4:.1f}" font-size="10.5" fill="{GREY}" text-anchor="end">{tv:g}</text>')
for s in [1000, 2000, 4000, 8000, 16000, 32000, 50000]:
    x = xm(s)
    o.append(f'<line x1="{x:.1f}" y1="{B}" x2="{x:.1f}" y2="{B+5}" stroke="{GRID}" stroke-width="1" opacity=".5"/>')
    o.append(f'<text x="{x:.1f}" y="{B+19}" font-size="10.5" fill="{GREY}" text-anchor="middle">{s//1000}K</text>')
o.append(f'<text x="18" y="{(T+B)/2:.0f}" font-size="11" fill="{GREY}" text-anchor="middle" '
         f'transform="rotate(-90 18 {(T+B)/2:.0f})">decode tok/s</text>')
o.append(f'<text x="{(L+R)/2:.0f}" y="{B+37}" font-size="11" fill="{GREY}" text-anchor="middle">'
         f'context length (tokens, log scale)</text>')

for col, dashed, lab, pts in SERIES:
    dash = ' stroke-dasharray="7 4"' if dashed else ""
    o.append('<polyline points="' + " ".join(f"{xm(s):.1f},{ym(v):.1f}" for s, v in pts) +
             f'" fill="none" stroke="{col}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"{dash}/>')
    for s, v in pts:
        o.append(f'<circle cx="{xm(s):.1f}" cy="{ym(v):.1f}" r="3" fill="{col}"/>')

for i, (col, dashed, lab, _) in enumerate(SERIES):
    cx = L + (i % 2) * ((R - L) / 2)
    cy = B + 58 + (i // 2) * 19
    dash = ' stroke-dasharray="7 4"' if dashed else ""
    o.append(f'<line x1="{cx}" y1="{cy-4}" x2="{cx+22}" y2="{cy-4}" stroke="{col}" stroke-width="3"{dash}/>')
    o.append(f'<text x="{cx+29}" y="{cy}" font-size="11.5" fill="{GREY}">{lab}</text>')
o.append("</svg>")

path = os.path.join(OUT, "spec-decode-45450-ladder.svg")
open(path, "w").write("\n".join(o))
print("wrote", os.path.relpath(path, os.path.join(HERE, "..", "..")))
