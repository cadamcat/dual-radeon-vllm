#!/usr/bin/env python3
"""Standalone SVG for the sliding-window block skip, in the same house style as
gen_charts.py: colours baked in so it reads on both GitHub themes, legend and
title inside the file so it works as a plain <img>.

Two panels. Left is milliseconds per generated token against context, dashed
before and solid after, with each model's own sliding window marked. Right is
the speed-up, whose shape is the point: 1.00x below the window because there is
nothing to skip, then monotonic."""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "docs", "assets")
DATA = os.path.join(HERE, "..", "sliding-window-block-skip.json")
os.makedirs(OUT, exist_ok=True)
d = json.load(open(DATA))

SERIES = [
    ("gemma-3-27b-it-quantized.w4a16", "#d99a24", "gemma-3-27b · w4a16 · window 1024", 1024),
    ("Muse-Glimmer-30B-INT4", "#8b6ee0", "Muse-Glimmer-30B · int4 · window 2048", 2048),
]
GREY = "#8a8a8a"
W, H = 900, 416
T, B = 94, 316


def panel(o, L, R, curves, vmax, ticks, ylab, title, ratio=False):
    xm = lambda s: L + (math.log10(s) - math.log10(450)) / (math.log10(34000) - math.log10(450)) * (R - L)
    ym = lambda v: T + (1 - (v - (1 if ratio else 0)) / (vmax - (1 if ratio else 0))) * (B - T)
    o.append(f'<text x="{L}" y="{T-16}" font-size="12.5" font-weight="600" fill="{GREY}">{title}</text>')
    for tv in ticks:
        y = ym(tv)
        o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{R}" y2="{y:.1f}" stroke="{GREY}" stroke-width="1" opacity=".28"/>')
        o.append(f'<text x="{L-8}" y="{y+4:.1f}" font-size="10.5" fill="{GREY}" text-anchor="end">{tv:g}</text>')
    for s in (512, 1024, 2048, 4096, 8192, 16384, 32768):
        x = xm(s)
        o.append(f'<line x1="{x:.1f}" y1="{B}" x2="{x:.1f}" y2="{B+5}" stroke="{GREY}" stroke-width="1" opacity=".5"/>')
        o.append(f'<text x="{x:.1f}" y="{B+19}" font-size="10" fill="{GREY}" text-anchor="middle">'
                 f'{f"{s//1024}K" if s >= 1024 else s}</text>')
    o.append(f'<text x="{L-44}" y="{(T+B)/2:.0f}" font-size="11" fill="{GREY}" text-anchor="middle" '
             f'transform="rotate(-90 {L-44} {(T+B)/2:.0f})">{ylab}</text>')
    for col, pts, dash, win in curves:
        if win is not None:
            x = xm(win)
            o.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{B}" stroke="{col}" stroke-width="1" '
                     f'stroke-dasharray="2 4" opacity=".55"/>')
        o.append('<polyline points="' + " ".join(f"{xm(s):.1f},{ym(v):.1f}" for s, v in pts) +
                 f'" fill="none" stroke="{col}" stroke-width="2.1"{dash}/>')
        for s, v in pts:
            o.append(f'<circle cx="{xm(s):.1f}" cy="{ym(v):.1f}" r="2.6" fill="{col}"/>')
    return xm, ym


def stamp():
    """date, software and kernel, from the data file rather than from here"""
    m = d["machine"]
    return (f'{m["measured"]} &#183; {m["userspace"]} &#183; kernel {m["kernel"]} '
            f'&#183; {m["gpu"]}')


def main():
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif">',
         f'<text x="52" y="26" font-size="16" font-weight="700" fill="{GREY}">'
         f'Skipping decode blocks the sliding-window mask would zero anyway</text>',
         f'<text x="52" y="45" font-size="11.5" fill="{GREY}" opacity=".85">'
         f'2× RX 7900 XT (gfx1100), TP=2, ROCM_ATTN. Median of three processes per point. '
         f'Vertical dashes mark each model’s own window.</text>',
         f'<text x="52" y="61" font-size="10.5" fill="{GREY}" opacity=".7">{stamp()}</text>']

    left, right = [], []
    for key, col, _lbl, win in SERIES:
        c = d["models_affected"][key]["depth_curve_n3"]
        left.append((col, [(r["depth"], r["before_median_ms"]) for r in c], ' stroke-dasharray="7 4"', win))
        left.append((col, [(r["depth"], r["after_median_ms"]) for r in c], "", None))
        right.append((col, [(r["depth"], r["speedup"]) for r in c], "", win))

    panel(o, 92, 452, left, 130, [0, 25, 50, 75, 100, 125], "ms per generated token",
          "cost — dashed before, solid after")
    panel(o, 548, 878, right, 3.3, [1, 1.5, 2, 2.5, 3], "speed-up (×)",
          "gain — 1.00× below the window, then monotonic", ratio=True)

    y = 340
    for i, (_key, col, lbl, _w) in enumerate(SERIES):
        x = 92 + i * 400
        o.append(f'<line x1="{x}" y1="{y-4}" x2="{x+26}" y2="{y-4}" stroke="{col}" stroke-width="2.4"/>')
        o.append(f'<text x="{x+33}" y="{y}" font-size="11.5" fill="{GREY}">{lbl}</text>')
    o.append(f'<text x="92" y="{y+22}" font-size="10.5" fill="{GREY}" opacity=".8">'
             f'Correctness is kernel-level: upstream’s own test file with no case changing outcome, '
             f'and 15 boundary cases bit-identical under torch.equal.</text>')
    o.append('</svg>')
    p = os.path.join(OUT, "sliding-window-block-skip.svg")
    open(p, "w").write("\n".join(o))
    print("wrote", p)


if __name__ == "__main__":
    main()
