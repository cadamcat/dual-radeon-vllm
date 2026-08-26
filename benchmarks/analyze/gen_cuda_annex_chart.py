#!/usr/bin/env python3
"""gen_cuda_annex_chart.py — the one chart of the CUDA A100 annex.

Grouped bars: decode tok/s for the four routing/speculation combinations at
30K and at 50K context, from ../cuda-a100/gemma4-mtp-backend-matrix.json.
Same conventions as gen_charts.py: colours baked in and readable on both
GitHub themes, neutral grey text, legend and stamp inside the SVG, output
byte-identical on every run. TRITON keeps gemma-4-31B's campaign colour
(#d99a24) since it is the same model on its forced default; FlashInfer takes
the palette's blue. The no-spec bar of each pair is the pale one; MTP solid.

    python3 gen_cuda_annex_chart.py
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "docs", "assets")
SRC = os.path.join(HERE, "..", "cuda-a100", "gemma4-mtp-backend-matrix.json")

STAMP = ("2026-08-26 &#183; Colab A100-SXM4-80GB &#183; vLLM 0.28.0 &#183; "
         "flashinfer 0.6.16.post3 &#183; single-run probes, two sessions")

TRITON, FLASHINFER = "#d99a24", "#3f8fd4"
GREY, GRID = "#8a8a8a", "#8a8a8a"

M = json.load(open(SRC))["decode_tok_s"]

# (label under the group, [(colour, pale?, value), ...] left to right)
GROUPS = [
    ("30K context", [
        (TRITON, True,  M["30000"]["triton_forced"]["nospec"]),
        (TRITON, False, M["30000"]["triton_forced"]["mtp"]),
        (FLASHINFER, True,  M["30000"]["flashinfer_explicit"]["nospec"]),
        (FLASHINFER, False, M["30000"]["flashinfer_explicit"]["mtp"]),
    ]),
    ("50K context", [
        (TRITON, True,  M["50000"]["triton_forced"]["nospec"]),
        (TRITON, False, M["50000"]["triton_forced"]["mtp"]),
        (FLASHINFER, True,  M["50000"]["flashinfer_explicit"]["nospec"]),
        (FLASHINFER, False, M["50000"]["flashinfer_explicit"]["mtp"]),
    ]),
]
VMAX = 80.0
TICKS = [0, 20, 40, 60, 80]

W, H = 780, 396
L, R, T, B = 62, 762, 76, 300
# two groups of four bars inside L..R = 700px: 40 + 274 + 72 + 274 + 40
BAR, GAP, GROUP_PAD, GROUP_SEP = 58, 14, 40, 72
ym = lambda v: T + (1 - v / VMAX) * (B - T)

o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
     f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif">',
     f'<text x="{L}" y="24" font-size="16" font-weight="700" fill="{GREY}">'
     f'gemma-4-31B: the MTP collapse is the routing, not the model</text>',
     f'<text x="{L}" y="42" font-size="11.5" fill="{GREY}" opacity=".85">'
     f'decode tok/s, batch 1, text-only serving; MTP = official drafter, k=1; '
     f'TRITON_ATTN is the forced default when FA4 is unavailable (vllm#52049)</text>',
     f'<text x="{L}" y="58" font-size="10.5" fill="{GREY}" opacity=".7">{STAMP}</text>']

for tv in TICKS:
    y = ym(tv)
    o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{R}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1" opacity=".28"/>')
    o.append(f'<text x="{L-8}" y="{y+4:.1f}" font-size="10.5" fill="{GREY}" text-anchor="end">{tv:g}</text>')
o.append(f'<text x="18" y="{(T+B)/2:.0f}" font-size="11" fill="{GREY}" text-anchor="middle" '
         f'transform="rotate(-90 18 {(T+B)/2:.0f})">decode tok/s</text>')

group_w = 4 * BAR + 3 * GAP
x0 = L + GROUP_PAD
for glabel, bars in GROUPS:
    for i, (col, pale, v) in enumerate(bars):
        x = x0 + i * (BAR + GAP)
        y = ym(v)
        fill = f' fill="{col}" opacity=".40"' if pale else f' fill="{col}"'
        o.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{BAR}" height="{B - y:.1f}"{fill}/>')
        o.append(f'<text x="{x + BAR/2:.1f}" y="{y - 6:.1f}" font-size="11" font-weight="600" '
                 f'fill="{GREY}" text-anchor="middle">{v:.2f}</text>')
    o.append(f'<text x="{x0 + group_w/2:.1f}" y="{B + 19}" font-size="11.5" fill="{GREY}" '
             f'text-anchor="middle">{glabel}</text>')
    x0 += group_w + GROUP_SEP

LEGEND = [
    (TRITON, True,  "TRITON_ATTN (default), MTP off"),
    (TRITON, False, "TRITON_ATTN (default), MTP on"),
    (FLASHINFER, True,  "FlashInfer, MTP off"),
    (FLASHINFER, False, "FlashInfer, MTP on"),
]
for i, (col, pale, lab) in enumerate(LEGEND):
    cx = L + (i % 2) * ((R - L) / 2)
    cy = B + 44 + (i // 2) * 19
    fill = f' fill="{col}" opacity=".40"' if pale else f' fill="{col}"'
    o.append(f'<rect x="{cx}" y="{cy - 10}" width="14" height="12"{fill}/>')
    o.append(f'<text x="{cx + 21}" y="{cy}" font-size="11.5" fill="{GREY}">{lab}</text>')
o.append("</svg>")

path = os.path.join(OUT, "gemma4-mtp-backend-matrix-a100.svg")
open(path, "w").write("\n".join(o))
print("wrote", os.path.relpath(path, os.path.join(HERE, "..", "..")))
