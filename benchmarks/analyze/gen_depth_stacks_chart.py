#!/usr/bin/env python3
"""gen_depth_stacks_chart.py — one checkpoint, three software stacks, and the
trade the third one makes.

Every other figure on the front page holds the stack still and varies the
model. This one holds the model still: Qwen3.8-27B AWQ on the same two cards
at TP=2, decoded on vLLM 0.23.1 (the arm the 2026-09-03 campaign published),
on vLLM 0.27.1 with the backend it picks for itself, and on 0.27.1 with
`--attention-backend TRITON_ATTN`. The left panel is milliseconds per decoded
token against context, whose slope is what a token of context costs; the
legend carries that slope, fitted over the rungs all three ladders share.

The right panel is why the flattest line is not simply the best one: the
Triton arm divided by the ROCM_ATTN arm at every rung, for decode and for
prefill. One goes up with depth and the other goes down, which is the shape of
a trade, not of one backend winning.

Same conventions as the other generators here: baked colours, grey text,
legend and notes inside the SVG, byte-identical output. `render()` returns the
file so the verifier can compare the committed SVG with what the rows draw.

    python3 gen_depth_stacks_chart.py

Reads `decode.jsonl` and `prefill.jsonl` -- the cross-machine projections,
which `build_decode.py --check` and `build_prefill.py --check` recompute from
their campaign sources -- rather than aggregating the raw rows a third time.
"""
import json
import math
import os

from chartlib import nice_ticks

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "docs", "assets")
DECODE = os.path.join(HERE, "..", "decode.jsonl")
PREFILL = os.path.join(HERE, "..", "prefill.jsonl")
FN = "depth-cost-three-stacks.svg"

MACHINE = "RX 7900 XT"
SHARED = 32000          # the deepest rung every arm reaches; the fit stops here

# cfg, date in the projection, colour, legend label. The two 0.27.1 arms are
# one session pair from campaign-2026-09-07; the 0.23.1 arm is the pair's own
# long campaign of 2026-09-03. Dates are shown without the sitting suffix.
ARMS = [
    ("D8-27B-tp2-long", "2026-09-03", "#3f8fd4",
     "vLLM 0.23.1 &#183; ROCm 7.14 &#183; ROCM_ATTN &#183; 2026-09-03"),
    ("D8-27B-tp2-long-027b", "2026-09-07b", "#d99a24",
     "vLLM 0.27.1 &#183; ROCm 10.0 &#183; ROCM_ATTN &#183; 2026-09-07"),
    ("D8-27B-tp2-triton-long-027b", "2026-09-07b", "#e05c48",
     "vLLM 0.27.1 &#183; ROCm 10.0 &#183; TRITON_ATTN &#183; 2026-09-07"),
]
NUM, DEN = ARMS[2], ARMS[1]          # the trade: Triton over what 0.27.1 picks
DEC_COL, PRE_COL = "#2ea36a", "#8b6ee0"
GREY, GRID = "#8a8a8a", "#8a8a8a"


def ladder(rows, cfg, date, key):
    """ctx -> (value, chart_grade) for one arm of one projection."""
    return {r["ctx"]: (r[key], bool(r["chart_grade"])) for r in rows
            if r["machine"] == MACHINE and r["cfg"] == cfg and r["date"] == date}


def slope_us(d, hi):
    """OLS of ms per token on context over the rungs up to `hi`, in
    microseconds per context token -- the unit the Findings quote."""
    xs = sorted(k for k in d if k <= hi)
    ys = [1000.0 / d[k][0] for k in xs]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    b = (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
         / sum((x - mx) ** 2 for x in xs))
    return b * 1000


def render():
    dec = [json.loads(l) for l in open(DECODE)]
    pre = [json.loads(l) for l in open(PREFILL)]
    arms = [(ladder(dec, cfg, date, "decode_tok_s"), col, lab)
            for cfg, date, col, lab in ARMS]
    for d, _, lab in arms:
        assert d and max(d) >= SHARED, lab
    slopes = [slope_us(d, SHARED) for d, _, _ in arms]
    num_d, den_d = ladder(dec, NUM[0], NUM[1], "decode_tok_s"), ladder(dec, DEN[0], DEN[1], "decode_tok_s")
    num_p, den_p = ladder(pre, NUM[0], NUM[1], "prefill_tok_s"), ladder(pre, DEN[0], DEN[1], "prefill_tok_s")
    rungs = sorted(k for k in num_d if k in den_d and k in num_p and k in den_p)
    trade = [(k, num_d[k][0] / den_d[k][0], num_d[k][1] and den_d[k][1],
              num_p[k][0] / den_p[k][0], num_p[k][1] and den_p[k][1]) for k in rungs]
    deepest = rungs[-1]
    end_dec = next(t[1] for t in trade if t[0] == deepest)
    end_pre = next(t[3] for t in trade if t[0] == deepest)
    hollow = sorted(k for k, _, gd, _, gp in trade if not (gd and gp))

    W = 780
    L, R, T, B = 62, 436, 76, 296            # left panel: ms per token
    L2, R2 = 512, 762                        # right panel: the ratio
    XMIN, XMAX = 450, 150000
    xm = lambda s, l, r: l + (math.log10(s) - math.log10(XMIN)) / (
        math.log10(XMAX) - math.log10(XMIN)) * (r - l)
    VMAX = 120.0
    ym = lambda v: T + (1 - v / VMAX) * (B - T)
    RLO, RHI = 0.25, 1.5
    ym2 = lambda v: T + (1 - (v - RLO) / (RHI - RLO)) * (B - T)

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {{H}}" width="{W}" '
         f'height="{{H}}" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,'
         f'Helvetica,Arial,sans-serif">',
         f'<text x="{L}" y="24" font-size="16" font-weight="700" fill="{GREY}">'
         f'One checkpoint, three software stacks: what a token of context costs</text>',
         f'<text x="{L}" y="42" font-size="11.5" fill="{GREY}" opacity=".85">'
         f'Qwen3.8-27B AWQ, 2x RX 7900 XT, TP=2 &#183; left: ms per decoded token against context '
         f'&#183; right: TRITON_ATTN &#247; ROCM_ATTN</text>',
         f'<text x="{L}" y="58" font-size="10.5" fill="{GREY}" opacity=".7">'
         f'same weights, same two cards, same ladder &#183; the 0.23.1 arm ends at '
         f'{max(arms[0][0]) // 1000} K, the 0.27.1 arms at {deepest // 1000} K &#183; '
         f'every decode point chart-grade</text>']

    # the span the slopes are fitted over, shaded before anything is drawn on it
    o.append(f'<rect x="{xm(500, L, R):.1f}" y="{T}" width="{xm(SHARED, L, R) - xm(500, L, R):.1f}" '
             f'height="{B - T}" fill="{GREY}" opacity=".06"/>')
    o.append(f'<text x="{xm(500, L, R) + 4:.1f}" y="{T + 12}" font-size="9.5" fill="{GREY}" '
             f'opacity=".8">slopes fitted here: 500 &#8211; 32 K</text>')

    for tv in nice_ticks(VMAX):
        y = ym(tv)
        o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{R}" y2="{y:.1f}" stroke="{GRID}" '
                 f'stroke-width="1" opacity=".28"/>')
        o.append(f'<text x="{L - 8}" y="{y + 4:.1f}" font-size="10.5" fill="{GREY}" '
                 f'text-anchor="end">{tv:g}</text>')
    for tv in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
        y = ym2(tv)
        op = ".6" if tv == 1.0 else ".28"
        o.append(f'<line x1="{L2}" y1="{y:.1f}" x2="{R2}" y2="{y:.1f}" stroke="{GRID}" '
                 f'stroke-width="1" opacity="{op}"/>')
        o.append(f'<text x="{L2 - 8}" y="{y + 4:.1f}" font-size="10.5" fill="{GREY}" '
                 f'text-anchor="end">{tv:g}&#215;</text>')
    for l, r in ((L, R), (L2, R2)):
        for s in (500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000):
            x = xm(s, l, r)
            o.append(f'<line x1="{x:.1f}" y1="{B}" x2="{x:.1f}" y2="{B + 5}" stroke="{GRID}" '
                     f'stroke-width="1" opacity=".5"/>')
            o.append(f'<text x="{x:.1f}" y="{B + 19}" font-size="10.5" fill="{GREY}" '
                     f'text-anchor="middle">{f"{s // 1000}K" if s >= 1000 else s}</text>')
        o.append(f'<text x="{(l + r) / 2:.0f}" y="{B + 37}" font-size="11" fill="{GREY}" '
                 f'text-anchor="middle">context length (tokens, log scale)</text>')
    o.append(f'<text x="18" y="{(T + B) / 2:.0f}" font-size="11" fill="{GREY}" '
             f'text-anchor="middle" transform="rotate(-90 18 {(T + B) / 2:.0f})">'
             f'ms per decoded token</text>')
    o.append(f'<text x="{L2 - 44}" y="{(T + B) / 2:.0f}" font-size="11" fill="{GREY}" '
             f'text-anchor="middle" transform="rotate(-90 {L2 - 44} {(T + B) / 2:.0f})">'
             f'TRITON_ATTN &#247; ROCM_ATTN</text>')

    # left: one line per stack
    for d, col, lab in arms:
        pts = [(k, 1000.0 / d[k][0]) for k in sorted(d)]
        o.append('<polyline points="' + " ".join(f"{xm(s, L, R):.1f},{ym(v):.1f}" for s, v in pts)
                 + f'" fill="none" stroke="{col}" stroke-width="2.4" stroke-linejoin="round" '
                 f'stroke-linecap="round"/>')
        for s, v in pts:
            o.append(f'<circle cx="{xm(s, L, R):.1f}" cy="{ym(v):.1f}" r="3" fill="{col}"/>')

    # right: the two ratios; a hollow point is a rung the ledger does not grade
    # on one of the arms it divides
    for idx, col, dashed in ((1, DEC_COL, False), (3, PRE_COL, True)):
        pts = [(t[0], t[idx], t[idx + 1]) for t in trade]
        dash = ' stroke-dasharray="7 4"' if dashed else ""
        o.append('<polyline points="' + " ".join(f"{xm(s, L2, R2):.1f},{ym2(v):.1f}" for s, v, _ in pts)
                 + f'" fill="none" stroke="{col}" stroke-width="2.4" stroke-linejoin="round" '
                 f'stroke-linecap="round"{dash}/>')
        for s, v, graded in pts:
            fill = col if graded else "#ffffff"
            o.append(f'<circle cx="{xm(s, L2, R2):.1f}" cy="{ym2(v):.1f}" r="3" fill="{fill}" '
                     f'stroke="{col}" stroke-width="1.5"/>')
    o.append(f'<text x="{xm(deepest, L2, R2) - 6:.1f}" y="{ym2(end_dec) - 8:.1f}" font-size="10.5" '
             f'font-weight="700" fill="{DEC_COL}" text-anchor="end">decode {end_dec:.2f}&#215;</text>')
    o.append(f'<text x="{xm(deepest, L2, R2) - 6:.1f}" y="{ym2(end_pre) + 14:.1f}" font-size="10.5" '
             f'font-weight="700" fill="{PRE_COL}" text-anchor="end">prefill {end_pre:.2f}&#215;</text>')

    # legends: the stacks with their fitted slopes on the left, the ratios on the right
    ly = B + 58
    for i, ((d, col, lab), b) in enumerate(zip(arms, slopes)):
        cy = ly + i * 19
        o.append(f'<line x1="{L}" y1="{cy - 4}" x2="{L + 22}" y2="{cy - 4}" stroke="{col}" '
                 f'stroke-width="3"/>')
        o.append(f'<text x="{L + 29}" y="{cy}" font-size="11.5" fill="{GREY}">{lab} &#8212; '
                 f'{b:.3f} &#181;s per context token</text>')
    for i, (col, dashed, lab) in enumerate((
            (DEC_COL, False, "right panel, decode: TRITON_ATTN &#247; ROCM_ATTN at each rung"),
            (PRE_COL, True, "right panel, prefill: the same ratio &#183; hollow: a rung the ledger "
                            "does not grade on one of the two arms"))):
        cy = ly + (3 + i) * 19
        dash = ' stroke-dasharray="7 4"' if dashed else ""
        o.append(f'<line x1="{L}" y1="{cy - 4}" x2="{L + 22}" y2="{cy - 4}" stroke="{col}" '
                 f'stroke-width="3"{dash}/>')
        o.append(f'<text x="{L + 29}" y="{cy}" font-size="11.5" fill="{GREY}">{lab}</text>')

    ny = ly + 5 * 19 + 8
    notes = [
        "slope: least squares of ms per decoded token on context over the rungs all three ladders share, "
        "500 to 32 000, which is the number",
        f"the Findings quote; the steepest is {max(slopes) / min(slopes):.2f}x the flattest. "
        "Patches, from the rows: the 0.23.1 arm carries vllm#45916 split-KV,",
        "the window block-skip and vllm#45450; the 0.27.1 arms carry vllm#45916 split-KV. "
        "The version step also changes ROCm and the weight",
        "kernel (Triton W4A16 to the native RDNA hybrid kernel); within 0.27.1 only the backend flag "
        f"changes. Hollow prefill rung{'s' if len(hollow) != 1 else ''}: "
        + ", ".join(f"{k // 1000} K" if k >= 1000 else str(k) for k in hollow) + ".",
        "decode.jsonl and prefill.jsonl, RX 7900 XT rows; campaign-2026-09-03 and campaign-2026-09-07 "
        "are the campaigns, and the latter holds the",
        "drift control that prices the session boundary between its two arms.",
    ]
    for n in notes:
        o.append(f'<text x="{L}" y="{ny}" font-size="10" fill="{GREY}" opacity=".75">{n}</text>')
        ny += 15
    o.append("</svg>")
    H = ny - 15 + 12
    return "\n".join(o).replace("{H}", str(H)) + "\n"


def main():
    path = os.path.join(OUT, FN)
    open(path, "w").write(render())
    print("wrote", os.path.relpath(path, os.path.join(HERE, "..", "..")))


if __name__ == "__main__":
    main()
