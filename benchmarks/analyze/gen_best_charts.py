#!/usr/bin/env python3
"""gen_best_charts.py — one line per model, the best this machine is known to do.

The per-campaign charts answer "what did that run measure?". A reader arriving
at this repository is asking "what does this box do?", and answering it with two
figures per question -- which is where the front page ended up, carrying two
versions of the same chart -- makes them do the subtraction themselves.

So the series here is not a campaign. It is a model, drawn from whichever stack
measured it best, with the stack travelling with the line instead of sitting in
one stamp over the whole figure:

  * solid  -- every point came from a released vLLM with no patch applied
  * dashed -- the line needs a patch that is not merged upstream, named below
    the chart

Selection, in order, per model at TP=2:
  1. candidates are the (date, vllm, rocm, patches) groups in the ledger
  2. score each at the deepest context it reaches
  3. take the best, but prefer an unpatched candidate when it is within 2% of
     it, because a chart should not send a reader to install a patch that buys
     them nothing
  4. drop points the ledger marks not chart-grade, and leave the gap visible

Two models are deliberately absent. Qwen3.6-27B is the same architecture as
Qwen3.8-27B and was superseded by it; gemma-3-27b runs within two tok/s of both
Muse-Glimmer and gemma-4-31B between 500 and 4000, so three lines read as one.
Both are in the ledger and in benchmarks.md.

    python3 gen_best_charts.py
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "docs", "assets")
LEDGER = os.path.join(HERE, "..", "ledger.jsonl")

COLOUR = {
    "gemma-4-26B-A4B": ("#8b6ee0", "gemma-4-26B-A4B &#183; MoE"),
    "Qwen3-8B": ("#3f8fd4", "Qwen3-8B &#183; BF16"),
    "gemma-4-12B-it": ("#2ea36a", "gemma-4-12B &#183; w4a16"),
    "Muse-Glimmer-30B": ("#d1519a", "Muse-Glimmer-30B &#183; sliding window"),
    "gemma-4-31B-it": ("#d99a24", "gemma-4-31B &#183; w4a16"),
    "Qwen3.8-27B": ("#e05c48", "Qwen3.8-27B &#183; hybrid SSM"),
}
ORDER = ["gemma-4-26B-A4B", "Qwen3-8B", "gemma-4-12B-it", "Muse-Glimmer-30B",
         "gemma-4-31B-it", "Qwen3.8-27B"]
GREY = GRID = "#8a8a8a"
PREFER_UNPATCHED_WITHIN = 0.02


def pick(rows):
    """One series per model: the best stack, unpatched preferred at a tie."""
    groups = {}
    for r in rows:
        if r["tp"] != 2 or r["model"] not in COLOUR:
            continue
        groups.setdefault(
            (r["model"], r["date"], r["vllm"], r["rocm"], tuple(r["patches"])),
            []).append(r)

    chosen = {}
    for key, pts in groups.items():
        model = key[0]
        graded = [p for p in pts if p["chart_grade"]]
        if not graded:
            continue
        deepest = max(graded, key=lambda p: p["ctx"])
        cand = (deepest["ctx"], deepest["decode_tok_s"], key, pts)
        best = chosen.get(model)
        if best is None:
            chosen[model] = cand
            continue
        # deeper context wins outright; at the same depth, the faster one does,
        # unless an unpatched candidate is close enough not to be worth a patch
        if cand[0] > best[0] or (cand[0] == best[0] and cand[1] > best[1]):
            if (best[2][4] == () and cand[2][4] != ()
                    and cand[1] <= best[1] * (1 + PREFER_UNPATCHED_WITHIN)):
                continue
            chosen[model] = cand
        elif (cand[2][4] == () and best[2][4] != () and cand[0] == best[0]
                and best[1] <= cand[1] * (1 + PREFER_UNPATCHED_WITHIN)):
            chosen[model] = cand
    return chosen


def nice_ticks(vmax):
    for step in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000):
        if 4 <= vmax / step <= 6:
            return [i * step for i in range(int(vmax // step) + 1)]
    return [0, vmax]


def build(fn, title, sub, series, vmax, ylab, notes):
    ticks = nice_ticks(vmax)
    rows_legend = math.ceil(len(series) / 2)
    W = 780
    H = 352 + rows_legend * 19 + len(notes) * 15
    L, R, T, B = 62, 762, 76, 282
    xm = lambda s: L + (math.log10(s) - math.log10(450)) / (
        math.log10(34000) - math.log10(450)) * (R - L)
    ym = lambda v: T + (1 - v / vmax) * (B - T)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
         f'height="{H}" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,'
         f'Helvetica,Arial,sans-serif">',
         f'<text x="{L}" y="24" font-size="16" font-weight="700" fill="{GREY}">{title}</text>',
         f'<text x="{L}" y="42" font-size="11.5" fill="{GREY}" opacity=".85">{sub}</text>',
         f'<text x="{L}" y="58" font-size="10.5" fill="{GREY}" opacity=".7">'
         f'2x RX 7900 XT &#183; TP=2 &#183; solid: released vLLM, no patch &#183; '
         f'dashed: needs an unmerged patch, named below</text>']
    for tv in ticks:
        y = ym(tv)
        o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{R}" y2="{y:.1f}" stroke="{GRID}" '
                 f'stroke-width="1" opacity=".28"/>')
        o.append(f'<text x="{L-8}" y="{y+4:.1f}" font-size="10.5" fill="{GREY}" '
                 f'text-anchor="end">{tv:g}</text>')
    for s in [500, 1000, 2000, 4000, 8000, 16000, 32000]:
        x = xm(s)
        o.append(f'<line x1="{x:.1f}" y1="{B}" x2="{x:.1f}" y2="{B+5}" stroke="{GRID}" '
                 f'stroke-width="1" opacity=".5"/>')
        o.append(f'<text x="{x:.1f}" y="{B+19}" font-size="10.5" fill="{GREY}" '
                 f'text-anchor="middle">{f"{s//1000}K" if s >= 1000 else s}</text>')
    o.append(f'<text x="18" y="{(T+B)/2:.0f}" font-size="11" fill="{GREY}" '
             f'text-anchor="middle" transform="rotate(-90 18 {(T+B)/2:.0f})">{ylab}</text>')
    o.append(f'<text x="{(L+R)/2:.0f}" y="{B+37}" font-size="11" fill="{GREY}" '
             f'text-anchor="middle">context length (tokens, log scale)</text>')

    for model, pts, patched in series:
        col = COLOUR[model][0]
        dash = ' stroke-dasharray="7 4"' if patched else ""
        # a gap is a gap: consecutive points only join if nothing was dropped
        # between them, so a missing cell reads as missing rather than as a line
        for i in range(len(pts) - 1):
            x0, y0, gap = pts[i]
            x1, y1, _ = pts[i + 1]
            if gap:
                continue
            o.append(f'<line x1="{xm(x0):.1f}" y1="{ym(y0):.1f}" x2="{xm(x1):.1f}" '
                     f'y2="{ym(y1):.1f}" stroke="{col}" stroke-width="2.4" '
                     f'stroke-linecap="round"{dash}/>')
        for x, y, _ in pts:
            o.append(f'<circle cx="{xm(x):.1f}" cy="{ym(y):.1f}" r="3" fill="{col}"/>')

    for i, (model, _, patched) in enumerate(series):
        col, lab = COLOUR[model]
        cx = L + (i % 2) * ((R - L) / 2)
        cy = B + 58 + (i // 2) * 19
        dash = ' stroke-dasharray="7 4"' if patched else ""
        o.append(f'<line x1="{cx}" y1="{cy-4}" x2="{cx+22}" y2="{cy-4}" stroke="{col}" '
                 f'stroke-width="3"{dash}/>')
        o.append(f'<text x="{cx+29}" y="{cy}" font-size="11.5" fill="{GREY}">{lab}</text>')

    ny = B + 58 + rows_legend * 19 + 8
    for n in notes:
        o.append(f'<text x="{L}" y="{ny}" font-size="10" fill="{GREY}" opacity=".75">{n}</text>')
        ny += 15
    o.append("</svg>")
    open(os.path.join(OUT, fn), "w").write("\n".join(o))
    return fn


def main():
    led = [json.loads(l) for l in open(LEDGER)]
    chosen = pick(led)

    series, notes = [], []
    for model in ORDER:
        if model not in chosen:
            continue
        _, _, key, pts = chosen[model]
        graded = sorted([p for p in pts if p["chart_grade"]], key=lambda p: p["ctx"])
        dropped = sorted(p["ctx"] for p in pts if not p["chart_grade"])
        # third element marks "a gap starts here", so build() can skip the join
        marked = []
        for i, p in enumerate(graded):
            nxt = graded[i + 1]["ctx"] if i + 1 < len(graded) else None
            gap = any(p["ctx"] < d < nxt for d in dropped) if nxt else False
            marked.append((p["ctx"], p["decode_tok_s"], gap))
        patched = bool(key[4])
        series.append((model, marked, patched))
        # the footnote has to fit 700px at 10px, so the vLLM version is cut at
        # its dev tag and the patch list keeps only the PR numbers
        vllm = key[2].split("+")[0]
        stack = f"{key[1]} &#183; vLLM {vllm} &#183; ROCm {key[3]}"
        if patched:
            short = ", ".join(x.split(" ")[0] for x in key[4])
            stack += f" &#183; needs {short}"
        note = f"{COLOUR[model][1].split(' &#183;')[0]} &#183; {stack}"
        if dropped:
            note += (" &#183; no point at "
                     + ", ".join(f"{d//1000}K" if d >= 1000 else str(d) for d in dropped)
                     + ": passes differ >6%")
        notes.append(note)

    mseries = [(m, [(x, 1000 / y, g) for x, y, g in pts], p) for m, pts, p in series]
    out = [
        build("decode-vs-context-best.svg", "Decode throughput vs context length",
              "one line per model, the best configuration measured here",
              series, 115, "decode tok/s", notes),
        build("decode-ms-per-token-best.svg", "Cost of one context token at decode time",
              "slope = ms added per token of context; a linear-attention model should be flat",
              # 40, not the 250 the per-campaign chart needed: with each model
              # drawn at its best the hybrid-SSM collapse is no longer on this
              # figure -- 235 ms/token at 32K was the unpatched Qwen3.6. That
              # contrast now lives in docs/hybrid-decode-on-rdna.md, which is
              # the cost of asking this chart "what does the machine do?"
              # instead of "what did that campaign measure?".
              mseries, 40, "ms per generated token", notes),
    ]
    for model in ORDER:
        if model in chosen:
            k = chosen[model][2]
            print(f"  {model:<18} <- {k[1]} vllm {k[2]} {list(k[4]) or 'stock'}")
    for f in out:
        print("wrote", f)


if __name__ == "__main__":
    main()
