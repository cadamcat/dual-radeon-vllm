#!/usr/bin/env python3
"""gen_svg.py — standalone SVG charts for the GitHub repo.
Colours are baked in (no CSS vars) and chosen to read on both light and dark
GitHub themes; axes/text use neutral grey. Legend and title live inside the SVG
so the file works as a plain <img> in markdown."""
import json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "docs", "assets")
os.makedirs(OUT, exist_ok=True)
rows = [json.loads(l) for l in open(os.path.join(HERE, "..", "results.jsonl")) if l.strip()]

C = {
    "E-26B-tp2":  ("#8b6ee0", "gemma-4-26B-A4B · MoE · TP2"),
    "B-8B-tp2":   ("#3f8fd4", "Qwen3-8B · BF16 · TP2"),
    "A-12B-tp2":  ("#2ea36a", "gemma-4-12B · w4a16 · TP2"),
    "C-31B-tp2":  ("#d99a24", "gemma-4-31B · w4a16 · TP2"),
    "D-27B-tp2":  ("#e05c48", "Qwen3.6-27B · hybrid SSM · TP2"),
    "B-8B-tp1":   ("#3f8fd4", "Qwen3-8B · BF16 · TP1 (single card)"),
    "A-12B-tp1":  ("#2ea36a", "gemma-4-12B · w4a16 · TP1 (single card)"),
}
GREY, GRID = "#8a8a8a", "#8a8a8a"

dec, pre = {}, {}
for r in rows:
    if r["kind"] == "decode" and r.get("decode_tps"):
        dec.setdefault(r["cfg"], {}).setdefault(r["target"], []).append(r["decode_tps"])
    elif r["kind"] == "prefill":
        pre.setdefault(r["cfg"], {}).setdefault(r["target"], []).append(r["prefill_tps"])
D = lambda c, t: sum(dec[c][t]) / len(dec[c][t]) if c in dec and t in dec[c] else None
P = lambda c, t: max(pre[c][t]) if c in pre and t in pre[c] else None
TARGETS = [500, 1000, 2000, 4000, 6000, 8000, 12000, 16000, 20000, 24000, 32000]

def build(fn, title, sub, series, vmax, ylab, ticks, ncol=2):
    rowsn = math.ceil(len(series) / ncol)
    W, H = 780, 330 + rowsn * 19
    L, R, T, B = 62, 762, 62, 268
    xm = lambda s: L + (math.log10(s) - math.log10(450)) / (math.log10(34000) - math.log10(450)) * (R - L)
    ym = lambda v: T + (1 - v / vmax) * (B - T)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif">',
         f'<text x="{L}" y="26" font-size="16" font-weight="700" fill="{GREY}">{title}</text>',
         f'<text x="{L}" y="45" font-size="11.5" fill="{GREY}" opacity=".85">{sub}</text>']
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

TP2 = ["E-26B-tp2", "B-8B-tp2", "A-12B-tp2", "C-31B-tp2", "D-27B-tp2"]
sd = [(k, [(t, D(k, t)) for t in TARGETS if D(k, t)]) for k in TP2]
sp = [(k, [(t, P(k, t)) for t in TARGETS if P(k, t)]) for k in TP2]
st = [(k, [(t, D(k, t)) for t in TARGETS if D(k, t)]) for k in
      ["B-8B-tp2", "B-8B-tp1", "A-12B-tp2", "A-12B-tp1"]]
sm = [(k, [(t, 1000 / D(k, t)) for t in TARGETS if D(k, t)]) for k in TP2]

for f in [
    build("decode-vs-context.svg", "Decode throughput vs context length",
          "2x RX 7900 XT, vLLM 0.23 + ROCm 7.14, TP=2, CUDA graph, 512-token outputs, mean of 2 runs",
          sd, 115, "decode tok/s", [0, 20, 40, 60, 80, 100]),
    build("prefill-vs-context.svg", "Prefill throughput vs context length",
          "max_tokens=1, throughput = prompt tokens / TTFT, best of 2 runs",
          sp, 4200, "prefill tok/s", [0, 1000, 2000, 3000, 4000]),
    build("tp1-vs-tp2.svg", "Single card vs dual card (TP=1 dashed, TP=2 solid)",
          "BF16 scales 1.70x; w4a16 only 1.19x - see 'why' in benchmarks.md",
          st, 90, "decode tok/s", [0, 20, 40, 60, 80]),
    build("decode-ms-per-token.svg", "Cost of one context token at decode time",
          "slope = ms added per token of context; a linear-attention model should be flat",
          sm, 250, "ms per generated token", [0, 50, 100, 150, 200, 250]),
]:
    print("wrote", f)
