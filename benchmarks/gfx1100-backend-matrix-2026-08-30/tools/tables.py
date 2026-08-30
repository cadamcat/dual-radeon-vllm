#!/usr/bin/env python3
"""Final tables for the four-arm gfx1100 matrix."""
import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-yaoxu-Documents-dual-radeon-vllm/2eed2b7e-71ff-49ba-a03a-28a95d771313/scratchpad')
from analyze52684 import load, fit

F = sys.argv[1] if len(sys.argv) > 1 else \
    '/private/tmp/claude-501/-Users-yaoxu-Documents-dual-radeon-vllm/2eed2b7e-71ff-49ba-a03a-28a95d771313/scratchpad/ab-results.jsonl'
A, B = 'Q38-rocm-nopatch-tp2', 'Q38-rocm-45916-tp2'
C, D = 'Q38-triton-stock-tp2', 'Q38-triton-52684-tp2'
NAME = {A: 'A  ROCM_ATTN, release', B: 'B  ROCM_ATTN +#45916',
        C: 'C  TRITON_ATTN, release', D: 'D  TRITON_ATTN +#52684'}
CTX = [500, 1000, 2000, 4000, 6000, 8000, 12000, 16000, 20000, 24000, 32000]

pre = load(F, 'prefill'); dec = load(F, 'decode')
P = {(r['cfg'], r['ctx']): r for r in pre}
Dd = {(r['cfg'], r['ctx']): r for r in dec}

def cell(idx, cfg, c, key):
    """None unless the rung is chart-grade -- a rung whose two rounds disagree
    by more than 8% is not a measurement and must not reach a table."""
    r = idx.get((cfg, c))
    if r is None or not r["chart_grade"]:
        return None
    return r[key]

print("### DECODE  tok/s\n")
print("| ctx | A release | B +#45916 | C TRITON | D TRITON +#52684 | C/A | D/B |")
print("|---|---:|---:|---:|---:|---:|---:|")
for c in CTX:
    a, b, cc, d = (cell(Dd, x, c, 'decode_tok_s') for x in (A, B, C, D))
    f = lambda v: "-" if v is None else f"{v:.2f}"
    ca = f"{cc/a:.2f}x" if (a and cc) else "-"
    db = f"{d/b:.3f}" if (b and d) else "-"
    print(f"| {c:,} | {f(a)} | {f(b)} | {f(cc)} | {f(d)} | **{ca}** | {db} |".replace(",", " "))

print("\n### PREFILL  tok/s\n")
print("| ctx | A release | B +#45916 | C TRITON | D TRITON +#52684 | D/C | D/B |")
print("|---|---:|---:|---:|---:|---:|---:|")
for c in CTX:
    a, b, cc, d = (cell(P, x, c, 'prefill_tok_s') for x in (A, B, C, D))
    f = lambda v: "-" if v is None else f"{v:.1f}"
    dc = f"{d/cc:.2f}x" if (cc and d) else "-"
    db = f"{d/b:.3f}" if (b and d) else "-"
    print(f"| {c:,} | {f(a)} | {f(b)} | {f(cc)} | {f(d)} | **{dc}** | {db} |".replace(",", " "))

print("\n### FITS  t(n) = a + b*n + c*n^2, chart-grade rungs, faster round\n")
print("| arm | rungs | a ms | b us/tok | c ns/tok^2 | r2 |")
print("|---|---:|---:|---:|---:|---:|")
for cfg in (A, B, C, D):
    rows = [r for r in pre if r['cfg'] == cfg]
    if not rows: continue
    res, n, tot = fit(rows)
    if res is None:
        print(f"| {NAME[cfg]} | {n}/{tot} | - | - | - | - |"); continue
    aa, bb, ccx, r2 = res
    print(f"| {NAME[cfg]} | {n}/{tot} | {aa*1e3:.1f} | {bb*1e6:.1f} | **{ccx*1e9:.2f}** | {r2:.4f} |")

print("\n### CHART-GRADE / ERROR AUDIT\n")
for cfg in (A, B, C, D):
    pr = [r for r in pre if r['cfg'] == cfg]; dr = [r for r in dec if r['cfg'] == cfg]
    if not pr: continue
    print(f"  {NAME[cfg]:28s} prefill {sum(r['chart_grade'] for r in pr)}/{len(pr)}  "
          f"decode {sum(r['chart_grade'] for r in dr)}/{len(dr)}  "
          f"worst prefill range {max(r['range_pct'] for r in pr):.2f}%  "
          f"worst decode range {max(r['range_pct'] for r in dr):.2f}%")
