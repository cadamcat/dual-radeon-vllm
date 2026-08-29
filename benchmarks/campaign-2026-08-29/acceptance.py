#!/usr/bin/env python3
"""Acceptance length per rung, from the serve logs the campaign already wrote.

handoff 5 records "Acceptance rate is the obvious suspect and was not
captured." It was captured -- vLLM logs `SpecDecoding metrics` every ten
seconds -- it was just never read. Each rung's decode rows carry a `ts`, and
each metrics line carries a timestamp, so a line belongs to the rung whose
measurement window contains it.

    python3 acceptance.py <results.jsonl> <serve-logs-dir> [cfg ...]

Rungs with no line inside their window are printed as "-" rather than filled
in from a neighbour: a ten-second cadence does not cover a two-second rung.
"""
import json, os, re, sys, statistics, collections, datetime

RES, LOGS = sys.argv[1], sys.argv[2]
want = sys.argv[3:] or None

M = re.compile(r"INFO (\d\d)-(\d\d) (\d\d):(\d\d):(\d\d).*?"
               r"Mean acceptance length: ([0-9.]+).*?"
               r"Per-position acceptance rate: ([0-9., ]+)")

rows = [json.loads(l) for l in open(RES) if l.strip()]
by_cfg = collections.defaultdict(list)
for r in rows:
    if r.get("kind") == "decode" and r.get("ts"):
        by_cfg[r["cfg"]].append(r)

for cfg in sorted(by_cfg):
    if want and cfg not in want:
        continue
    p = os.path.join(LOGS, cfg + ".log")
    if not os.path.exists(p):
        continue
    marks = []
    year = datetime.datetime.fromtimestamp(by_cfg[cfg][0]["ts"],
                                           datetime.timezone.utc).year
    for line in open(p, errors="replace"):
        m = M.search(line)
        if not m:
            continue
        mo, d, hh, mm, ss = (int(x) for x in m.groups()[:5])
        t = datetime.datetime(year, mo, d, hh, mm, ss,
                              tzinfo=datetime.timezone.utc).timestamp()
        pos = [float(x) for x in m.group(7).split(",") if x.strip()]
        marks.append((t, float(m.group(6)), pos))
    if not marks:
        continue
    pts = sorted(by_cfg[cfg], key=lambda r: r["ts"])
    print(f"\n{cfg}  ({len(marks)} metrics lines)")
    print(f"  {'ctx':>7} {'accept_len':>11} {'per-position':>26}")
    prev = pts[0]["ts"] - 60
    seen = {}
    for r in pts:
        lo, hi = prev, r["ts"]
        inside = [m for m in marks if lo < m[0] <= hi]
        prev = r["ts"]
        if not inside:
            continue
        seen.setdefault(r["target"], []).extend(inside)
    for ctx in sorted(seen):
        al = statistics.mean(m[1] for m in seen[ctx])
        pos = seen[ctx][-1][2]
        print(f"  {ctx:>7} {al:>11.2f}   {', '.join(f'{x:.3f}' for x in pos):>24}")
    allm = statistics.mean(m[1] for m in marks)
    print(f"  overall mean acceptance length {allm:.2f}")
