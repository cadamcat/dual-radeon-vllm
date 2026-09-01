#!/usr/bin/env python3
"""build_decode.py — every decode measurement, on every machine, one row format.

`ledger.jsonl` is the Radeon box's decode projection and stays that way: it has
a `rocm` column, eighteen Radeon configuration ids and no room for a machine,
which is why the front page builds its A100 series from a hand-written function
instead of from the ledger. This is the cross-machine projection, the decode
half of what `build_prefill.py` does for prefill, and it carries `machine`.

    python3 build_decode.py            # write ../decode.jsonl
    python3 build_decode.py --check    # fail if the committed file is stale,
                                       # or if it disagrees with the ledger

**The two files must not fork.** Both project the same Radeon rows, so `--check`
recomputes the overlap and fails if any cell's values differ. The campaign
table, the model metadata and the arm table are imported from
`build_prefill.py` rather than restated, so a source added for one projection is
a source for both.
"""
import argparse
import json
import os
import statistics
import sys

from build_ledger import RANGE_CUT, latest_session, B, PROBES, MODELS, CFG
from build_prefill import SOURCES, meta_for, arm_for, routes_from_source


DECODE = B("decode.jsonl")
LEDGER = B("ledger.jsonl")


def aggregate(values, tokens, **row):
    v = sorted(values)
    row["values"] = v
    row["runs"] = len(v)
    row["decode_tok_s"] = statistics.mean(v)
    # probe rows record a context, not a measured prompt length
    row["prompt_tokens"] = round(statistics.mean(tokens)) or None
    row["range_pct"] = (v[-1] - v[0]) / statistics.mean(v) * 100 if len(v) > 1 else None
    row["chart_grade"] = row["runs"] >= 2 and row["range_pct"] <= RANGE_CUT
    if not row["chart_grade"]:
        row["chart_grade_note"] = (
            f"{row['runs']} run(s), range {row['range_pct']:.2f}%"
            if row["runs"] > 1 else "one run")
    return row


def build():
    rows = []
    for s in SOURCES:
        path = B(s["file"])
        if not os.path.exists(path):
            continue
        by = {}
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("kind") != "decode" or not r.get("decode_tps"):
                continue
            by.setdefault((r["cfg"], r["target"]), []).append(
                (r["ts"], r["decode_tps"], r["prompt_tokens"]))
        # why the backend was chosen, from the serve logs beside this source.
        # `backend` itself still comes from the arm table here, unlike prefill,
        # which reads the log first -- that asymmetry predates this and is not
        # settled by adding a column.
        routed = routes_from_source(path)
        for (cfg, target), trips in sorted(by.items()):
            name, quant, arch, tp = meta_for(cfg)
            spec, backend = arm_for(cfg)
            over = s.get("per_cfg", {}).get(cfg, {})
            vals, superseded = latest_session([(t, (d, pt)) for t, d, pt in trips])
            extra = {"superseded_values": sorted(x[0] for x in superseded)} \
                if superseded else {}
            rows.append(aggregate(
                [x[0] for x in vals], [x[1] for x in vals], **extra,
                machine=s["machine"], model=name, quant=quant, arch=arch, tp=tp,
                ctx=target, date=s["date"], vllm=over.get("vllm", s["vllm"]),
                rocm=over.get("rocm", s["rocm"]), cuda=over.get("cuda", s["cuda"]),
                driver=over.get("driver", s.get("driver")),
                kernel=over.get("kernel", s["kernel"]),
                patches=over.get("patches", s["patches"]),
                harness="campaign-server", source=s["file"], cfg=cfg,
                spec=spec, attn_backend=backend, route=routed.get(cfg),
                prefix_caching=over.get("prefix_caching", s.get("prefix_caching"))))
    # The probe sources have decode and no prefill, so they are not in the
    # shared SOURCES table -- but the ledger carries them, and this projection
    # has to be a superset of the ledger or the cross-check below compares two
    # different questions. They are arm-structured: the arm decides `patches`,
    # not the file.
    for prb in PROBES:
        by = {}
        for f in prb["files"]:
            path = B(f)
            if not os.path.exists(path):
                continue
            for line in open(path):
                if not line.strip():
                    continue
                r = json.loads(line)
                by.setdefault((MODELS[r["model"]], r["arm"], r["ctx"]), []).append(
                    r["decode_tok_s"])
        for (cfg, arm, ctx), vals in sorted(by.items()):
            name, quant, arch, tp = CFG[cfg]
            rows.append(aggregate(
                vals, [0], machine="RX 7900 XT", model=name, quant=quant, arch=arch,
                tp=tp, ctx=ctx, date=prb["date"], vllm=prb["vllm"], rocm=prb["rocm"],
                cuda=None, driver=None, kernel=prb["kernel"],
                patches=prb["arms"][arm], harness="probe-t8t64",
                source=", ".join(prb["files"]), cfg=cfg, spec=None,
                attn_backend=None, prefix_caching=None))

    rows.sort(key=lambda r: (r["machine"], r["model"], r["tp"], r["date"],
                             ",".join(r["patches"]), r["ctx"]))
    return rows


def ledger_disagreements(rows):
    """Cells both projections cover must carry the same numbers.

    The ledger is built from a subset of these same files, so every one of its
    rows should appear here with identical values. If it does not, one of the
    two was edited by hand or their source tables have drifted apart, and the
    repository has two answers to the same question.
    """
    if not os.path.exists(LEDGER):
        return ["ledger.jsonl is missing"]
    # `patches` is part of the key, not decoration: the probe sources measured
    # the same cell with and without vllm#45916, so (cfg, ctx, date) names two
    # different rows and comparing on it put a stock arm against a patched one
    # -- 12.57 against 40.14 at 8K, which is the patch's whole effect.
    key = lambda r: (r["cfg"], r["ctx"], r["date"], ",".join(r["patches"]))
    mine = {key(r): r for r in rows}
    bad = []
    for line in open(LEDGER):
        L = json.loads(line)
        k = key(L)
        m = mine.get(k)
        if m is None:
            bad.append(f"{k} is in ledger.jsonl and not here")
        elif m["values"] != L["values"]:
            bad.append(f"{k} values differ: {L['values']} vs {m['values']}")
    return bad


def dump(rows):
    return "".join(json.dumps(r) + "\n" for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    rows = build()
    text = dump(rows)

    if a.check:
        have = open(DECODE).read() if os.path.exists(DECODE) else ""
        if have != text:
            print("decode.jsonl is stale; re-run build_decode.py", file=sys.stderr)
            return 1
        bad = ledger_disagreements(rows)
        if bad:
            print("decode.jsonl disagrees with ledger.jsonl:", file=sys.stderr)
            for b in bad[:10]:
                print("  " + b, file=sys.stderr)
            return 1
        machines = sorted({r["machine"] for r in rows})
        print(f"decode.jsonl matches its sources and the ledger: "
              f"{len(rows)} rows, {len(machines)} machines ({', '.join(machines)})")
        return 0

    open(DECODE, "w").write(text)
    print(f"wrote {DECODE}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
