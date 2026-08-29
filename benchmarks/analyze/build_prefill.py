#!/usr/bin/env python3
"""build_prefill.py — project every prefill measurement onto one row format.

The decode ledger is one machine's. `ledger.jsonl` carries a `rocm` column and
eighteen configuration ids that are all Radeon, because it was built to answer
"what does this box do?". Prefill is being asked across five machines, so it
needs a column the ledger deliberately does not have -- which is why this is a
second projection rather than more rows in the first. `ledger.jsonl` stays
decode-only and its 258 rows and every gate that counts them are untouched.

What a row is: one (machine, configuration, rung) point, aggregated over the
rounds that measured it, carrying the machine and the stack it was measured on.

    python3 build_prefill.py            # write ../prefill.jsonl
    python3 build_prefill.py --check    # fail if the committed file is stale
    python3 build_prefill.py --fits     # the a/b/c decomposition per config

Three rules are encoded here rather than left to the chart:

  * **Rungs are grouped by `target`, not by `prompt_tokens`.** Both rounds of a
    rung report the same `prompt_tokens` on CUDA and counts that differ by one
    to three tokens on ROCm, because the ladders were cut by different methods
    -- `cut_prompts.py` searches sentence boundaries, `a100_run.py` truncates
    token ids. Grouping on the measured count therefore gives eleven paired
    points on one machine and nineteen mostly-unpaired ones on the other, and
    `min()` over a bucket of one is not "the quieter of two rounds". Every row
    carries `target`, which is exact on both. Measured on 2026-08-30: the
    choice moves the fitted fixed cost `a` by -43 % to +50 % and the throughput
    peak by up to 24 %, while `b` and `c` move by under 2.6 %.
  * `runs` and `range_pct` travel with the point, as they do in the ledger. The
    same cut applies, for the same reason.
  * `attn_backend` is read from the serve log, and vLLM 0.28 writes it two
    ways: `Using AttentionBackendEnum.TRITON_ATTN backend.` from one branch and
    `Using FLASH_ATTN attention backend out of potential backends: [...]` from
    another, in the same source file. A regex for one silently misses the
    other, which is why the 2026-08-29 A100 campaign recorded no backend at
    all. The vision-tower lines (`for vit attention`, `MMEncoderAttention`) are
    a third form and must not be mistaken for the decoder's.
"""
import argparse
import json
import os
import re
import statistics
import sys

from build_ledger import CFG, ARMS, RANGE_CUT, latest_session, B

PREFILL = B("prefill.jsonl")

# The CUDA half. `build_ledger.CFG` is Radeon-only by construction, so the
# A100's arms and this round's are named here, in the same grammar: the
# numeric format first, then how the checkpoint was produced.
CFG_CUDA = {
    "A100-G12":                ("gemma-4-12B-it",  "w4a16 QAT", "dense", 1),
    "A100-G26A4B":             ("gemma-4-26B-A4B", "int4 AWQ",  "MoE, 128 experts", 1),
    "A100-G26A4B-mtp":         ("gemma-4-26B-A4B", "int4 AWQ",  "MoE, 128 experts", 1),
    "A100-G26A4B-mtp-p45450":  ("gemma-4-26B-A4B", "int4 AWQ",  "MoE, 128 experts", 1),
    "A100-G31":                ("gemma-4-31B-it",  "w4a16 QAT", "dense", 1),
    "A100-G31-mtp":            ("gemma-4-31B-it",  "w4a16 QAT", "dense", 1),
    "A100-G31-mtp-p45450":     ("gemma-4-31B-it",  "w4a16 QAT", "dense", 1),
    "A100-MG30":               ("Muse-Glimmer-30B", "int4",     "sliding window 2048", 1),
    "A100-MG30-dflash":        ("Muse-Glimmer-30B", "int4",     "sliding window 2048", 1),
    "A100-Q38":                ("Qwen3.8-27B",     "int4 AWQ",  "hybrid SSM", 1),
    "A100-Q38-mtp":            ("Qwen3.8-27B",     "int4 AWQ",  "hybrid SSM", 1),
    "A100-Q38-mtp-p45450":     ("Qwen3.8-27B",     "int4 AWQ",  "hybrid SSM", 1),
    # 2026-08-30, one 7900 XT. The id carries its utilisation because no other
    # row in either projection was measured at 0.95.
    "E26-tp1-u95":             ("gemma-4-26B-A4B", "int4 AWQ",  "MoE, 128 experts", 1),
}

MTP3 = "mtp k=3"
DRAFT3 = "draft_model k=3"
DFLASH8 = "dflash k=8"

# spec, attn_backend. The backends are read from the serve logs kept in
# cuda-a100/campaign-2026-08-29/logs/, and from the campaign's own reading
# where a log was not kept: gemma-4 goes to TRITON_ATTN and Qwen3.8 to
# FLASH_ATTN on this machine, which is the whole point of that campaign's
# central finding. Muse-Glimmer's log was not kept and its backend is null
# rather than guessed.
ARMS_CUDA = {
    "A100-G12":               (None,    "TRITON_ATTN"),
    "A100-G26A4B":            (None,    "TRITON_ATTN"),
    "A100-G26A4B-mtp":        (DRAFT3,  "TRITON_ATTN"),
    "A100-G26A4B-mtp-p45450": (DRAFT3,  "TRITON_ATTN"),
    "A100-G31":               (None,    "TRITON_ATTN"),
    "A100-G31-mtp":           (DRAFT3,  "TRITON_ATTN"),
    "A100-G31-mtp-p45450":    (DRAFT3,  "TRITON_ATTN"),
    "A100-MG30":              (None,    None),
    "A100-MG30-dflash":       (DFLASH8, None),
    "A100-Q38":               (None,    "FLASH_ATTN"),
    "A100-Q38-mtp":           (MTP3,    "FLASH_ATTN"),
    "A100-Q38-mtp-p45450":    (MTP3,    "FLASH_ATTN"),
    "E26-tp1-u95":            (None,    "TRITON_ATTN"),
}

# Every prefill source, and the machine it ran on. The Radeon entries mirror
# build_ledger.CAMPAIGNS; they are restated rather than imported because this
# file adds a machine to each and the ledger has no place to put one.
SOURCES = [
    dict(file="results.jsonl", machine="RX 7900 XT", date="2026-07-25",
         vllm="0.23", rocm="7.14", cuda=None, kernel="7.0.0-28", patches=[]),
    dict(file="results-2026-08-24.jsonl", machine="RX 7900 XT", date="2026-08-24",
         vllm="0.23.1.dev1+g9ddef7117", rocm="7.14", cuda=None, kernel="7.0.0-30",
         patches=["vllm#45916 split-KV", "window block-skip"]),
    # prefix_caching is read from the serve logs, and it is not uniform even
    # within this campaign: the gemma-4 arms ran on the 0.23 container with it
    # True, the Qwen3.8 arms on 0.27 with it False. It is recorded rather than
    # acted on, because on this machine True did not produce hits -- G31-tp2's
    # two rounds agree to 1.00x at 32 K. What actually gates the fit is
    # `chart_grade`, which does not need to know the cause.
    dict(file="campaign-2026-08-29/results.jsonl", machine="RX 7900 XT",
         date="2026-08-29", vllm="0.27.1.dev5+gf46a9dfe2", rocm="10.0", cuda=None,
         kernel="7.0.0-30", patches=["vllm#45916 split-KV"],
         prefix_caching=False,
         per_cfg={
             "Q38-mtp-p45450-tp2": dict(
                 patches=["vllm#45916 split-KV", "vllm#45450 3D admission"]),
             "Q38-mtp-triton-p45450-tp2": dict(
                 patches=["vllm#45916 split-KV", "vllm#45450 3D admission"]),
             "G31-tp2": dict(
                 vllm="0.23.1.dev1+g9ddef7117", rocm="7.14", prefix_caching=True,
                 patches=["vllm#45916 split-KV", "window block-skip",
                          "vllm#45450 3D admission"]),
             "G31-mtp-p45450-tp2": dict(
                 vllm="0.23.1.dev1+g9ddef7117", rocm="7.14", prefix_caching=True,
                 patches=["vllm#45916 split-KV", "window block-skip",
                          "vllm#45450 3D admission"]),
         }),
    # enable_prefix_caching=True, and here it did hit: every rung is a strict
    # prefix of the next, so 130 of these 132 rungs fail the repeatability cut.
    # This campaign's prefill cannot be used and has to be measured again.
    # 2026-08-30. gemma-4-26B-A4B on ONE 7900 XT, at util 0.95 and a ladder the
    # card cut short: 16.96 GiB of weights resident on a 19.98 GiB card left
    # 0.93 GiB of KV, 13 149 tokens, so seven rungs of the eleven. The util
    # 0.92 attempt is in the same file as a config_failed row -- 1536 tokens --
    # and its serve log is beside this one. `enable_prefix_caching=True` here
    # and, as on every Radeon arm, it produced no hits: the two rounds of the
    # 12 000 rung are 6.209 s and 6.207 s.
    dict(file="campaign-2026-08-30/results.jsonl", machine="RX 7900 XT",
         date="2026-08-30", vllm="0.23.1.dev1+g9ddef7117.d20260715", rocm="7.14",
         cuda=None, kernel="7.0.0-30", patches=[], prefix_caching=True),
    dict(file="cuda-a100/campaign-2026-08-29/results.jsonl", prefix_caching=True,
         machine="A100-SXM4-80GB", date="2026-08-29", vllm="0.28.0",
         rocm=None, cuda="13.0", kernel=None, patches=[],
         per_cfg={
             "A100-G31-mtp-p45450":    dict(patches=["vllm#45450 3D admission"]),
             "A100-G26A4B-mtp-p45450": dict(patches=["vllm#45450 3D admission"]),
             "A100-Q38-mtp-p45450":    dict(patches=["vllm#45450 3D admission"]),
         }),
]

BACKEND_RE = re.compile(
    r"Using (?:AttentionBackendEnum\.)?([A-Z0-9_]+)(?: attention)? backend")
VIT_RE = re.compile(r"vit attention|MMEncoderAttention")


def backend_from_log(path):
    """The decoder's backend, from either of the two forms vLLM 0.28 writes."""
    if not os.path.exists(path):
        return None
    for line in open(path, errors="ignore"):
        if VIT_RE.search(line):
            continue                      # the vision tower, not the decoder
        m = BACKEND_RE.search(line)
        if m:
            return m.group(1)
    return None


def meta_for(cfg):
    if cfg in CFG:
        return CFG[cfg]
    if cfg in CFG_CUDA:
        return CFG_CUDA[cfg]
    raise KeyError(f"unknown cfg {cfg!r} — add it to CFG_CUDA")


def arm_for(cfg):
    if cfg in ARMS:
        return ARMS[cfg]
    return ARMS_CUDA.get(cfg, (None, None))


def aggregate(values, tokens, **row):
    """values are TTFT in seconds; `tokens` the measured prompt length."""
    v = sorted(values)
    row["values"] = v
    row["runs"] = len(v)
    row["ttft_s"] = statistics.mean(v)
    row["prompt_tokens"] = round(statistics.mean(tokens))
    row["prefill_tok_s"] = row["prompt_tokens"] / row["ttft_s"] if row["ttft_s"] else None
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
            if r.get("kind") != "prefill" or not r.get("ttft"):
                continue
            by.setdefault((r["cfg"], r["target"]), []).append(
                (r["ts"], r["ttft"], r["prompt_tokens"]))
        for (cfg, target), trips in sorted(by.items()):
            name, quant, arch, tp = meta_for(cfg)
            spec, backend = arm_for(cfg)
            over = s.get("per_cfg", {}).get(cfg, {})
            vals, superseded = latest_session([(t, (tt, pt)) for t, tt, pt in trips])
            extra = {"superseded_values": sorted(x[0] for x in superseded)} \
                if superseded else {}
            rows.append(aggregate(
                [x[0] for x in vals], [x[1] for x in vals], **extra,
                machine=s["machine"], model=name, quant=quant, arch=arch, tp=tp,
                ctx=target, date=s["date"], vllm=over.get("vllm", s["vllm"]),
                rocm=over.get("rocm", s["rocm"]), cuda=over.get("cuda", s["cuda"]),
                kernel=over.get("kernel", s["kernel"]),
                patches=over.get("patches", s["patches"]),
                harness="campaign-server", source=s["file"], cfg=cfg,
                spec=spec, attn_backend=backend,
                prefix_caching=over.get("prefix_caching", s.get("prefix_caching"))))
    rows.sort(key=lambda r: (r["machine"], r["model"], r["tp"], r["date"],
                             ",".join(r["patches"]), r["ctx"]))
    return rows


def solve(A, y):
    m = len(A)
    M = [row[:] + [y[i]] for i, row in enumerate(A)]
    for col in range(m):
        p = max(range(col, m), key=lambda r: abs(M[r][col]))
        M[col], M[p] = M[p], M[col]
        for r in range(m):
            if r != col and M[col][col]:
                f = M[r][col] / M[col][col]
                for k in range(col, m + 1):
                    M[r][k] -= f * M[col][k]
    return [M[i][m] / M[i][i] for i in range(m)]


def fits(rows):
    """T(S) = a + b*S + c*S^2 per (machine, cfg, date, patches).

    **Only chart-grade rungs are fitted.** A rung whose two rounds disagree is
    not a measurement of anything, and on the 2026-08-29 A100 campaign that is
    130 of 132 rungs: prefix caching was on, every rung is a strict prefix of
    the next, and round 2 of the 32 K rung took 0.201 s against round 1's
    2.932 s. Fitting through that produced b = 4.1 us/tok against the Radeon's
    446 -- a hundredfold gap no hardware explains -- because `min()` selects
    the cached round. The repeatability cut the ledger already applies catches
    this without needing to know the cause, which is why it is the gate here.

    A configuration needs four rungs to determine three coefficients, so
    shorter ladders are reported and not fitted.
    """
    out, by = [], {}
    for r in rows:
        # date and patches, not just the id: gemma-4-12B at TP=1 was measured
        # on 2026-07-25 and again on 2026-08-24, and those are two
        # configurations of the same arm rather than four rounds of one. Fitting
        # them together produced a 22-point curve and a fixed cost belonging to
        # neither campaign.
        by.setdefault((r["machine"], r["cfg"], r["date"],
                       ",".join(r["patches"])), []).append(r)
    for (machine, cfg, date, patches), rs in sorted(by.items()):
        rs_all = sorted(rs, key=lambda r: r["ctx"])
        rs = [r for r in rs_all if r["chart_grade"]]
        S = [r["prompt_tokens"] for r in rs]
        T = [min(r["values"]) for r in rs]
        n = len(S)
        rec = {"machine": machine, "cfg": cfg, "model": rs_all[0]["model"],
               "tp": rs_all[0]["tp"], "spec": rs_all[0]["spec"],
               "attn_backend": rs_all[0]["attn_backend"], "rungs": n,
               "rungs_measured": len(rs_all), "date": date,
               "patches": rs_all[0]["patches"]}
        if n < 4:
            rec["note"] = (f"{n} of {len(rs_all)} rungs chart-grade, "
                           f"too few to fit three coefficients")
            out.append(rec)
            continue
        P = [[sum(s ** (i + j) for s in S) for j in range(3)] for i in range(3)]
        q = [sum(T[k] * S[k] ** i for k in range(n)) for i in range(3)]
        a, b, c = solve(P, q)
        pred = [a + b * s + c * s * s for s in S]
        ss_res = sum((T[k] - pred[k]) ** 2 for k in range(n))
        mean_t = statistics.mean(T)
        ss_tot = sum((t - mean_t) ** 2 for t in T)
        rec |= {"a_ms": a * 1000, "b_us_tok": b * 1e6, "c_ns_tok2": c * 1e9,
                "r2": 1 - ss_res / ss_tot if ss_tot else None,
                "s_star": (a / c) ** 0.5 if c > 0 and a > 0 else None}
        out.append(rec)
    return out


def dump(rows):
    return "".join(json.dumps(r) + "\n" for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--fits", action="store_true")
    a = ap.parse_args()
    rows = build()

    if a.fits:
        print(f"{'machine':<16} {'cfg':<26} {'date':<11} {'rungs':>7} {'a ms':>8} "
              f"{'b us/tok':>9} {'c ns/tok2':>10} {'S*':>7} {'r2':>7}  backend")
        for f in fits(rows):
            if "a_ms" not in f:
                print(f"{f['machine']:<16} {f['cfg']:<26} {f['date']:<11} "
                      f"{f['rungs']:>3}/{f['rungs_measured']:<3}  {f['note']}")
                continue
            ss = f"{f['s_star']:7.0f}" if f["s_star"] else "      -"
            print(f"{f['machine']:<16} {f['cfg']:<26} {f['date']:<11} "
                  f"{f['rungs']:>3}/{f['rungs_measured']:<3} {f['a_ms']:8.1f} "
                  f"{f['b_us_tok']:9.1f} {f['c_ns_tok2']:10.2f} {ss} {f['r2']:7.4f}  "
                  f"{f['attn_backend'] or '-'}")
        return 0

    text = dump(rows)
    if a.check:
        have = open(PREFILL).read() if os.path.exists(PREFILL) else ""
        if have != text:
            print("prefill.jsonl is stale; re-run build_prefill.py", file=sys.stderr)
            return 1
        print(f"prefill.jsonl matches its sources: {len(rows)} rows")
        return 0
    open(PREFILL, "w").write(text)
    print(f"wrote {PREFILL}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
