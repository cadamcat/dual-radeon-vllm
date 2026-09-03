"""The index's "what this machine does today" figure.

One line per model, each the fastest configuration that model has been measured
in. That is not one experiment: five of the lines come from a single campaign
and are directly comparable, and two do not, because a later stack or
speculative decoding beats that campaign by more than its own spread. Which
line is which is data, not a footnote -- every series carries the stack it
needed, and this script refuses to emit a pick it cannot show is the best one
in the repository.

Run it; do not hand-edit figures-index.json.
"""
import json, pathlib, re, sys

R = pathlib.Path(__file__).resolve().parents[2]
B = R / "benchmarks"
led = [json.loads(l) for l in open(B / "ledger.jsonl")]

# The backbone: one day, one stack, one harness, so these lines are one
# experiment and may be read against each other.
CAMPAIGN = "2026-08-24"
BACKBONE = ["gemma-4-26B-A4B", "Qwen3-8B", "Muse-Glimmer-30B", "gemma-4-12B-it"]
# What the reader is most likely to be here for, lit without being asked.
LIT = ["gemma-4-31B-it", "gemma-4-26B-A4B", "gemma-4-12B-it", "Qwen3.8-27B",
       "Muse-Glimmer-30B"]
# Deliberately absent: Qwen3.6-27B exists in the ledger only as the superseded
# 2026-07-25 stock run whose collapse the split-KV work fixed, so it is not a
# statement about today. gemma-3-27b-it is one sitting on one machine
# (2026-08-24, F-27B-tp2) of a checkpoint gemma-4 replaced; it stays in the
# projections and in the 2026-08-24 write-ups as the control it was measured
# as, and since 2026-09-03 it is not offered on the front page, where a model
# with nothing to compare against is a button that does nothing.
OMIT = ["Qwen3.6-27B", "gemma-3-27b-it"]
# each omission names the page string that explains it, so the two languages
# can say why in their own words and a third omission cannot borrow a reason
OMIT_WHY = {"Qwen3.6-27B": "bestOmitSuperseded", "gemma-3-27b-it": "bestOmitControl"}
assert set(OMIT_WHY) == set(OMIT)

sid = lambda r: (r["model"], r["tp"], r["vllm"], tuple(r["patches"]), r["harness"], r["date"])


def ledger_series(model, tp=2, date=None, vllm=None, patches=None, cfg=None):
    rows = [r for r in led if r["model"] == model and r["tp"] == tp
            and (date is None or r["date"] == date)
            and (vllm is None or r["vllm"] == vllm)
            and (patches is None or r["patches"] == patches)
            and (cfg is None or r["cfg"] == cfg)]
    assert rows, (model, tp, date, vllm, patches, cfg)
    ids = {sid(r) for r in rows}
    assert len(ids) == 1, f"{model}: {len(ids)} series match, not one"
    rows.sort(key=lambda r: r["ctx"])
    i = ids.pop()
    return {"model": model, "tp": tp, "vllm": i[2], "patches": list(i[3]),
            "harness": i[4], "date": i[5], "quant": rows[0]["quant"],
            "arch": rows[0]["arch"], "spec": rows[0]["spec"] is not None,
            "spec_desc": rows[0]["spec"], "attn_backend": rows[0]["attn_backend"],
            "cfg": rows[0]["cfg"],
            "source": "benchmarks/ledger.jsonl",
            "points": [{"ctx": r["ctx"], "tok_s": r["decode_tok_s"], "runs": r["runs"],
                        "range_pct": r["range_pct"], "graded": r["chart_grade"]}
                       for r in rows]}


series = [ledger_series(m, 2, CAMPAIGN) for m in BACKBONE]

# --- the two models the campaign does not represent -------------------------
# Both from 2026-08-29, eleven rungs and two rounds a cell, neither speculating.
# They used to be a three-point probe and a four-point k=1 *speculative* arm,
# which meant one line in a chart captioned "what this machine does today" had
# MTP on and the rest did not, with nothing on the page saying so. Speculation
# is a button now, below.
#
# Qwen3.8-27B: the campaign ran it on 0.23.1, which has no native gfx1100 W4A16
# kernel for an asymmetric checkpoint. 0.27 does. The kernel is pinned with
# --attention-backend because ROCm's own selector takes ROCM_ATTN and that is
# 15.0% slower at 32K -- one flag, no patch, and the check below would fail if
# the faster of the two were left undrawn.
series.append(ledger_series("Qwen3.8-27B", 2, "2026-08-29", cfg="Q38-triton-tp2"))
series.append(ledger_series("gemma-4-31B-it", 2, "2026-08-29", cfg="G31-tp2"))

for s in series:
    s["machine"] = "rdna3"
    s["lit"] = s["model"] in LIT
    s["rungs_capped"] = None
    s["alt"] = None

# --- speculation, as its own layer -----------------------------------------
# One arm per model that has one, measured the same day as that model's line
# above and against it as a control. Off until the MTP button is pressed: it is
# a different way of running the same machine, not a faster reading of the same
# thing, and on Qwen3.8 it is a net loss past 8K.
MTP = [("gemma-4-31B-it", "G31-mtp-p45450-tp2"),
       ("Qwen3.8-27B", "Q38-mtp-triton-p45450-tp2")]
for model, cfg in MTP:
    m = ledger_series(model, 2, "2026-08-29", cfg=cfg)
    m["machine"] = "rdna3"
    m["lit"] = False
    m["rungs_capped"] = None
    m["alt"] = None
    assert m["spec"], f"{cfg} is not a speculative arm"
    series.append(m)

# --- the backend ROCm picks for itself, as its own layer ---------------------
# Q38-tp2 is the only campaign-server decode series in this repository with a
# sibling on the same machine, the same day and the same stack that is not a
# later run superseding it: `Q38-triton-tp2` differs from it in one flag. The
# line above is the Triton one because it is faster at decode -- +0.2 % at 500
# rising to +15.0 % at 32 K -- and until 2026-08-30 that was the whole story.
#
# It is not. Both arms recorded prefill too, and prefill goes the other way and
# by more: 1 099 against 760 tok/s at 32 K, `ROCM_ATTN` 1.45x ahead, fitted
# quadratic terms 4.46 against 17.27. (Those are the 2026-09-02 re-measurement
# on a full-width link; the 2026-08-29 pair read 969 against 690 and 1.40x, on
# a link that biased both.) So the flag is a trade, and a front page
# that draws only the decode-faster arm tells a reader half of it. This is that
# other half, on a switch, off until asked for -- the same shape as the
# speculative arms: an alternative way of running the line it sits beside,
# measured against it as its own control, not a competitor for "fastest".
ALT = [("Qwen3.8-27B", "Q38-tp2", "2026-08-29")]
for model, cfg, date in ALT:
    a = ledger_series(model, 2, date, cfg=cfg)
    a["machine"] = "rdna3"
    a["lit"] = False
    a["rungs_capped"] = None
    a["alt"] = "backend"
    assert not a["spec"], f"{cfg} is a speculative arm, not a backend alternative"
    series.append(a)

# --- the other machine ------------------------------------------------------
# All twelve A100 configurations are the ladder the Radeon lines use -- eleven
# rungs, two rounds a cell -- so the campaign is drawable whole, not just the
# one model that happened to exist on both machines first. Five stock lines and
# the four speculative arms measured beside them. It used to be five single-run
# points from a validation log, speculative, with no control beside it, so the
# one cross-machine comparison on the front page was between a speculative A100
# and a stock Radeon. Now the default is stock against stock.
#
# `quant` and `arch` are read out of the ledger by model name rather than typed
# here. The two machines serve the same checkpoints -- the campaign's setup.log
# pulls google/gemma-4-31B-it-qat-w4a16-ct, cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit
# and cyankiwi/Qwen3.8-27B-AWQ-INT4, which are the paths bench_runner.py serves
# -- so there is one place in this repository that says what a checkpoint is,
# and it is not this file.
#
# `spec` records what the *engine* resolved, read off the serve logs in
# cuda-a100/campaign-2026-08-29/logs/, and not what the flag asked for: the two
# gemma arms request method "draft_model" and vLLM reports
# SpeculativeConfig(method='mtp', model=.../-assistant, num_spec_tokens=3),
# which is exactly what the ledger records for the Radeon gemma arm. Qwen3.8
# carries its head in its own weights and so has no drafter. Muse-Glimmer's arm
# is method 'dflash' at k=8 -- a block-diffusion drafter, not MTP -- and the
# switch on that model takes its name from this field rather than assuming, so
# it does not say MTP on the one model where MTP would be a lie.
#
# Which of each pair: the patched arm, throughout, stated rather than inherited.
# vllm#45450 nearly doubles decode at 32K on the two models vLLM routes onto the
# Triton kernel and does nothing at all on the one it does not -- Qwen3.8 is
# served by FLASH_ATTN here, the probe never prints, and its two arms agree to a
# mean of -0.08%, 20.51 against 20.52 at 32K. So on Qwen3.8 the patched arm and
# the unpatched one are the same measurement and either would draw the same
# line; on the two gemmas the patched arm is the one the article reports.
#
# `attn_backend` is only filled where a serve log survives to say so. The VM was
# reclaimed four times mid-campaign and the logs of nine configurations went
# with it; the results did not, because the harvester had them. An inference
# from the model family would be a good one -- vLLM forces TRITON_ATTN for
# Gemma4's heterogeneous head dimensions -- but it would be an inference, and
# this column is for what was read.
import statistics as _st, collections as _ct

_QA = {r["model"]: (r["quant"], r["arch"]) for r in led}
_A100_RAW = _ct.defaultdict(lambda: _ct.defaultdict(list))
for _line in open(B / "cuda-a100" / "campaign-2026-08-29" / "results.jsonl"):
    _r = json.loads(_line)
    if _r.get("kind") == "decode" and _r.get("decode_tps"):
        _A100_RAW[_r["cfg"]][_r["target"]].append(_r["decode_tps"])

P45450 = ["vllm#45450 3D admission"]
MTP3 = {"method": "mtp", "k": 3}
DRAFT3 = lambda d: {"method": "mtp", "drafter": d, "k": 3}
# cfg, model, spec descriptor, patches, backend read from a log, lit
A100 = [
    ("A100-G31",               "gemma-4-31B-it",   None, [], "TRITON_ATTN"),
    ("A100-G12",               "gemma-4-12B-it",   None, [], None),
    ("A100-G26A4B",            "gemma-4-26B-A4B",  None, [], None),
    ("A100-Q38",               "Qwen3.8-27B",      None, [], None),
    ("A100-MG30",              "Muse-Glimmer-30B", None, [], None),
    ("A100-G31-mtp-p45450",    "gemma-4-31B-it",
     DRAFT3("gemma-4-31B-it-assistant"), P45450, "TRITON_ATTN"),
    ("A100-G26A4B-mtp-p45450", "gemma-4-26B-A4B",
     DRAFT3("gemma-4-26B-A4B-it-assistant"), P45450, "TRITON_ATTN"),
    ("A100-Q38-mtp-p45450",    "Qwen3.8-27B",      MTP3, P45450, "FLASH_ATTN"),
    ("A100-MG30-dflash",       "Muse-Glimmer-30B",
     {"method": "dflash", "drafter": "Muse-Glimmer-30B-assistant", "k": 8}, [], None),
]
# A100-G31 is the one stock line whose backend the campaign README states from a
# log that no longer exists; it is kept because the patched arm's surviving log
# says TRITON_ATTN for the same model on the same stack and the forcing is
# architectural, printed as "Gemma4 model has heterogeneous head dimensions".

def _a100(cfg, model, spec_desc, patches, backend, lit):
    by = _A100_RAW[cfg]
    assert len(by) == 11, f"{cfg}: {len(by)} rungs"
    quant, arch = _QA[model]
    pts = []
    for ctx in sorted(by):
        v = by[ctx]
        m = _st.mean(v)
        rng = (max(v) - min(v)) / m * 100.0
        pts.append({"ctx": ctx, "tok_s": m, "runs": len(v),
                    "range_pct": rng, "graded": len(v) >= 2 and rng <= 8.0})
    return {"model": model, "machine": "a100", "tp": 1, "lit": lit,
            "vllm": "0.28.0", "patches": list(patches),
            "harness": "campaign-server", "date": "2026-08-29", "quant": quant,
            "arch": arch, "spec": bool(spec_desc), "spec_desc": spec_desc,
            "attn_backend": backend, "cfg": cfg, "rungs_capped": None, "alt": None,
            "source": "benchmarks/cuda-a100/campaign-2026-08-29/results.jsonl",
            "points": pts}

for cfg, model, spec_desc, patches, backend in A100:
    series.append(_a100(cfg, model, spec_desc, patches, backend,
                        not spec_desc and model in LIT))

# --- the single-card layer --------------------------------------------------
# The A100 lines above are already one card; what the chart had no way to show
# is the same question on the other two vendors, and on one of these Radeons
# rather than the pair. Three machines, one card each, measured 2026-08-30
# except where the ledger already had the answer.
#
# Two of these lines are short, and that is the finding rather than a gap. A
# card that cannot hold the KV for a rung cannot measure it:
#
#   Qwen3-8B on one 7900 XT     1.16 GiB of KV = 8 442 tokens, so 8 000 (which
#                               needs ~8 522 with the generation) is out
#   gemma-4-26B-A4B, same card  16.96 GiB resident of 19.98, leaving 0.93 GiB
#                               = 13 149 tokens at util 0.95, so 16 000 is out
#
# `rungs_capped` carries that so the figure can say why a line stops instead of
# leaving a reader to assume the run was abandoned.
CAPPED = {
    "B-8B-tp1": {"kv_gib": 1.16, "kv_tokens": 8442,
                 "why": "1.16 GiB of KV on a 19.98 GiB card holds 8 442 tokens"},
    "E26-tp1-u95": {"kv_gib": 0.93, "kv_tokens": 13149,
                    "why": "16.96 GiB of weights leaves 0.93 GiB of KV, 13 149 tokens"},
}

SINGLE = [("gemma-4-12B-it", "2026-08-24", "A-12B-tp1"),
          ("gemma-4-26B-A4B", "2026-08-30", "E26-tp1-u95"),
          ("Qwen3-8B", "2026-08-24", "B-8B-tp1")]
for model, date, cfg in SINGLE:
    x = ledger_series(model, 1, date, cfg=cfg)
    x["machine"] = "rdna3-1"
    x["lit"] = False
    x["rungs_capped"] = CAPPED.get(cfg)
    x["alt"] = None
    series.append(x)

# The L4 is this round's own file, the same runner as the A100 half, and the
# first CUDA rows in this repository measured with prefix caching off.
_L4_RAW = _ct.defaultdict(lambda: _ct.defaultdict(list))
_L4_META = {}
for _line in open(B / "cuda-l4" / "campaign-2026-08-30" / "results.jsonl"):
    _r = json.loads(_line)
    if _r.get("kind") == "decode" and _r.get("decode_tps"):
        _L4_RAW[_r["cfg"]][_r["target"]].append(_r["decode_tps"])
    elif _r.get("kind") == "model_meta":
        _L4_META[_r["cfg"]] = _r

L4 = [("G12", "gemma-4-12B-it"), ("G26A4B", "gemma-4-26B-A4B")]


def _campaign_series(raw, meta, cfg, model, machine, date, vllm, source, rungs=11):
    by = raw[cfg]
    assert len(by) == rungs, f"{cfg}: {len(by)} rungs, expected {rungs}"
    quant, arch = _QA[model]
    pts = []
    for ctx in sorted(by):
        v = by[ctx]
        m = _st.mean(v)
        rng = (max(v) - min(v)) / m * 100.0
        pts.append({"ctx": ctx, "tok_s": m, "runs": len(v),
                    "range_pct": rng, "graded": len(v) >= 2 and rng <= 8.0})
    md = meta.get(cfg, {})
    return {"model": model, "machine": machine, "tp": 1, "lit": False,
            "vllm": vllm, "patches": [], "harness": "campaign-server",
            "date": date, "quant": quant, "arch": arch, "spec": False,
            "spec_desc": None, "attn_backend": md.get("backend"), "cfg": cfg,
            "rungs_capped": None, "alt": None, "source": source, "points": pts}


for cfg, model in L4:
    series.append(_campaign_series(
        _L4_RAW, _L4_META, cfg, model, "l4", "2026-08-30", "0.28.0",
        "benchmarks/cuda-l4/campaign-2026-08-30/results.jsonl"))

# --- the 2026-08-30 four-machine round --------------------------------------
# These come out of `decode.jsonl` rather than out of a campaign's raw file,
# because that projection is the one this repository gates: `build_decode.py
# --check` recomputes it from the sources AND asserts it against the ledger
# wherever both cover a cell. Re-aggregating the raw rows here a third time
# would be a third answer to a question that already has one.
_DEC = [json.loads(l) for l in open(B / "decode.jsonl")]


def _decode_series(machine_name, cfg, date, mid, rungs, capped=None, lit=False):
    rows = sorted([r for r in _DEC if r["machine"] == machine_name
                   and r["cfg"] == cfg and r["date"] == date],
                  key=lambda r: r["ctx"])
    assert len(rows) == rungs, f"{cfg}: {len(rows)} rungs, expected {rungs}"
    r0 = rows[0]
    return {"model": r0["model"], "machine": mid, "tp": r0["tp"], "lit": lit,
            "vllm": r0["vllm"], "patches": list(r0["patches"]),
            "harness": r0["harness"], "date": date, "quant": r0["quant"],
            "arch": r0["arch"], "spec": r0["spec"] is not None,
            "spec_desc": r0["spec"], "attn_backend": r0["attn_backend"],
            "cfg": cfg, "rungs_capped": capped, "alt": None,
            "source": "benchmarks/decode.jsonl",
            "points": [{"ctx": r["ctx"], "tok_s": r["decode_tok_s"],
                        "runs": r["runs"], "range_pct": r["range_pct"],
                        "graded": r["chart_grade"]} for r in rows]}


# --- 2026-09-03, the rented sweep -------------------------------------------
# Six more machine configurations, every one of them off by default. Figures 1
# and 2 ask what happens between 500 and 32 000 tokens, and these ladders run
# four times further; the rungs past 32 000 are Figures 3 and 4's subject and
# are cut here rather than allowed to stretch this figure's axis. Nothing is
# re-aggregated: the points come out of decode.jsonl, the projection this
# repository gates.
RENTED = [("h100",     "H100-80GB-HBM3",            1),
          ("h200",     "H200-143GB-HBM3e",          1),
          ("b300",     "B300-SXM6",                 1),
          ("pro6000",  "RTX-PRO-6000-Blackwell",    1),
          ("h100x2",   "H100-80GB-HBM3-x2",         2),
          ("pro6000x2", "RTX-PRO-6000-Blackwell-x2", 2)]
FIG12_MAX_CTX = 32000


def _rented_decode(machine_name, mid, cfg):
    """one line, from decode.jsonl, cut to what Figures 1 and 2 are about"""
    rows = sorted([r for r in _DEC if r["machine"] == machine_name
                   and r["cfg"] == cfg and r["date"] == "2026-09-03"
                   and r["ctx"] <= FIG12_MAX_CTX], key=lambda r: r["ctx"])
    assert len(rows) == 11, f"{mid}/{cfg}: {len(rows)} rungs at or below 32 000"
    r0 = rows[0]
    return {"model": r0["model"], "machine": mid, "tp": r0["tp"], "lit": False,
            "vllm": r0["vllm"], "patches": list(r0["patches"]),
            "harness": r0["harness"], "date": r0["date"], "quant": r0["quant"],
            "arch": r0["arch"], "spec": False, "spec_desc": None,
            "attn_backend": r0["attn_backend"], "cfg": cfg,
            "rungs_capped": None, "alt": None,
            "source": "benchmarks/decode.jsonl",
            "points": [{"ctx": r["ctx"], "tok_s": r["decode_tok_s"],
                        "runs": r["runs"], "range_pct": r["range_pct"],
                        "graded": r["chart_grade"]} for r in rows]}


# The control arms are deliberately not drawn. `G31-mml33` and the two `Q38`
# arms exist to measure what `mml` and `max_num_seqs` are worth -- 0.10 % and
# 0.33 %, in cuda-h100/campaign-2026-09-03b -- so on a figure asking how fast a
# machine is they would be three near-identical lines for one model, and the
# thing they establish is a number in a README rather than a curve. Named, not
# pattern-matched, so a new control has to be looked at rather than swept in.
FIG12_CONTROLS = {"G31-mml33", "G12-mml33", "G26A4B-mml33",
                  "Q38-mml33", "Q38-mml33-mns16"}

_RENTED_CFGS = {}
for _mid, _mname, _cards in RENTED:
    _cfgs = sorted({r["cfg"] for r in _DEC if r["machine"] == _mname
                    and r["date"] == "2026-09-03"} - FIG12_CONTROLS)
    assert _cfgs, _mname
    assert not (set(_cfgs) & FIG12_CONTROLS), _mname
    _RENTED_CFGS[_mid] = _cfgs
    for _cfg in _cfgs:
        series.append(_rented_decode(_mname, _mid, _cfg))

# The fifth machine. It exists here at all because of vllm#39018: without it the
# engine does not start on sm75, dying at kernel load asking 98 304 bytes of
# shared memory against Turing's 65 536. **Decode is unaffected by that patch**
# -- it only changes `TILE_PREFILL` -- so this line answers Figure 1's question
# on the same terms as every other. Figure 2's is a different matter and is
# handled there.
series.append(_decode_series("T4", "G12", "2026-08-30", "t4", 11))

# Two more on the L4, from the same round.
#
# `B8` stops at 24 000 and that is the capacity retry rather than the card
# refusing: 33 000 tokens wanted 4.53 GiB of KV against 4.40 available, so the
# runner stepped `max_model_len` to 31 680 and the 32 000 rung plus its 512
# generated tokens no longer fits inside it.
#
# `G31-eager` stops at 1 000 and that IS the card. gemma-4-31B does not start on
# this L4 at all with CUDA graphs on -- `Available KV cache memory: -0.8 GiB` --
# and `--enforce-eager` turns that into +1.71 GiB, 2 020 tokens. It carries its
# own configuration id because eager is a different engine, not another arm of
# `G31`, and two rungs is what a 2 020-token pool reaches.
L4_ROUND = [
    ("B8", 10, {"kv_gib": 4.4, "kv_tokens": 32000,
                "why": "4.40 GiB of KV holds 32 000 tokens, 0.13 GiB short of "
                       "what 33 000 needed, so the runner stepped "
                       "max_model_len to 31 680 and the 32 000 rung plus its "
                       "512 generated tokens no longer fits"}),
    ("G31-eager", 2, {"kv_gib": 1.71, "kv_tokens": 2020,
                      "why": "18.7 GiB of weights leave a negative KV budget with "
                             "CUDA graphs on; --enforce-eager buys 2.51 GiB, "
                             "1.71 GiB of KV, 2 020 tokens"}),
]
for cfg, rungs, capped in L4_ROUND:
    series.append(_decode_series("L4", cfg, "2026-08-30", "l4", rungs, capped))

# Deliberately NOT drawn, and each for a reason that is about the figure's own
# grammar rather than about the data:
#
#   L4 `Q38S`   RedHatAI/Qwen3.8-27B, symmetric compressed-tensors at group 128.
#               It is a different CHECKPOINT of Qwen3.8-27B, not another arm of
#               the AWQ one this figure already draws, and on gfx1100 the two
#               differ by 1.27-3.24x on decode. The chart keys a label on the
#               model name and asserts one `quant` per model; drawing this would
#               put "INT4 AWQ" and "INT4 SYM CT" behind one legend entry. It has
#               no counterpart on any other machine either, so it would be a
#               line with nothing to be read against.
#
#   7900 XT `B8-tp1-u95`   Qwen3-8B on one card on the 0.27 image, stock. Decode
#               is the same as the 2026-08-24 line already drawn to 0.21 % at
#               every rung, and prefill is 1.24x better on b and 1.82x on c --
#               so it is the better measurement, and it still cannot replace
#               that line here. `tp_gain` below prices the second card by
#               comparing this machine's one-card line against its two-card
#               line, and those have to be the same stack: swapping in an 0.27
#               single-card arm against an 0.23.1 pair would turn a
#               1.23x/2.08x topology result into a 1.04x/1.03x stack result and
#               call it the second card. The arm is in both projections and has
#               its own README; it is not on this chart.
NOT_DRAWN = [
    {"machine": "l4", "cfg": "Q38S", "model": "Qwen3.8-27B",
     "why": "a different checkpoint of the same model, with no counterpart on "
            "any other machine"},
    {"machine": "rdna3-1", "cfg": "B8-tp1-u95", "model": "Qwen3-8B",
     "why": "decode is identical to the line already drawn, and swapping it in "
            "would make the second-card comparison a stack comparison"},
]

# --- the control the A100 lines now have ------------------------------------
# The same two models were measured again on the A100 on 2026-08-30 with prefix
# caching off, because that campaign's *prefill* was measured through a warm
# cache and cannot be used. Decode was never in question -- it is read from the
# stream after the first token -- and this is the measurement that says so
# rather than the assertion. It is not drawn: it would be a second stock line
# for a model that already has one, on the same machine, which is exactly what
# the "no faster measurement left undrawn" rule exists to prevent.
_A100_30 = _ct.defaultdict(lambda: _ct.defaultdict(list))
for _line in open(B / "cuda-a100" / "campaign-2026-08-30" / "results.jsonl"):
    _r = json.loads(_line)
    if _r.get("kind") == "decode" and _r.get("decode_tps"):
        _A100_30[_r["cfg"]][_r["target"]].append(_r["decode_tps"])
cache_control = []
for _c30, _c29 in (("G12", "A100-G12"), ("G26A4B", "A100-G26A4B")):
    a = {c: _st.mean(v) for c, v in _A100_RAW[_c29].items()}
    b_ = {c: _st.mean(v) for c, v in _A100_30[_c30].items()}
    shared_ctx = sorted(set(a) & set(b_))
    assert len(shared_ctx) == 11, (_c30, len(shared_ctx))
    d = [abs(b_[c] - a[c]) / a[c] * 100.0 for c in shared_ctx]
    cache_control.append({"cfg_on": _c29, "cfg_off": _c30, "rungs": len(shared_ctx),
                          "worst_pct": max(d), "median_pct": sorted(d)[len(d) // 2]})
assert max(x["worst_pct"] for x in cache_control) < 8.0, cache_control

# --- what a label says ------------------------------------------------------
# The chart names a model by the format its checkpoint is in as well as by its
# name, because "gemma-4-31B-it" does not tell a reader whether they are looking
# at a 4-bit model or a 16-bit one and the two are not the same claim. The
# string is the ledger's own `quant` with its first token upper-cased and the
# qualifier left alone -- "w4a16 QAT" -> "W4A16 QAT", "int4 AWQ" -> "INT4 AWQ"
# -- which is one rule rather than a table, and works because build_ledger.py
# writes that field to one grammar. It is not a strings-table key: it is a
# machine string and reads the same in both languages.
#
# The speculative switch is named for what the engine resolved rather than for
# the button it replaces. Three of the four arms are mtp; Muse-Glimmer's is a
# block-diffusion drafter at k=8, method dflash, and is a net loss at every
# depth -- so a switch labelled MTP would be wrong on the one model where it
# matters most to be right.
qlabel = lambda q: q.split(" ")[0].upper() + q[len(q.split(" ")[0]):]
SPEC_LABEL = {"mtp": "MTP", "dflash": "DFlash"}
for x in series:
    x["quant_label"] = qlabel(x["quant"])
    x["spec_label"] = SPEC_LABEL[x["spec_desc"]["method"]] if x["spec"] else None
    # An alternative arm is named for what it *is* -- the backend the engine
    # resolved -- for the same reason the speculative switch is named for the
    # method it ran rather than for the button it replaces.
    x["alt_label"] = x["attn_backend"] if x.get("alt") else None

# A label is per model, so every line a model owns has to agree about it --
# otherwise the legend would have to pick one and the chart would say something
# no row does. The checkpoints are the same on both machines, so this holds; the
# assertion is what would catch it if a later campaign served a different one.
labels = {}
for x in series:
    prev = labels.setdefault(x["model"], {"quant": x["quant"],
                                          "quant_label": x["quant_label"],
                                          "spec_label": None, "alt_label": None})
    assert prev["quant"] == x["quant"], \
        f'{x["model"]}: {prev["quant"]!r} on one line and {x["quant"]!r} on another'
    if x["spec"]:
        assert prev["spec_label"] in (None, x["spec_label"]), \
            f'{x["model"]}: two speculative methods, {prev["spec_label"]} and {x["spec_label"]}'
        prev["spec_label"] = x["spec_label"]
    if x.get("alt"):
        assert prev["alt_label"] in (None, x["alt_label"]), \
            f'{x["model"]}: two alternative arms, {prev["alt_label"]} and {x["alt_label"]}'
        prev["alt_label"] = x["alt_label"]

# --- how well this machine repeats a whole campaign -------------------------
# The same models were run twice, thirty days apart, on the same box. Their
# disagreement is this machine's campaign-to-campaign reproducibility, measured
# rather than assumed, and it is the slack the check below is entitled to.
PRIOR = "2026-07-25"
c1 = {(r["model"], r["tp"], r["ctx"]): r["decode_tok_s"] for r in led if r["date"] == PRIOR}
c2 = {(r["model"], r["tp"], r["ctx"]): r["decode_tok_s"] for r in led if r["date"] == CAMPAIGN}
shared = sorted(set(c1) & set(c2))
assert len(shared) >= 40, f"only {len(shared)} cells shared by the two campaigns"
diffs = sorted(abs(c1[k] - c2[k]) / max(c1[k], c2[k]) * 100 for k in shared)
REPRO = {"cells": len(shared), "worst_pct": diffs[-1], "median_pct": diffs[len(diffs) // 2],
         "campaigns": [PRIOR, CAMPAIGN]}

# the backbone is this campaign because it is the one that measured all of them
for m in BACKBONE:
    assert any(r["model"] == m and r["date"] == CAMPAIGN for r in led), m
assert not all(any(r["model"] == m and r["date"] == PRIOR for r in led) for m in BACKBONE), \
    "the earlier campaign also covers every backbone model; say why this one"

# --- the pick has to survive the rest of the repository ----------------------
# For every model on the Radeons, no other series in the ledger may beat the one
# chosen here at a depth they share, by more than this machine repeats itself. A
# faster run that exists and is not drawn would make "today's best" a lie.
# Like against like: a speculative row cannot beat a line drawn without
# speculation, and does not answer the same question. Each layer is checked
# against the rows of its own kind, so both the default view and the MTP one
# have to be the fastest of their sort.
beaten = []
for spec_layer in (False,):
    picked = {s["model"]: s for s in series
              if s["machine"] == "rdna3" and s["spec"] == spec_layer
              and not s.get("alt")}
    for model, s in picked.items():
        mine = {p["ctx"]: p["tok_s"] for p in s["points"]}
        for r in led:
            if r["model"] != model or r["tp"] != s["tp"]:
                continue
            if (r["spec"] is not None) != spec_layer:
                continue
            if sid(r) == (model, s["tp"], s["vllm"], tuple(s["patches"]),
                          s["harness"], s["date"]) and r["cfg"] == s.get("cfg"):
                continue
            if r["ctx"] not in mine:
                continue
            slack = max(r["range_pct"] or 0.0, REPRO["worst_pct"]) / 100.0 * r["decode_tok_s"]
            if r["decode_tok_s"] - mine[r["ctx"]] > slack:
                beaten.append((("MTP " if spec_layer else "") + model, r["ctx"],
                               round(r["decode_tok_s"], 2), round(mine[r["ctx"]], 2),
                               r["date"], r["cfg"]))
assert not beaten, "a faster measurement exists and is not drawn:\n  " + \
                   "\n  ".join(map(str, beaten))

# The speculative layer is held to a different rule, because "the fastest
# speculative measurement" is not what it answers. Neither Qwen3.8 arm dominates
# -- ROCM_ATTN is faster at 500 (91.43 against 75.10) and far slower at 32K
# (24.34 against 34.02) -- so picking by speed would draw an arm whose stock
# control is not on the chart. What a switch is for is what happens to *this
# line* when speculation goes on, so every speculative series has to be its own
# line's companion, on its own machine: same day, same kernel, same stack apart
# from the speculation. That promise now covers both machines, which is what
# stops the A100 arms being drawn beside a control that is not on the page.
#
# The kernel is compared only where both sides recorded one. Nine of the twelve
# A100 configurations lost their serve logs to the reclaims, so most of that
# machine's stock lines carry no backend to compare against -- an equality test
# there would be testing that two nulls match, and asserting it as though it
# were the kernel would be worse than saying nothing.
alt_pairs = []
for a in [x for x in series if x.get("alt")]:
    base = next(x for x in series if x["machine"] == a["machine"]
                and not x["spec"] and not x.get("alt") and x["model"] == a["model"])
    assert a["date"] == base["date"], (a["cfg"], a["date"], base["date"])
    assert a["vllm"] == base["vllm"], (a["cfg"], a["vllm"])
    assert a["attn_backend"] != base["attn_backend"], (a["cfg"], a["attn_backend"])
    alt_pairs.append({"model": a["model"], "machine": a["machine"], "kind": a["alt"],
                      "base_cfg": base["cfg"], "alt_cfg": a["cfg"],
                      "base_backend": base["attn_backend"],
                      "alt_backend": a["attn_backend"], "date": a["date"],
                      "delta_pct": [{"ctx": p["ctx"],
                                     "pct": (q["tok_s"] / p["tok_s"] - 1) * 100.0}
                                    for p, q in zip(base["points"], a["points"])]})

mtp_pairs = []
for m in [x for x in series if x["spec"]]:
    base = next(x for x in series if x["machine"] == m["machine"]
                and not x["spec"] and x["model"] == m["model"])
    assert m["date"] == base["date"], (m["cfg"], m["date"], base["date"])
    assert m["vllm"] == base["vllm"], (m["cfg"], m["vllm"])
    if m["attn_backend"] and base["attn_backend"]:
        assert m["attn_backend"] == base["attn_backend"], (m["cfg"], m["attn_backend"])
    mtp_pairs.append({"model": m["model"], "machine": m["machine"],
                      "label": m["spec_label"],
                      "base_cfg": base["cfg"], "mtp_cfg": m["cfg"],
                      "attn_backend": m["attn_backend"], "date": m["date"],
                      "spec": m["spec_desc"],
                      "delta_pct": [
                          {"ctx": a["ctx"],
                           "pct": (b["tok_s"] / a["tok_s"] - 1) * 100.0}
                          for a, b in zip(base["points"], m["points"])]})
for pr in mtp_pairs:
    pr["crosses_zero"] = (pr["delta_pct"][0]["pct"] > 0) != (pr["delta_pct"][-1]["pct"] > 0)
    pr["at_shortest_pct"] = pr["delta_pct"][0]["pct"]
    pr["at_deepest_pct"] = pr["delta_pct"][-1]["pct"]

# Each override has to earn its place against the campaign line it replaces --
# but "faster" is not the only way to earn it, and pretending otherwise is what
# put a speculative arm in this chart without a label. Two reasons count:
#
#   faster        Qwen3.8 on 0.27, which the 0.23 campaign had no native
#                 gfx1100 W4A16 kernel for
#   reproduces    gemma-4-31B, whose 2026-08-29 line lands within this
#                 machine's own campaign-to-campaign spread of the 2026-08-24
#                 one, and is drawn instead because it is the same-day control
#                 for the MTP arm the button reveals. A pair measured five days
#                 apart is not a pair.
over = []
for model in ("Qwen3.8-27B", "gemma-4-31B-it"):
    camp = ledger_series(model, 2, CAMPAIGN)
    c = {p["ctx"]: p["tok_s"] for p in camp["points"]}
    mine = {p["ctx"]: p["tok_s"] for p in picked[model]["points"]}
    near = [(x, min(c, key=lambda k: abs(k - x))) for x in mine]
    gains = [mine[x] / c[k] for x, k in near if abs(x - k) / x < 0.06]
    assert gains, f"{model}: override shares no depth with the campaign"
    faster = min(gains) > 1.0
    reproduces = (1 - min(gains)) * 100.0 <= REPRO["worst_pct"]
    assert faster or reproduces, (
        f"{model}: override neither beats the campaign nor reproduces it "
        f"(worst {(min(gains) - 1) * 100:.2f}% against a {REPRO['worst_pct']:.2f}% spread)")
    over.append({"model": model, "min": min(gains), "max": max(gains),
                 "why": "faster" if faster else "reproduces",
                 "campaign_deepest": c[max(c)], "picked_deepest": mine[max(mine)]})

for m in OMIT:
    assert any(r["model"] == m for r in led), f"{m} is not in the ledger to omit"
    assert not any(s["model"] == m for s in series), f"{m} was not omitted"

# --- where the axis is written ----------------------------------------------
# The depths that double: 500 and each doubling of it the ladder actually has.
# Both ends are labelled, which is the point -- the left edge used to carry no
# label at all, so an axis whose first mark was 1K read as though it started at
# zero, and the lines looked like they began somewhere they do not. And no tick
# is allowed outside the range: the list used to end at 50 000, which is past
# the deepest rung measured and drew itself beyond the right-hand edge of the
# frame. Derived rather than typed, so it cannot outlive the ladder again.
CTX_TICKS = [c for c in sorted({p["ctx"] for s in series for p in s["points"]})
             if c in {500 * 2 ** i for i in range(8)}]
assert CTX_TICKS[0] == min(p["ctx"] for s in series for p in s["points"]), CTX_TICKS
assert CTX_TICKS[-1] == max(p["ctx"] for s in series for p in s["points"]), CTX_TICKS

# --- Figure 2: what one card of each kind does to a prompt -------------------
# The decode figure answers "how fast does it generate"; this one answers the
# other half, and it is the half where the three cards do not agree about which
# is better. Two models that all three ran, one card each, prefill throughput
# against prompt length.
#
# Only chart-grade rungs are drawn. A prefill rung whose two rounds disagree is
# not a measurement of anything, and on the shallowest rung of most of these
# configurations they disagree a great deal: the first request to a freshly
# started engine absorbs the first CUDA-graph replay, the first allocation out
# of the KV pool and lazy JIT, and until 2026-08-30 the CUDA runner measured it
# instead of discarding it. Drawing the mean of a cell whose rounds are 2.06 s
# and 0.29 s would put a visibly wrong point on the curve; dropping it and
# saying so is the honest version, and `dropped` carries which and why.
#
# The fit is imported from build_prefill.py rather than repeated here, so the
# coefficients this figure states and the ones `prefill.jsonl --fits` reports
# cannot drift apart.
sys.path.insert(0, str(B / "analyze"))
import build_prefill as _bp

_PF = [json.loads(l) for l in open(B / "prefill.jsonl")]

# Machine id here is the figure's, and matches Figure 1's so a reader carries
# the same stroke and the same name between the two. The TP=2 lines are the
# same configurations Figure 1 draws, model for model and day for day, so the
# two figures describe the same runs rather than two picks of the same box.
#
# Everything in this repository that has a chart-grade prefill ladder is here.
# Prefill was recorded beside decode in every campaign since July and had never
# been drawn; there is no reason for eight models' worth of it to sit unread.
PF_LINES = [
    # one card
    ("a100",    "A100-SXM4-80GB", "gemma-4-12B-it",   "2026-08-30", "G12"),
    ("a100",    "A100-SXM4-80GB", "gemma-4-26B-A4B",  "2026-08-30", "G26A4B"),
    ("rdna3-1", "RX 7900 XT",     "gemma-4-12B-it",   "2026-08-24", "A-12B-tp1"),
    ("rdna3-1", "RX 7900 XT",     "gemma-4-26B-A4B",  "2026-08-30", "E26-tp1-u95"),
    ("rdna3-1", "RX 7900 XT",     "Qwen3-8B",         "2026-08-24", "B-8B-tp1"),
    ("l4",      "L4",             "gemma-4-12B-it",   "2026-08-30", "G12"),
    ("l4",      "L4",             "gemma-4-26B-A4B",  "2026-08-30", "G26A4B"),
    # 2026-08-30, second pass. The 2026-08-29 A100 campaign measured these three
    # through a warm prefix cache, so they had no usable prefill until they were
    # measured again; each now has a card-against-cards line where before it had
    # only the pair's.
    ("a100",    "A100-SXM4-80GB", "gemma-4-31B-it",   "2026-08-30", "G31"),
    ("a100",    "A100-SXM4-80GB", "Qwen3.8-27B",      "2026-08-30", "Q38"),
    ("a100",    "A100-SXM4-80GB", "Muse-Glimmer-30B", "2026-08-30", "MG30"),
    # the pair, one line per model, the configurations Figure 1 draws
    ("rdna3",   "RX 7900 XT",     "gemma-4-12B-it",   "2026-08-24", "A-12B-tp2"),
    ("rdna3",   "RX 7900 XT",     "gemma-4-26B-A4B",  "2026-08-24", "E-26B-tp2"),
    ("rdna3",   "RX 7900 XT",     "Qwen3-8B",         "2026-08-24", "B-8B-tp2"),
    ("rdna3",   "RX 7900 XT",     "Muse-Glimmer-30B", "2026-08-24", "G-30B-tp2"),
    # gemma-3-27b-it's 2026-08-24 line is in prefill.jsonl and not here: OMIT
    # above says why, and the OMIT check below holds this figure to it too.
    # 2026-09-03, the rented sweep: every line off by default, cut at 32 000.
    # Built below from prefill.jsonl rather than typed, because six machines
    # times six models is thirty-four lines and a hand-typed table of that
    # length is a table that drifts.
    # 2026-09-02, not 2026-08-29: those two sittings ran with one card trained
    # at PCIe 3.0 x8, and this figure's argument is the split between b and c,
    # which is exactly what a narrowed link moves. Re-measured on the restored
    # link the 31B's b is 722.6 against 868.7 and Qwen3.8's is 758.5 against
    # 846.2. Same container, same vLLM build, same serve arguments to the byte,
    # same #45450 state -- the configuration is what has to match, and it does;
    # the day does not, and Figure 1 still draws these two models' decode from
    # 2026-08-29, which the link moves by 1.0% at most.
    ("rdna3",   "RX 7900 XT",     "Qwen3.8-27B",      "2026-09-02", "Q38-triton-tp2-x16"),
    ("rdna3",   "RX 7900 XT",     "gemma-4-31B-it",   "2026-09-02", "G31-tp2-x16"),
    # 2026-08-30, the four-machine round.
    #
    # Qwen3-8B on the L4 is a third card for a model that had one and the pair.
    # Ten rungs: the capacity retry stepped `max_model_len` to 31 680.
    ("l4",      "L4",             "Qwen3-8B",         "2026-08-30", "B8"),
    # The T4, which is here on different terms from every other line and says so
    # in `caveat`. Its rows are the only ones in either projection measured with
    # a patch that changes an attention kernel's tile size: vllm#39018 halves
    # `TILE_PREFILL` on the head_size 512 layers, and this figure's argument is
    # the split between b and c. So its c is not this card against the others,
    # it is this card with a different kernel.
    ("t4",      "T4",             "gemma-4-12B-it",   "2026-08-30", "G12"),
]

# Lines whose coefficients must not be read as this figure reads the others.
# Carried on the series so the tooltip and the caption say it, rather than the
# figure drawing a line it quietly does not mean.
PF_CAVEAT = {
    ("t4", "G12"): {
        "kind": "patched-kernel",
        "patch": "vllm#39018",
        "what": "halves TILE_PREFILL on the head_size 512 layers, which is the "
                "quadratic term this figure decomposes -- without it the engine "
                "does not start on sm75 at all",
        # And the second reason, which is this line's own measurement rather
        # than the patch: b is not determined by this ladder. The 32 000 rung
        # was measured on a different VM from the ten below it, the two agree to
        # 4.61 % there, and swapping which one supplies it moves b by 29.9 % and
        # c by 12.8 %. The curve is quadratic-dominated in a way no other line
        # here is -- 224 s against 97 s at 32 K -- so the linear term absorbs it.
        "b_undetermined_pct": 29.9,
        "c_undetermined_pct": 12.8,
    },
}
# Lit to start: the same five models Figure 1 lights, so a reader who has
# learnt the five colours there reads them here. The other two are one click
# away in the key.
PF_LIT = set(LIT)

# Figures 1 and 2 stop at 32 000, so the coefficients they display are fitted
# on the rungs they draw. For every machine that measured no further the two
# fits are the same object of study, and that is asserted rather than assumed:
# if a pre-2026-09-03 line's fit moved when the ladder was cut, the cut is
# reaching rungs it should not.
_PF32 = [r for r in _PF if r["ctx"] <= FIG12_MAX_CTX]
_fits = {(f["machine"], f["cfg"], f["date"]): f for f in _bp.fits(_PF32)}
_fits_full = {(f["machine"], f["cfg"], f["date"]): f for f in _bp.fits(_PF)}
for _k, _f in _fits.items():
    if _k[2] == "2026-09-03":
        continue
    _g = _fits_full.get(_k)
    if _g and "b_us_tok" in _f and "b_us_tok" in _g:
        assert abs(_f["b_us_tok"] - _g["b_us_tok"]) < 1e-9, _k
for _mid, _mname, _cards in RENTED:
    for _cfg in _RENTED_CFGS[_mid]:
        _r0 = next(r for r in _PF if r["machine"] == _mname and r["cfg"] == _cfg
                   and r["date"] == "2026-09-03")
        PF_LINES.append((_mid, _mname, _r0["model"], "2026-09-03", _cfg))

pf_series = []
for mid, machine, model, date, cfg in PF_LINES:
    rows = sorted([r for r in _PF if r["machine"] == machine and r["cfg"] == cfg
                   and r["date"] == date and r["ctx"] <= FIG12_MAX_CTX],
                  key=lambda r: r["ctx"])
    assert rows, (machine, cfg, date)
    good = [r for r in rows if r["chart_grade"]]
    assert len(good) >= 4, (cfg, len(good))
    f = _fits[(machine, cfg, date)]
    assert "b_us_tok" in f, (cfg, f.get("note"))
    quant, arch = _QA[model]
    # how many rungs this arm actually measured, against how many this figure
    # draws -- B8 stops at 32 000 because its own config.json does, so its
    # ladder is not truncated here and must not say it is
    _n_measured = len([r for r in _PF if r["machine"] == machine
                       and r["cfg"] == cfg and r["date"] == date])
    pf_series.append({
        "machine": mid, "machine_name": machine, "model": model, "date": date,
        # A caveated line is never lit by default: it has to be asked for, so a
        # reader who has it on has seen the row that says why it is different.
        "lit": model in PF_LIT and (mid, cfg) not in PF_CAVEAT,
        "caveat": PF_CAVEAT.get((mid, cfg)),
        "cfg": cfg, "quant": quant, "quant_label": qlabel(quant), "arch": arch,
        "tp": rows[0]["tp"], "vllm": rows[0]["vllm"],
        "attn_backend": rows[0]["attn_backend"],
        "prefix_caching": rows[0]["prefix_caching"],
        "source": rows[0]["source"],
        # Fitted on the rungs this figure draws, which for a 2026-09-03 line is
        # eleven of the sixteen measured. `fit_scope` says so on the line
        # itself, because a coefficient fitted to 32 000 and one fitted to
        # 128 000 are different numbers for the same arm and the tooltip must
        # not present them as one.
        "fit": {"a_ms": f["a_ms"], "b_us_tok": f["b_us_tok"],
                "c_ns_tok2": f["c_ns_tok2"], "r2": f["r2"], "rungs": f["rungs"]},
        "fit_scope": ("%d rungs to 32 000 of the %d measured"
                      % (len(rows), _n_measured)
                      if _n_measured > len(rows) else "the whole ladder"),
        "dropped": [{"ctx": r["ctx"], "range_pct": r["range_pct"]}
                    for r in rows if not r["chart_grade"]],
        "points": [{"ctx": r["ctx"], "tokens": r["prompt_tokens"],
                    "tok_s": r["prefill_tok_s"], "ttft_s": r["ttft_s"],
                    "runs": r["runs"], "range_pct": r["range_pct"]}
                   for r in good],
    })

# The comparison the figure exists to make: against one 7900 XT, how much of
# each machine's advantage is the linear term and how much the quadratic. b is
# GEMM throughput -- the compute -- and c is how badly attention scales, and on
# these lines they do not move together. Card against card only; the pair is a
# different question and is answered below.
for m in OMIT:
    assert not any(x["model"] == m for x in pf_series), f"{m} was not omitted from Figure 2"

pf_cmp = []
for model in ("gemma-4-12B-it", "gemma-4-26B-A4B"):
    ref = next(x for x in pf_series if x["model"] == model and x["machine"] == "rdna3-1")
    for x in pf_series:
        if x["model"] != model or x["machine"] in ("rdna3-1", "rdna3"):
            continue
        # A card-against-card ratio is the point of this table, and a line whose
        # kernel was patched is not that card against the others. The T4 is
        # drawn and excluded here, which is the honest pair of things to do.
        if x.get("caveat"):
            continue
        pf_cmp.append({"model": model, "machine": x["machine"],
                       "b_ratio": ref["fit"]["b_us_tok"] / x["fit"]["b_us_tok"],
                       "c_ratio": ref["fit"]["c_ns_tok2"] / x["fit"]["c_ns_tok2"]})

# --- what the second card buys, which is not what section 4 said it did -----
# docs/benchmarks.md's section 4 priced the second card off the fitted
# intercept -- "+76 ms, 72 all-reduces at 1.05 ms each" -- and that was
# withdrawn on 2026-08-30 because `a` does not survive being measured twice.
# What does survive is b and c, and three models here have both topologies, so
# the claim can be made on the coefficients that reproduce instead of the one
# that does not. It is the same shape on all three: the second card buys about
# half again on compute and about twice on attention, because attention needs
# no communication and the GEMMs do.
tp_gain = []
for model in ("gemma-4-12B-it", "gemma-4-26B-A4B", "Qwen3-8B"):
    one = next(x for x in pf_series if x["model"] == model and x["machine"] == "rdna3-1")
    two = next(x for x in pf_series if x["model"] == model and x["machine"] == "rdna3")
    tp_gain.append({"model": model,
                    "one_cfg": one["cfg"], "two_cfg": two["cfg"],
                    "b_gain": one["fit"]["b_us_tok"] / two["fit"]["b_us_tok"],
                    "c_gain": one["fit"]["c_ns_tok2"] / two["fit"]["c_ns_tok2"]})
assert all(g["c_gain"] > g["b_gain"] for g in tp_gain), tp_gain

# --- the flag that is not free ----------------------------------------------
# Pinning Qwen3.8 to TRITON_ATTN is worth about 15 % of decode at 32 K against
# the backend ROCm picks for itself, and this repository called that "nothing
# but a flag" until 2026-08-30. Both arms recorded prefill too, on the same day,
# differing in that flag and nothing else -- and prefill goes the other way and
# by more. It is a trade, not a free win.
#
# The 2026-09-02 sitting, not 2026-08-29: the earlier one had one card at PCIe
# 3.0 x8, which biases prefill and is half of what this panel reports. Both arms
# were re-measured on the restored link on one day, so the "same day, one flag
# apart" property this panel rests on is intact. The trade barely moves --
# 1.40x -> 1.45x on prefill -- which is what "both arms on the same link"
# predicted, and the absolutes move 10-13 %.
_bt = {}
for cfg in ("Q38-tp2-x16", "Q38-triton-tp2-x16"):
    pre = {r["ctx"]: r for r in _PF if r["cfg"] == cfg and r["chart_grade"]}
    dec = {r["ctx"]: r for r in json.loads("[" + ",".join(
        open(B / "decode.jsonl").read().strip().splitlines()) + "]")
        if r["cfg"] == cfg and r["chart_grade"]}
    _bt[cfg] = (pre, dec)
_deep = max(set(_bt["Q38-tp2-x16"][0]) & set(_bt["Q38-triton-tp2-x16"][0]))
_deep_d = max(set(_bt["Q38-tp2-x16"][1]) & set(_bt["Q38-triton-tp2-x16"][1]))
backend_tradeoff = {
    "model": "Qwen3.8-27B", "date": "2026-09-02", "ctx": _deep, "ctx_decode": _deep_d,
    "rocm_cfg": "Q38-tp2-x16", "triton_cfg": "Q38-triton-tp2-x16",
    "prefill_rocm": _bt["Q38-tp2-x16"][0][_deep]["prefill_tok_s"],
    "prefill_triton": _bt["Q38-triton-tp2-x16"][0][_deep]["prefill_tok_s"],
    "decode_rocm": _bt["Q38-tp2-x16"][1][_deep_d]["decode_tok_s"],
    "decode_triton": _bt["Q38-triton-tp2-x16"][1][_deep_d]["decode_tok_s"],
    "c_rocm": next(f for f in _bp.fits(_PF)
                   if f["cfg"] == "Q38-tp2-x16")["c_ns_tok2"],
    "c_triton": next(f for f in _bp.fits(_PF)
                     if f["cfg"] == "Q38-triton-tp2-x16")["c_ns_tok2"],
}
backend_tradeoff["prefill_gain"] = (backend_tradeoff["prefill_rocm"]
                                    / backend_tradeoff["prefill_triton"])
backend_tradeoff["decode_gain"] = (backend_tradeoff["decode_triton"]
                                   / backend_tradeoff["decode_rocm"])
# the two go opposite ways, which is the whole point of recording it
assert backend_tradeoff["prefill_gain"] > 1 and backend_tradeoff["decode_gain"] > 1
# The L4 is behind one 7900 XT on the linear term and ahead on the quadratic --
# the one crossing in the set, and the reason a single prefill tok/s number
# cannot state what these cards do.
_l4_12 = next(c for c in pf_cmp if c["machine"] == "l4" and c["model"] == "gemma-4-12B-it")
assert _l4_12["b_ratio"] < 1.0 < _l4_12["c_ratio"], _l4_12
# Where a model's lines do not share a kernel, the difference between them is
# not only the card. gemma-4 goes to TRITON_ATTN on both machines, so those
# comparisons are clean; Qwen3.8 and Muse-Glimmer run FLASH_ATTN on the A100
# and TRITON_ATTN (or an unrecorded backend) on the Radeons, and a reader who
# reads their c ratio as hardware would be reading a kernel as well. The figure
# says which is which rather than leaving it to be noticed.
pf_backend_mixed = []
for model in sorted({x["model"] for x in pf_series}):
    got = [x for x in pf_series if x["model"] == model]
    known = {x["attn_backend"] for x in got if x["attn_backend"]}
    if len(got) > 1:
        unrec = sum(1 for x in got if not x["attn_backend"])
        # three states, not two. "No contradiction recorded" is not the same
        # claim as "known to be the same kernel", and only the first of those
        # is true where a serve log did not survive.
        kind = ("different" if len(known) > 1
                else "same" if unrec == 0 and len(known) == 1
                else "unknown")
        pf_backend_mixed.append({
            "model": model, "backends": sorted(known), "kernel": kind,
            "machines": sorted(x["machine"] for x in got), "unrecorded": unrec})

PF_TICKS = [c for c in sorted({p["ctx"] for x in pf_series for p in x["points"]})
            if c in {500 * 2 ** i for i in range(8)}]

_lit_models = [m for m in dict.fromkeys(x["model"] for x in series if x["lit"])]
# The models Figure 2 lights by default, not every model it draws: once it drew
# all seven, "a Figure 2 model" stopped distinguishing anything and the pair
# that has to be legible without a click fell back onto m1/m7, 17.8 apart.
_pf_models = sorted(PF_LIT)
_tail = [m for m in dict.fromkeys(x["model"] for x in series) if m not in _lit_models]
MODEL_ORDER = _lit_models + sorted(_tail, key=lambda m: (m not in _pf_models,
                                                         _tail.index(m)))
assert len(MODEL_ORDER) == len(set(MODEL_ORDER)) == len({x["model"] for x in series})

# --- Figures 3 and 4: past 32 000 ------------------------------------------
# Figures 1 and 2 stop at FIG12_MAX_CTX and say so. These two take the rungs
# above it, and they have a subject: the pair this repository is about, on
# the 2026-09-03 ladder that runs to 128 000 -- the first Radeon rows past
# 32 000 anywhere here. The rented machines from the same day are the
# background, off until asked for, exactly as they are in Figures 1 and 2.
# The whole ladder is drawn, 500 upward, because a curve's shape at depth is
# only readable against where it started; the cut is on the machine, not on
# the rung.
#
# Everything comes out of the two projections this repository gates, and a
# line has to reach past 32 000 to be here at all: a ladder that stops at
# 32 000 is Figure 1's and Figure 2's, not these.
LONG_DATE = "2026-09-03"
LONG_PAIR = ["A-12B-tp2-long", "B-8B-tp2-long", "E-26B-tp2-long",
             "G-30B-tp2-long", "D8-27B-tp2-long", "C-31B-tp2-long"]
# A configuration the campaign ran and this figure cannot draw has to say why,
# here, or the missing line reads as an abandoned run. Qwen3-8B's own config
# caps its positions at 40 960, so on this cut its ladder is the eleven rungs
# Figure 1 already has -- on every machine, which is the not_drawn rule below.
LONG_PAIR_ABSENT = {
    "B-8B-tp2-long": "its own config caps context at 40 960, so its ladder "
                     "stops at 32 000 and is Figure 1's",
}
# The L4 ran gemma-4-12B to 128 000 on the same day, as the platform control
# that lets every rented ratio be read as a card difference. It is a real
# sixteen-rung ladder on a card Figure 1 already names, so it is offered here
# too, off by default like every other background line.
LONG_EXTRA = [("l4", "L4", "G12")]
LONG_TICKS_ALL = [500 * 2 ** i for i in range(9)]   # 500 .. 128 000


def _long_series(rows_all, machine_name, mid, cfg, date, lit, kind):
    rows = sorted([r for r in rows_all if r["machine"] == machine_name
                   and r["cfg"] == cfg and r["date"] == date], key=lambda r: r["ctx"])
    assert rows, (machine_name, cfg, date)
    assert rows[-1]["ctx"] > FIG12_MAX_CTX, \
        f"{machine_name}/{cfg}: stops at {rows[-1]['ctx']}, which is Figure 1's subject"
    r0 = rows[0]
    quant, arch = _QA[r0["model"]]
    s = {"model": r0["model"], "machine": mid, "machine_name": machine_name,
         "tp": r0["tp"], "lit": lit, "vllm": r0["vllm"], "patches": list(r0["patches"]),
         "harness": r0["harness"], "date": date, "quant": quant,
         "quant_label": qlabel(quant), "arch": arch, "attn_backend": r0["attn_backend"],
         "prefix_caching": r0.get("prefix_caching"), "cfg": cfg,
         "source": "benchmarks/decode.jsonl" if kind == "decode" else "benchmarks/prefill.jsonl",
         "rungs_measured": len(rows)}
    if kind == "decode":
        s["points"] = [{"ctx": r["ctx"], "tok_s": r["decode_tok_s"], "runs": r["runs"],
                        "range_pct": r["range_pct"], "graded": r["chart_grade"]}
                       for r in rows]
        # what the line keeps from its shallowest rung to its deepest, on the
        # rungs that are measurements: a cell whose two rounds disagree is not
        # one, and the H200 has two such cells at the shallow end
        good = [r for r in rows if r["chart_grade"]]
        assert len(good) >= 4, (machine_name, cfg, len(good))
        s["retained"] = {"from_ctx": good[0]["ctx"], "to_ctx": good[-1]["ctx"],
                         "from_tok_s": good[0]["decode_tok_s"],
                         "to_tok_s": good[-1]["decode_tok_s"],
                         "change_pct": (good[-1]["decode_tok_s"] / good[0]["decode_tok_s"]
                                        - 1) * 100.0}
    else:
        good = [r for r in rows if r["chart_grade"]]
        assert len(good) >= 4, (machine_name, cfg, len(good))
        f = _fits_full[(machine_name, cfg, date)]
        assert "b_us_tok" in f, (cfg, f.get("note"))
        s["points"] = [{"ctx": r["ctx"], "tokens": r["prompt_tokens"],
                        "tok_s": r["prefill_tok_s"], "ttft_s": r["ttft_s"],
                        "runs": r["runs"], "range_pct": r["range_pct"]} for r in good]
        s["dropped"] = [{"ctx": r["ctx"], "range_pct": r["range_pct"]}
                        for r in rows if not r["chart_grade"]]
        s["fit"] = {"a_ms": f["a_ms"], "b_us_tok": f["b_us_tok"],
                    "c_ns_tok2": f["c_ns_tok2"], "r2": f["r2"], "rungs": f["rungs"]}
        # Fitted on every chart-grade rung of the ladder, which is what makes
        # these coefficients different objects from Figure 2's for the same
        # arm: that figure fits the rungs it draws, to 32 000.
        s["fit_scope"] = "the whole ladder"
        # Where the quadratic term overtakes the linear one -- b*S = c*S^2 at
        # S = b/c -- and what share of the deepest rung's time it is. Both are
        # arithmetic on the fit, so they are stated here rather than read off
        # a curve by eye.
        a, b, c = f["a_ms"] / 1e3, f["b_us_tok"] / 1e6, f["c_ns_tok2"] / 1e9
        S = good[-1]["prompt_tokens"]
        s["crossover_tokens"] = b / c if c > 0 else None
        s["quadratic_share_deepest"] = {
            "ctx": good[-1]["ctx"], "tokens": S,
            "pct": c * S * S / (a + b * S + c * S * S) * 100.0} if c > 0 else None
    return s


def _long_figure(rows_all, kind):
    CFG_MODEL = {r["cfg"]: r["model"] for r in rows_all}
    out, cfgs_on_pair = [], sorted({r["cfg"] for r in rows_all
                                    if r["machine"] == "RX 7900 XT"
                                    and r["date"] == LONG_DATE
                                    and r["cfg"].endswith("-long")})
    assert set(cfgs_on_pair) <= set(LONG_PAIR), cfgs_on_pair
    # every configuration the campaign named is either drawn or explained;
    # a line that is neither is the assertion, not a quiet gap. Two ways to be
    # absent: no rows at all, or rows that never pass 32 000 -- B8's ladder on
    # this cut is eleven rungs on the pair as on every rented card, because
    # its config.json caps context at 40 960. Both have to be in
    # LONG_PAIR_ABSENT, and the second is also listed with the not-drawn
    # ladders below, so the "left out on every machine" rule sees it.
    deepest_on_pair = {c: max(r["ctx"] for r in rows_all if r["machine"] == "RX 7900 XT"
                              and r["date"] == LONG_DATE and r["cfg"] == c)
                       for c in cfgs_on_pair}
    missing = [c for c in LONG_PAIR if c not in cfgs_on_pair]
    shallow = [c for c in cfgs_on_pair if deepest_on_pair[c] <= FIG12_MAX_CTX]
    assert set(missing) | set(shallow) == set(LONG_PAIR_ABSENT), \
        f"the pair's long campaign: {missing} absent, {shallow} never past 32 000, {sorted(LONG_PAIR_ABSENT)} explained"
    not_drawn = []
    # the pair first, in the order the campaign ran them
    for cfg in LONG_PAIR:
        if cfg in LONG_PAIR_ABSENT:
            if cfg in shallow:
                not_drawn.append({"machine": "rdna3", "cfg": cfg, "model": CFG_MODEL[cfg],
                                  "deepest": deepest_on_pair[cfg]})
            continue
        out.append(_long_series(rows_all, "RX 7900 XT", "rdna3", cfg, LONG_DATE,
                                CFG_MODEL[cfg] in LIT, kind))
    # the rented machines, from the same day, every one off by default. B8 is
    # not here on any of them and that is arithmetic: its config.json caps
    # context at 40 960, so its ladder is Figure 1's eleven rungs everywhere.
    for mid, mname, _cards in RENTED:
        for cfg in _RENTED_CFGS[mid]:
            deepest = max(r["ctx"] for r in rows_all if r["machine"] == mname
                          and r["cfg"] == cfg and r["date"] == LONG_DATE)
            if deepest <= FIG12_MAX_CTX:
                not_drawn.append({"machine": mid, "cfg": cfg,
                                  "model": next(r["model"] for r in rows_all
                                                if r["machine"] == mname and r["cfg"] == cfg),
                                  "deepest": deepest})
                continue
            out.append(_long_series(rows_all, mname, mid, cfg, LONG_DATE, False, kind))
    for mid, mname, cfg in LONG_EXTRA:
        out.append(_long_series(rows_all, mname, mid, cfg, LONG_DATE, False, kind))
    return out, not_drawn


long_dec, long_dec_nd = _long_figure(_DEC, "decode")
long_pre, long_pre_nd = _long_figure(_PF, "prefill")
# the same machines and models on both, or a reader carrying a colour or a
# stroke between the two figures is carrying it to the wrong line
assert [(x["machine"], x["cfg"]) for x in long_dec] == [(x["machine"], x["cfg"]) for x in long_pre]
assert long_dec_nd == long_pre_nd, (long_dec_nd, long_pre_nd)
# only Qwen3-8B is left out, and only for its own ceiling
assert {n["model"] for n in long_dec_nd} == {"Qwen3-8B"}, long_dec_nd
assert all(n["deepest"] == FIG12_MAX_CTX for n in long_dec_nd), long_dec_nd
# a model's label is the same on every line, as the check above insists for
# Figure 1, and every model here is already one Figure 1 draws
for x in long_dec + long_pre:
    assert x["model"] in labels and labels[x["model"]]["quant"] == x["quant"], x["model"]


def _long_ticks(ser):
    """the ladder's own doubling depths, by the rung's nominal length -- a
    prefill point is drawn at its measured token count, which is never exactly
    the rung, so the ticks come from `ctx` on both figures as Figure 2's do"""
    cs = sorted({p["ctx"] for x in ser for p in x["points"]})
    ticks = [c for c in cs if c in LONG_TICKS_ALL]
    assert ticks[0] == cs[0] and ticks[-1] == cs[-1], (ticks, cs[0], cs[-1])
    assert ticks[-1] == 128000, ticks
    return ticks


# The card behind a machine id, whatever the count of them. In the one-model
# view colour is the family and the dash is the count, so two H100s and one
# read as kin; the Radeons are the subject and take the page's text colour.
FAMILY = {"rdna3": "radeon", "rdna3-1": "radeon", "a100": "a100", "l4": "l4", "t4": "t4",
          "h100": "h100", "h100x2": "h100", "h200": "h200", "b300": "b300",
          "pro6000": "pro6000", "pro6000x2": "pro6000"}
# the order the seven NVIDIA families take the seven palette slots in; the
# Radeons are outside the palette on purpose
FAMILY_ORDER = ["h100", "h200", "b300", "pro6000", "a100", "l4", "t4"]
LONG_MACHINES = ([{"id": "rdna3", "default": True, "cards": 2, "family": "radeon"}]
                 + [{"id": m, "default": False, "cards": c, "family": FAMILY[m]}
                    for m, _n, c in RENTED]
                 + [{"id": "l4", "default": False, "cards": 1, "family": "l4"}])
assert [m["id"] for m in LONG_MACHINES] == list(dict.fromkeys(
    [x["machine"] for x in long_dec])), "the machine row is not the lines' order"
LONG_OUT = {
    "decode": {"series": long_dec, "ticks": _long_ticks(long_dec),
               "ctx_min": min(p["ctx"] for x in long_dec for p in x["points"]),
               "ctx_max": max(p["ctx"] for x in long_dec for p in x["points"]),
               "not_drawn": long_dec_nd},
    # the prefill axis spans the measured token counts, and the 500 tick has
    # to stay on it whether or not any line's shallowest prompt fell short of
    # 500 -- an axis whose first label is 1K reads as starting at zero
    "prefill": {"series": long_pre, "ticks": _long_ticks(long_pre),
                "ctx_min": min([500] + [p["tokens"] for x in long_pre for p in x["points"]]),
                "ctx_max": max(p["tokens"] for x in long_pre for p in x["points"]),
                "not_drawn": long_pre_nd},
    "machines": LONG_MACHINES,
    "family_order": FAMILY_ORDER,
    "pair_cfgs": [c for c in LONG_PAIR if c not in LONG_PAIR_ABSENT],
    "pair_absent": [{"cfg": c, "why": w} for c, w in LONG_PAIR_ABSENT.items()],
    "date": LONG_DATE,
    "fig12_max_ctx": FIG12_MAX_CTX,
}

out = {
    "_what": "The index's best-measured-today figure. One line per model per machine, "
             "each the fastest configuration that model has been measured in; five of "
             "the Radeon lines share one campaign and two do not, and the A100 side is "
             "the whole 2026-08-29 campaign rather than one model of it. Speculation "
             "is a switch on each model that has an arm, named for the method the "
             "engine resolved. Derived by site/src/genfig-index.py from "
             "benchmarks/ledger.jsonl, benchmarks/speculative-decoding/ and "
             "benchmarks/cuda-a100/.",
    "prefill": {
        "series": pf_series,
        "compare": pf_cmp,
        "tp_gain": tp_gain,
        "backend_tradeoff": backend_tradeoff,
        "backend_mixed": pf_backend_mixed,
        "ticks": PF_TICKS,
        "ctx_min": min(p["tokens"] for x in pf_series for p in x["points"]),
        "ctx_max": max(p["tokens"] for x in pf_series for p in x["points"]),
    },
    "long": LONG_OUT,
    "best": {
        "series": series,
        "campaign": {"date": CAMPAIGN, "models": len(BACKBONE),
                     "vllm": series[0]["vllm"], "patches": series[0]["patches"]},
        "repro": REPRO,
        "overrides": over,
        "labels": labels,
        "mtp_pairs": mtp_pairs,
        "alt_pairs": alt_pairs,
        "omitted": OMIT,
        "omitted_why": OMIT_WHY,
        # Measured, in both projections, and deliberately absent from this
        # chart. `omitted` is a model this repository has nothing current to say
        # about; this is a series it has plenty to say about that this figure's
        # own grammar cannot hold.
        "not_drawn": NOT_DRAWN,
        # Order is the order the row is drawn in, and the two-card Radeon is
        # first because it is what this repository is about. The three
        # single-card machines are off by default: the figure's first question
        # is still "what does this box do", and a reader who wants the
        # cross-vendor single-card comparison presses for it -- or reads
        # Figure 2, which asks only that.
        "machines": [{"id": "rdna3", "default": True, "cards": 2, "family": "radeon"},
                     {"id": "a100", "default": False, "cards": 1, "family": "a100"},
                     {"id": "rdna3-1", "default": False, "cards": 1, "family": "radeon"},
                     {"id": "l4", "default": False, "cards": 1, "family": "l4"},
                     {"id": "t4", "default": False, "cards": 1, "family": "t4"}]
                    + [{"id": m, "default": False, "cards": c, "family": FAMILY[m]}
                       for m, _n, c in RENTED],
        "family_order": FAMILY_ORDER,
        "cache_control": cache_control,
        "ctx_min": min(p["ctx"] for s in series for p in s["points"]),
        "ctx_max": max(p["ctx"] for s in series for p in s["points"]),
        # The colour order, decided here rather than in the page, because
        # Figure 2 draws two of the same models and a reader carries the colour
        # between the two figures. Lit models first -- they take the four
        # furthest apart in both themes -- then the rest.
        #
        # Within the tail, the models Figure 2 *lights by default* come first,
        # and that is not cosmetic. colour[m] is var(--m{i%7+1}) and the palette's closest pair
        # by CIE76 is m1 against m7 at dE 17.8, where every other pair is 40 or
        # more. Figure 2 has exactly two models on one chart, one of them lit
        # here and so on m1; leaving the other at the end of the tail put it on
        # m7 and drew both of that figure's lit models in the same blue. Pulling
        # it forward costs Figure 1 nothing -- it swaps two models that are both
        # off in its default view -- and gives Figure 2 a pair 78.7 apart.
        "model_order": MODEL_ORDER,
        "ctx_ticks": CTX_TICKS,
        "fastest": max(p["tok_s"] for s in series for p in s["points"]),
    },
}
json.dump(out, open(pathlib.Path(__file__).parent / "figures-index.json", "w"),
          ensure_ascii=False, indent=1)

print(f"{len(series)} series, {sum(len(s['points']) for s in series)} points")
for s in series:
    print(f'  {s["machine"]:6s} {s["model"]:18s} {s["quant_label"]:10s} '
          f'{"lit " if s["lit"] else "    "}'
          f'{len(s["points"]):2d} pts  {s["points"][0]["tok_s"]:6.1f} -> '
          f'{s["points"][-1]["tok_s"]:6.1f}  '
          f'{(s["spec_label"] + " ") if s["spec"] else ""}'
          f'{"+".join(s["patches"]) or "stock"}')
print("overrides:", [(o["model"], round(o["min"], 2), round(o["max"], 2)) for o in over])
print(f'the two campaigns agree on {REPRO["cells"]} cells to '
      f'{REPRO["worst_pct"]:.2f}% at worst, {REPRO["median_pct"]:.2f}% median')
print("no faster measurement is left undrawn")

# --- one small figure per article card -------------------------------------
# Each card's numbers are read out of that article's own figures-*.json, so a
# card and the page it links to cannot disagree about what the article found --
# the same rule the one-line summary already follows. Nothing here is typed.
#
# Where the article's finding IS a comparison, the card draws both sides rather
# than the ratio between them: a lone ratio line tells a reader the shape and
# not the thing. A name beginning "@" is a strings-table key, because it is
# prose; a bare name is a machine string and reads the same in both languages.
D = pathlib.Path(__file__).parent
fig = lambda n: json.load(open(D / n))

A_HYB, A_A100, A_SPEC = fig("figures.json"), fig("figures-a100.json"), fig("figures-spec.json")
A_W4, A_MEAS, A_MOE = fig("figures-w4a16.json"), fig("figures-measure.json"), fig("figures-moe.json")
A_MODAL = fig("figures-modal.json")
A_LOAD, A_RCCL, A_RD = fig("figures-loader.json"), fig("figures-rccl.json"), fig("figures-rdna3.json")
A_GQA, A_65 = fig("figures-gqa.json"), fig("figures-6565.json")

_hyb = [s for s in A_HYB["fig1"]["series"] if s["arch"] == "hybrid SSM"][0]
_dense = [s for s in A_HYB["fig1"]["series"]
          if s["arch"] == "dense" and len(s["points"]) == len(_hyb["points"])][0]
_gqa023 = A_GQA["fig1"]["versions"][0]
_gqaex = [r for r in _gqa023["rows"] if not r["admitted"]]
_meas = A_MEAS["fig2"]["rows"][0]

cards = {
 # the rented sweep: the most memory-bound model's gain in each of the five
 # settings, the article's Figure 1 in one bar each; a bar below parity is
 # the slower card (the PRO 6000), drawn as such
 "mem-busy-orders-five-settings": {
   "form": "bars", "unit": "cRatioX", "rule": 1.0, "ruleT": "@cParity",
   # the card's label column is narrow; the article's own long names truncate
   # there, so each setting gets its short form, keyed by the setting id
   "bars": [{"label": {"radeon2": "second 7900 XT", "h200": "H200 / H100", "b300": "B300 / H100",
                       "pro6000": "PRO 6000 / H100", "h100x2": "second H100"}.get(st["id"], st["what"]),
             "v": st["models"][0]["ratio"],
             "kind": "bad" if st["models"][0]["ratio"] < 1 else None}
            for st in A_MODAL["fig1"]["settings"]],
   "src": "figures-modal.json fig1"},
 # what this article compares is the SLOPE -- "fourteen to forty times steeper
 # than any dense model" -- so the card draws each model against its own rate at
 # the shortest depth. On absolute tok/s the hybrid's 12.1 sits so far under the
 # dense model's 79.6 that its whole collapse reads as a flat line near zero.
 "hybrid-ssm-collapse": {
   "form": "line", "unit": "cRetained", "xlog": True, "y0": 0, "xctx": True,
   "series": [{"name": _hyb["model"], "kind": "bad",
               "pts": [[p["ctx"], p["tok_s"] / _hyb["points"][0]["tok_s"] * 100.0]
                       for p in _hyb["points"]]},
              {"name": _dense["model"],
               "pts": [[p["ctx"], p["tok_s"] / _dense["points"][0]["tok_s"] * 100.0]
                       for p in _dense["points"]]}],
   "src": "figures.json fig1"},
 "a100-vs-two-radeons": {
   "form": "line", "unit": "cTokS", "xlog": True, "y0": 0, "xctx": True,
   "series": [{"name": "@cRadeons",
               "pts": [[r["ctx"], r["radeons"]] for r in A_A100["fig1"]["rows"]]},
              {"name": "@cA100", "alt": True,
               "pts": [[r["a100_ctx"], r["a100"]] for r in A_A100["fig1"]["rows"]]}],
   "src": "figures-a100.json fig1"},
 "speculative-decoding-net-loss": {
   "form": "line", "unit": "cTokS", "xlog": True, "y0": 0, "xctx": True,
   "series": [{"name": "@cNoSpec",
               "pts": [[r["ctx"], r["nospec"]] for r in A_SPEC["fig1"]["rows"]]},
              {"name": "MTP", "kind": "bad",
               "pts": [[r["ctx"], r["mtp"]] for r in A_SPEC["fig1"]["rows"]]}],
   "src": "figures-spec.json fig1"},
 "w4a16-two-problems": {
   "form": "line", "unit": "cMsStep", "xlog": True, "y0": 0, "xctx": True,
   "series": [{"name": "@cAsym", "kind": "bad",
               "pts": [[c["ctx"], c["asym_ms"]] for c in A_W4["fig1"]["cells"]]},
              {"name": "@cSym",
               "pts": [[c["ctx"], c["sym_ms"]] for c in A_W4["fig1"]["cells"]]}],
   "src": "figures-w4a16.json fig1"},
 "measuring-decode": {
   "form": "line", "unit": "cRun", "y0": 0, "xrun": True,
   "rule": _meas["converged"], "ruleT": "@cConverged",
   "series": [{"name": "@cFourRuns",
               "pts": [[i + 1, v] for i, v in enumerate(_meas["runs"])]}],
   "src": "figures-measure.json fig2"},
 "gqa-gate-costs-nothing": {
   "form": "line", "unit": "cRatio", "xlog": True, "rule": 1.0, "ruleT": "@cParity",
   "xctx": True,
   "series": [{"name": r["shape"], "pts": [[c["ctx"], c["ratio"]] for c in r["cells"]]}
              for r in _gqaex],
   "src": "figures-gqa.json fig1"},
 "moe-written-off-by-eager": {
   "form": "bars", "unit": "cTokS",
   "bars": ([{"label": b["model"], "v": b["tok_s"]} for b in A_MOE["fig1"]["bars"]]
            + [{"label": A_MOE["fig1"]["bars"][0]["model"], "note": "@cEager",
                "v": A_MOE["fig1"]["eager"]["tok_s"], "kind": "bad"}]),
   "src": "figures-moe.json fig1"},
 "weight-loading-19x": {
   "form": "bars", "unit": "cMsLog", "log": True,
   # the article calls these kernels -28 and -30; the label is derived from the
   # kernel string rather than typed, and says which of the two -28s this is
   "bars": [{"label": re.search(r"-\d+", s["kernel"]).group(0)
                      + (" stock" if s["shipped"] else " +342981f"),
             "v": [c for c in s["cases"] if c["key"] == "rw_p_resident"][0]["ms"],
             "kind": "bad" if i == 0 else None}
            for i, s in enumerate(A_LOAD["fig1"]["states"])],
   "src": "figures-loader.json fig1"},
 "rdna3-second-class": {
   "form": "bars", "unit": "cFindings",
   "bars": [{"label": "@cRdna3", "v": A_RD["fig1"]["counts"]["rdna3"], "kind": "bad"},
            {"label": "@cNotRdna3",
             "v": A_RD["fig1"]["total"] - A_RD["fig1"]["counts"]["rdna3"]}],
   "src": "figures-rdna3.json fig1"},
 "reporting-a-non-reproduction": {
   "form": "bars", "unit": "cInits",
   "bars": [{"label": a["arm"], "v": a["n"]} for a in A_65["fig1"]["arms"]],
   "src": "figures-6565.json fig1"},
 "rccl-atomics-hostcall": {
   "form": "status", "unit": "cHostcall",
   "rows": [{"label": s["rccl"], "ok": s["behaviour"] == "works",
             "note": "0" if s["hostcall"] == "0" else "N"}
            for s in A_RCCL["shipped"]],
   "src": "figures-rccl.json shipped"},
}
out["cards"] = cards
json.dump(out, open(D / "figures-index.json", "w"), ensure_ascii=False, indent=1)
print(f"cards: {len(cards)} "
      f"({sum(1 for c in cards.values() if c['form'] == 'line')} line, "
      f"{sum(1 for c in cards.values() if c['form'] == 'bars')} bars, "
      f"{sum(1 for c in cards.values() if c['form'] == 'status')} status)")
