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
  * `attn_backend` is resolved per **(source, cfg)**, in this order: the row's
    own `model_meta.backend` (which the CUDA runners extract from the serve log
    in-process), then the serve log itself, then the `ARMS` / `ARMS_CUDA`
    tables. Until 2026-08-30 it came from the tables alone -- `backend_from_log`
    had no callers anywhere in the repository, while this docstring claimed the
    log was read -- and the tables are keyed on a cfg id that is **not unique
    across machines**: `G12` and `G26A4B` each already name a row on two
    machines, and this round adds a third.

    vLLM writes the line four ways, and a regex for one silently misses the
    others:

        Using AttentionBackendEnum.TRITON_ATTN backend.
        Using FLASH_ATTN attention backend out of potential backends: [...]
        Using TRITON_ATTN backend (selected via --attention-backend).
        Found incompatible backend(s) [TURBOQUANT] with AttentionType.DECODER.
          Overriding with ROCM_ATTN out of potential backends: [...]

    The fourth is the ROCm override, and it is the **only** backend line in
    `campaign-0830d/serve-logs/B8-tp1-u95.log`. The vision-tower lines
    (`for vit attention`, `MMEncoderAttention`) are a fifth form and must not
    be mistaken for the decoder's. A log carries the drafter's selection as
    well as the target's on a speculative arm, so the FIRST decoder line is the
    target's and is the one taken.
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
    # 2026-08-30. The CUDA runner names its arms by model alone; `machine` is
    # what separates the L4's G12 from the A100's, so the id does not repeat it.
    "G12":                     ("gemma-4-12B-it",  "w4a16 QAT", "dense", 1),
    "G26A4B":                  ("gemma-4-26B-A4B", "int4 AWQ",  "MoE, 128 experts", 1),
    # 2026-08-30, second pass: the three whose prefill the 2026-08-29 campaign
    # measured through a warm prefix cache and therefore never measured.
    "G31":                     ("gemma-4-31B-it",  "w4a16 QAT", "dense", 1),
    "Q38":                     ("Qwen3.8-27B",     "int4 AWQ",  "hybrid SSM", 1),
    "MG30":                    ("Muse-Glimmer-30B", "int4",     "sliding window 2048", 1),
    # 2026-08-30, the four-machine round.
    "B8":                      ("Qwen3-8B",        "bf16",      "dense", 1),
    # **A different checkpoint, not a different arm of `Q38`.** `Q38` is
    # cyankiwi's AWQ; this is RedHatAI/Qwen3.8-27B, compressed-tensors with
    # symmetric int4 at group 128. On gfx1100 the two land on different kernels
    # and differ by 1.27-3.24x on decode (`w4a16-symmetry/w4a16-ab.jsonl`), so
    # they must not share a table row anywhere.
    "Q38S":                    ("Qwen3.8-27B",     "int4 sym CT", "hybrid SSM", 1),
    # `--enforce-eager`, which is a different engine and therefore its own
    # configuration: gemma-4-31B does not start on a 23 GiB L4 without it. Four
    # capacity retries from mml 33000 down to 2062 all reported no room for KV
    # at all; eager at the same utilisation gave a 2020-token pool on the first
    # try, because what was consuming the budget was the CUDA graphs.
    "G31-eager":               ("gemma-4-31B-it",  "w4a16 QAT", "dense", 1),
    # Same fallback, same reason, if Qwen3.8 needs it too. Present so a run that
    # takes the eager branch does not stop the projection with a KeyError:
    # `meta_for` raises on an unknown id by design, and the id a run produces is
    # not known until the run has produced it.
    "Q38-eager":               ("Qwen3.8-27B",     "int4 AWQ",  "hybrid SSM", 1),
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
    "G12":                    (None,    "TRITON_ATTN"),
    "G26A4B":                 (None,    "TRITON_ATTN"),
    # read from this run's own model_meta, not inferred: the 2026-08-29 campaign
    # recorded no backend at all, and Muse-Glimmer's was null there because its
    # serve log went with a reclaimed VM.
    "G31":                    (None,    "TRITON_ATTN"),
    "Q38":                    (None,    "FLASH_ATTN"),
    "MG30":                   (None,    "FLASH_ATTN"),
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
    # 2026-08-30. The spine's fourth machine, and the first CUDA rows in this
    # repository measured with prefix caching off. Both configurations are 11
    # rungs x 2 rounds, 22 measurements, 0 errors. `driver` is from nvidia-smi
    # on the VM; no log records a torch or CUDA version, so those stay null.
    dict(file="cuda-l4/campaign-2026-08-30/results.jsonl", machine="L4",
         date="2026-08-30", vllm="0.28.0", rocm=None, cuda=None,
         driver="580.82.07", kernel=None, patches=[], prefix_caching=False),
    # 2026-08-30. The A100 measured again, with prefix caching off, because the
    # 2026-08-29 rows below are not prefill measurements: 32 K there is 2.932 s
    # and 0.201 s for the same cell; here it is 8.3826 s and 8.3796 s, a spread
    # of 0.04 %. Round 1 of the old data was wrong too, by 2.9x -- every rung
    # is a strict prefix of the next, so ascending the ladder was itself a
    # sequence of cache hits.
    #
    # No serve log: the VM was reclaimed after the run finished and the logs
    # went with it. What survives is what `harvester.py` had already pulled,
    # which is the rows and their `model_meta` -- backend, quant kernel, KV
    # size, and the `run_meta` row's versions. The logs themselves are gone.
    dict(file="cuda-a100/campaign-2026-08-30/results.jsonl",
         machine="A100-SXM4-80GB", date="2026-08-30", vllm="0.28.0", rocm=None,
         cuda="13.0", driver="580.82.07", kernel=None, patches=[],
         prefix_caching=False),
    # 2026-08-30, the four-machine round. Qwen3-8B on ONE 7900 XT at util 0.95,
    # on the 0.27 image and **fully stock**: the container's
    # `triton_unified_attention.py`, `triton_attn.py` and
    # `chunked_prefill_paged_decode.py` were restored to the image's own bytes
    # and asserted by md5 before the run, so this row carries no patches.
    #
    # It was run to lift the July ladder's 6 000 ceiling by raising utilisation
    # and did not: the pool came out at 1.13 GiB and 8 236 tokens against July's
    # 8 442, and the runner stepped `max_model_len` to 8 157. Five rungs.
    #
    # `rocm` is the image tag (`rocm/vllm:rocm10.0.0_ubuntu24.04_...`). `kernel`
    # is **carried from `campaign-2026-08-30`** -- the same box, the same day,
    # no reboot between them -- and is not read from this run's own log, which
    # records neither. `enable_prefix_caching=True` and, as on every Radeon arm,
    # it produced no hits: the serve log is 0.0 % at every one of its samples.
    dict(file="campaign-2026-08-30b/results.jsonl", machine="RX 7900 XT",
         date="2026-08-30", vllm="0.27.1.dev5+gf46a9dfe2.d20260827", rocm="10.0",
         cuda=None, kernel="7.0.0-30", patches=[], prefix_caching=True),
    # 2026-08-30. The L4's second pass, after `cuda_run.py` gained the capacity
    # retry the Radeon runner has had since rev2. `B8` reaches 24 000 rather
    # than 32 000 because the retry stepped `max_model_len` to 31 680; `Q38S`
    # reaches 8 000 on a 10 090-token pool. `G31` is a `config_failed` row in
    # the same file -- it is measured in `campaign-2026-08-30c` instead, and
    # only with `--enforce-eager`.
    dict(file="cuda-l4/campaign-2026-08-30b/results.jsonl", machine="L4",
         date="2026-08-30", vllm="0.28.0", rocm=None, cuda="13.0",
         driver="580.82.07", kernel=None, patches=[], prefix_caching=False),
    # 2026-08-30. The fifth machine, which the pre-flight had recorded as a
    # wall. **These are the only rows in either projection measured with a
    # patch that changes an attention kernel's tile size**: without vllm#39018
    # the engine does not start at all on sm75, dying at kernel load asking
    # 98 304 bytes of shared memory against Turing's 65 536. The patch halves
    # `TILE_PREFILL` on the head_size 512 layers only, so `c` on these rows is
    # not comparable with any other machine's; decode is untouched, which the
    # recorder confirmed in both states.
    # 2026-08-30, the L4's third pass: the two arms the second could not fit,
    # at `max_num_seqs=1` with an `--enforce-eager` fallback. `G31` needed the
    # fallback and becomes `G31-eager`, which is its own configuration and not
    # an arm of `G31` -- a different engine. Four capacity retries from
    # mml 33000 to 2062 all reported `Available KV cache memory: -0.8 GiB`; eager
    # at the same utilisation reported 1.71 GiB, 2 020 tokens. The 2.51 GiB
    # between them is CUDA graphs, and both serve logs are in `logs/`.
    dict(file="cuda-l4/campaign-2026-08-30c/results.jsonl", machine="L4",
         date="2026-08-30", vllm="0.28.0", rocm=None, cuda="13.0",
         driver="580.82.07", kernel=None, patches=[], prefix_caching=False),
    dict(file="cuda-t4/campaign-2026-08-30/results.jsonl", machine="T4",
         date="2026-08-30", vllm="0.28.0", rocm=None, cuda="13.0",
         driver="580.82.07", kernel=None, patches=["vllm#39018"],
         prefix_caching=False),
    dict(file="cuda-a100/campaign-2026-08-29/results.jsonl", prefix_caching=True,
         machine="A100-SXM4-80GB", date="2026-08-29", vllm="0.28.0",
         rocm=None, cuda=None, driver=None, kernel=None, patches=[],
         per_cfg={
             "A100-G31-mtp-p45450":    dict(patches=["vllm#45450 3D admission"]),
             "A100-G26A4B-mtp-p45450": dict(patches=["vllm#45450 3D admission"]),
             "A100-Q38-mtp-p45450":    dict(patches=["vllm#45450 3D admission"]),
         }),
]

# (source, cfg, read-from-this-source, ARMS-table). Reported, not resolved.
BACKEND_MISMATCH = []

BACKEND_RE = re.compile(
    r"Using (?:AttentionBackendEnum\.)?([A-Z0-9_]+)(?: attention)? backend"
    r"|Overriding with ([A-Z0-9_]+) out of potential backends")
VIT_RE = re.compile(r"vit attention|MMEncoderAttention")


def backend_from_log(path):
    """The decoder's backend, from any of the forms vLLM 0.28 writes.

    The first decoder line wins. On a speculative arm the log carries two: the
    target's, then the drafter's after `Loading drafter model...`, and they
    disagree -- `--attention-backend TRITON_ATTN` reaches the target and not the
    drafter (vllm#53450). Taking the first is taking the target's, which is what
    the arm is named for; a drafter that differs makes the arm a mixture and is
    a fact about the arm rather than about this column.
    """
    if not os.path.exists(path):
        return None
    for line in open(path, errors="ignore"):
        if VIT_RE.search(line):
            continue                      # the vision tower, not the decoder
        m = BACKEND_RE.search(line)
        if m:
            return m.group(1) or m.group(2)
    return None


# --- how the backend was chosen, not only which one won ----------------------
# The serve logs carry more than the winner: the candidate set an override
# picked from, the reason a choice was forced, the head dimensions that forced
# it, and which quantisation kernel the checkpoint landed on. None of that is in
# either projection today -- FlashInfer and FLEX_ATTENTION appear in the logs and
# in no row -- so a question about routing has to be answered by grepping logs
# by hand. These lift it into the data.
#
# `gqa_ratio` is deliberately absent: vLLM does not print the head counts, so it
# is not derivable from what we keep. It would have to come from the model
# config, which this repository does not check in.
ROUTE_RES = {
    "candidates": re.compile(
        r"Overriding with [A-Z0-9_]+ out of potential backends: \[([^\]]*)\]"),
    # two phrasings, and the reason sits on opposite sides of the verb:
    #   "... Forcing TRITON_ATTN backend to prevent mixed-backend divergence."
    #   "... FA4 not available, forcing TRITON_ATTN backend."
    "forced": re.compile(r"(?:([^.,\n]{3,60}), )?[Ff]orcing ([A-Z0-9_]+) backend"
                         r"(?: to ([^.\n]+))?"),
    "head_dim": re.compile(r"head_dim=(\d+)"),
    "global_head_dim": re.compile(r"global_head_dim=(\d+)"),
    "head_dims": re.compile(r"(\{'(?:sliding|full)_attention'[^}]*\})"),
    "kv_cache_dtype": re.compile(r"kv_cache_dtype=([A-Za-z0-9_.]+)"),
    "quant_kernel": re.compile(r"Using (\w+Kernel) for (\w+)"),
}


def route_from_log(path):
    """Why this arm got the backend it got, as far as the log says.

    Returns None when the log yields nothing, so a row whose log was never kept
    is distinguishable from one whose log simply said little.
    """
    if not os.path.exists(path):
        return None
    out = {}
    for line in open(path, errors="ignore"):
        if VIT_RE.search(line):
            continue
        if "candidates" not in out:
            m = ROUTE_RES["candidates"].search(line)
            if m:
                out["candidates"] = [x.strip().strip("'\"")
                                     for x in m.group(1).split(",") if x.strip()]
                out["decision"] = "override"
        if "forced_reason" not in out:
            m = ROUTE_RES["forced"].search(line)
            if m:
                out["decision"] = "forced"
                out["forced_reason"] = ((m.group(3) or "").strip()
                                        or (m.group(1) or "").strip()
                                        or "unstated")
        for k in ("head_dim", "global_head_dim"):
            if k not in out:
                m = ROUTE_RES[k].search(line)
                if m:
                    out[k] = int(m.group(1))
        if "head_dims" not in out:
            m = ROUTE_RES["head_dims"].search(line)
            if m:
                # a python-repr dict in the log; keep it as data, not a string
                out["head_dims"] = {k: int(v) for k, v in
                                    re.findall(r"'(\w+)':\s*(\d+)", m.group(1))}
        if "kv_cache_dtype" not in out:
            m = ROUTE_RES["kv_cache_dtype"].search(line)
            if m:
                out["kv_cache_dtype"] = m.group(1)
        if "quant_kernel" not in out:
            m = ROUTE_RES["quant_kernel"].search(line)
            if m:
                out["quant_kernel"] = m.group(1)
                out["quant_scheme"] = m.group(2)
    if out and "decision" not in out:
        out["decision"] = "default"
    return out or None


def routes_from_source(path):
    """{cfg: route} for one results file, from the serve logs beside it."""
    out = {}
    for name in ("logs", "serve-logs"):
        d = os.path.join(os.path.dirname(path), name)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".log"):
                continue
            cfg = fn[:-4]
            cfg = cfg[6:] if cfg.startswith("serve-") else cfg
            if cfg in out:
                continue
            r = route_from_log(os.path.join(d, fn))
            if r:
                out[cfg] = r
    return out


# --- host PCIe link, the measurement condition nobody recorded -----------------
# On 2026-09-02 the Proxmox host's persistent journal showed card 0b:00.0
# (guest card1, the runners' p1/v1) trained at x8 from the boot of 2026-08-29
# 21:48 CST, after ten boots at x16 and a hard stop at 06:28 that day:
#
#   boot -2 (Aug 27 18:56 .. Aug 29 06:28)  "limited by 8.0 GT/s PCIe x16 link at 0000:00:03.1"
#   boot -1 (Aug 29 21:48 ..)               "limited by 8.0 GT/s PCIe x8 link at 0000:00:03.1"
#
# The other card was x16 at every boot. The guest's own sysfs reports 16 GT/s
# x16 throughout, because that is the on-card bridge link, so no run could have
# seen it. TP=2 is bounded by the narrower card; TP=1 on 2026-08-30 ran on card1
# (its VRAM rose, card2's did not) but has no all-reduce, so only weight loading
# crosses that link. Decode is not measurably affected at either width; prefill
# at TP=2 and depth is -- the 31B's fitted `b` is 743.9 and 736.0 on the two
# x16 sittings and 868.7 on the x8 one.
HOST_LINK_X8_FROM = "2026-08-29"


def host_link(machine, date):
    if machine != "RX 7900 XT":
        return None                               # rented VMs: unknown, and not ours
    return "x8/x16" if date >= HOST_LINK_X8_FROM else "x16/x16"


def backends_from_source(path):
    """{cfg: backend} for one results file, from its own metadata and logs.

    `model_meta.backend` is the CUDA runners' in-process read of the serve log
    and is exact. The Radeon runners and the 2026-08-29 A100 campaign do not
    write it, so their logs are read here -- under both directory names this
    repository has used (`logs/`, `serve-logs/`) and both file conventions
    (`<cfg>.log`, `serve-<cfg>.log`).
    """
    out = {}
    if os.path.exists(path):
        for line in open(path, errors="ignore"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("kind") == "model_meta" and r.get("backend"):
                out[r["cfg"]] = r["backend"]
    for name in ("logs", "serve-logs"):
        d = os.path.join(os.path.dirname(path), name)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".log"):
                continue
            cfg = fn[:-4]
            cfg = cfg[6:] if cfg.startswith("serve-") else cfg
            if cfg in out:
                continue                  # model_meta already settled it
            b = backend_from_log(os.path.join(d, fn))
            if b:
                out[cfg] = b
    return out


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
        measured = backends_from_source(path)
        routed = routes_from_source(path)
        for (cfg, target), trips in sorted(by.items()):
            name, quant, arch, tp = meta_for(cfg)
            spec, tabled = arm_for(cfg)
            # This source's own reading first. `ARMS`/`ARMS_CUDA` are keyed on a
            # cfg id that repeats across machines, so they are the fallback and
            # not the authority; where both exist and disagree, that is recorded
            # rather than silently resolved.
            backend = measured.get(cfg) or tabled
            if tabled and measured.get(cfg) and measured[cfg] != tabled:
                BACKEND_MISMATCH.append((s["file"], cfg, measured[cfg], tabled))
            over = s.get("per_cfg", {}).get(cfg, {})
            vals, superseded = latest_session([(t, (tt, pt)) for t, tt, pt in trips])
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
                host_link=host_link(s["machine"], s["date"]),
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
