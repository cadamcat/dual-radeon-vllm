#!/usr/bin/env python3
"""Derive the campaign's provenance from the serve logs, rather than typing it.

The ledger's `attn_backend` column is a claim about which kernel served each
arm. A claim needs its evidence in the repository, so this reads the logs the
runs actually wrote and emits both the value and the line it came from. It is
the same discipline hybrid-splitkv-027/provenance.json uses for #45916's md5s:
what cannot be recomputed from a committed file is not recorded.

    python3 make_provenance.py <serve-logs-dir> <serve-scripts-dir> > provenance.json
"""
import json, os, re, sys

LOGS, SCRIPTS = sys.argv[1], sys.argv[2]

BACKEND_RE = re.compile(
    r"(Overriding with ([A-Z_]+) out of potential backends: \[[^\]]*\]"
    r"|Using ([A-Z_]+) backend"
    r"|Forcing ([A-Z_]+) backend)")
VLLM_RE = re.compile(r"v(0\.\d+\.\d+(?:\.dev\d+)?\+g[0-9a-f]+(?:\.d\d+)?)")
SPEC_RE = re.compile(r"SpeculativeConfig\(([^)]*)\)")


def arm(name):
    p = os.path.join(LOGS, name + ".log")
    if not os.path.exists(p):
        return None
    txt = open(p, errors="replace").read()
    # Two different lines, and they are not the same claim.
    #
    #   rocm.py:606  "Using TRITON_ATTN backend (selected via --attention-backend)"
    #   rocm.py:651  "Found incompatible backend(s) [TURBOQUANT] with
    #                 AttentionType.DECODER. Overriding with ROCM_ATTN ..."
    #
    # In the forced arms both appear, and the timeline says what each one is:
    # the selection lands right after "Starting to load model", the overrides
    # land after the "based on the speculative decoding settings" warning, one
    # per tensor-parallel worker. They are the *draft* head, not the target.
    # A no-speculation forced arm (Q38-triton-tp2) has the selection and no
    # override at all, which is the control for that reading. So even with the
    # kernel chosen explicitly, Qwen3.8's mtp head still falls back to
    # ROCM_ATTN, while the target -- whose verify step is what #45450 admits
    # and where the probe fires -- runs on Triton.
    #
    # attn_backend is therefore the target's: the explicit selection when there
    # is one, and otherwise the override, which in an unforced arm is the
    # target's own.
    sel, ovr, lines = [], [], []
    for m in BACKEND_RE.finditer(txt):
        b = m.group(2) or m.group(3) or m.group(4)
        (ovr if m.group(2) else sel).append(b)
        if m.group(0) not in lines:
            lines.append(m.group(0))
    backends = sel or ovr
    vs = VLLM_RE.findall(txt)
    # only the engine's own resolved line, never the "**self.speculative_config"
    # that appears inside a traceback from a run that never resolved one
    sc = next((m for m in SPEC_RE.finditer(txt) if "method=" in m.group(1)), None)
    sp = os.path.join(SCRIPTS, f"serve-{name}.sh")
    out = {
        "serve_log": f"logs/{name}.log",
        "attn_backend": backends[0] if backends else None,
        "attn_backend_draft": (ovr[0] if (sel and ovr) else None),
        "attn_backend_evidence": lines,
        "vllm": sorted(set(vs))[0] if vs else None,
        "speculative_config": sc.group(1) if sc else None,
        # once per process, so TP=2 prints it twice. The 45450-validation
        # README's "exactly once" was measured on a TP=1 A100.
        "probe_3d_spec_active": txt.count("PROBE_3D_SPEC_ACTIVE"),
        "reached_startup_complete": "Application startup complete" in txt,
    }
    st = STATE.get(name)
    if st:
        out["container"] = st[0]
        out["p45450"] = st[1]
        out["p45450_md5"] = {
            "triton_unified_attention.py": MD5[st][0],
            "triton_attn.py": MD5[st][1],
        }
    if os.path.exists(sp):
        cmd = [l for l in open(sp) if l.startswith("exec vllm serve")]
        out["serve_command"] = cmd[0].split(" > ")[0].strip() if cmd else None
    return out


ARMS = ["Q38-tp2", "Q38-mtp-tp2", "Q38-mtp-p45450-tp2",
        "Q38-triton-tp2", "Q38-mtp-triton-p45450-tp2", "Q38-mtp-triton-tp2",
        "G31-tp2", "G31-mtp-p45450-tp2",
        "G31-mtp-tp2", "G31-tp2-on-027",
        "MG30-dflash-tp2", "MG30-dflash-method-dflash"]

# Which container, and what state its two #45450 files were in while the arm
# ran. This cannot come from the serve log: `Q38-mtp-triton-tp2` and
# `Q38-mtp-triton-p45450-tp2` issue byte-identical serve commands and differ
# only in the bytes of two files inside the container. Each md5 below was read
# off the running container with `docker cp` + md5sum, before and after each
# change, by revert45450.py, whose assertions refuse to proceed on anything
# else. "stock" is the image's own file; "patched" is inject_45450.py applied.
MD5 = {
    ("vllm-027", "stock"):   ("49fab3b6", "f0a1379d"),
    ("vllm-027", "patched"): ("9416a868", "8bd13173"),
    ("vllm-tp2", "patched"): ("4a14f86d", "7e275cdc"),
}
STATE = {
    # measured before inject_45450.py was ever run on this host
    "Q38-tp2":                   ("vllm-027", "stock"),
    "Q38-mtp-tp2":               ("vllm-027", "stock"),
    # after the injection
    "Q38-mtp-p45450-tp2":        ("vllm-027", "patched"),
    "Q38-triton-tp2":            ("vllm-027", "patched"),
    "Q38-mtp-triton-p45450-tp2": ("vllm-027", "patched"),
    # reverted to the image's own bytes for this one arm, then put back
    "Q38-mtp-triton-tp2":        ("vllm-027", "stock"),
    "G31-tp2":                   ("vllm-tp2", "patched"),
    "G31-mtp-p45450-tp2":        ("vllm-tp2", "patched"),
    # the three that produced no data. Recorded anyway, so a reader is not left
    # wondering whether the state was unknown or merely uninteresting.
    "G31-mtp-tp2":               ("vllm-027", "stock"),
    "MG30-dflash-tp2":           ("vllm-027", "stock"),
    "MG30-dflash-method-dflash": ("vllm-027", "patched"),
    "G31-tp2-on-027":            ("vllm-027", "patched"),
}

doc = {
    "what": "The 2026-08-29 ladder campaign on the Radeon host: which kernel "
            "served each arm, what speculation the engine actually resolved, "
            "and whether vllm#45450's probe fired.",
    "why": "results-2026-08-29.jsonl carries eight arms across two vLLM images "
           "and two attention backends. Two of them differ only in whether "
           "#45450 is installed, and the patch cannot act on one of them, so "
           "the backend is part of what the row claims and needs its evidence "
           "committed rather than asserted.",
    "arms": {a: v for a in ARMS if (v := arm(a)) is not None},
}
print(json.dumps(doc, indent=1))
