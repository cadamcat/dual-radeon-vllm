"""Figures for rdna3-second-class.html.

This article classifies the eight findings the others establish, so its data is
mostly cross-references. What is derived rather than asserted: the magnitude
column, recomputed from the same files the individual articles draw, and the
gate comparison, extracted from the ecosystem list in docs/architecture-notes.md.
"""
import json, pathlib, re, sys
R = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "benchmarks" / "analyze"))
import verify_doc_figures as V

B = R / "benchmarks"
jl = lambda p: [json.loads(l) for l in open(p)]
jul = V.decode(str(B / "results.jsonl"))
lad = lambda fn: {r["depth"]: r["tok_per_s"] for r in
                  json.load(open(B / "speculative-decoding" / fn))["rows"]}

# ---- the magnitudes, each recomputed from the file its own article draws ---
ab = {(r["arm"], r["ctx"]): r for r in jl(B / "w4a16-symmetry" / "w4a16-ab.jsonl")}
led = jl(B / "ledger.jsonl")
q = lambda patches, ctx: next(
    r["decode_tok_s"] for r in led if r["model"] == "Qwen3.8-27B" and r["tp"] == 2
    and r["vllm"] == "0.27.1.dev5+gf46a9dfe2" and r["patches"] == patches
    and r["date"] == "2026-08-28" and r["ctx"] == ctx)
g1 = jl(B / "vllm-50603" / "stage1-rocm-paths.jsonl")
gr = [r["triton"]["median_ms"] / r["ck"]["median_ms"] for r in g1]
s3 = {(r["arm"], r["ctx"]): r["decode_tok_s"]
      for r in jl(B / "vllm-50603" / "stage3-endtoend.jsonl")}
hmm = json.load(open(B / "hmm-kernel-three-states.json"))["states"]
lf = json.load(open(B / "loader-flag-kernel-30.json"))
mm = {(r["model"], r["cache"], r["mode"]): r["median_s"] for r in lf["medians_seconds"]}
p45, ns45 = lad("mtp-31b-p45450.json"), lad("splitkv-31b-stock.json")
mtp, stk = lad("mtp-31b-mtp.json"), lad("mtp-31b-stock45450.json")
# the A100's own 2D retention, from the legs the comparison article draws
VD = B / "cuda-a100" / "45450-validation" / "logs"
leg = lambda fn: float(re.search(r"RESULT decode_tok_s=([\d.]+)",
                                 open(VD / fn).read()).group(1))
leg1k, leg30 = leg("C1K.log"), leg("C30.log")

FINDINGS = [
    {"slug": "hybrid-ssm-collapse", "title": "A hybrid-SSM model that decodes slower the longer you talk to it",
     "axis": "rdna3", "mechanism": "the few full-attention layers fall off the ROCm paged-attention fast path",
     "magnitude": V.tps(jul, "D-27B-tp2", 500) / V.tps(jul, "D-27B-tp2", 32000),
     "unit": "x lost from 500 to 32K tokens",
     "why": "the split-KV fallback that fixes it is being added for gfx11 specifically; the path exists elsewhere",
     "upstream": ["vllm#45916", "vllm#50264"]},
    {"slug": "w4a16-two-problems", "title": "Twelve tokens a second was two problems",
     "axis": "rdna3", "mechanism": "an asymmetric int4 checkpoint misses the native gfx1100 W4A16 kernel on a type gate",
     "magnitude": ab[("sym", 1024)]["decode_tok_s"] / ab[("asym", 1024)]["decode_tok_s"],
     "unit": "x at 1K, symmetric against asymmetric packaging",
     "why": "eight of twelve asymmetric configurations still have no kernel at all on gfx1100",
     "upstream": ["vllm#40977"]},
    {"slug": "gqa-gate-costs-nothing", "title": "A gate that costs 2 to 7 times and buys nothing",
     "axis": "rdna3", "mechanism": "the gfx11 branch requires gqa_ratio >= 3; the CDNA branch of the same function requires >= 1",
     "magnitude": max(gr),
     "unit": "x, the excluded range's best case at the kernel",
     "why": "one line, one architecture, and the reason given for it does not hold here",
     "upstream": ["vllm#54210", "vllm#50603"]},
    {"slug": "weight-loading-19x", "title": "Loading weights was slower than the disk, twice over",
     "axis": "amd", "mechanism": "KFD reads the access mode off the VMA, so a read-only copy asks for write and breaks copy-on-write",
     "magnitude": mm[("gemma-4-31B-w4a16", "cold", "baseline")]
                  / mm[("gemma-4-31B-w4a16", "cold", "flag")],
     "unit": "x on a checkpoint that does not fit in RAM",
     "why": "kfd_svm.c is not architecture-specific and AMD confirmed it as such",
     "upstream": ["ROCm#6523", "vllm#49991"]},
    {"slug": "moe-written-off-by-eager", "title": "The fastest model here was written off at 15 tok/s",
     "axis": "neutral", "mechanism": "vLLM assigns TORCHINDUCTOR_COMPILE_THREADS=1 on import, so an export cannot override it",
     "magnitude": V.tps(jul, "E-26B-tp2", 500) / 15.0,
     "unit": "x between eager and compiled on this host",
     "why": "the assignment is in env_override.py and runs on every platform",
     "upstream": ["vllm#53891", "vllm#53892"]},
    {"slug": "speculative-decoding-net-loss", "title": "One boolean costs 71% on a Radeon and 61% on an A100",
     "axis": "neutral", "mechanism": "max_seqlen_q > 1 drops the Triton attention kernel from 128 workgroups to 8",
     "magnitude": ns45[32768] / mtp[32768],
     "unit": "x slower with speculation on at 32K",
     "why": "measured on an A100 as well, where the same kernel loses 61% at 50K",
     "upstream": ["vllm#45450", "vllm#48076"]},
    {"slug": "a100-vs-two-radeons", "title": "Two consumer Radeons against one A100",
     "axis": "tp", "mechanism": "the 2D attention path launches one workgroup per KV head, and tensor parallelism halves them",
     "magnitude": (leg30 / leg1k) / (stk[32768] / stk[1024]),
     "unit": "x worse retention on two cards than on one",
     "why": "a property of splitting the work, not of the silicon; the single A100 runs the same kernel",
     "upstream": []},
    {"slug": "rccl-atomics-hostcall", "title": "The RCCL crash was never about RCCL",
     "axis": "platform", "mechanism": "no PCIe AtomicOps means no hostcall buffer, and the dispatch is refused",
     "magnitude": None,
     "unit": "every collective fails at dispatch",
     "why": "reproduced on bare metal by someone else; the trigger is the root port, not the GPU",
     "upstream": ["ROCm#6520"]},
]
AXES = {"rdna3": "RDNA3 specifically", "amd": "AMD-wide", "neutral": "vendor-neutral",
        "tp": "tensor parallelism", "platform": "platform capability"}
fig1 = {"findings": FINDINGS, "axes": AXES,
        "counts": {a: sum(1 for f in FINDINGS if f["axis"] == a) for a in AXES},
        "total": len(FINDINGS)}
fig1["rdna3_share"] = fig1["counts"]["rdna3"] / fig1["total"] * 100.0

# ---- fig2: the gate, and the ecosystem list it sits in --------------------
an = (R / "docs/architecture-notes.md").read_text()
tail = an.split("## The common cause", 1)[1]
gaps = [{"what": re.sub(r"[*`]", "", m.group(1)).strip(),
         "why": re.sub(r"\s+", " ", re.sub(r"[*`]", "", m.group(2))).strip()}
        for m in re.finditer(r"^- \*\*([^*]+)\*\*([^\n]*(?:\n  [^\n]*)*)", tail, re.M)]
fig2 = {"gaps": gaps,
        "gate": {"gfx11": "gqa_ratio >= 3", "cdna": "gqa_ratio >= 1",
                 "function": "use_rocm_custom_paged_attention",
                 "file": "vllm/platforms/rocm.py"},
        "measured": {"excluded_low": min(r["triton"]["median_ms"] / r["ck"]["median_ms"]
                                         for r in g1 if not r["gate_as_shipped"]),
                     "excluded_high": max(r["triton"]["median_ms"] / r["ck"]["median_ms"]
                                          for r in g1 if not r["gate_as_shipped"]),
                     "admitted_low": min(r["triton"]["median_ms"] / r["ck"]["median_ms"]
                                         for r in g1 if r["gate_as_shipped"]),
                     "admitted_high": max(r["triton"]["median_ms"] / r["ck"]["median_ms"]
                                          for r in g1 if r["gate_as_shipped"]),
                     "end_to_end_32k": s3[("widened", 32768)] / s3[("stock", 32768)]},
        # the document wraps that sentence, so match on the half that carries it
        "extends_to_rdna4": "extends to RDNA4." in an}

# ---- fig3: where each thread stands, as of the date on the page -----------
STATUS = [
    {"id": "vllm#40977", "kind": "PR", "author": "mgehre-amd", "state": "merged",
     "when": "2026-07-14", "ours": False},
    {"id": "vllm#45916", "kind": "PR", "author": "feiyehua", "state": "open",
     "when": "verified here 2026-07-30", "ours": False},
    {"id": "vllm#45450", "kind": "PR", "author": "jinhuang12", "state": "open, conflicted",
     "when": "validated here 2026-08-26", "ours": False},
    {"id": "vllm#53856", "kind": "PR", "author": "aoshen02", "state": "open",
     "when": "gfx11 evidence supplied 2026-08-28", "ours": False},
    {"id": "vllm#49588", "kind": "PR", "author": "hec-ovi", "state": "open, draft",
     "when": "second evidence body 2026-08-24", "ours": False},
    {"id": "vllm#48076", "kind": "issue", "author": "tuananhlfc", "state": "open",
     "when": "second trigger posted 2026-08-01", "ours": False},
    {"id": "vllm#50603", "kind": "issue", "author": "AIwork4me", "state": "open",
     "when": "measured 2026-08-27", "ours": False},
    {"id": "vllm#50264", "kind": "issue", "author": "cadamcat", "state": "open",
     "when": "2026-07-30", "ours": True},
    {"id": "vllm#49991", "kind": "PR", "author": "cadamcat", "state": "open",
     "when": "2026-07-27", "ours": True},
    {"id": "vllm#53891", "kind": "issue", "author": "cadamcat", "state": "open",
     "when": "2026-08-26", "ours": True},
    {"id": "vllm#53892", "kind": "PR", "author": "cadamcat", "state": "open",
     "when": "2026-08-26", "ours": True},
    {"id": "vllm#53930", "kind": "PR", "author": "cadamcat", "state": "open",
     "when": "2026-08-27", "ours": True},
    {"id": "vllm#54210", "kind": "PR", "author": "cadamcat", "state": "open",
     "when": "2026-08-28", "ours": True},
    {"id": "ROCm#6520", "kind": "issue", "author": "cadamcat", "state": "open",
     "when": "2026-07-26", "ours": True},
    {"id": "ROCm#6523", "kind": "issue", "author": "cadamcat", "state": "open",
     "when": "fixed by Ubuntu 7.0.0-30", "ours": True},
    {"id": "ROCm#6565", "kind": "issue", "author": "BoJl4apa", "state": "open",
     "when": "contrast cell 2026-08-28", "ours": False},
]
fig3 = {"threads": STATUS, "checked": "2026-08-29",
        "ours": sum(1 for s in STATUS if s["ours"]),
        "others": sum(1 for s in STATUS if not s["ours"]),
        "merged": sum(1 for s in STATUS if s["state"] == "merged")}

out = {"_what": "Every figure in rdna3-second-class.html. The classification is "
                "the article's own; every magnitude beside it is recomputed from "
                "the file the individual article draws. Derived by "
                "site/src/genfig-rdna3.py.",
       "fig1": fig1, "fig2": fig2, "fig3": fig3}
json.dump(out, open(pathlib.Path(__file__).parent / "figures-rdna3.json", "w"),
          ensure_ascii=False, indent=1)
print("fig1 counts:", fig1["counts"], f'rdna3 share {fig1["rdna3_share"]:.0f}%')
for f in FINDINGS:
    m = f"{f['magnitude']:8.2f}" if f["magnitude"] is not None else "       -"
    print(f'  {f["axis"]:<9} {m}  {f["slug"]}')
print("fig2 gaps:", [g["what"] for g in gaps])
print("fig2 measured:", {k: round(v, 2) for k, v in fig2["measured"].items()},
      "rdna4", fig2["extends_to_rdna4"])
print("fig3:", fig3["ours"], "ours,", fig3["others"], "others,", fig3["merged"], "merged")
print("bytes:", len(json.dumps(out)))
