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
    {"slug": "hybrid-ssm-collapse", "title": "One kernel accounts for the whole 12.1 to 4.2 tok/s fall",
     "axis": "rdna3", "mechanism": "the few full-attention layers fall off the ROCm paged-attention fast path",
     "magnitude": V.tps(jul, "D-27B-tp2", 500) / V.tps(jul, "D-27B-tp2", 32000),
     "unit": "x lost from 500 to 32K tokens",
     "why": "the split-KV fallback that fixes it is being added for gfx11 specifically; the path exists elsewhere",
     "upstream": ["vllm#45916", "vllm#50264"]},
    {"slug": "w4a16-two-problems", "title": "A flat 60 ms per step, under a cost that grows with context",
     "axis": "rdna3", "mechanism": "an asymmetric int4 checkpoint misses the native gfx1100 W4A16 kernel on a type gate",
     "magnitude": ab[("sym", 1024)]["decode_tok_s"] / ab[("asym", 1024)]["decode_tok_s"],
     "unit": "x at 1K, symmetric against asymmetric packaging",
     "why": "eight of twelve asymmetric configurations still have no kernel at all on gfx1100",
     "upstream": ["vllm#40977"]},
    {"slug": "gqa-gate-costs-nothing", "title": "The excluded kernel wins all sixty cells, and the two bands overlap",
     "axis": "rdna3", "mechanism": "the gfx11 branch requires gqa_ratio >= 3; the CDNA branch of the same function requires >= 1",
     "magnitude": max(gr),
     "unit": "x, the excluded range's best case at the kernel",
     "why": "one line, one architecture, and the reason given for it does not hold here",
     "upstream": ["vllm#54210", "vllm#50603"]},
    {"slug": "weight-loading-19x", "title": "A read-only copy asks for write access, and one kernel charged a second for it",
     "axis": "amd", "mechanism": "KFD reads the access mode off the VMA, so a read-only copy asks for write and breaks copy-on-write",
     "magnitude": mm[("gemma-4-31B-w4a16", "cold", "baseline")]
                  / mm[("gemma-4-31B-w4a16", "cold", "flag")],
     "unit": "x on a checkpoint that does not fit in RAM",
     "why": "kfd_svm.c is not architecture-specific and AMD confirmed it as such",
     "upstream": ["ROCm#6523", "vllm#49991"]},
    {"slug": "moe-written-off-by-eager", "title": "Eager mode recorded 107.8 tok/s as 15, and invented two findings on the way",
     "axis": "neutral", "mechanism": "vLLM assigns TORCHINDUCTOR_COMPILE_THREADS=1 on import, so an export cannot override it",
     "magnitude": V.tps(jul, "E-26B-tp2", 500) / 15.0,
     "unit": "x between eager and compiled on this host",
     "why": "the assignment is in env_override.py and runs on every platform",
     "upstream": ["vllm#53891", "vllm#53892"]},
    {"slug": "speculative-decoding-net-loss", "title": "Speculation's second query row costs 120 of 128 workgroups",
     "axis": "neutral", "mechanism": "max_seqlen_q > 1 drops the Triton attention kernel from 128 workgroups to 8",
     "magnitude": ns45[32768] / mtp[32768],
     "unit": "x slower with speculation on at 32K",
     "why": "measured on an A100 as well, where the same kernel loses 61% at 50K",
     "upstream": ["vllm#45450", "vllm#48076"]},
    {"slug": "a100-vs-two-radeons", "title": "Two consumer RX 7900 XTs against one A100",
     "axis": "tp", "mechanism": "the 2D attention path launches one workgroup per KV head, and tensor parallelism halves them",
     "magnitude": (leg30 / leg1k) / (stk[32768] / stk[1024]),
     "unit": "x worse retention on two cards than on one",
     "why": "a property of splitting the work, not of the silicon; the single A100 runs the same kernel",
     "upstream": []},
    {"slug": "rccl-atomics-hostcall", "title": "No PCIe atomics, no hostcall buffer, and every collective fails at dispatch",
     "axis": "platform", "mechanism": "no PCIe AtomicOps means no hostcall buffer, and the dispatch is refused",
     "magnitude": None,
     "unit": "every collective fails at dispatch",
     "why": "reproduced on bare metal by someone else; the trigger is the root port, not the GPU",
     "upstream": ["ROCm#6520"]},
]
# The Chinese half of the same eight rows. The figure block is byte-identical
# across the language pair, so both live here and the shared script picks by
# document.documentElement.lang; the index's one-line summary reads the same
# mechanism field, which is what keeps the synthesis article and the index from
# describing a finding two different ways.
ZH = {
    "hybrid-ssm-collapse": {
        "title": "\u4e00\u4e2a kernel \u5c31\u89e3\u91ca\u4e86 12.1 \u5230 4.2 tok/s \u7684\u5168\u90e8\u4e0b\u6ed1",
        "mechanism": "\u5c11\u6570\u51e0\u5c42\u5168\u6ce8\u610f\u529b\u6389\u51fa\u4e86 ROCm paged-attention \u7684\u5feb\u8def\u5f84",
        "unit": "\u00d7\uff0c\u4ece 500 \u5230 32K token \u635f\u5931\u7684\u500d\u6570",
        "why": "\u4fee\u597d\u5b83\u7684 split-KV \u515c\u5e95\u662f\u4e13\u95e8\u4e3a gfx11 \u52a0\u7684\uff1b\u8fd9\u6761\u8def\u5f84\u5728\u522b\u5904\u65e9\u5c31\u6709"},
    "w4a16-two-problems": {
        "title": "\u6bcf\u6b65\u56fa\u5b9a 60 ms\uff0c\u538b\u5728\u4e00\u7b14\u968f\u4e0a\u4e0b\u6587\u589e\u957f\u7684\u5f00\u9500\u4e0b\u9762",
        "mechanism": "\u975e\u5bf9\u79f0 int4 checkpoint \u5728\u7c7b\u578b\u5224\u65ad\u4e0a\u9519\u8fc7\u4e86 gfx1100 \u7684\u539f\u751f W4A16 kernel",
        "unit": "\u00d7\uff0c1K \u4e0a\u5bf9\u79f0\u4e0e\u975e\u5bf9\u79f0\u6253\u5305\u4e4b\u6bd4",
        "why": "\u5341\u4e8c\u79cd\u975e\u5bf9\u79f0\u914d\u7f6e\u91cc\u6709\u516b\u79cd\u5728 gfx1100 \u4e0a\u81f3\u4eca\u6ca1\u6709\u4efb\u4f55 kernel"},
    "gqa-gate-costs-nothing": {
        "title": "\u88ab\u6392\u9664\u7684 kernel \u5728\u516d\u5341\u4e2a\u683c\u5b50\u91cc\u5168\u8d62\uff0c\u800c\u4e24\u4e2a\u533a\u95f4\u8fd8\u4e92\u76f8\u91cd\u53e0",
        "mechanism": "gfx11 \u5206\u652f\u8981\u6c42 gqa_ratio >= 3\uff0c\u540c\u4e00\u51fd\u6570\u7684 CDNA \u5206\u652f\u53ea\u8981\u6c42 >= 1",
        "unit": "\u00d7\uff0c\u88ab\u6392\u9664\u533a\u95f4\u5728 kernel \u5c42\u9762\u7684\u6700\u597d\u60c5\u51b5",
        "why": "\u4e00\u884c\u4ee3\u7801\u3001\u4e00\u79cd\u67b6\u6784\uff0c\u800c\u7ed9\u51fa\u7684\u7406\u7531\u5728\u8fd9\u91cc\u4e0d\u6210\u7acb"},
    "weight-loading-19x": {
        "title": "\u53ea\u8bfb\u7684\u62f7\u8d1d\u53bb\u7533\u8bf7\u5199\u6743\u9650\uff0c\u800c\u67d0\u4e2a\u5185\u6838\u4e3a\u6b64\u6bcf\u6b21\u591a\u6536\u4e00\u79d2",
        "mechanism": "KFD \u4ece VMA \u8bfb\u8bbf\u95ee\u6a21\u5f0f\uff0c\u4e8e\u662f\u53ea\u8bfb\u7684\u62f7\u8d1d\u4e5f\u8981\u5199\u6743\u9650\uff0c\u7834\u574f\u4e86 copy-on-write",
        "unit": "\u00d7\uff0c\u5728\u4e00\u4e2a\u88c5\u4e0d\u8fdb\u5185\u5b58\u7684 checkpoint \u4e0a",
        "why": "kfd_svm.c \u4e0d\u662f\u67b6\u6784\u7279\u5b9a\u7684\uff0cAMD \u4e5f\u8fd9\u4e48\u786e\u8ba4\u4e86"},
    "moe-written-off-by-eager": {
        "title": "eager \u6a21\u5f0f\u628a 107.8 tok/s \u8bb0\u6210 15\uff0c\u987a\u624b\u8fd8\u9020\u51fa\u4e24\u4e2a\u7ed3\u8bba",
        "mechanism": "vLLM \u5728 import \u65f6\u5c31\u8d4b\u503c TORCHINDUCTOR_COMPILE_THREADS=1\uff0cexport \u8986\u76d6\u4e0d\u6389",
        "unit": "\u00d7\uff0c\u672c\u673a\u4e0a eager \u4e0e\u7f16\u8bd1\u4e4b\u6bd4",
        "why": "\u8fd9\u6761\u8d4b\u503c\u5728 env_override.py \u91cc\uff0c\u6bcf\u4e2a\u5e73\u53f0\u90fd\u4f1a\u6267\u884c"},
    "speculative-decoding-net-loss": {
        "title": "\u6295\u673a\u7684\u7b2c\u4e8c\u884c query\uff0c\u8981\u4ed8\u6389 128 \u4e2a workgroup \u91cc\u7684 120 \u4e2a",
        "mechanism": "max_seqlen_q > 1 \u628a Triton \u6ce8\u610f\u529b kernel \u4ece 128 \u4e2a workgroup \u964d\u5230 8 \u4e2a",
        "unit": "\u00d7\uff0c32K \u4e0a\u5f00\u6295\u673a\u6bd4\u4e0d\u5f00\u6162",
        "why": "\u5728 A100 \u4e0a\u4e5f\u6d4b\u4e86\uff0c\u540c\u4e00\u4e2a kernel \u5728 50K \u4e0a\u635f\u5931 61%"},
    "a100-vs-two-radeons": {
        "title": "\u4e24\u5f20\u6d88\u8d39\u7ea7 Radeon 7900 XT \u5bf9\u4e00\u5f20 A100",
        "mechanism": "2D \u6ce8\u610f\u529b\u8def\u5f84\u6bcf\u4e2a KV head \u8d77\u4e00\u4e2a workgroup\uff0c\u5f20\u91cf\u5e76\u884c\u628a\u5b83\u4eec\u51cf\u534a",
        "unit": "\u00d7\uff0c\u4e24\u5361\u7684\u4fdd\u6301\u7387\u6bd4\u5355\u5361\u5dee",
        "why": "\u8fd9\u662f\u62c6\u5206\u5de5\u4f5c\u7684\u6027\u8d28\uff0c\u4e0d\u662f\u7845\u7247\u7684\uff1b\u5355\u5f20 A100 \u8dd1\u7684\u662f\u540c\u4e00\u4e2a kernel"},
    "rccl-atomics-hostcall": {
        "title": "\u6ca1\u6709 PCIe atomics \u5c31\u6ca1\u6709 hostcall buffer\uff0c\u6bcf\u4e00\u6b21\u96c6\u5408\u901a\u4fe1\u90fd\u5728 dispatch \u5904\u5931\u8d25",
        "mechanism": "\u6ca1\u6709 PCIe AtomicOps \u5c31\u6ca1\u6709 hostcall buffer\uff0cdispatch \u88ab\u62d2",
        "unit": "\u6bcf\u4e00\u6b21\u96c6\u5408\u901a\u4fe1\u90fd\u5728 dispatch \u5904\u5931\u8d25",
        "why": "\u522b\u4eba\u5728\u88f8\u673a\u4e0a\u590d\u73b0\u4e86\uff1b\u89e6\u53d1\u6761\u4ef6\u662f root port\uff0c\u4e0d\u662f GPU"},
}
for _f in FINDINGS:
    for _k, _v in ZH[_f["slug"]].items():
        _f[_k + "_zh"] = _v

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
