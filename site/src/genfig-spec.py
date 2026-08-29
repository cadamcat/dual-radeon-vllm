"""Figures for speculative-decoding-net-loss.html.

Four committed sources: the ROCm ladders in benchmarks/speculative-decoding/,
the direct kernel sweeps in the same directory, the A100 matrix in
benchmarks/cuda-a100/, and the profiler summary that names the traces it came
from but does not contain them.
"""
import json, pathlib
R = pathlib.Path(__file__).resolve().parents[2]
S = R / "benchmarks" / "speculative-decoding"
A = R / "benchmarks" / "cuda-a100"
jl = lambda p: json.load(open(p))
DEPTHS = (1024, 8192, 16384, 32768)
ladder = lambda name: {r["depth"]: r["tok_per_s"] for r in jl(S / name)["rows"]}

# ---- fig1: the ladder, and the replicate the MTP arm turns out to have -----
nospec, mtp, mtp2 = (ladder("splitkv-31b-stock.json"), ladder("mtp-31b-mtp.json"),
                     ladder("mtp-31b-stock45450.json"))
fig1 = {"rows": [{"ctx": d, "nospec": nospec[d], "mtp": mtp[d],
                  "delta_pct": (mtp[d] / nospec[d] - 1) * 100.0,
                  # the same configuration, measured again on 2026-08-26 as the
                  # stock baseline of the vllm#45450 comparison
                  "mtp_repeat": mtp2[d],
                  "repeat_spread_pct": abs(mtp2[d] / mtp[d] - 1) * 100.0}
                 for d in DEPTHS],
        "source": {"nospec": "splitkv-31b-stock.json", "mtp": "mtp-31b-mtp.json",
                   "repeat": "mtp-31b-stock45450.json"}}
fig1["best"] = max(r["delta_pct"] for r in fig1["rows"])
fig1["worst"] = min(r["delta_pct"] for r in fig1["rows"])
fig1["monotonic"] = all(a["delta_pct"] > b["delta_pct"]
                        for a, b in zip(fig1["rows"], fig1["rows"][1:]))
long_rep = [r["repeat_spread_pct"] for r in fig1["rows"] if r["ctx"] >= 8192]
fig1["repeat_worst_long_pct"] = max(long_rep)
fig1["repeat_1k_pct"] = fig1["rows"][0]["repeat_spread_pct"]

# ---- fig2: the control that leaves only one explanation -------------------
# Two independent constructions of the same sweep, kernel called directly.
sweeps = {}
for tag, fn in (("A", "kbench-0.json"), ("B", "kbench2-0.json")):
    sweeps[tag] = {(r["kv_len"], r["q_len"]): r["us"] for r in jl(S / fn)}
QS = (1, 2, 4, 8)
KVS = (1024, 8192, 16384, 32768)
fig2 = {"sweeps": [{"id": t, "file": fn,
                    "rows": [{"kv": kv, "us": [sweeps[t][(kv, q)] for q in QS]}
                             for kv in KVS]}
                   for t, fn in (("A", "kbench-0.json"), ("B", "kbench2-0.json"))],
        "q_lens": list(QS)}
fig2["q1_to_q2_32k_pct"] = [(sweeps[t][(32768, 2)] / sweeps[t][(32768, 1)] - 1) * 100.0
                            for t in ("A", "B")]
# the two constructions do not agree everywhere, and saying where matters more
# than the average: within each sweep the q dimension is what is being read,
# and between sweeps the absolute level is not.
fig2["between_constructions_pct"] = [
    {"kv": kv,
     "q1": abs(sweeps["B"][(kv, 1)] / sweeps["A"][(kv, 1)] - 1) * 100.0,
     "q2plus_max": max(abs(sweeps["B"][(kv, q)] / sweeps["A"][(kv, q)] - 1) * 100.0
                       for q in (2, 4, 8)),
     "within_A": (sweeps["A"][(kv, 2)] / sweeps["A"][(kv, 1)] - 1) * 100.0,
     "within_B": (sweeps["B"][(kv, 2)] / sweeps["B"][(kv, 1)] - 1) * 100.0}
    for kv in KVS]
fig2["constructions_agree_at_32k_pct"] = [
    r["q1"] for r in fig2["between_constructions_pct"] if r["kv"] == 32768][0]
fig2["worst_within_disagreement_pct"] = max(
    abs(r["within_A"] - r["within_B"]) for r in fig2["between_constructions_pct"])
fig2["worst_across_q_32k_pct"] = [
    (max(sweeps[t][(32768, q)] for q in QS) / min(sweeps[t][(32768, q)] for q in QS) - 1)
    * 100.0 for t in ("A", "B")]
# the launch grids the gate chooses between, from the document
fig2["grids"] = {"2d": [1, 8], "3d": [1, 8, 16]}
fig2["workgroups"] = {"2d": 1 * 8, "3d": 1 * 8 * 16}

# ---- fig3: the other vendor ----------------------------------------------
mat = jl(A / "gemma4-mtp-backend-matrix.json")
pct = lambda a, b: (a / b - 1) * 100.0
cells = []
for depth in ("30000", "50000"):
    for backend, label in (("triton_forced", "TRITON_ATTN, forced"),
                           ("flashinfer_explicit", "FLASHINFER, explicit"),
                           ("auto_selector_47547", "auto, mixed")):
        c = mat["decode_tok_s"][depth].get(backend)
        if not c:
            continue
        cells.append({"ctx": int(depth), "backend": backend, "label": label,
                      "mtp": c["mtp"], "nospec": c["nospec"],
                      "delta_pct": pct(c["mtp"], c["nospec"]),
                      "session": c.get("session") or
                                 f'{c.get("mtp_session")}/{c.get("nospec_session")}'})
fig3 = {"cells": cells, "machine": mat["machine"]["gpu"], "vllm": mat["machine"]["vllm"],
        "measured": "2026-08-26",
        "rocm_reference_pct": mat["readings"]["mtp_delta_on_triton_pct"]["rocm_32k_reference"],
        "sessions_caveat": mat["sessions_caveat"],
        "runs_per_cell": 1}

# ---- fig4: what admitting speculation back into the 3D path is worth ------
stock, ported = ladder("mtp-31b-stock45450.json"), ladder("mtp-31b-p45450.json")
forced = jl(S / "mtp32k-spec3d.json")
fig4 = {"rows": [{"ctx": d, "stock": stock[d], "ported": ported[d],
                  "ratio": ported[d] / stock[d]} for d in DEPTHS],
        "hand_forced_32k": forced["tok_per_s"],
        "hand_vs_ported_pct": abs(forced["tok_per_s"] / ported[32768] - 1) * 100.0,
        "nospec_32k": nospec[32768]}
fig4["net_positive_everywhere"] = all(ported[d] > nospec[d] for d in DEPTHS)
kc = jl(S / "kcorrect-45450.json")
fig4["correctness"] = {
    "cases": len(kc),
    "deterministic": sum(1 for c in kc if c["det2"] and c["det3"]),
    "wrote_3d": sum(1 for c in kc if c["segm_touched"]),
    "max_abs_diff": max(c["max_abs_diff"] for c in kc),
    "bf16_ulp_at_1": 2 ** -8}

# ---- fig4's second layer: the same A/B, five times, on two vendors ---------
# The four points above are one model on one machine at k=1. The 2026-08-29
# campaign ran the same comparison at eleven rungs and two rounds a cell, five
# times, and the five do not agree -- which is the finding. Where the Triton
# kernel this patch edits is on the path, the patched arm wins by a widening
# margin. Where the engine routes the model somewhere else, the two arms are
# the same measurement twice. The probe says which case a reader is in before
# any of it is measured.
import collections as _c, statistics as _st, json as _j, re as _re2

def _ladder(path, cfg):
    by = _c.defaultdict(list)
    for line in open(path):
        r = _j.loads(line)
        if r.get("kind") == "decode" and r.get("decode_tps") and r["cfg"] == cfg:
            by[r["target"]].append(r["decode_tps"])
    return {t: _st.mean(v) for t, v in by.items()}

_RES = R / "benchmarks/campaign-2026-08-29/results.jsonl"
_AES = R / "benchmarks/cuda-a100/campaign-2026-08-29/results.jsonl"
_PROV = _j.load(open(R / "benchmarks/campaign-2026-08-29/provenance.json"))["arms"]

def _probe_a100(name):
    f = R / f"benchmarks/cuda-a100/campaign-2026-08-29/logs/serve-{name}.log"
    return open(f, errors="replace").read().count("PROBE_3D_SPEC_ACTIVE") if f.exists() else None

PAIRS = [
    ("gemma-4-31B-it", "A100", "TRITON_ATTN", _AES, "A100-G31-mtp",
     "A100-G31-mtp-p45450", _probe_a100("A100-G31-mtp-p45450")),
    ("gemma-4-26B-A4B", "A100", "TRITON_ATTN", _AES, "A100-G26A4B-mtp",
     "A100-G26A4B-mtp-p45450", _probe_a100("A100-G26A4B-mtp-p45450")),
    ("Qwen3.8-27B", "2x RX 7900 XT", "TRITON_ATTN", _RES, "Q38-mtp-triton-tp2",
     "Q38-mtp-triton-p45450-tp2",
     _PROV["Q38-mtp-triton-p45450-tp2"]["probe_3d_spec_active"]),
    ("Qwen3.8-27B", "2x RX 7900 XT", "ROCM_ATTN", _RES, "Q38-mtp-tp2",
     "Q38-mtp-p45450-tp2", _PROV["Q38-mtp-p45450-tp2"]["probe_3d_spec_active"]),
    ("Qwen3.8-27B", "A100", "FLASH_ATTN", _AES, "A100-Q38-mtp",
     "A100-Q38-mtp-p45450", _probe_a100("A100-Q38-mtp-p45450")),
]

_pairs = []
for model, mach, backend, src, a, b, probe in PAIRS:
    st, po = _ladder(src, a), _ladder(src, b)
    rungs = sorted(set(st) & set(po))
    rows = [{"ctx": t, "stock": round(st[t], 2), "ported": round(po[t], 2),
             "ratio": po[t] / st[t]} for t in rungs]
    deltas = [(po[t] / st[t] - 1) * 100.0 for t in rungs]
    _pairs.append({
        "model": model, "machine": mach, "attn_backend": backend,
        "probe": probe, "rows": rows,
        "ratio_at_deepest": rows[-1]["ratio"],
        "mean_delta_pct": _st.mean(deltas),
        "worst_delta_pct": max(deltas, key=abs),
        # "acted" is the probe's claim, not a threshold on the numbers
        "acted": bool(probe)})

fig4["campaign"] = {
    "date": "2026-08-29", "k": 3, "harness": "campaign-server",
    "runs_per_cell": 2, "pairs": _pairs,
    "acted": [p["model"] + " / " + p["attn_backend"] for p in _pairs if p["acted"]],
    "inert": [p["model"] + " / " + p["attn_backend"] for p in _pairs if not p["acted"]],
    "source": {"radeons": "benchmarks/campaign-2026-08-29/results.jsonl",
               "a100": "benchmarks/cuda-a100/campaign-2026-08-29/results.jsonl"}}
# the whole point: the probe and the outcome agree in all five
fig4["campaign"]["probe_predicts"] = all(
    (p["probe"] > 0) == (abs(p["mean_delta_pct"]) > 5.0) for p in _pairs)
fig4["campaign"]["inert_worst_pct"] = max(
    (abs(p["worst_delta_pct"]) for p in _pairs if not p["acted"]), default=None)

# ---- the profiler summary, which is derived from traces not in the repo ---
tr = jl(S / "trace-unified-attention.json")
prof = {"reproducible_from_repo": False,
        "source": "benchmarks/speculative-decoding/trace-unified-attention.json",
        "note": tr["what"],
        "runs": {k: {kk: v[kk] for kk in
                     ("calls", "median_us", "p75_us", "max_us", "mean_us",
                      "hip_graph_launch")}
                 for k, v in tr["runs"].items()}}
n, m = prof["runs"]["no-speculation"], prof["runs"]["mtp"]
prof["ratios"] = {k: m[k] / n[k] for k in ("median_us", "p75_us", "max_us", "mean_us")}

out = {"_what": "Every figure in speculative-decoding-net-loss.html. Derived from "
                "benchmarks/speculative-decoding/ and benchmarks/cuda-a100/ by "
                "site/src/genfig-spec.py; edit the data, not this file.",
       "fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "profile": prof}
json.dump(out, open(pathlib.Path(__file__).parent / "figures-spec.json", "w"),
          ensure_ascii=False, indent=1)
print("fig1:", [(r["ctx"], round(r["delta_pct"], 1)) for r in fig1["rows"]],
      "monotonic", fig1["monotonic"])
print(f'fig1 repeat: 1K {fig1["repeat_1k_pct"]:.2f}%, longest three within '
      f'{fig1["repeat_worst_long_pct"]:.2f}%')
print("fig2 q1->q2 at 32K:", [f"{x:.2f}%" for x in fig2["q1_to_q2_32k_pct"]],
      "| spread over all q:", [f"{x:.2f}%" for x in fig2["worst_across_q_32k_pct"]])
print("fig3:", [(c["ctx"], c["backend"], round(c["delta_pct"], 1)) for c in cells])
print("fig4:", [(r["ctx"], round(r["ratio"], 2)) for r in fig4["rows"]],
      "net positive everywhere:", fig4["net_positive_everywhere"])
print(f'fig4 hand-forced {fig4["hand_forced_32k"]} vs ported '
      f'{ported[32768]} ({fig4["hand_vs_ported_pct"]:.1f}%)')
print("profile ratios:", {k: round(v, 1) for k, v in prof["ratios"].items()})
print("bytes:", len(json.dumps(out)))
