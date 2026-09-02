#!/usr/bin/env python3
"""derive.py — what the measured all-reduce implies for a served decode step.

Reads `results.jsonl` (this directory) and `../decode.jsonl`, and prints three
things, in increasing order of how much they assume:

  1. **Measured.** One all-reduce at the shapes each model reduces, in three
     timing modes. Nothing derived.
  2. **Arithmetic.** Per decode step = 2 x layers x one collective. This assumes
     every decoder layer issues exactly two `RowParallelLinear` reductions and
     that each costs what an isolated one does.
  3. **A cross-check with an assumption in it.** If TP=2 halved the bytes each
     card reads and added nothing but the collectives, the step would be
     `tp1/2 + all_reduce`. Compared against the measured TP=2 step, the residual
     is what neither perfect halving nor the collective explains.

Step 3 is the one that pays for the campaign, and its assumption is load-bearing:
"halves perfectly" is true only where decode is purely weight-read bandwidth.
It is reported as a residual, never as an explanation.

    python3 derive.py            # table
    python3 derive.py --json     # the same numbers, machine-readable
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

#: layers and hidden size read from each checkpoint's config.json on the Radeon
#: box on 2026-09-02, and the decode.jsonl `model` string each corresponds to.
#: The withdrawn claim in docs/benchmarks.md said "36 layers x 2 all-reduces"
#: for a hidden-3840 model; 3840 is the 12B, which has 48. 36 is the 8B's.
MODELS = [
    # key,     decode.jsonl model, hidden, layers, tp1 cfg,      tp2 cfg
    ("8B", "Qwen3-8B", 4096, 36, "B-8B-tp1", "B-8B-tp2"),
    ("12B", "gemma-4-12B-it", 3840, 48, "A-12B-tp1", "A-12B-tp2"),
    # the MoE has no TP=1 arm in the 2026-08-24 sitting; its only one is
    # `E26-tp1-u95`, 2026-08-30, util 0.95 and on the narrowed link. Decode is
    # not measurably affected by the link (<1% of a step) so the subtraction is
    # still worth printing, but it crosses two sittings and is flagged as such.
    ("26B-A4B", "gemma-4-26B-A4B", 2816, 30, "E26-tp1-u95", "E-26B-tp2",
     "2026-08-30"),
    ("31B", "gemma-4-31B-it", 5376, 60, None, "C-31B-tp2"),
    ("27B", "Qwen3.8-27B", 5120, 64, None, "D8-27B-tp2"),
]

#: the sitting the cross-check reads. 2026-08-24 is a full x16 sitting with both
#: arms of the 8B and the 12B measured in one campaign on one stack, which is
#: what makes a TP=1 against TP=2 subtraction legitimate at all.
SITTING = "2026-08-24"
CTX = 500


def load():
    ar = [json.loads(l) for l in open(os.path.join(HERE, "results.jsonl"))
          if '"allreduce"' in l]
    dec = [json.loads(l) for l in open(os.path.join(HERE, "..", "decode.jsonl"))]
    return ar, dec


def at(ar, hidden, ntok=1):
    m = [r for r in ar if r["hidden"] == hidden and r["ntok"] == ntok]
    if not m:
        raise KeyError(f"no all-reduce row for hidden={hidden} ntok={ntok}")
    return m[0]


def step_ms(dec, model, cfg, date=SITTING, ctx=CTX):
    if cfg is None:
        return None
    m = [r for r in dec if r["machine"] == "RX 7900 XT" and r["model"] == model
         and r["cfg"] == cfg and r["date"] == date and r["ctx"] == ctx]
    return 1000.0 / m[0]["decode_tok_s"] if m else None


def rows():
    ar, dec = load()
    out = []
    for m in MODELS:
        key, model, hidden, layers, c1, c2 = m[:6]
        d1 = m[6] if len(m) > 6 else SITTING
        r = at(ar, hidden)
        n_coll = 2 * layers
        t1, t2 = step_ms(dec, model, c1, date=d1), step_ms(dec, model, c2)
        d = {"key": key, "model": model, "hidden": hidden, "layers": layers,
             "collectives_per_step": n_coll,
             "us_graph": r["t_graph_us"], "us_stream": r["t_stream_us"],
             "us_sync_median": r["t_sync_us_median"],
             "ms_per_step_graph": n_coll * r["t_graph_us"] / 1000.0,
             "ms_per_step_stream": n_coll * r["t_stream_us"] / 1000.0,
             "tp1_ms": t1, "tp2_ms": t2, "tp1_sitting": d1,
             "same_sitting": d1 == SITTING}
        if t2:
            d["ar_pct_of_tp2_step"] = d["ms_per_step_graph"] / t2 * 100
        if t1 and t2:
            d["speedup"] = t1 / t2
            d["predicted_tp2_ms"] = t1 / 2 + d["ms_per_step_graph"]
            d["residual_ms"] = t2 - d["predicted_tp2_ms"]
            d["residual_pct_of_step"] = d["residual_ms"] / t2 * 100
            gap = t2 - t1 / 2                       # everything TP=2 did not win
            d["ar_share_of_gap_pct"] = d["ms_per_step_graph"] / gap * 100
        out.append(d)
    return out


def main():
    rs = rows()
    if "--json" in sys.argv:
        print(json.dumps(rs, indent=1))
        return
    print("MEASURED — one all-reduce, batch-1 decode shape [1, hidden], bf16, "
          "RCCL 2.27.7, PCIe 3.0 x16/x16\n")
    print(f"{'model':>8} {'hidden':>7} {'graph us':>9} {'stream us':>10} "
          f"{'sync us':>9}")
    for d in rs:
        print(f"{d['key']:>8} {d['hidden']:>7} {d['us_graph']:>9.1f} "
              f"{d['us_stream']:>10.1f} {d['us_sync_median']:>9.1f}")
    print("\nARITHMETIC — 2 x layers x one graph-replayed collective\n")
    print(f"{'model':>8} {'layers':>7} {'colls':>6} {'ms/step':>8} "
          f"{'tp2 step ms':>12} {'% of step':>10}")
    for d in rs:
        pc = d.get("ar_pct_of_tp2_step")
        print(f"{d['key']:>8} {d['layers']:>7} {d['collectives_per_step']:>6} "
              f"{d['ms_per_step_graph']:>8.2f} "
              f"{(f'{d['tp2_ms']:12.2f}' if d['tp2_ms'] else '           -')} "
              f"{(f'{pc:10.1f}' if pc else '         -')}")
    print(f"\nCROSS-CHECK — {SITTING}, ctx {CTX}, assuming TP=2 halves the "
          "bytes each card reads\n")
    print(f"{'model':>8} {'tp1 ms':>7} {'tp2 ms':>7} {'speedup':>8} "
          f"{'predicted':>10} {'residual':>9} {'AR share of gap':>16}")
    for d in rs:
        if "speedup" not in d:
            continue
        print(f"{d['key']:>8} {d['tp1_ms']:>7.2f} {d['tp2_ms']:>7.2f} "
              f"{d['speedup']:>8.3f} {d['predicted_tp2_ms']:>10.2f} "
              f"{d['residual_ms']:>9.2f} {d['ar_share_of_gap_pct']:>15.1f}%"
              + ("" if d["same_sitting"] else f"   [tp1 from {d['tp1_sitting']}]"))
    # The point of the whole campaign, in one comparison that assumes nothing:
    # two models on one box, one library, collectives that cost within 0.6 ms of
    # each other per step -- and the second card is worth 1.70x to one and 1.19x
    # to the other. Whatever limits the second one, it is not the wire.
    a = {d["key"]: d for d in rs}
    if "speedup" in a["8B"] and "speedup" in a["12B"]:
        print(f"\nCONTRAST — 8B {a['8B']['ms_per_step_graph']:.2f} ms/step of "
              f"all-reduce ({a['8B']['ar_pct_of_tp2_step']:.1f}% of its step) "
              f"and {a['8B']['speedup']:.2f}x from the second card;\n"
              f"           12B {a['12B']['ms_per_step_graph']:.2f} ms/step "
              f"({a['12B']['ar_pct_of_tp2_step']:.1f}%) and "
              f"{a['12B']['speedup']:.2f}x. Same box, same library, same "
              f"collective cost.")


if __name__ == "__main__":
    main()
