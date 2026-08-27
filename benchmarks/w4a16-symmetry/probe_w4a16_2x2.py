"""Does symmetry or group size decide the W4A16 kernel? Ask the registry the 2x2.

The A/B in `probe_w4a16_ab.py` measures two real checkpoints of the same model,
and they differ in two ways at once: the asymmetric one is group_size 32, the
symmetric one is group_size 128. That confound cannot be removed by measurement
without a checkpoint that does not exist, but it can be removed for the question
the A/B is actually about — *which kernel gets selected* — because selection is
a pure function of the quantization parameters and the registry will answer it
with no model loaded and no weights in memory.

So this asks all four corners:

    symmetric  group_size   what it is
    false      32           our AWQ checkpoint, arm A of the A/B
    true       128          the RedHatAI checkpoint, arm B of the A/B
    true       32           counterfactual: symmetry at arm A's group size
    false      128          counterfactual: asymmetry at arm B's group size

If the two counterfactual rows follow their `symmetric` flag rather than their
group size, then group size does not decide kernel selection, and the A/B's
group-size confound does not reach the conclusion about *why* the native kernel
is skipped. It still reaches accuracy and memory traffic, which is why the
write-up discloses it rather than dismissing it.

Ground truth for the two real rows is not this script — it is
`kernels-<arm>-<ctx>.txt`, recorded from inside the TP workers while the model
actually ran. This script is the counterfactual instrument, and its two real
rows exist so they can be checked against those records.

It then asks the same question of every quantized checkpoint the benchmark
campaign actually used, reading each one's real `quantization_config` off disk
rather than restating it here. That matters for one specific confound: three of
the campaign's fastest models are group_size 32, the same group size as the
asymmetric checkpoint, so group size cannot be what separates them.

The campaign rows are evaluated at a representative weight shape, so the
shape-dependent conditions (group divides K, output features a multiple of the
pack factor) are answered for that shape and not for each model's real layers.
The type gate, which is the one under test, does not depend on shape.

Superset of the four cases in `probe_w4a16.py`; writes a separate file so that
the earlier artifact is not overwritten.
"""
import glob
import json
import os

import torch
from vllm.model_executor.kernels.linear import _POSSIBLE_KERNELS
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearLayerConfig,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

# how compressed_tensors_wNa16.py maps the checkpoint's `symmetric` flag.
# Confirmed at runtime, not only here: the worker-side records show the real
# layers of the asymmetric checkpoint arriving as uint4 with zero_points set.
SYM_MAP = {4: scalar_types.uint4b8, 8: scalar_types.uint8b128}
ZP_MAP = {4: scalar_types.uint4, 8: scalar_types.uint8}

CASES = [
    # name, num_bits, group_size, symmetric, has_g_idx, corner
    ("Qwen3.8-27B-AWQ-INT4  (ours, A/B arm A)", 4, 32, False, False, "asym g32"),
    ("Qwen3.8-27B-INT4  (RedHatAI, A/B arm B)", 4, 128, True, False, "sym g128"),
    ("counterfactual: symmetric at arm A's group size", 4, 32, True, False, "sym g32"),
    ("counterfactual: asymmetric at arm B's group size", 4, 128, False, False, "asym g128"),
    ("gemma-3-27b-it-w4a16  (the fast 27B here)", 4, 128, True, False, "sym g128"),
    ("Qwen/Qwen3.5-27B-GPTQ-Int4  sym=True desc_act=False", 4, 128, True, False, "sym g128"),
    ("...same but if the GPTQ path sets has_g_idx", 4, 128, True, True, "sym g128"),
]


def campaign_cases():
    """every checkpoint on this box, with its real quantization parameters."""
    rows = []
    for d in sorted(glob.glob("/data/incoming/*/")):
        cfg = os.path.join(d, "config.json")
        if not os.path.exists(cfg):
            continue
        try:
            c = json.load(open(cfg))
        except Exception:
            continue
        q = c.get("quantization_config")
        if not q:
            continue
        groups = q.get("config_groups") or {}
        if not groups:
            continue
        w = next(iter(groups.values())).get("weights", {})
        if w.get("num_bits") != 4 or w.get("symmetric") is None:
            continue
        rows.append((os.path.basename(d.rstrip("/")), 4, w["group_size"],
                     bool(w["symmetric"]), False,
                     ("sym" if w["symmetric"] else "asym") + f" g{w['group_size']}"))
    return rows


def verdict_for(kernels, bits, gs, sym, gidx):
    wt = (SYM_MAP if sym else ZP_MAP)[bits]
    cfg = MPLinearLayerConfig(
        full_weight_shape=(4096, 4096),
        partition_weight_shape=(4096, 2048),   # TP=2
        weight_type=wt,
        act_type=torch.bfloat16,
        group_size=gs,
        zero_points=not sym,
        has_g_idx=gidx,
    )
    chosen, rows = None, []
    for k in kernels:
        try:
            ok, why = k.can_implement(cfg)
        except Exception as exc:
            ok, why = False, f"{type(exc).__name__}: {exc}"
        rows.append((k.__name__, bool(ok), why))
        if ok and chosen is None:
            chosen = k.__name__
    return wt, chosen, rows


def main():
    cc = current_platform.get_device_capability()
    cc = cc[0] * 10 + cc[1] if cc else None
    print(f"platform={current_platform._enum} capability={cc}", flush=True)
    kernels = _POSSIBLE_KERNELS[current_platform._enum]
    print(f"kernels registered for this platform: {[k.__name__ for k in kernels]}\n",
          flush=True)

    out = []
    for name, bits, gs, sym, gidx, corner in CASES:
        wt, chosen, rows = verdict_for(kernels, bits, gs, sym, gidx)
        print(f"=== {name}")
        print(f"    num_bits={bits} group_size={gs} symmetric={sym} "
              f"-> weight_type={wt} zero_points={not sym} has_g_idx={gidx}")
        for kname, ok, why in rows:
            mark = "SELECTED" if (ok and chosen == kname) else ("ok" if ok else "no")
            print(f"      {mark:<9} {kname:<28} {'' if ok else (why or '')[:96]}")
        print(f"    -> first kernel that can implement: {chosen}\n", flush=True)
        out.append({"case": name, "corner": corner, "num_bits": bits, "group_size": gs,
                    "symmetric": sym, "has_g_idx": gidx, "weight_type": str(wt),
                    "zero_points": not sym, "chosen": chosen,
                    "verdicts": [{"kernel": a, "ok": b, "reason": c} for a, b, c in rows]})
    with open("/work/w4a16-selection-2x2.json", "w") as fh:
        json.dump(out, fh, indent=1)

    # state the 2x2 verdict in the log too, so the run is readable without jq
    by = {r["corner"]: r["chosen"] for r in out}
    print("2x2 by corner:")
    for corner in ("asym g32", "sym g32", "asym g128", "sym g128"):
        print(f"    {corner:<10} -> {by.get(corner)}")
    native = "RDNA3W4A16LinearKernel"
    tracks_symmetry = (by.get("sym g32") == native and by.get("sym g128") == native
                       and by.get("asym g32") != native and by.get("asym g128") != native)
    print(f"SELECTION_TRACKS_SYMMETRY_NOT_GROUP_SIZE={tracks_symmetry}", flush=True)

    # every quantized checkpoint the campaign used, from its own config.json
    print("\ncampaign checkpoints on this box:")
    camp = []
    for name, bits, gs, sym, gidx, corner in campaign_cases():
        wt, chosen, rows = verdict_for(kernels, bits, gs, sym, gidx)
        print(f"    {name:<34} sym={str(sym):<5} g{gs:<4} -> {chosen}")
        camp.append({"checkpoint": name, "num_bits": bits, "group_size": gs,
                     "symmetric": sym, "weight_type": str(wt), "chosen": chosen,
                     "verdicts": [{"kernel": a, "ok": b, "reason": c} for a, b, c in rows]})
    with open("/work/w4a16-campaign-selection.json", "w") as fh:
        json.dump(camp, fh, indent=1)
    g32 = sorted({c["chosen"] for c in camp if c["group_size"] == 32})
    print(f"    group_size 32 checkpoints select: {g32}")
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
