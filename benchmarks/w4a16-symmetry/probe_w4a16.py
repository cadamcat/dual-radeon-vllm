"""Which W4A16 linear kernel does each Qwen/gemma checkpoint actually get?

vllm#50264's profile shows `triton_w4a16_gemm_kernel` accounting for ~80 ms of
Qwen3.8-27B's 85.85 ms decode step on this box, while attention costs 5.7 ms.
vLLM has had a native gfx1100 W4A16 kernel since #41394 (merged 2026-05-29,
so it is in this container's 0.23.1.dev), and it is registered for the same
mixed-precision path compressed-tensors WNA16 uses.

This asks the registry directly, with no model loaded, why the native kernel is
or is not selected for each checkpoint's real quantization parameters.
"""
import json

import torch
from vllm.model_executor.kernels.linear import _POSSIBLE_KERNELS
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearLayerConfig,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

# how compressed_tensors_wNa16.py maps the checkpoint's `symmetric` flag
SYM_MAP = {4: scalar_types.uint4b8, 8: scalar_types.uint8b128}
ZP_MAP = {4: scalar_types.uint4, 8: scalar_types.uint8}

CASES = [
    # name, num_bits, group_size, symmetric, has_g_idx
    ("Qwen3.8-27B-AWQ-INT4  (ours, compressed-tensors)", 4, 32, False, False),
    ("gemma-3-27b-it-w4a16  (the fast 27B here)", 4, 128, True, False),
    ("Qwen/Qwen3.5-27B-GPTQ-Int4  sym=True desc_act=False", 4, 128, True, False),
    ("...same but if the GPTQ path sets has_g_idx", 4, 128, True, True),
]


def main():
    cc = current_platform.get_device_capability()
    cc = cc[0] * 10 + cc[1] if cc else None
    print(f"platform={current_platform._enum} capability={cc}", flush=True)
    kernels = _POSSIBLE_KERNELS[current_platform._enum]
    print(f"kernels registered for this platform: {[k.__name__ for k in kernels]}\n",
          flush=True)

    out = []
    for name, bits, gs, sym, gidx in CASES:
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
        print(f"=== {name}")
        print(f"    num_bits={bits} group_size={gs} symmetric={sym} "
              f"-> weight_type={wt} zero_points={not sym} has_g_idx={gidx}")
        chosen = None
        rows = []
        for k in kernels:
            try:
                ok, why = k.can_implement(cfg)
            except Exception as exc:
                ok, why = False, f"{type(exc).__name__}: {exc}"
            rows.append((k.__name__, bool(ok), why))
            if ok and chosen is None:
                chosen = k.__name__
            mark = "SELECTED" if (ok and chosen == k.__name__) else ("ok" if ok else "no")
            print(f"      {mark:<9} {k.__name__:<28} {'' if ok else (why or '')[:96]}")
        print(f"    -> first kernel that can implement: {chosen}\n", flush=True)
        out.append({"case": name, "num_bits": bits, "group_size": gs,
                    "symmetric": sym, "has_g_idx": gidx, "weight_type": str(wt),
                    "zero_points": not sym, "chosen": chosen,
                    "verdicts": [{"kernel": a, "ok": b, "reason": c} for a, b, c in rows]})
    with open("/work/w4a16-selection.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
