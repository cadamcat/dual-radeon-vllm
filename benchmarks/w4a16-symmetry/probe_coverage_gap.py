"""Where does the RDNA3 patch actually buy something? Ask the registry.

On the configurations Hybrid accepts, patching RDNA3 to take `uint4` is a loss:
it outranks Hybrid and is 5.9% slower (see w4a16-027.jsonl). But Hybrid refuses
two things RDNA3 does not, so there is a region where the patch is the
difference between the native kernel and the Triton fallback rather than
between two native kernels:

    Hybrid                          RDNA3
    SUPPORTED_GROUP_SIZES           group_size > 0 and divides K
      = [32, 64, 128]
    has_g_idx -> reject             act-order supported unless the input dim
                                    is TP-partitioned

This sweeps asymmetric (uint4 + zero points) configurations over group size and
act-order and records, for each, what the registry picks with the patch and
without it. `SUPPORTED_QUANT_TYPES` is a class attribute, so both answers can
be had in one process without rebuilding anything.

Nothing is executed here -- this is `can_implement` only, so it costs no GPU
time and says nothing about speed. It only measures how large the region is.

    python3 probe_coverage_gap.py [out.json]
"""

import json
import sys

import torch

OUT = sys.argv[1] if len(sys.argv) > 1 else "/work/coverage-gap.json"

from vllm.model_executor.kernels.linear import (
    _POSSIBLE_KERNELS,
    choose_mp_linear_kernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.mixed_precision.rdna3_w4a16 import (
    RDNA3W4A16LinearKernel,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

STOCK_TYPES = list(RDNA3W4A16LinearKernel.SUPPORTED_QUANT_TYPES)
PATCHED_TYPES = STOCK_TYPES + [scalar_types.uint4]

# K and N from a real layer of the AWQ checkpoint, so divisibility is realistic
K, N = 17408, 5120
GROUP_SIZES = [16, 32, 64, 128, 256, 512]
ACT_ORDER = [False, True]


def cfg_for(gs, g_idx, tp=1):
    return MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K // tp, N),
        weight_type=scalar_types.uint4,          # asymmetric
        act_type=torch.bfloat16,
        group_size=gs,
        zero_points=True,
        has_g_idx=g_idx,
    )


def verdicts(cfg):
    out = {}
    for k in _POSSIBLE_KERNELS[current_platform._enum]:
        try:
            ok, why = k.can_implement(cfg)
        except Exception as exc:
            ok, why = False, f"{type(exc).__name__}: {exc}"
        out[k.__name__] = {"ok": bool(ok), "reason": None if ok else (why or "")[:80]}
    return out


def main():
    rows = []
    print(f"asymmetric (uint4 + zero points), K={K} N={N}, TP=1\n")
    hdr = f"{'group':>6} {'act_order':>9} | {'stock picks':<28} | {'patched picks':<28} | verdict"
    print(hdr); print("-" * len(hdr))
    for gs in GROUP_SIZES:
        for g_idx in ACT_ORDER:
            if K % gs:
                continue
            cfg = cfg_for(gs, g_idx)

            def pick():
                try:
                    return choose_mp_linear_kernel(cfg).__name__
                except ValueError:
                    return "(none: layer would not load)"

            RDNA3W4A16LinearKernel.SUPPORTED_QUANT_TYPES = STOCK_TYPES
            v_stock = verdicts(cfg)
            stock_pick = pick()

            RDNA3W4A16LinearKernel.SUPPORTED_QUANT_TYPES = PATCHED_TYPES
            v_patched = verdicts(cfg)
            patched_pick = pick()

            RDNA3W4A16LinearKernel.SUPPORTED_QUANT_TYPES = STOCK_TYPES

            hybrid_ok = v_stock.get("RDNAHybridW4A16LinearKernel", {}).get("ok")
            rdna3_ok = v_patched.get("RDNA3W4A16LinearKernel", {}).get("ok")
            triton_ok = v_stock.get("TritonW4A16LinearKernel", {}).get("ok")
            if not hybrid_ok and rdna3_ok and triton_ok:
                verdict = "GAP: patch replaces Triton"
            elif not hybrid_ok and rdna3_ok and not triton_ok:
                verdict = "GAP+: patch makes the layer loadable at all"
            elif hybrid_ok and rdna3_ok:
                verdict = "overlap: patch displaces Hybrid"
            elif not rdna3_ok:
                verdict = "patch does not apply"
            else:
                verdict = "-"
            rows.append({"group_size": gs, "has_g_idx": g_idx,
                         "stock_pick": stock_pick, "patched_pick": patched_pick,
                         "hybrid_accepts": bool(hybrid_ok),
                         "triton_accepts": bool(triton_ok),
                         "rdna3_patched_accepts": bool(rdna3_ok),
                         "hybrid_reason": v_stock.get(
                             "RDNAHybridW4A16LinearKernel", {}).get("reason"),
                         "region": verdict})
            print(f"{gs:>6} {str(g_idx):>9} | {stock_pick:<28} | {patched_pick:<28} | {verdict}")

    with open(OUT, "w") as fh:
        json.dump(rows, fh, indent=1)
    gap = [r for r in rows if r["region"].startswith("GAP")]
    unloadable = [r for r in rows if r["region"].startswith("GAP+")]
    overlap = [r for r in rows if r["region"].startswith("overlap")]
    print()
    print(f"  configurations where the patch replaces Triton  : {len(gap)}")
    print(f"  configurations where it displaces Hybrid        : {len(overlap)}")
    print(f"  of those, ones nothing can serve today       : {len(unloadable)}")
    for r in gap:
        print(f"    gap: group_size={r['group_size']} act_order={r['has_g_idx']}"
              f" triton={r['triton_accepts']}  (hybrid: {r['hybrid_reason']})")
    print("COVERAGE_DONE")


if __name__ == "__main__":
    main()
