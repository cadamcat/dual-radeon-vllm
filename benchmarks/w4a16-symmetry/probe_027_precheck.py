"""Before trusting anything from the 0.27.0 image: does it serve gfx1100 at all?

The tag is `rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0`, with no
`_rdna` suffix -- the 7.14 line split into `_rdna` and `_cdna` builds, so a
combined tag may or may not carry the RDNA3 kernels. This checks that before
any measurement, because a silent fallback would look like a Hybrid-vs-RDNA3
result and be nothing of the sort.

Nothing here loads a model. Cheap enough to run while services are up.
"""

import json
import sys

import torch

OUT = sys.argv[1] if len(sys.argv) > 1 else "/work/precheck-027.json"

out = {}


def note(k, v):
    out[k] = v
    print(f"  {k:<42} {v}")


import vllm
from vllm.platforms import current_platform

note("vllm_version", vllm.__version__)
note("torch_version", torch.__version__)
note("hip_version", getattr(torch.version, "hip", None))
note("device_count", torch.cuda.device_count())
try:
    note("device_name", torch.cuda.get_device_name(0))
except Exception as exc:
    note("device_name", f"ERROR {exc}")

import vllm.platforms.rocm as R

for fn in ("on_gfx1100", "on_gfx1x", "on_gfx1151", "on_gfx12x"):
    try:
        note(fn, getattr(R, fn)() if hasattr(R, fn) else "ABSENT")
    except Exception as exc:
        note(fn, f"ERROR {exc}")

# the two C++ ops the two kernels need
for op in ("gptq_gemm_rdna3", "wvSplitK_int4_g"):
    present = hasattr(torch.ops, "_rocm_C") and hasattr(torch.ops._rocm_C, op)
    if not present:
        import vllm._custom_ops as ops
        present = hasattr(ops, op)
    note(f"op:{op}", present)

# the registry, and what it does with our real AWQ parameters
from vllm.model_executor.kernels.linear import (
    _POSSIBLE_KERNELS,
    choose_mp_linear_kernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearLayerConfig,
)
from vllm.scalar_type import scalar_types

kernels = _POSSIBLE_KERNELS[current_platform._enum]
note("registry", [k.__name__ for k in kernels])
note("has_rdna3", any("RDNA3W4A16" in k.__name__ for k in kernels))
note("has_hybrid", any("Hybrid" in k.__name__ for k in kernels))

cfg = MPLinearLayerConfig(
    full_weight_shape=(17408, 5120),
    partition_weight_shape=(17408, 2560),   # TP=2
    weight_type=scalar_types.uint4,
    act_type=torch.bfloat16,
    group_size=32,
    zero_points=True,
    has_g_idx=False,
)
verdicts = []
for k in kernels:
    try:
        ok, why = k.can_implement(cfg)
    except Exception as exc:
        ok, why = False, f"{type(exc).__name__}: {exc}"
    verdicts.append({"kernel": k.__name__, "ok": bool(ok), "reason": why})
    print(f"    {'YES' if ok else 'no ':<4} {k.__name__:<32} {'' if ok else (why or '')[:66]}")
out["awq_verdicts"] = verdicts
note("awq_selected", choose_mp_linear_kernel(cfg).__name__)

# and the symmetric config, for the same reason
cfg_sym = MPLinearLayerConfig(
    full_weight_shape=(17408, 5120),
    partition_weight_shape=(17408, 2560),
    weight_type=scalar_types.uint4b8,
    act_type=torch.bfloat16,
    group_size=128,
    zero_points=False,
    has_g_idx=False,
)
note("sym_selected", choose_mp_linear_kernel(cfg_sym).__name__)

with open(OUT, "w") as fh:
    json.dump(out, fh, indent=1)

ok = (out["has_rdna3"] and out["has_hybrid"]
      and out["on_gfx1100"] is True
      and out["awq_selected"] == "RDNAHybridW4A16LinearKernel")
print(f"PRECHECK_OK={ok}")
print("PRECHECK_DONE")
