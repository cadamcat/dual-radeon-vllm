#!/usr/bin/env python3
"""Is W4A16 slow here because of the card, or because of the kernel chosen?

Both quantised models on this box decode about ten tokens a second against
105-139 on the rented CUDA cards, while bf16 Qwen3-8B matches an H200. Both
serve logs say `Using TritonW4A16LinearKernel for CompressedTensorsWNA16`, and
the image's own int4 GEMMs (`wvSplitK_int4_hf_*`, 600 of the 628 kernels that
declare a hostcall buffer) are never launched. AMD's AITER kernels are in this
image behind `VLLM_ROCM_USE_AITER`, off by default.

Two arms at three rungs, ids of their own so the rows never mix with the
ladder's: the environment is the only difference, and each arm's serve log
records which kernel it chose.

    BENCH_WORK=/work BENCH_MODELS=/models BENCH_MACHINE=MI300X \
    BENCH_VLLM_EXPECT=0.27 python3 aiter_probe.py
"""
import importlib.util, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load_runner(path=None):
    path = path or os.environ.get("BENCH_RUNNER") or os.path.join(HERE, "runner.py")
    spec = importlib.util.spec_from_file_location("runner", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = load_runner()
MODEL = os.environ.get("BENCH_AB_MODEL", "Qwen3.8-27B-AWQ-INT4")
RUNGS = [500, 8000, 32000]
ARMS = [
    ("Q38-aiter", {"VLLM_ROCM_USE_AITER": "1"}),
    ("Q38-noaiter", {"VLLM_ROCM_USE_AITER": "0"}),
]


def main():
    R.preflight_stack()
    done = R.done_keys()
    for cid, env in ARMS:
        for k, v in env.items():
            os.environ[k] = v            # the serve script inherits this process's environment
        R.log(f"AITER probe: {cid} with {env}")
        R.run_cfg(dict(id=cid, model=MODEL, tp=1, targets=RUNGS), done)
    R.log("AITER probe done")


if __name__ == "__main__":
    main()
