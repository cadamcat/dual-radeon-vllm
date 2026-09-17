#!/usr/bin/env python3
"""The attention A/B of 2026-09-07, on the other AMD architecture.

On gfx1100 the selector ranks ROCM_ATTN above TRITON_ATTN for a custom paged
attention kernel that is not instantiated at head_size 256, and the measured
consequence — `benchmarks/campaign-2026-09-07`, reported in vllm#54440 — is that
TRITON_ATTN decodes 1.48x faster at 128 000 tokens while prefilling at 0.44x.
gfx942 is the architecture that kernel was written for, so the same comparison
here says whether the ranking is right where its reason holds.

Three arms in one sitting, as that campaign ran them: ROCM_ATTN, TRITON_ATTN,
then ROCM_ATTN again at three rungs, so session drift can be bounded rather
than assumed. Each arm is its own serve; the runner's ladder, rounds, warm-up
and telemetry are unchanged, and only `--attention-backend` differs.

    BENCH_MACHINE=MI300X BENCH_WORK=/work BENCH_MODELS=/models \
    BENCH_VLLM_EXPECT=0.27 BENCH_AB_MODEL=Qwen3-8B BENCH_AB_MML=40960 \
    python3 ab_attn.py
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
# The id prefix is the ladder's for that model, so a row says which model it is
# without a join, and the arms never collide with the ladder's own ids.
TAG = os.environ.get("BENCH_AB_TAG") or ("B8" if MODEL.startswith("Qwen3-8B") else "Q38")
# Qwen3-8B carries a position cap of 40960; passing it here saves the runner a
# start, a refusal and a retry at $1.99/h.
MML = int(os.environ["BENCH_AB_MML"]) if os.environ.get("BENCH_AB_MML") else None
DRIFT = [500, 8000, 32000]


def arm(suffix, backend, **kw):
    cfg = dict(id=f"{TAG}-{suffix}", model=MODEL, tp=1,
               extra=f"--attention-backend {backend}", **kw)
    return {k: v for k, v in cfg.items() if v is not None}


ARMS = [
    arm("rocm", "ROCM_ATTN", mml=MML),
    arm("triton", "TRITON_ATTN", mml=MML),
    # the drift control: arm A again, after arm B, at three rungs
    arm("rocm-b", "ROCM_ATTN", mml=MML, targets=DRIFT),
]


def main(which=None):
    R.preflight_stack()
    done = R.done_keys()
    for cfg in ARMS:
        if which and cfg["id"] not in which:
            continue
        R.log(f"attention A/B: {cfg['id']} ({cfg['extra']})")
        R.run_cfg(cfg, done)
    R.log("attention A/B done")


if __name__ == "__main__":
    want = (sys.argv[1].split(",") if len(sys.argv) > 1
            else (os.environ["BENCH_CFGS"].split(",") if os.environ.get("BENCH_CFGS") else None))
    main(want)
