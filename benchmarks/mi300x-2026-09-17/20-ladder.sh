#!/bin/bash
# Items 3 and 5a: the sixteen-rung ladder, six models, two rounds, batch 1.
# runner_rocm.py unchanged — it asserts the stack, pre-flights telemetry and
# the checkpoints, discards a warm-up request per serve and checkpoints itself,
# so a dropped connection costs the current rung.
set -euo pipefail
: "${BENCH_WORK:=/work}" "${BENCH_MODELS:=/models}" "${BENCH_MACHINE:=MI300X}"
: "${BENCH_VLLM_EXPECT:?set it to the version this image ships, e.g. 0.28 — the run must refuse to start on a mismatch}"
cd "$BENCH_WORK"
export BENCH_WORK BENCH_MODELS BENCH_MACHINE BENCH_VLLM_EXPECT
export BENCH_VOLUME="${BENCH_VOLUME:-$BENCH_WORK/volume.json}"
# telemetry.py lives in the repository clone, not beside the runner
export PYTHONPATH="${PYTHONPATH:-}${PYTHONPATH:+:}$BENCH_WORK/repo/benchmarks"
# an empty positional argument selects no configuration at all, which is how
# the first attempt "finished" in three seconds
nohup python3 runner_rocm.py ${1:+"$1"} > "$BENCH_WORK/ladder.log" 2>&1 &
echo "ladder started (pid $!). Watch: tail -f $BENCH_WORK/PROGRESS.txt"
