#!/bin/bash
# Items 4 and 5b: batch 1/2/4/8 x five repeats at three depths, and eight
# greedy repeats for two models. One serve per model, reusing the max-model-len
# and max-num-seqs the ladder already found for it.
set -euo pipefail
: "${BENCH_WORK:=/work}" "${BENCH_MODELS:=/models}" "${BENCH_MACHINE:=MI300X}"
: "${BENCH_VLLM_EXPECT:?set it as for the ladder}"
cd "$BENCH_WORK"
export BENCH_WORK BENCH_MODELS BENCH_MACHINE BENCH_VLLM_EXPECT
export BENCH_VOLUME="${BENCH_VOLUME:-$BENCH_WORK/volume.json}"
# telemetry.py lives in the repository clone, not beside the runner
export PYTHONPATH="${PYTHONPATH:-}${PYTHONPATH:+:}$BENCH_WORK/repo/benchmarks"
export BENCH_RUNNER="$BENCH_WORK/runner_rocm.py"
nohup python3 batch_greedy_rocm.py ${1:+"$1"} > "$BENCH_WORK/batch.log" 2>&1 &
echo "batch/greedy pass started (pid $!). Watch: tail -f $BENCH_WORK/PROGRESS.txt"
