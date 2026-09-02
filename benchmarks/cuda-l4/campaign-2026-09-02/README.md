# L4, 2026-09-02: the first CUDA cells with hardware telemetry

**Nine cells, one configuration, and the first time any CUDA row in this
repository carries clocks, power and temperature.** `gemma-4-12B-it` (`G12`) on
one NVIDIA L4, vLLM 0.28.0 pinned to the stack the 2026-08-30 rows ran on, the
500-token rung five times and 8 000 / 32 000 twice, through
[`harness/telemetry.py`](../../harness/telemetry.py).

Three things came out, and the third was not what the run was for.

## The 500 rung is stable here

Five rounds of decode at 500 tokens: **27.94, 28.30, 27.96, 27.94, 27.97 tok/s**.
Round 1 sits 0.36 % under the mean of the other four; the range over all five is
1.28 %. The 22 % two-round spread the Radeon 8B showed at this rung, and the
13.9 % the A100's 2026-08-29 sitting showed, are **not** a universal first-request
effect. Whatever they are, this card and this stack do not have it.

## The L4 is power-capped in every cell, and the published rows were too

| cell | tok/s | vs 2026-08-30 | sclk / cap | power / cap |
|---|--:|--:|--:|--:|
| 500 ×5 | 27.94–28.30 | −1.1 … +0.2 % | 70–78 % | 72.0–72.1 / 72 W |
| 8 000 ×2 | 27.24 / 27.25 | +0.2 % | 76 % | 72.1 / 72 W |
| 32 000 ×2 | 25.16 / 25.18 | +0.4 % | 78–79 % | 72.4–73.3 / 72 W |

Power sits on the 72 W cap in all nine cells and the SM clock at 69.9–79.4 % of
its 2 040 MHz ceiling. The 2026-08-30 rows agree with these to within 1.1 % at
decode and 2.3 % at prefill from 8 000 up, so they were taken in the same state.
The caveat every L4 number on this site should have carried is now a
measurement rather than a suspicion: **these are a 72 W L4's figures.**

## Prefill at 500 tokens falls 15 % across five rounds, and it is heat

| round | prefill tok/s | temp | sclk |
|--:|--:|--:|--:|
| 1 | 1 924.5 | 68 °C | 1 590 MHz |
| 2 | 1 756.8 | 74 °C | 1 530 MHz |
| 3 | 1 718.1 | 78 °C | 1 470 MHz |
| 4 | 1 639.0 | 81 °C | 1 425 MHz |
| 5 | 1 642.0 | 79 °C | 1 470 MHz |

Decode over the same five rounds moves 1.3 %. Prefill is compute-bound and
follows the clock; decode is bandwidth-bound and does not. The ramp takes about
ninety seconds of sustained load, so **a short prefill cell measures the card's
thermal history as much as the model**, and the 8 000 and 32 000 cells agree
with 2026-08-30 because they are long enough to reach the steady state. This
is why the 500-token prefill rung has no chart-grade row on the L4 in the
projection, and it is the mechanism, not a guess.

## What is here

    results.jsonl        22 rows: run_meta, telemetry_meta, model_meta, 9 prefill,
                         9 decode, config_complete -- every measured row carries
                         the harness/SCHEMA.md block
    logs/serve-G12.log   TRITON_ATTN, MarlinLinearKernel, prefix caching off
    run.py               harness/runner_cuda.py with one config and the
                         per-rung round counts; the argv gate is removed, because
                         under `colab exec` sys.argv is the kernel's own and the
                         first attempt of this run skipped every configuration
    setup.py             vllm==0.28.0 pinned, one checkpoint, the ladder cut by
                         benchmarks/prompts/cut_prompts.py

    NVIDIA L4, 23 034 MiB, sm89 · driver 580.82.07 · vLLM 0.28.0 · torch 2.13.0+cu130
