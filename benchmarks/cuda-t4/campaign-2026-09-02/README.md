# T4, 2026-09-02: the same nine cells, and a different answer at the 500 rung

**`gemma-4-12B-it` on one Tesla T4 with
[vllm#39018](https://github.com/vllm-project/vllm/pull/39018) applied and md5-asserted,
vLLM 0.28.0 pinned, the 500-token rung five times and 8 000 / 32 000 twice,
through [`harness/telemetry.py`](../../harness/telemetry.py).** Nine cells, no
errors. Read beside [`../../cuda-l4/campaign-2026-09-02/`](../../cuda-l4/campaign-2026-09-02/),
which ran the same protocol an hour earlier on an L4.

## Round 1 is the high outlier here, and it is heat again

| round | decode tok/s | sclk / cap | temp |
|--:|--:|--:|--:|
| 1 | **20.62** | 83.0 % | — |
| 2 | 20.36 | 78.3 % | — |
| 3 | 19.57 | 74.5 % | — |
| 4 | 19.46 | 69.8 % | — |
| 5 | 19.58 | 75.5 % | — |

Round 1 sits 4.45 % above the mean of the other four and the five fall
monotonically with the clock. On the L4 an hour earlier the same five rounds
moved 1.28 % with round 1 in the middle. So the shallow rung is not unstable
for a software reason that a warm-up call would fix — **it measures where the
card is on its thermal ramp**, and a 70 W T4 ramps within ninety seconds where
the 72 W L4 held its decode clock. The 2026-08-30 row for this rung, 20.28
tok/s, is inside this envelope; it was one sample of the same ramp.

## At 32 K the T4 is not bandwidth-bound, and the telemetry shows it

| ctx | tok/s | vs 2026-08-30 | gpu busy | **mem busy** | sclk / cap |
|--:|--:|--:|--:|--:|--:|
| 500 | 19.5–20.6 | −4.0 … +1.7 % | 100 % | 79–86 % | 70–83 % |
| 8 000 | 14.40 | +3.0 % | 100 % | 59 % | 92.5 % |
| 32 000 | 8.98 | −0.1 % | 100 % | **35 %** | 98.1 % |

The memory controller is busy 80 % of the time streaming weights at 500 tokens
and 35 % at 32 000, while the SMs stay at 100 % and the clock rises to 98 % of
its ceiling. The T4's decode falling to 0.36× of the L4's at depth — the only
card here whose decode more than halves across the ladder — is **attention
compute on sm75**, which has no FlashAttention 2 and runs the Triton kernel,
not memory bandwidth. Nothing before this campaign could have said which.

## A reading not to trust, and what it changes in the schema

`power_w_max` reads 90 W at 8 000 and **105.7 W at 32 000 against a 70 W cap**.
A T4 does not draw 105 W. That is the maximum of instantaneous NVML samples over
a three-minute cell, and NVML power on this part is spiky; the L4's readings
sat at 72.0–73.3 W against its 72 W cap and were credible. `summarise()` keeps
the maximum because it is what tells a throttle from a slow kernel, but it
needs a mean beside it, and that field is added after this batch lands so the
three 2026-09-02 campaigns share one emitted shape.

## What is here

    results.jsonl        22 rows, every measured row with the harness/SCHEMA.md block
    logs/serve-G12.log   TRITON_ATTN forced ("FA4 not available"), MarlinLinearKernel
    PROVENANCE.json      vllm#39018: md5 of both files before and after, patch rc 0
    pr39018.diff         the patch as applied
    chain.py             pin vllm==0.28.0, apply and assert the patch, fetch the checkpoint
    run.py               harness/runner_cuda.py with one config and the per-rung
                         round counts; the argv gate removed (see the L4 README)

    Tesla T4, 15 360 MiB, sm75 · driver 580.82.07 · vLLM 0.28.0 · torch 2.13.0+cu130
