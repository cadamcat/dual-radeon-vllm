# gemma-4-31B and Qwen3.8-27B on one NVIDIA L4 — what fits and what does not

Two configurations the L4's earlier passes could not start, retried at
`--max-num-seqs 1` with an `--enforce-eager` fallback. **One of them now runs, on
two rungs; the other does not run at all.** Both outcomes are measurements, and
the four serve logs in `logs/` are where they are read from.

    NVIDIA L4 · 23034 MiB · sm89 · driver 580.82.07 · vLLM 0.28.0
    torch 2.13.0+cu130 · CUDA 13.0 · util 0.95 · max_num_seqs 1
    enable_prefix_caching=False

| cfg | weights | KV budget | outcome |
|---|--:|--:|---|
| `G31` gemma-4-31B w4a16 QAT | 18.7 GiB | **−0.8 GiB** | four capacity retries, then the eager fallback |
| `G31-eager` | 18.7 GiB | **+1.71 GiB**, 2 020 tok | 2 rungs, 4 measurements, 0 errors |
| `Q38` Qwen3.8-27B int4 AWQ | 19.24 GiB | **−0.39 GiB** | four retries, then the eager fallback |
| `Q38-eager` | 19.24 GiB | **−0.34 GiB** | four more retries, `config_failed` |

## Halving `max_model_len` cannot fix a negative budget

Both configurations spent four engine starts stepping `max_model_len` 33 000 →
16 500 → 8 250 → 4 125 → 2 062 and moved nothing, because what vLLM reports is
not "the pool is too small for one request at this length" but

    gpu_worker.py:578  Available KV cache memory: -0.8 GiB
    core.py            ValueError: No available memory for the cache blocks

At `max_num_seqs=1` the length does not enter the activation peak enough to
bring a **negative** budget back above zero. The runner's other branch — a
positive-but-too-small token count, where it retries at `0.99 × kv_max_len` — is
the one the length is the right lever for, and `Available KV cache memory` is
the log line that tells the two apart.

## What `--enforce-eager` buys is per-model, and the two here differ by 50×

Same card, same utilisation, same `max_num_seqs`:

| | KV budget, graphs on | KV budget, eager | eager buys |
|---|--:|--:|--:|
| `gemma-4-31B` | −0.8 GiB | **+1.71 GiB** | **2.51 GiB** |
| `Qwen3.8-27B` | −0.39 GiB | −0.34 GiB | **0.05 GiB** |

So gemma-4-31B's problem was the CUDA graphs and eager solved it; **Qwen3.8-27B's
problem is the weights**, and eager cannot solve it. 19.24 GiB resident on a
22.49 GiB card at util 0.95 leaves 2.13 GiB for everything that is not KV, and
this model's everything-else is more than that. Its logs also carry

    interface.py:911  Setting attention block size to 784 tokens to ensure that
                      attention page size is >= mamba page size

which is the hybrid-SSM page alignment, and is a second reason its floor is
high.

**`Qwen3.8-27B-AWQ-INT4 does not fit on a 23 GB L4.`** That is a ceiling like
the two this repository already publishes, not a configuration that was tried
badly: `--enforce-eager` and `max_num_seqs 1` are both applied, and raising
`util` past 0.95 is deliberately not tried — the runner's rev2 note records that
these cards keep scratch above it, and on the Radeon it produced
`HSA_STATUS_ERROR_OUT_OF_RESOURCES` rather than a bigger pool.

## `G31-eager` is its own configuration, not an arm of `G31`

`--enforce-eager` is a different engine, so it carries its own id everywhere.
Two rungs is what a 2 020-token pool reaches:

| rung | prefill r1 / r2 | range | decode r1 / r2 | range |
|---|---|--:|---|--:|
| 500 | 0.7205 / 0.8124 s | 11.99 % — **ungraded** | 11.0273 / 11.1579 | 1.18 % |
| 1 000 | 1.4874 / 1.4932 s | 0.39 % | 11.0071 / 11.1090 | 0.92 % |

One chart-grade prefill rung and two chart-grade decode rungs. Too few to fit
three coefficients, so this configuration is reported and not fitted — which is
the point of keeping it: **it is a ceiling, recorded as one.**

For scale, the same model on an A100-SXM4-80GB the same day, with no eager
fallback needed, decodes at 58.51 tok/s at 500 against this card's 11.09.

    TRITON_ATTN · MarlinLinearKernel · KV 1.71 GiB / 2 020 tokens · mml 1 980
