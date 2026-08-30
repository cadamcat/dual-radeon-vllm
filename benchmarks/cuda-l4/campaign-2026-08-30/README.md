# gemma-4 on one NVIDIA L4 — 2026-08-30

The spine's fourth machine, and **the first CUDA rows in this repository
measured with prefix caching off**. Two configurations, eleven rungs each, two
rounds a rung: **22 measurements each, 0 errors**.

`results.jsonl` is the raw data, `run.py` produced it, `setup.py` built the
machine, `logs/` holds a serve log per configuration.

    NVIDIA L4 · 23034 MiB · sm89 · driver 580.82.07 · vLLM 0.28.0
    TRITON_ATTN · MarlinLinearKernel · bfloat16 · enable_prefix_caching=False

| cfg | model | KV | reaches |
|---|---|--:|--:|
| `G12` | gemma-4-12B-it w4a16 QAT | 10.84 GiB | 172 732 tok |
| `G26A4B` | gemma-4-26B-A4B int4 AWQ | 2.54 GiB | 52 009 tok |

## Why prefix caching is off, and why that makes this the reference

Every rung of the ladder is a strict prefix of the next. With the cache on, a
rung's prefill is charged only for the tokens the rung below it did not already
leave in the KV, and a *repeat* of a rung is charged for almost nothing. That
is what happened to the 2026-08-29 A100 campaign, whose 32 K rung took 2.932 s
on round 1 and **0.201 s** on round 2, recording a prefill of 159 299 tok/s;
130 of its 132 rungs fail the repeatability cut and its prefill cannot be used.

These rows repeat: 32 K is 25.3553 s and 25.3731 s on `G12`, 12.2564 s and
12.2759 s on `G26A4B`.

## The one ungraded rung, and why it is the harness

The 500 rung's two prefill rounds are 2.0636 s and 0.2866 s on `G12`,
2.0104 s and 0.1280 s on `G26A4B` — a 151 % and 176 % spread on configurations
whose every other rung agrees to better than 0.1 %. It is the **first request to
a cold engine** absorbing the first CUDA-graph replay, the first allocation out
of the KV pool and lazy JIT. The Radeon runner has always discarded one request
first, as its health gate; `a100_run.py` never did, so every CUDA configuration
in this repository carries one ungraded rung for a reason that belongs to the
harness rather than the machine.

`run.py` here has the warmup added, but it was added **after** this run, so
these two configurations still show it. Decode is untouched — all eleven rungs
of both configurations are chart-grade, because decode rate is measured from
the stream after the first token.

## What it says against one RX 7900 XT

Same model, same TP=1, both fitted on `target` and gated on repeatability:

| | b µs/tok | c ns/tok² | TTFT at 32 K |
|---|--:|--:|--:|
| RX 7900 XT, gemma-4-12B | 446–479 | 24.2–25.2 | 40.03 s |
| L4, gemma-4-12B | **534.7** | **8.03** | **25.36 s** |

**The Radeon wins the linear term and loses the quadratic by three times.** The
L4 is 1.58× faster at 32 K and would be *slower* at a short enough prompt; the
decomposition is what separates the two, and a single prefill tok/s number
hides it. Reconstructing TTFT from each fit gives 40.1 s and 25.3 s against
40.03 and 25.36 measured.
