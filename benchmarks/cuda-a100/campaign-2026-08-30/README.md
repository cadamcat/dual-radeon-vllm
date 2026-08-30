# The A100, measured again with prefix caching off — 2026-08-30

`campaign-2026-08-29/` next door has eleven rungs of prefill for twelve
configurations and **none of it is a prefill measurement**. This is the
correction, for the two models the 2026-08-30 round needs.

Two configurations, eleven rungs, two rounds: **22 measurements each, 0 errors**.

    NVIDIA A100-SXM4-80GB · 81920 MiB · sm80 · driver 580.82.07
    vLLM 0.28.0 · torch 2.13.0+cu130 · transformers 5.15.1 · CUDA 13.0
    TRITON_ATTN · MarlinLinearKernel · enable_prefix_caching=False

| cfg | model | KV | reaches |
|---|---|--:|--:|
| `G12` | gemma-4-12B-it w4a16 QAT | 60.61 GiB | 965 712 tok |
| `G26A4B` | gemma-4-26B-A4B int4 AWQ | 53.99 GiB | 1 106 907 tok |

## What was wrong with the old rows

Every rung of the ladder is a strict prefix of the next — `run.py` cuts by
truncating token ids — so with the cache on, a rung's prefill is charged only
for the tokens the rung below it did not already leave in the KV, and a repeat
of a rung is charged for almost nothing.

| gemma-4-12B, 32 K | round 1 | round 2 | spread |
|---|--:|--:|--:|
| 2026-08-29, caching **on** | 2.9320 s | **0.2010 s** | 174 % |
| 2026-08-30, caching **off** | 8.3826 s | 8.3796 s | **0.04 %** |

**Round 1 of the old data was wrong too, by 2.9×** — ascending the ladder was
itself a sequence of cache hits, so no round of that campaign measured a cold
prefill. Its recorded `prefill_tps` at 32 K was 159 299.

130 of that campaign's 132 prefill rungs fail the repeatability cut, which is
how `build_prefill.py` excludes them without needing to know the cause.

**Decode is unaffected and was never in question**: decode rate is measured from
the stream after the first token, and every one of those rungs is chart-grade.
The 2026-08-29 decode rows stand.

## The warmup

`run.py` now discards one request after startup, which the Radeon runner has
always done as its health gate. It works: the 500 rung, which on the L4 was
2.0636 s against 0.2866 s, is 0.1026 s against 0.0861 s here, and
`G26A4B`'s is inside the cut at 7.48 %. One rung of `G12` is still ungraded at
17.49 % — the shallowest rung is where a fixed cost of tens of milliseconds is
the whole measurement.

## No serve logs

The VM was reclaimed after the run finished; the logs went with it. What
survives is what `harvester.py` had already pulled to the Mac — the rows, each
configuration's `model_meta` (backend, quant kernel, KV size, load time) and the
`run_meta` row's versions. That is the provenance the projections use, but the
raw logs for this campaign do not exist. `harvester.py` is kept here beside the
data because it is the reason the data is here at all: it was running before the
first measurement, and this is the sixth reclaim across two days.

## What it says about single-card prefill

`T(S) = a + b·S + c·S²`, fitted on `target`, chart-grade rungs only:

| machine | model | b µs/tok | c ns/tok² |
|---|---|--:|--:|
| A100 80GB | gemma-4-12B | **145.7** | **3.62** |
| L4 | gemma-4-12B | 534.7 | 8.03 |
| RX 7900 XT | gemma-4-12B | 446–479 | 24.2–25.2 |
| A100 80GB | gemma-4-26B-A4B | 62.6 | 2.30 |
| L4 | gemma-4-26B-A4B | 204.4 | 5.53 |
| RX 7900 XT | gemma-4-26B-A4B | 360.0 | 13.13 |

The A100 leads one 7900 XT by about **3× on the linear term and 7× on the
quadratic**: the gap in how attention scales is twice the gap in compute. The
Radeon *beats* the L4 on the linear term and loses three times over on the
quadratic — which is why the L4 is 1.58× faster at 32 K and would lose at a
short enough prompt. A single prefill tok/s number states none of this.

Reconstructed from the fit, `G12` at 32 K is 8.379 s against 8.3796 measured.
