# The A100, measured again with prefix caching off — 2026-08-30

`campaign-2026-08-29/` next door has eleven rungs of prefill for twelve
configurations and **none of it is a prefill measurement**. This is the
correction, for the two models the 2026-08-30 round needs.

**Five** configurations, in two passes on two VMs, eleven rungs and two rounds
each: **22 measurements per configuration, 0 errors**. The first pass took the
two models the 2026-08-30 round's spine needed; the second took the three whose
prefill the 2026-08-29 campaign had recorded through a warm cache and therefore
never measured.

    NVIDIA A100-SXM4-80GB · 81920 MiB · sm80 · driver 580.82.07
    vLLM 0.28.0 · torch 2.13.0+cu130 · transformers 5.15.1 · CUDA 13.0
    TRITON_ATTN · MarlinLinearKernel · enable_prefix_caching=False

| cfg | model | backend | KV | reaches |
|---|---|---|--:|--:|
| `G12` | gemma-4-12B-it w4a16 QAT | TRITON_ATTN | 60.61 GiB | 965 712 tok |
| `G26A4B` | gemma-4-26B-A4B int4 AWQ | TRITON_ATTN | 53.99 GiB | 1 106 907 tok |
| `G31` | gemma-4-31B-it w4a16 QAT | TRITON_ATTN | 49.28 GiB | 252 587 tok |
| `Q38` | Qwen3.8-27B int4 AWQ | **FLASH_ATTN** | 49.59 GiB | 743 217 tok |
| `MG30` | Muse-Glimmer-30B int4 | **FLASH_ATTN** | 48.23 GiB | 2 493 233 tok |

**Only the stock arms were re-measured.** Speculation changes what happens after
the first token, not the forward pass over the prompt, so a speculative arm's
prefill answers no question this round asked; the decode those arms produced was
never in doubt and stands.

The `backend` column is read from each run's own `model_meta`, and two of these
are the first record of it: the 2026-08-29 campaign captured no backend at all —
its regex matched one of the two forms vLLM 0.28 writes — and Muse-Glimmer's
serve log went with a reclaimed VM. **vLLM sends Qwen3.8 and Muse-Glimmer to
FLASH_ATTN here and gemma-4 to TRITON_ATTN**, which matters for the cross-machine
comparison: a `c` ratio between an A100 FLASH_ATTN line and a Radeon TRITON_ATTN
one is a kernel difference as well as a card one. `gemma-4-31B` is the only model
in this repository whose lines on both machines are recorded and on the same
kernel.

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
