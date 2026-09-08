# Retention and depth-cost rankings across the measured machines

The Radeon [depth-cost article](articles/depth-cost-is-the-stacks.html) showed
that retaining a larger fraction of short-context throughput can coexist with
a steeper absolute cost of context. Applying its calculation to the committed
September ladders gives **6.7–53.3 % discordant model pairs across seven
configurations**. This is a reproduction using the per-cell medians and all
recorded rungs. The exact percentages depend on the repeat, the cost statistic
and which models are included; some curves do not support a linear cost model.

The tables below keep those distinctions together. They compare rankings
**within each configuration**, not throughput between different machines.

## What was compared

For each model, let `v(S)` be the median decode tok/s at nominal rung `S`.
Retention is `v(32000) / v(500)`, larger first. Depth cost is the OLS slope of
`1e6 / v(S)` against `S`, smaller first, in microseconds per generated token
per additional context token. Every fit uses the same eleven nominal rungs
from **500 to 32 000**, with **two rounds per cell**. This matches the article's
shared-span calculation; using the server's actual `prompt_tokens` instead is
a separate sensitivity check below.

A model pair is discordant when the model with higher retention also has the
higher cost. All unordered pairs enter the denominator: six models give fifteen
pairs, five give ten. Exact ties are counted separately and are absent here;
near-ties are not promoted to statistically resolved ranks.

The baseline enters even without a fit. If `T = 1 / v`, then
`retention = T(short) / (T(short) + ΔT)`. A slower baseline can retain more of
its rate while paying more absolute time for context. A slope describes a
constant marginal cost only where a line describes the curve.

The source set is fixed by `(file, cfg)` in
[`depth_order.py`](../benchmarks/analyze/depth_order.py):

| Configuration | Raw files and selected arms |
|---|---|
| RX 7900 XT TP2 | [`campaign-2026-09-03/results.jsonl`](../benchmarks/campaign-2026-09-03/results.jsonl), the six `*-long` arms |
| H100 TP1 | [`results.jsonl`](../benchmarks/cuda-h100/campaign-2026-09-03/results.jsonl): B8, G12, G26A4B, G31; [`results-q38.jsonl`](../benchmarks/cuda-h100/campaign-2026-09-03/results-q38.jsonl): Q38; [`09-03b/results.jsonl`](../benchmarks/cuda-h100/campaign-2026-09-03b/results.jsonl): MG30 only |
| H200 TP1 | [`results.jsonl`](../benchmarks/cuda-h200/campaign-2026-09-03/results.jsonl), all six arms |
| B300 TP1 | [`results.jsonl`](../benchmarks/cuda-b300/campaign-2026-09-03/results.jsonl), all six arms |
| RTX PRO 6000 TP1 | [`results.jsonl`](../benchmarks/cuda-pro6000/campaign-2026-09-03/results.jsonl), all six arms |
| H100 TP2 | [`results.jsonl`](../benchmarks/cuda-h100/campaign-2026-09-03-tp2/results.jsonl), five arms; no Muse-Glimmer |
| RTX PRO 6000 TP2 | [`results.jsonl`](../benchmarks/cuda-pro6000/campaign-2026-09-03-tp2/results.jsonl), five arms; no Muse-Glimmer |

The H100's `mml` and `max-num-seqs` control arms are excluded: they are repeat
configurations of existing checkpoints, not additional models. No speculative
arms enter. The four-H100 campaign has only one model and cannot supply a
model-pair ranking. The Radeon rows use the original stack; September's newer
Qwen3.8 arms are not substituted into that same-stack comparison.

The [campaign records](../benchmarks/cuda-modal/README.md) carry versions,
checkpoint revisions and serve-log backend selections. They include different
attention and weight kernels. This analysis neither matches those kernels
across hardware nor attributes a rank difference to hardware alone.

## Reproducing the median-based ranking

Baseline spread is the largest fitted intercept divided by the smallest
within the configuration. Min r² is the poorest model fit in that row.
Percent is the discordant fraction, not a confidence level.

| Configuration | Models | Discordant / pairs | Percent | Baseline spread | Min r² |
|---|--:|--:|--:|--:|--:|
| RX 7900 XT TP2 | 6 | 8/15 | 53.3 | 8.429 | 0.3145 |
| H100 TP1 | 6 | 1/15 | 6.7 | 2.755 | 0.7162 |
| H200 TP1 | 6 | 5/15 | 33.3 | 2.596 | 0.0280 |
| B300 TP1 | 6 | 3/15 | 20.0 | 2.391 | 0.4980 |
| RTX PRO 6000 TP1 | 6 | 5/15 | 33.3 | 3.188 | 0.3870 |
| H100 TP2 | 5 | 2/10 | 20.0 | 1.945 | 0.7058 |
| RTX PRO 6000 TP2 | 5 | 2/10 | 20.0 | 2.467 | 0.9755 |

**This table retains ungraded cells.** Two H200 cells exceed the repository's
decode repeat-spread cut: G31 at 500 and G26A4B at 1 000. The worst spread is
**17.75 %**. Keeping them reproduces the all-rung calculation; it does not make
the H200 ranking chart-grade. In particular, its MoE fit has **r² 0.0280**:
a fitted number here is not evidence of a constant marginal depth cost.
Muse-Glimmer's bounded window is another reason an OLS line can be a poor
description, even when repeats agree.

Discordance and baseline spread have **Pearson r = 0.809** across these
configurations; excluding the Radeon pair gives **r = 0.305**. These are
descriptive associations of quantities computed from the same fits. The
configurations share models and hardware families, and the pair has both the
widest baseline spread and the most disagreement. This is not a calibrated
cross-machine predictor.

## What changes under other readings

Each entry below is discordant pairs / all model pairs.

| Configuration | Actual tokens | Round 1 | Round 2 | Endpoint cost | Without Muse-Glimmer |
|---|--:|--:|--:|--:|--:|
| RX 7900 XT TP2 | 8/15 | 7/15 | 7/15 | 9/15 | 7/10 |
| H100 TP1 | 1/15 | 1/15 | 2/15 | 2/15 | 1/10 |
| H200 TP1 | 5/15 | 1/15 | 4/15 | 0/15 | 3/10 |
| B300 TP1 | 3/15 | 1/15 | 4/15 | 2/15 | 2/10 |
| RTX PRO 6000 TP1 | 5/15 | 4/15 | 5/15 | 4/15 | 4/10 |
| H100 TP2 | 2/10 | 0/10 | 2/10 | 1/10 | 2/10 |
| RTX PRO 6000 TP2 | 2/10 | 2/10 | 2/10 | 2/10 | 2/10 |

- **Actual tokens:** fit against the mean server-reported context length in
  each cell. Every pair's concordant/discordant verdict stays the same.
- **One round at a time:** recompute both retention and slope from that round
  only. These are repeat diagnostics, not confidence intervals. H100 TP2's
  first round has no disagreement. The median-based percentage is not a lower
  bound on what every repeat will show.
- **Endpoint cost:** replace OLS with the latency difference between endpoints
  divided by their nominal context span. This measures average additional cost
  over the interval without asserting linearity inside it. H200's two rankings
  now agree, while the Radeon pair disagrees on **60.0 %** of pairs.
- **Without Muse-Glimmer:** omit the bounded-window model wherever present.
  The resulting range is **10.0–70.0 %**. The model set changes the denominator
  as well as the observed ordering; other low-r² fits remain in this check.

The cross-machine extension supports the arithmetic warning about retention:
the chosen baseline and cost statistic affect the ranking. It does not support
a stable discordance percentage for a hardware family, or treating every fitted
slope as a property of a model. Reporting absolute latency alongside retention,
the context interval, backend and repeat spread makes those distinctions visible.

## Recompute

```bash
python3 benchmarks/analyze/depth_order.py          # the two tables
python3 benchmarks/analyze/depth_order.py --json   # every fit, pair and source
python3 benchmarks/analyze/verify_doc_figures.py   # checks this page against the rows
```

The analysis reads the original JSONL files directly and writes no projection.
It rejects missing or duplicate `(target, round)` cells in its selected ladders.
No new GPU measurements were made for this page.
