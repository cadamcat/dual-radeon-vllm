# The 500-token rung, five rounds — 2026-09-02b

Every ladder in this repository measures each rung twice, and the shallowest
rung is the one whose two rounds disagree. On this box it disagreed by **22.13%**
on `B-8B-tp2` at 500 tokens in July and by **18.24%** on `B-8B-tp1` in August,
and a published claim rested on that one cell: *below roughly 1 K tokens of
prompt, TP=2 hurts time-to-first-token.* It was withdrawn on 2026-08-30 because
the residual landed on TP=2 in July and on TP=1 in August — a coin flip, not a
topology.

Two readings of the coin flip were on the table. The 2026-08-30 note called it a
**first-request cost** — the first CUDA-graph replay, the first allocation out of
the KV pool, lazy JIT. The 2026-09-02 CUDA campaigns called it a **thermal
ramp**: on the T4 round 1 was +4.45% and the five rounds fell monotonically with
the clock, 83% → 70% of cap.

Five rounds at that rung, both arms of the 8B, twice — and **neither reading
survives on this machine.**

---

## What the five rounds did

Two sittings fifteen minutes apart. `results-attempt1.jsonl` is the first;
`results.jsonl` is the second, with the sampler fixed (below). Prefill is the
time-to-first-token in seconds; decode is tok/s.

| arm | sitting | round 1 | 2 | 3 | 4 | 5 | spread | fastest |
|---|---|--:|--:|--:|--:|--:|--:|:--|
| `B8-tp2-r5` prefill | 1 | 0.1239 | 0.1407 | 0.1256 | 0.1241 | 0.1251 | 13.56% | round 1 |
| `B8-tp2-r5` prefill | 2 | 0.1241 | 0.1282 | 0.1262 | 0.1421 | 0.1421 | 14.50% | round 1 |
| `B8-tp1-r5` prefill | 1 | 0.1439 | 0.1499 | 0.1742 | 0.1490 | 0.1737 | 21.06% | round 1 |
| `B8-tp1-r5` prefill | 2 | 0.1713 | 0.1479 | 0.1474 | 0.1700 | 0.1458 | 17.49% | round 5 |
| `B8-tp2-r5` decode | 2 | 79.29 | 79.16 | 79.22 | 79.24 | 79.05 | **0.30%** | |
| `B8-tp1-r5` decode | 2 | 46.80 | 46.84 | 46.83 | 46.88 | 46.79 | **0.19%** | |

**It is not a first-request cost.** Round 1 is the *fastest* of the five in
three of the four arm-by-sitting combinations, and the slowest in the fourth.
The runner discards a health-check request before any measurement, so the
engine's genuine first request is already absorbed; whatever varies here varies
after that.

**It is not a thermal ramp.** Across the five prefill rounds the hottest card
moves 41 → 44 °C on the pair and 49 → 52 °C on the single card — one to three
degrees — and the ordering of the timings is not monotone in it. On the T4 the
five fell monotonically with the clock over a much larger swing. That is not
what is happening here.

**And the same cell, measured for eleven seconds instead of a tenth of one,
repeats to 0.19–0.30%.** The decode rounds run on the same card, in the same
session, with the same one-second gap between requests. The only thing that
differs is how long the measurement lasts.

## What it is, on the pair: the clock the card had reached

The sampler now records the *lowest* clock seen inside the cell as well as the
highest. On `B8-tp2-r5` the five rounds order **exactly** by it:

| `sclk_mhz_min` | 436 | 463 | 702 | 802 | 1543 |
|---|--:|--:|--:|--:|--:|
| ttft (s) | 0.1421 | 0.1421 | 0.1282 | 0.1262 | 0.1241 |

No exception, in either direction. The card idles between requests — the runner
sleeps one second, and gfx1100 at idle reads `S: 0Mhz *` in `pp_dpm_sclk` — and
a 0.12 s prefill is shorter than the ramp back up. Every round reaches 2 863 to
2 896 MHz and 100% busy before it ends; what differs is how much of the cell was
spent getting there.

**On the single card it does not hold.** `B8-tp1-r5`'s five order 818, 927, 964,
1 162, 1 784 MHz against 0.1479, 0.1700, 0.1474, 0.1458, 0.1713 s — no relation.
That arm is in a different regime: one card at 100% busy peaking at 317–328 W
against a 265 W cap and topping out at 2 516–2 594 MHz where the pair reach
2 863–2 896. Its timings are also **bimodal rather than spread** — 0.1458,
0.1474, 0.1479 against 0.1700, 0.1713, and 0.1439, 0.1490, 0.1499 against
0.1737, 0.1742 in the first sitting: two levels about 16% apart, reproduced
across both sittings, not a continuum. What sets which level a round lands in is
**not determined here.**

## What it says about the claim that was withdrawn

The withdrawn claim was that at 500 tokens **one card prefills faster than
two**. Three sittings now answer it, and the five-round pair is the only one
whose cells are more than a coin toss wide:

| sitting | TP=1 @500 tok/s | TP=2 @500 tok/s | faster | rounds |
|---|--:|--:|:--|--:|
| 2026-07-25 | 3 444 (0.87%) | 2 019 (**22.13%**) | one card | 2 |
| 2026-08-24 | 3 265 (**18.24%**) | 3 690 (1.72%) | two cards | 2 |
| 2026-09-02 | 3 272 (16.30%) | 3 871 (13.58%) | two cards | 5 |

Two of three say two cards, and the one that says otherwise is the sitting whose
TP=2 cell disagrees with itself by 22%. But the September cells disagree with
themselves by 14–16% *with five rounds each*, so what this adds is not a verdict:
it is that a 500-token cell on this machine carries roughly 15% of noise
whatever you do to it, and a 3 272-against-3 871 gap of 18% sits barely outside
it. **The crossover stays unmeasured**, and measuring it needs a rung whose cell
is long enough to be a measurement.

## The conclusion that matters for the ladder

At 500 tokens the prefill measurement is shorter than the card's clock ramp, so
the cell measures the ramp as much as it measures the prompt. That is enough to
say no claim should rest on a single shallow-rung cell — which is exactly what
the withdrawn crossover claim did — and it is *not* enough to say the shallow
rung is worthless: five rounds separate the levels cleanly, and the deeper rungs
of the same ladders repeat to 0.02–0.3%.

## Why there are two sittings

The first sitting is kept because it is the evidence for a defect in the
harness, not because it is a second sample.

The telemetry sampler ran at 1.5 s. A 500-token prefill here takes 0.12 s, so
four of the five rounds caught exactly one sample and it landed in the idle gap
*before* the request: those rows record `sclk_mhz_max` **0** against a
`sclk_mhz_cap` of 2 075, while the one round that happened to sample inside the
request read 2 897 against 2 897. Nothing in schema v1 could tell "the card was
idle" from "we looked at the wrong moment" — and this campaign exists to ask
what the card was doing during that 0.12 s.

Schema **v2** answers it: `tele_period_s` on every row, the Radeon template
sampling prefill at 0.02 s, and `sclk_mhz_first` / `sclk_mhz_min` / `temp_c_first`
taken from the **untrimmed** samples, because for a cell shorter than the ramp
the steady-state window is the wrong window.

> **A caveat the same cell produced.** `sclk_mhz_cap` on gfx1100 is not a fixed
> ceiling. `pp_dpm_sclk` at idle reads `S: 0Mhz *` / `1: 500Mhz` / `2: 2075Mhz`,
> and under load its top row reads 2 892. So `sclk_pct_of_cap` is a percentage of
> a moving denominator on this platform, and is not the throttle indicator it is
> on NVML, where the cap comes from `nvmlDeviceGetMaxClockInfo`. Radeon rows
> carry it; do not read it as NVML's.

## Two differences between the sittings, recorded

* `B8-tp1-r5` settled at `mml` **8 363** (8 448 KV tokens) in the first sitting
  and **15 792** (15 952) in the second. Both are the runner's capacity retry
  finding what fit at util 0.90; the first matches August's `B-8B-tp1` exactly
  (8 363 / 8 442). So the two sittings of that arm are not the same
  configuration, and only the second is registered in the projections.
* `B8-tp2-r5` is `mml` 33 000 / 128 800 KV tokens in both.

## Also confirmed in passing

TP=1 runs on **card1 alone** — card2 reads sclk 0, 0% busy, 7–8 W in every row
of `B8-tp1-r5`. That was inferred on 2026-09-02 from VRAM moving on one card;
here it is read directly.

## Files

    runner.py                 copied from harness/runner_radeon.py; CFGS, paths
    results.jsonl             the second sitting, schema v2 — the registered one
    results-attempt1.jsonl    the first, schema v1, kept as the defect's evidence
    PROGRESS.txt              / PROGRESS-attempt1.txt
    logs/ , logs-attempt1/    one serve log per arm per sitting
    serve-B8-tp*-r5.sh        the exact serve commands
    host_link.json            preflight, both root ports x16, before the run
