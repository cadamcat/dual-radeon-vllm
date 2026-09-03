# The prediction, written before the data — 2026-09-03 02:36 CEST

`campaign-2026-09-02d` found that what the second Radeon is worth is decided by
`mem_busy`, the memory controller's busy fraction: 90 % on Qwen3-8B and 1.70×
from the second card, 56 % on gemma-4-12B and 1.20×. `cuda-h100/campaign-2026-09-03`
found the same two models at 87 % and 53 % on an H100 — two vendors, three
points apart — which made `mem_busy` a quantity that travels rather than a fact
about one box.

It has never been used to **predict** anything. This run is the test, because
an H200 is an H100 with the same compute capability, the same 700 W cap, the
same SM clock ceiling, the same PCIe generation, and more memory.

## The model

If a decode step spends a fraction `f` of its time waiting on memory and the
rest on everything else, then multiplying memory speed by `r` gives

    speedup = 1 / ((1 - f) + f / r)

taking `f` = `mem_busy_pct_max` measured on the H100 at the 500 rung, and `r`
from the two cards' **measured** maximum memory clocks — 2 619 MHz on the H100
and 3 201 MHz on the H200, from `modal-2026-09-02/machines.jsonl`, so
**r = 1.222**. The vendors' bandwidth figures (3.35 and 4.8 TB/s, r = 1.43)
are not used: they are not something this repository measured.

| model | `mem_busy` on H100 | predicted H200 / H100 |
|---|--:|--:|
| `B8` Qwen3-8B bf16 | 87 % | **1.188×** |
| `G31` gemma-4-31B w4a16 | 67 % | **1.139×** |
| `Q38` Qwen3.8-27B int4 SSM | 65 % | 1.134× |
| `G12` gemma-4-12B w4a16 | 53 % | 1.107× |
| `MG30` Muse-Glimmer-30B int4 | 51 % | 1.102× |
| `G26A4B` gemma-4-26B-A4B MoE | 38 % | **1.074×** |

## What would falsify it

The discriminating quantity is the **spread**: 0.114 between `B8` and
`G26A4B`. If the six models come back within a few per cent of each other, or
in the wrong order, `mem_busy` predicts nothing about bandwidth and the
2026-09-02d reading stays a correlation across five cells.

At the time of writing, `G31` had reached the 32 000 rung at **1.123×**
against the 1.139× above — the only model this file's author had seen. The
other five had not started.

## What this cannot settle

`mem_busy` is a busy **fraction**, not achieved bandwidth; `harness/telemetry.py`
records achieved bandwidth as absent on both platforms and says why. A busy
fraction of 87 % at a lower absolute rate is not the same thing as 87 % at a
higher one, and nothing here separates them.

---

# The verdict, written after — 2026-09-03

**Ordering: right at both ends. Magnitude: wrong, by about a factor of two on
the spread.**

| model | `mem_busy` | predicted | measured | rung used |
|---|--:|--:|--:|--:|
| `B8` Qwen3-8B bf16 | 87 % | 1.188× | **1.254×** | 500 |
| `G31` gemma-4-31B | 67 % | 1.139× | 1.120× | 2 000 |
| `Q38` Qwen3.8-27B SSM | 65 % | 1.134× | 1.124× | 500 |
| `G12` gemma-4-12B | 53 % | 1.107× | 1.072× | 500 |
| `MG30` Muse-Glimmer-30B | 51 % | 1.102× | 1.089× | 500 |
| `G26A4B` gemma-4-26B-A4B | 38 % | 1.074× | **1.044×** | 500 |

Each row uses the shallowest rung whose two rounds agree within 1 % **on both
machines** — a rule fixed before the numbers were looked at. `G31` uses 2 000
because its H200 500-rung rounds came back 84.62 and 95.84, 13.3 % apart; every
other rung on that ladder agrees to 0.2 %, and the projection marks it not
chart-grade on its own repeatability rule.

**What passed.** The two ends are in the right order and clearly separated:
1.254× at 87 % against 1.044× at 38 %. The two order violations are
`G31`/`Q38` and `G12`/`MG30`, and each is a pair whose `mem_busy` differs by
two points — inside the resolution of the predictor itself.

**What failed.** Measured spread 0.210 against 0.114 predicted. Fitting `r`
with `f` held at `mem_busy` gives 1.227, all but identical to the 1.222 the
prediction used — so `r` is not the error. The error is in `f`, and it is not a
constant scaling: `B8`'s 1.254× **exceeds** `r` itself, which
`1/((1-f) + f/r)` cannot produce for any `f ≤ 1`. Either the effective
bandwidth ratio is above the memory-clock ratio (the vendors' 3.35 and
4.8 TB/s give 1.43, and at that `r` the same form over-predicts every row), or
the H200 differs from the H100 in something this file assumed away.

**Standing conclusion: `mem_busy` is an ordinal predictor of what bandwidth is
worth, not a cardinal one.** It has now ordered five settings correctly and
sized none of them.

An independent check that the mechanism is right even where the arithmetic is
not: `mem_busy` on the H200 is **lower than on the H100 for every one of the six** —
87→76, 67→53, 65→52, 53→39, 51→39, 38→28 at the 500 rung, no exceptions and no
crossings. Faster memory doing the same work leaves the controller busy for
less of the time, which is what the model says should happen and is not
something it was fitted to.
