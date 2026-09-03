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
