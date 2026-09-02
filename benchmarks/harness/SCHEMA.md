# Campaign record schema — v2, 2026-09-02

Every campaign writes JSONL. One record per line, `kind` says which.
`harness/telemetry.py` produces the hardware half on both platforms; it was
smoke-tested on the two-card gfx1100 guest and on a Colab T4 the day it was
written, and every field below returned a real value on at least one of them.

## Why this exists

The two runner families had drifted into recording disjoint sets:

| | Radeon runners | CUDA runners |
|---|---|---|
| per-card power, VRAM | yes | **no** |
| `wall_s`, `gen_tokens` | **no** | yes |
| `machine` | **no** | only the last few |
| clocks, temperature, busy % | **no** | **no** |
| serve log kept | sometimes | sometimes |

Neither was a superset, so a cross-machine question had to start by asking which
campaign the row came from. And neither sampled clocks: a Colab T4 measured on
2026-09-02 ran a 300-step matmul at **1245 MHz against a 1590 MHz ceiling** while
drawing **69.9 W against a 70 W cap** — power-throttled from end to end. No field
in either schema could tell that from a slow kernel.

## `kind: decode` / `kind: prefill`

Measurement, unchanged from before plus the telemetry block:

    cfg target round ts prompt_tokens
    decode_tps | ttft prefill_tps
    wall_s gen_tokens            required on both platforms now
    machine                      required

## The telemetry block, on every measured row

    tele_schema          2
    tele_samples         how many polls landed inside the cell
    tele_period_s        the polling period this cell was sampled at (v2)
    gpu_busy_pct_max     amdgpu gpu_busy_percent | NVML utilization.gpu
    mem_busy_pct_max     amdgpu mem_busy_percent | NVML utilization.memory
    power_w_max          hwmon power1_average    | NVML power usage -- the maximum of
                         instantaneous samples: it tells a throttle from a slow
                         kernel, and on a Colab T4 it read 105.7 W against a 70 W
                         cap, which is not a draw. Read the next two for the draw.
    power_w_mean         mean of the same samples, all cards      (added 2026-09-02)
    power_w_median       median of the same samples, all cards    (added 2026-09-02)
    power_w_sum_max      summed across cards -- what the wall sees
    power_w_sum_min
    power_cap_w          hwmon power1_cap        | NVML enforced limit
    sclk_mhz_max         pp_dpm_sclk starred row | NVML SM clock
    sclk_mhz_cap         pp_dpm_sclk top row     | NVML max SM clock
    sclk_pct_of_cap      derived; below ~95 means the card was throttled
    mclk_mhz_max         pp_dpm_mclk starred row | NVML MEM clock
    temp_c_max           hwmon temp1_input       | NVML GPU temperature
    vram_used_b_max      mem_info_vram_used      | NVML memory used
    pcie_tx_kbs_max      null on Radeon          | NVML PCIe TX
    pcie_rx_kbs_max      null on Radeon          | NVML PCIe RX
    sclk_mhz_first       the FIRST sample's clock, untrimmed          (v2)
    temp_c_first         the FIRST sample's temperature, untrimmed    (v2)
    sclk_mhz_min         the lowest clock in the cell, untrimmed      (v2)
    per_card             the same keys, one entry per card

Sampling is a background thread. The default period is 1.5 s; the Radeon
template drops to **0.02 s for prefill**, and `tele_period_s` records which was
used. The first third and last sixth of the samples are dropped, as the Radeon
runners always did: the head is warm-up and the tail is the request draining.

## v2, and the cell that forced it — 2026-09-02b

A 500-token prefill on gfx1100 takes **0.12 s**. At a 1.5 s period that is one
sample, and in four of the five rounds of `B8-tp2-r5` it landed in the idle gap
*before* the request: the row recorded `sclk_mhz_max` **0** against a
`sclk_mhz_cap` of 2075, while the round that happened to sample inside the
request read 2897 against 2897. Nothing in v1 distinguished "the card was idle"
from "we looked at the wrong moment", and the campaign existed to ask what the
card was doing during that 0.12 s.

So v2 adds two things and neither is a new quantity — both are about *when* the
quantity was read. `tele_period_s` puts the sampling rate on the row instead of
in a docstring. `sclk_mhz_first`, `temp_c_first` and `sclk_mhz_min` come from
the **untrimmed** samples, because for a cell shorter than the card's clock ramp
the steady-state window is empty and the only answerable question is what state
the cell started in.

v1 rows stay v1. `tele_schema` is on every row so a reader never has to date
one to know which set it carries.

## `kind: telemetry_meta`, once per configuration

`n_cards`, each card's `slot`, `vram_total_b`, `link_speed`, `link_width`, and
`absent` — the machine-readable list of what this platform cannot measure.

## The two templates

`runner_radeon.py` and `runner_cuda.py` are the files a new campaign copies.
Each is the most recent runner of its family with telemetry wired in and nothing
else touched, because the runners in campaign directories are **records** — each
has exactly one commit, on its own campaign's date, and editing one would
falsify what actually ran.

    runner_radeon.py   campaign-2026-08-30b/runner.py, 77 lines changed
    runner_cuda.py     cuda-l4/campaign-2026-08-30c/run.py, 38 lines changed

The Radeon template also fixes something the old one did that no schema could
have caught: it started the sampler **only for decode**, so every prefill row in
this repository carries no hardware reading at all. Both kinds are sampled now.

Both resolve `harness.telemetry` whether they sit in `harness/` or are copied to
`campaign-*/`, which was checked from both locations.

## Required artefacts

* `results.jsonl` with the above
* one serve log per configuration, under `logs/` or `serve-logs/`, named
  `<cfg>.log` or `serve-<cfg>.log`. The backend, the routing decision and the
  quantisation kernel are parsed from it and exist nowhere else; two campaigns
  that kept none have no `attn_backend` at all and never will.

## What is deliberately absent, and why

**Achieved memory bandwidth.** Not obtainable on the Radeon box. `rocprofv3`
runs in the VFIO guest and its kernel trace is correct — names, grid sizes and
durations all land — but the memory-side counters read zero: over six dispatches
of a 2048² fp32 matmul, `FETCH_SIZE`, `GRBM_COUNT`, `GPUBusy` and
`SQ_INSTS_VALU` each summed to **0.0** while `SQ_WAVES` on the same run returned
**6120**. Some counter blocks survive passthrough; the ones that would answer
this do not. On CUDA it is obtainable — `ncu` is present on Colab and
`dram__bytes.sum` returned 18.71 MB — but only for a microbenchmark, not during a
served run.

So the bandwidth-utilisation figures this repository publishes (75 %, 63 %, 38 %)
remain **derived** from tok/s × bytes/token, and are not cross-checked against
any hardware counter on the machine they describe. `mem_busy_pct` is the nearest
available independent reading and is not the same quantity.

**`pcie_bw` on Radeon.** amdgpu exposes no such node on Navi 31; both cards were
checked. NVML has the equivalent, so the field is in the schema and null on the
Radeon side rather than dropped — the asymmetry is a fact about the machines.

**A second PMC counter in one `rocprofv3` pass.** Aborts with SIGABRT on this
box, at 512² and at 2048² alike; one counter per pass works at both. The axis is
the counter count, not the workload.
