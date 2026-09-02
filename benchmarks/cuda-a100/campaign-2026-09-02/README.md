# A100-SXM4-40GB, 2026-09-02: what the derived bandwidth figures are worth

**This repository's memory-bandwidth utilisation figures — 75 %, 63 %, 38 % in
[`benchmarks.md §3`](../../../docs/benchmarks.md), and every one in the a100
article's Figure 7 — are *derived*: decode tok/s multiplied by the checkpoint's
size, on the assumption that a decode step reads every weight byte from memory
once. Nothing had ever checked that assumption on any machine.** It cannot be
checked on the Radeon box at all: `rocprofv3` runs in the VFIO guest and its
memory counters read zero. Here it can, and the assumption is wrong by 17–23 %.

## What was measured

`gemma-4-12B-it` and `gemma-4-31B-it`, TP=1, 500-token prompt, eager, under
`ncu` with the profile scoped to an NVTX range around `generate`. vLLM's prefill
emits the first token, so `GEN=1` is a bare prefill and `GEN=8` is that prefill
plus seven decode steps; the two are differenced, which is how the seven decode
steps are isolated from the prefill. `dram__bytes_read.sum` counts what the
memory system actually moved.

| | prefill (500 tok) | per decode step | checkpoint | measured ÷ checkpoint |
|---|--:|--:|--:|--:|
| gemma-4-12B-it | 9.180 GB | **8.375 GB** | 10.265 GB | **81.6 %** |
| gemma-4-31B-it | 20.376 GB | **19.914 GB** | 23.268 GB | **85.6 %** |

Both models read less than their whole checkpoint per token. The direction is
the same and the size is not: 0.816 against 0.856, a 4.7 % spread, so **there is
no single correction factor to apply** — it is a property of the model, not a
constant of the method.

## What that does to the figures on this card

Serving rates are from the telemetry rows in `results.jsonl`, same card, same
day, 500-token context.

| | tok/s | derived GB/s | measured GB/s | derived % | **measured %** |
|---|--:|--:|--:|--:|--:|
| gemma-4-12B-it | 100.50 | 1031.6 | 842.4 | 66.3 % | **54.2 %** |
| gemma-4-31B-it | 50.15 | 1167.0 | 999.5 | 75.0 % | **64.3 %** |

## What is not established

**Nothing here transfers to the Radeon figures.** Those are w4a16 through
`RDNA3W4A16LinearKernel`, these are through `MarlinLinearKernel`; a kernel that
reads its weights differently would have a different factor, and gfx1100's
counters cannot be read to find out. The published Radeon numbers keep their
derived value and now carry a note saying what that means.

**Why the gap exists is a hypothesis, not a result.** The checkpoint contains
weights that a decode step does not read in full — gemma-4's 262 144-entry
embedding table is looked up one row at a time — but the unread amount is
1.89 GB on the 12B and 3.35 GB on the 31B, and this campaign did not attribute
it kernel by kernel.

**The rate comparison is not like for like in time.** `ncu` replays each kernel
in isolation and the profiled run is eager, so its own kernel-time totals
(113.4 ms for the 12B's eight tokens, 196.3 ms for the 31B's) are not serving
times. The bytes are what carries across; the GB/s above uses the *serving*
step time from the telemetry rows, not ncu's.

**One card, one context depth, one round each.**

## The 40 GB part is not the published 80 GB one

Colab Pro had lapsed on this account, the high-memory shape was refused five
times, and a plain A100 is the SXM4-40GB: 1 555 GB/s of HBM against the 80 GB
part's 2 039. Rows land as machine `A100-SXM4-40GB` and are never merged with
the 80 GB rows. Interesting on its own: at 500 tokens this card reaches
**75.0 %** of its narrower HBM on the 31B where the 80 GB part reaches 66.8 % of
its wider one, and the tok/s ratio is 0.857 against an HBM ratio of 0.763 — the
80 GB part is not converting its extra bandwidth into tokens.

## What is here

    results.jsonl       43 rows: 18 measured cells (G31 and G12, 500 x5 /
                        8000 x2 / 32000 x2) with the harness/SCHEMA.md block,
                        plus run_meta, telemetry_meta and model_meta
    ncu-summary.json    the four profiles' totals and per-kernel aggregates.
                        The raw CSVs are 3 327 to 27 771 rows and stayed on the VM
    logs/               both serve logs: TRITON_ATTN, MarlinLinearKernel
    run.py              harness/runner_cuda.py, two configs, per-rung rounds
    setup.py            vllm==0.28.0 pinned, both checkpoints, the ladder
    decode_step.py      the profiled workload, NVTX-scoped
    ncu_decode2.py      the ncu driver. VLLM_ENABLE_V1_MULTIPROCESSING=0 is
                        load-bearing: the first attempt profiled nothing at all
                        because vLLM's V1 engine runs in a child process and
                        ncu did not follow it

    A100-SXM4-40GB · 40 960 MiB · sm80 · 400 W · driver 580.82.07
    vLLM 0.28.0 · torch 2.13.0+cu130 · CUDA 13.0
