# The TP=2 all-reduce, timed — 2026-09-02

This repository published three numbers for the cost of a TP=2 all-reduce on
this box and **withdrew all three** on 2026-08-30, because every one of them
came out of a fitted intercept rather than a clock. The a100 article has said
since then, correctly, that *no all-reduce was timed here.*

Now one has been.

    one all-reduce, batch-1 decode shape, RX 7900 XT pair, PCIe 3.0 x16/x16

    graph replay   16.6 - 21.5 us      <- what a served decode step pays
    back to back   55.2 - 58.8 us
    one at a time  79.1 - 89.1 us

    withdrawn 2026-08-30, from a fit:  1050 us

The withdrawn figure is **49 to 63 times** the measured one. It was not
slightly wrong; it was a different quantity.

---

## What was run

`allreduce.py`, two ranks under `torchrun`, inside the `vllm-tp2` container —
the same container, the same RCCL, and the same environment variables the TP=2
deployment serves with. It reduces a `[ntok, hidden]` bfloat16 tensor over the
five hidden sizes this repository's checkpoints actually use and eleven token
counts from 1 to 16 384, and times each cell three ways.

**The library is the one vLLM calls, checked rather than assumed.** After the
first collective, `allreduce.py` reads `/proc/self/maps` and records what is
mapped:

    /opt/python/.../lib/librccl.so.1
      md5                       ab5b50f0d84806ed7fbe0f4f560151ff
      version string in file    RCCL version 2.27.7
      hidden_hostcall_buffer    0            <- the no-hostcall build

That matches `G31-tp2`'s serve log — `pynccl.py: vLLM is using nccl==2.27.7`,
and `cuda_communicator.py: Using ['PYNCCL'] all-reduce backends (in dispatch
order) for group 'tp:0' out of potential backends: ['NCCL_SYMM_MEM',
'QUICK_REDUCE', 'FLASHINFER', 'CUSTOM', 'SYMM_MEM', 'PYNCCL']`, with
`disable_custom_all_reduce=True` in the engine config. Every faster path is
rejected on this topology. What this script times is not a stand-in for vLLM's
collective; it is the same call into the same library.

> `torch.cuda.nccl.version()` reports **2.30.4** in this container and that is
> wrong for the loaded library — it is a compile-time constant of the torch
> build, and 2.30.4 is the version that does not work on this box at all
> ([open-questions §0](../../docs/open-questions.md)). The results file records
> both, under names that say which is which.

**The link was full width, checked two ways.** `preflight_host_link.sh` read
both host root ports before the run — `8GT/s, Width x16` on each, in
`host_link.json`. Independently, `pcie_probe.py` measured **13.86–13.94 GB/s**
host-to-device and 12.45–14.26 GB/s device-to-host on both cards, which a
PCIe 3.0 x8 link (7.9 GB/s ceiling) cannot reach. The guest's own sysfs cannot
see this — it reports the on-card bridge link, 16 GT/s x16 always — which is
how one card spent 2026-08-29 to 2026-09-02 at x8 unnoticed.

## Three timings, and which one is the answer

| mode | what it includes | batch-1 cost |
|---|---|---|
| `t_graph_us` | N collectives captured in a HIP graph and replayed | **16.6–21.5 µs** |
| `t_stream_us` | N back to back on one stream, one host sync | 55.2–58.8 µs |
| `t_sync_us_median` | one at a time, host sync after each | 79.1–89.1 µs |

The graph number is the one a decode step pays, and that is not a judgement
call. `G31-tp2`'s serve log records `cudagraph_mode:
CUDAGraphMode.FULL_AND_PIECEWISE` and then `Capturing CUDA graphs (decode,
FULL)` — a batch-1 decode step is replayed as one captured graph with its
collectives inside it, which is exactly what `t_graph_us` measures. The
difference between 19 µs and 56 µs is ~37 µs of per-call host dispatch that
graph replay removes; publishing the eager number would have overstated the
collective by 3×, in the same direction as the claim it replaces.

A trivial local kernel of the same size, launched the same way, costs 7.4–8.3 µs
over every cell at 64 tokens or fewer (`t_local_kernel_us`) — with one 80.1 µs
outlier at the very first cell measured, which is the JIT of that kernel and not
a reading. So the collective is not merely a launch.

**The two ranks agree to 0.43%** at every cell (`results.rank1.jsonl` is rank
1's own file, written so this is checkable rather than asserted).

## What it costs a step, and what that explains

`derive.py` does the arithmetic and labels how much each row assumes.

| model | hidden | layers | collectives/step | ms/step | measured TP=2 step | share |
|---|--:|--:|--:|--:|--:|--:|
| Qwen3-8B | 4096 | 36 | 72 | 1.20 | 12.58 ms | 9.5% |
| gemma-4-12B | 3840 | 48 | 96 | 1.83 | 16.70 ms | 11.0% |
| gemma-4-26B-A4B | 2816 | 30 | 60 | 1.29 | 9.29 ms | 13.9% |
| gemma-4-31B | 5376 | 60 | 120 | 2.30 | 23.34 ms | 9.9% |
| Qwen3.8-27B | 5120 | 64 | 128 | 2.46 | 81.27 ms | 3.0% |

Layer counts and hidden sizes are read from each checkpoint's `config.json` on
this box, not remembered. The withdrawn claim's "36 layers × 2 all-reduces = 72
collectives" was applied to a hidden-3840 model; 3840 is the 12B, which has
**48** layers. 36 is the 8B's.

**The finding this campaign exists for.** Two models on one box, one library,
collectives that cost within 0.6 ms of each other per step and take ~10% of the
step in both cases:

| | all-reduce per step | share of step | what the second card buys at decode |
|---|--:|--:|--:|
| Qwen3-8B | 1.20 ms | 9.5% | **1.70×** |
| gemma-4-12B | 1.83 ms | 11.0% | **1.18×** |

Whatever holds the 12B to 1.18×, it is not the wire. If a fixed all-reduce
were eating most of what a second card contributes, the 8B — which pays the
same collective, on the same link, in the same engine — could not be getting
1.70×.

The subtraction says the same thing with an assumption attached. Take
"TP=2 halves the bytes each card reads and adds nothing but the collectives":

| model | TP=1 | TP=2 | predicted TP=2 | residual | all-reduce's share of the shortfall |
|---|--:|--:|--:|--:|--:|
| Qwen3-8B | 21.42 ms | 12.58 ms | 11.91 ms | +0.68 ms | 63.9% |
| gemma-4-12B | 19.78 ms | 16.70 ms | 11.72 ms | +4.97 ms | 26.9% |

For the 8B the measured collective plus perfect halving lands within 5.4% of
the measured step, and the collective is most of the small shortfall. For the
12B the same subtraction leaves 4.97 ms — a third of its step — that neither
the collective nor perfect halving accounts for. **What that 4.97 ms is, this
campaign does not say.** It is a residual, not an explanation, and the
"halves perfectly" premise is itself doing work in it.

> **Answered the same day, and the premise was the problem.**
> [`../campaign-2026-09-02d/`](../campaign-2026-09-02d/README.md) put the
> sampler on both arms of both models in one sitting: the 8B's single card runs
> at **90% memory-controller busy** and the 12B's at **56%**. The bytes do halve
> under TP=2 — 90 → 77% and 56 → 35% — but only a step that was *waiting* on
> them gets faster when they arrive sooner. The 8B was waiting, which is why its
> subtraction lands within 5.4%. The 12B was not, so its 4.97 ms is not a
> missing cost: it is this null model being wrong for a model whose memory
> controller is half idle. The power cap was the other candidate and is also
> out — both TP=1 arms sit at 51.4–52.2% of 265 W at every depth.

## The size curve, and why the floor is not flat

Under graph replay the cost rises with the message from the first token
onward — ×1.44 to ×1.94 by 8 tokens, ×3.25 to ×5.80 by 64 — and reaches a plateau
of **7.35–7.54 GB/s** of ring bus bandwidth at 4 096 tokens and above. It looks
flat in the eager columns only because ~37 µs of host dispatch swamps the first
few sizes there. Anyone reading the eager numbers would have concluded the
collective is pure latency up to 16 tokens; it is not.

The plateau is close to half the 13.9 GB/s one-way ceiling `pcie_probe.py`
measured, which is what a `NCCL_P2P_DISABLE=1` path predicts — with P2P off the
route is device → host → device, so every byte crosses the link twice. That
consistency is worth noting; it is not itself a measurement of the route.

## What is still assumed

**Two collectives per decoder layer.** Every per-step figure above multiplies by
`2 × layers`, on the standard reading that each layer reduces once after
attention's `o_proj` and once after the MLP's `down_proj`, both
`RowParallelLinear`. That is architecture, not something this campaign
measured, and it has not been counted on this stack.

**Isolated cost equals in-step cost.** A collective inside a captured decode
graph sits between real kernels; this one sits between other collectives. Graph
replay removes the host-side difference, but not any device-side one.

## Files

    allreduce.py            the sweep; run under torchrun, 2 ranks
    pcie_probe.py           per-card pinned H2D/D2H, the one-way ceiling
    derive.py               measured -> per step -> cross-check, each labelled
    results.jsonl           rank 0: meta, telemetry, 55 cells, completion
    results.rank1.jsonl     rank 1's own 55 cells
    pcie.jsonl              10 cells, two cards x five sizes
    host_link.json          preflight, both root ports, before the run
    logs/ar.out             the sweep's console output
    logs/pcie.out           the probe's

The run took 7.5 s of measurement. Telemetry over it: both cards at 2 914–2 948
MHz against a 2 942 MHz cap, 123–125 W against a 265 W cap, 30–31 °C. Nothing
was throttled and nothing was warm — an all-reduce is not a thermal workload,
and the numbers are a cold card's.
