# Does `--enforce-eager` remove the greedy non-determinism? — 2026-08-30

**No.** Four cells, one engine each, eight identical greedy generations after a
warm-up. **No cell that varied with CUDA graphs on became stable with eager, and
one cell that was stable became unstable.**

| model | ctx | graphs on | `--enforce-eager` | |
|---|--:|--:|--:|---|
| Muse-Glimmer-30B-INT4 | 512 | **5** of 8 | **5** of 8 | unchanged |
| | 8 192 | 1 of 8 | **3** of 8 | *was* stable |
| gemma-3-27b-it-w4a16 | 512 | 1 of 8 | 1 of 8 | stable either way |
| | 8 192 | **5** of 8 | **2** of 8 | fewer, still varying |

The counts themselves are one process each and carry the run-to-run spread the
original campaign documented — 2 to 3 of 8 at one depth across four independent
engines. **What is read here is the binary, varies or does not**, and on that the
four cells are unambiguous.

## Why this was measured

[`../gfx1100-greedy-nondeterminism.json`](../gfx1100-greedy-nondeterminism.json)
established that greedy decoding is not reproducible on this box, and that the
kernel is not the source: 15 fixed-input cases run in two separate processes
agreed bit for bit. That left the engine, and CUDA graphs are the obvious thing
in the engine between the sampler and the kernel.

Then on 2026-08-29 the reporter of
[vllm#50603](https://github.com/vllm-project/vllm/issues/50603) posted that they
could **not** reproduce either symptom on a rebuilt vLLM 0.25.1, with
`enforce_eager` on. The published cells here ran with it **off** — `nondet.py`
never passes it, and the run logs record `enforce_eager=False` alongside 58
`Capturing CUDA graph` lines — so graphs were an axis that differed between the
two results and had never been tested. Now it has been.

## What it rules out, and what is left

* the kernel — ruled out by the earlier fixed-input result
* CUDA graph capture and replay — ruled out here

Which leaves the vLLM version (0.23.1.dev1 here against their 0.25.1) and the
topology. **The topology axis is not testable from this box**: their W7900D has
48 GB, while Muse-Glimmer-30B-INT4 is 21 GB and gemma-3-27b-it-w4a16 is 19 GB on
disk against 19.98 GiB of card here, so neither model runs at TP=1.

## One qualitative difference worth recording

At `muse` / 512, the eight generations split differently under the two states.
With graphs on, all eight begin with the same token and diverge at index 9 or 23.
With eager, the first three generations begin `303` and the last five begin `258`
— identical from token 1 onward — as though something settles after three
generations. The harness warms up before all eight, so whatever settles is not
covered by that warm-up.

One process, so this is an observation and not a mechanism. It is recorded
because vllm#50603's original claim is that a warm-up call fixes first-call
non-determinism, and this is the shape that claim describes, appearing *after* a
warm-up.

## What is here

    eager-ab.json                 the four pairs, the eight cells, and the caveats
    nondet-eager-<model>-e<0|1>-p1.json   one file per cell, with every token sequence
    logs/                         the four run logs, including the engine config lines
    nondet-eager-ab.log           the driver's own log: order, timings, restore
    nondet_eager.py               the runner
    nondet.py                     the original, so the two can be diffed
    run_eager_ab.sh               the driver

`nondet_eager.py` is `nondet.py` with four executable changes and nothing else:
one extra argument, `enforce_eager=` passed to `LLM`, the print string, and the
output filename. Every measurement constant is identical — the `MODELS` through
`REPEATS` block diffs empty — which is why the graphs-on arm reproduces the
published cells rather than merely resembling them: `muse` at 512 gave 5 of 8
here, inside the 5-to-8 band recorded across four engines before.

## The state the box was left in

`run_eager_ab.sh` stops `ollama` and `llamacpp-hub`, runs, and restores them from
an `EXIT` trap, so an interrupted run still gives the machine back. The kernel
state was left exactly as found — the window-skip patch present, 3 `first_block`
sites — because the original campaign found the effect symmetric between kernel
states, so it is not a variable here either. After the run: both cards back to
27 971 584 bytes of VRAM, both services active, no containers running.
