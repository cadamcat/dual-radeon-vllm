# What the second card buys, and what decides it — 2026-09-02d

`allreduce-2026-09-02` measured the collective a TP=2 decode step pays and found
it is not what limits the second card: Qwen3-8B gets **1.70×** from it, gemma-4-12B
**1.19×**, on collectives within 0.6 ms of each other per step. Subtracting
perfect halving and the measured collective left **+0.68 ms** on the 8B and
**+4.97 ms** on the 12B, and that 4.97 was published as a residual with no
explanation.

It has one now, and it is the memory controller. One sitting, four arms, the
sampler on every cell:

| model | ctx | TP=1 tok/s | **TP=1 mem busy** | TP=2 tok/s | TP=2 mem busy | second card |
|---|--:|--:|--:|--:|--:|--:|
| gemma-4-12B w4a16 | 500 | 50.45 | **56 %** | 60.45 | 35 % | **1.198×** |
| | 8 000 | 44.02 | 52 % | 51.73 | 32 % | 1.175× |
| | 32 000 | 36.47 | 44 % | 41.50 | 27 % | 1.138× |
| Qwen3-8B bf16 | 500 | 46.81 | **90 %** | 79.38 | 77 % | **1.696×** |
| | 8 000 | 43.42 | 89 % | 73.25 | 75 % | 1.687× |

**The 8B's single card sits at 90 % memory-controller busy and the 12B's at 56 %,
and the second card is worth 1.70× to the first and 1.20× to the second.** Across
all five cells the gain follows `mem_busy` and nothing else in the row.

That is the whole answer to the residual. The subtraction assumed "TP=2 halves
the bytes each card reads, so the step halves". The bytes *do* halve — the 12B's
`mem_busy` goes 56 % → 35 %, the 8B's 90 % → 77 % — but only a step that was
**waiting** on those bytes gets faster when they arrive sooner. The 8B was
waiting; its subtraction lands within 5.4 %. The 12B was not; its 4.97 ms is not
a missing cost, it is the null model being wrong for a model whose memory
controller is half idle.

## Three candidates, all now eliminated with counters

| candidate | verdict | the number |
|---|---|---|
| the all-reduce | **no** | 1.83 ms of the 12B's 16.70 ms step, and the 8B pays the same and scales 1.70× |
| the power cap | **no** | both models' TP=1 arms sit at **51.4–52.2 % of the 265 W cap** at every depth |
| unmet memory bandwidth | **no, and that is the point** | the 12B's TP=1 memory controller is 44–56 % busy; there was nothing to halve |

The power candidate was worth testing: the same day's campaigns found this pair
at 100.0–100.4 % of its cap at the 32 000 prefill rung, so "the cards are
power-limited, and a power-limited card cannot halve its step when its bytes
halve" was a live explanation. It is dead at decode. **Prefill and decode are in
different regimes on this box** — prefill at depth is at the power limit, decode
is at half of it — which is itself worth knowing and was invisible before the
harness sampled both.

## What the 12B's card is doing instead

At TP=1 it reads 93–100 % `gpu_busy` against 44–56 % `mem_busy`, at 2 897–2 913 MHz
and 52 % of its power cap. The shaders are occupied and neither the memory
controller nor the power budget is the limit. On a w4a16 checkpoint the work
between those two is the dequantise-and-GEMM kernel — `RDNA3W4A16LinearKernel`,
which the route column records for this model — but **this campaign does not
attribute the time kernel by kernel**, and `mem_busy` plus `gpu_busy` cannot do
it. What is established is which three things it is *not*.

## What this settles in the published text

`README.md` has said since 2026-08-27 that w4a16 scales only 1.19 % "because the
quantised model was never bandwidth-bound in the first place". That was an
inference from the scaling itself — the thing it was explaining. It is now a
measurement: **56 % against 90 % on the memory controller**, same two cards, same
sitting, same harness. The a100 article's competing explanation, that the second
card's bandwidth is spent on the wire, was withdrawn earlier the same day.

## The arms reproduce

Decode at the 500 rung, against 2026-08-24:

    A-12B-tp2   59.89 -> 60.45   +0.94 %
    A-12B-tp1   50.56 -> 50.45   -0.22 %
    B-8B-tp2    79.47 -> 79.38   -0.11 %
    B-8B-tp1    46.70 -> 46.81   +0.25 %

KV pools match too: 355 078 against 351 680 tokens on the 12B's TP=2 arm,
151 808 against 149 528 on its TP=1.

## Not the same configuration as 2026-08-24, and the ids say so

`vllm-tp2` carries what August's entry lists — `vllm#45916` split-KV and the
window block-skip, 3 `first_block` sites, cppd md5 `63f0505d` — **and one more
thing it does not: `#45450`, patched** (`4a14f86d` / `7e275cdc`), installed on
2026-08-29 and never reverted. gemma-4 is forced onto TRITON_ATTN by vLLM, and
#45450 patches exactly the two Triton files, so on the 12B it is on the path;
the 8B routes to ROCM_ATTN, where it is not. Hence `-p45450` on all four ids:
these are new configurations, not new rounds. Both arms of each model carry the
same state, so the TP=1 against TP=2 comparison — the whole question — is
internally clean.

Every patch state was read out of the container before the run, which is the
rule `campaign-2026-09-02c` earned the hard way.

## A third instance of an old open item

`B8-tp1-p45450` settled at `mml` **15 792**, so its 32 000 rung does not exist.
The same arm settled at 8 363 in one sitting and 15 792 in another fifteen
minutes apart on 2026-09-02b, and `E26-tp1-u95` gave 32 064, then 13 248, then
13 149 across three starts in seven minutes. Same flags, same container,
different reachable ladder. Still unexplained, and it decides which rungs a
campaign can measure.

## Files

    runner.py                 copied from harness/runner_radeon.py
    results.jsonl             44 measurements, 0 errors, telemetry on every one
    logs/                     one serve log per arm
    serve-*.sh                the exact commands, as the runner wrote them
    host_link.json            preflight, both root ports x16, before the run
    PROGRESS.txt
