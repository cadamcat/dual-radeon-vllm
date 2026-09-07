# The depth cost was the software's: 0.350 → 0.111 µs/tok, same card, same weights — 2026-09-07

[`campaign-2026-09-06`](../campaign-2026-09-06/README.md) showed that the
published 27B depth curve's *percentage* belonged to vllm 0.23.1 and that its
*slope* did not move with it nearly as much. This one asks how much of the slope
is the software too, and the answer is: nearly all of it.

Three sittings in one run, same checkpoint, same two cards, same TP=2, same
split-KV patch, same `mml 130000 / util 0.92 / max-num-seqs 16`, sixteen rungs to
128 000 and two rounds a cell. **140 measurements in 70 cells, 0 errors.**

| over 500 → 128 000, decode | intercept | slope | r² | 500 → 128 000 |
|---|--:|--:|--:|--:|
| **A** `ROCM_ATTN` — what vLLM picks here | 20.08 ms | **0.235 µs/tok** | 1.0000 | −59.7 % |
| **B** `TRITON_ATTN` — the other candidate | 19.79 ms | **0.111 µs/tok** | 0.9998 | −41.7 % |

**The intercepts agree to 1.5 % and the slopes differ by 2.11×.** The backend
moves the depth term and leaves the baseline alone, which is what makes this an
attribution rather than a comparison: it is not that one backend is faster, it
is that one costs less than half as much per token of context.

## Why that is the backend and not the session

A and B cannot share a serve session — the backend is chosen at startup — so an
A–B difference is a backend difference only if the session boundary costs less
than it does. **C is the same arm as A, repeated at three rungs after B**, and it
is not an optional part of this campaign: `rccl-ndebug-ab-2026-09-04` published a
3.6 % "arm effect" that turned out to be session order, and this is the design
that catches it.

| rung | A | C, one session later | drift |
|---:|--:|--:|--:|
| 500 | 49.36 | 49.56 | +0.40 % |
| 8 000 | 45.54 | 45.62 | +0.18 % |
| 32 000 | 36.24 | 36.42 | +0.50 % |

**At most 0.50 %, against an A–B gap that reaches 48 %.** Two orders of
magnitude apart, so the attribution holds.

## The trade, and that it has not turned over at 128 000

| rung | decode T/R | prefill T/R |
|---:|--:|--:|
| 500 | 1.02× | 0.99× |
| 8 000 | 1.06× | 0.89× |
| 32 000 | 1.18× | 0.69× |
| 64 000 | 1.30× | 0.56× |
| 96 000 | 1.40× | 0.48× |
| **128 000** | **1.48×** | **0.44×** |

`campaign-2026-09-02c` measured these two to 32 000 and found them trading; both
gaps were still widening where it stopped, and **both are still widening at
128 000**. Neither backend is the better one. They divide the cost between
prefill and decode, and which division a deployment wants is a question about its
traffic, not about this machine.

## What this does to the ordering the front page draws

The six arms of `campaign-2026-09-03`, on the span all six share (500 → 32 000),
with this checkpoint's two new arms placed among them by the same fit:

| slope µs/tok | arm |
|--:|---|
| 0.066 | Muse-Glimmer-30B — **but see below** |
| 0.113 | Qwen3-8B |
| **0.117** | **Qwen3.8-27B, 0.27 + TRITON_ATTN** |
| 0.148 | gemma-4-26B-A4B |
| **0.233** | **Qwen3.8-27B, 0.27 + ROCM_ATTN** |
| 0.254 | gemma-4-12B-it |
| 0.342 | gemma-4-31B-it |
| **0.350** | **Qwen3.8-27B, 0.23.1 — the published arm** |

**The same checkpoint occupies the first, fifth and last places of its own
ranking**, and nothing but software moved between them. `campaign-2026-09-03`
reads its ordering as a fact about attention structure; on this evidence the
ordering is a fact about attention structure *and the kernel that implements it*,
and here the kernel is the larger term.

**Muse-Glimmer's 0.066 is not the same kind of number.** Its ms/tok curve fits a
line at **r² 0.3145** where every other arm is above 0.978: a 2 048-token window
bounds its cost instead of growing it, so a linear slope cannot describe it. The
figure is listed, but it is not comparable.

## What this does not establish

- **Nothing about the other five checkpoints.** Only this one was measured on
  both backends. gemma-4 cannot be served on the 0.27 image at all — its Quark
  plugin reads a heterogeneous `head_dim` and dies before loading — so the six
  cannot be re-run together on this stack in any case.
- **Nothing about why.** The two arms differ in the decoder's attention kernel
  and this campaign does not open either. That prefill moves the other way is
  the shape of a trade, not an explanation of one.
- **Nothing about stock Triton.** Arm B ran the Triton path carrying vllm#45450, the
  state this container has held since `campaign-2026-08-29`; the image's own Triton is
  `49fab3b6` and was not measured here. What is compared is two backends in one container,
  not two upstream defaults.
- **Nothing about batch.** Every cell here is batch 1. `max-num-seqs` is 16 for
  capacity, not for load; a serving deployment at depth is a different question.
- **Two sittings of arm A, on different days, differ by up to 1.23 %**
  (`campaign-2026-09-06`'s eight rungs against A's). Within one run the boundary
  is worth ≤ 0.50 %. Both are small beside what is claimed here.

## Provenance

    container    vllm-027, rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0
    vllm         0.27.1.dev5+gf46a9dfe2.d20260827        ROCm 10.0, kernel 7.0.0-30
    patch        the two attention paths are independent files, and this run asserted only
                 one of them before measuring -- the other was read out of the container
                 afterwards, on 2026-09-07:
                   chunked_prefill_paged_decode.py  84c6d4f9...  vllm#45916   ROCM_ATTN's path
                   triton_unified_attention.py      9416a868...  vllm#45450   TRITON_ATTN's
                   triton_attn.py                   8bd13173...  vllm#45450   TRITON_ATTN's
                 The two Triton md5s are what `campaign-2026-09-02c/runner.py` records for
                 vllm#45450, and they are the state its own Triton arm ran, so the two
                 campaigns' Triton arms are the same arm. **Arm B is not stock Triton**, and
                 a reading of it as "TRITON_ATTN out of the box" would be wrong.
                 Both arms ran in one container with those three files in one state; only
                 `--attention-backend` differed between them, which is what makes A-B the
                 backend and not the patch state. `run.sh` asserting one path and not the
                 other is a gap in the runner, not in this comparison
    checkpoint   /data/incoming/Qwen3.8-27B-AWQ-INT4
    serve        TP=2, mml 130000, util 0.92, max-num-seqs 16; arm B adds
                 --attention-backend TRITON_ATTN and nothing else
    host link    both cards x16 at the host root ports, read before the run (host_link.json)
    prompts      /data/rccl-build/v2/prompts-qwen, the 2026-09-03 cut — the same sixteen
                 rungs campaign-2026-09-03 measured, so the ladders are one ladder
    runner       runner.py, campaign-2026-09-06/runner.py with three arms and a per-cfg
                 rung list; the drift arm asks for three rungs rather than sixteen
