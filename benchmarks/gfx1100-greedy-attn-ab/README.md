# Is the greedy non-determinism the attention backend? — 2026-09-02

**No.** Forced onto `TRITON_ATTN`, with the W4A16 quantisation kernel held at
`RDNA3W4A16LinearKernel` and every run log checked to prove it stayed there,
**both unstable models are still unstable** — each still produces varying greedy
output in at least one of its two cells, where the two models the published
reading called stable produced none in four cells each.

| model | ctx | `ROCM_ATTN` | `TRITON_ATTN` | |
|---|--:|--:|--:|---|
| Muse-Glimmer-30B-INT4 | 512 | **6** of 8 | **7** of 8 | unchanged |
| | 8 192 | **4** of 8 | 1 of 8 | flipped |
| gemma-3-27b-it-w4a16 | 512 | 1 of 8 | 1 of 8 | stable either way |
| | 8 192 | **4** of 8 | **3** of 8 | unchanged |

Three of the four cells give the same verdict either way. The fourth, `muse` at
8 192, flips — and it is the same cell the `--enforce-eager` A/B already found
flipping in the other direction (1 of 8 with graphs, 3 of 8 eager), so it is
borderline rather than a signal. Counting varying cells: **3 of 4 on
`ROCM_ATTN` against 2 of 4 on `TRITON_ATTN`**, on counts that carry the
run-to-run spread this box's earlier campaigns documented — `muse` at 512 has
read 5, 5, 6 and 7 of 8 across four independent engines. **What is read is the
binary, varies or does not**, and on that neither model is rescued by the
backend.

## The confound this was built to break

`../gfx1100-greedy-nondeterminism.json` read the quantisation kernel out of no
serve log. Reading it out of all of them:

| model | W4A16 kernel | attention | unstable cells |
|---|---|---|--:|
| gemma-3-27b-it-w4a16 | `RDNA3W4A16LinearKernel` | `ROCM_ATTN` | 3 of 14 |
| Muse-Glimmer-30B-INT4 | `RDNA3W4A16LinearKernel` | `ROCM_ATTN` | 7 of 14 |
| gemma-4-31B | `RDNA3W4A16LinearKernel` | `TRITON_ATTN` | 0 of 4 |
| Qwen3.8-27B | `RDNAHybridW4A16LinearKernel` | `ROCM_ATTN` | 0 of 4 |

Every unstable cell is `RDNA3W4A16` **and** `ROCM_ATTN`, and each of the two
stable models differs from the unstable ones on a *different* one of those two
axes. That set is consistent with either being the cause and cannot choose
between them. It was read as if it could, twice and in opposite directions: the
published `reading` blamed the attention backend, and a handoff's open
prediction blamed the quantisation kernel.

Holding one axis and moving the other is the whole experiment. The kernel stays
`RDNA3W4A16LinearKernel` in all eight runs — the driver greps every log and
prints it beside the result — and the attention backend moves.

## What this withdraws, and what it leaves

**Withdrawn:** *"the affected models are on `ROCM_ATTN` and its Triton
`kernel_paged_attention_2d`."* They are still affected on `TRITON_ATTN` — 7 of 8
for `muse` at 512 and 3 of 8 for `gemma3` at 8 192 — and the reading attributed
`gemma-4-31B`'s stability to that backend. The sentence is moved into
`reading_withdrawn_2026-09-02` in `../gfx1100-greedy-nondeterminism.json`
rather than deleted.

**Not established:** that the quantisation kernel is the cause. It is the
remaining named candidate, and `gemma-4-31B` runs the *same*
`RDNA3W4A16LinearKernel` and was stable in all four of its cells. Four cells
against Muse's fourteen is thin — at a true rate of one in two, four stable
cells in a row has a 6% chance — but it is what there is, and this campaign does
not resolve it.

**What would:** [vllm#54706](https://github.com/vllm-project/vllm/pull/54706)
replaces that kernel's CAS-atomic split-K epilogue with FP32 partials and a
fixed-order reduction — a deterministic epilogue for exactly this kernel. If it
removes the variation the kernel is the source; if it does not, neither named
candidate survives. **It cannot be A/B'd by swapping a file:** it changes
`csrc/rocm/q_gemm_rdna3.cu` and `q_gemm_rdna3_wmma.cu`, so it needs vLLM's ROCm
extension compiled for gfx1100, and the PR carries no runtime switch. Budget a
build, not an hour.

## The attempt that measured nothing, and why it is kept

`logs-attempt1/` is a complete run of `muse` under both arms in which **both
arms ran on `ROCM_ATTN`**. It set `VLLM_ATTENTION_BACKEND=TRITON_ATTN` before
importing vLLM, on the strength of "0.27 removed it in favour of
`--attention-backend`" from an earlier campaign — which is true of 0.27 and says
nothing about whether 0.23's ROCm selector consults it. It does not:
`rocm.py`'s selector reaches `Overriding with ROCM_ATTN` without ever looking.

The two arms therefore differed in nothing, and gave 6 of 8 and 6 of 8 at 512,
1 and 1 at 8 192 — a result that would have read as *"the backend makes no
difference"* and been right by accident.

What caught it is that `run_attn_ab.sh` greps every run log for the backend
actually chosen and prints it beside the number. Both said `Overriding with
ROCM_ATTN`, where a forced arm says `Using TRITON_ATTN backend`. The corrected
runner passes `attention_backend=AttentionBackendEnum[...]` to `LLM(...)` —
0.23 does have it, `arg_utils.py:905` — and all four logs then name the backend
they were asked for.

> **The rule:** an arm that asks for a configuration is not an arm that got it.
> Grep the log for what the engine chose, print it beside the number, and let
> the reader see the two together.

Kept as a two-sitting repeat of the `ROCM_ATTN` control, which is what it is:
`muse` at 512 gave 6 of 8 in both sittings, at 8 192 one of 8 in that sitting
against four in the next.

## The container's state

`vllm-tp2`, vLLM 0.23.1.dev1+g9ddef7117.d20260715 — the container and version
the published nondeterminism cells ran on. `chunked_prefill_paged_decode.py`
carries the window-skip patch: **3** `first_block` sites, which is the state the
`--enforce-eager` A/B recorded and left. Its md5 is
`63f0505d770aec04476f0127f506e2ac`, read here for the first time — that campaign
recorded only the count, and a count is not an identity. Nothing was patched or
reverted for this one.

> Read because of what happened to `vllm-027` the same day: it had silently lost
> `vllm#45916` and a campaign measured the wrong arm for an hour before the
> numbers gave it away (see `../campaign-2026-09-02c/`). A container's state is
> not what a campaign's notes say it was.

## Files

    nondet_attn.py                nondet_eager.py with the backend as its third
                                  argument, passed as an engine arg
    run_attn_ab.sh                the driver; stops and restores ollama and
                                  llamacpp-hub from an EXIT trap, and greps the
                                  backend and the quantisation kernel out of
                                  every log
    nondet-attn-<model>-<backend>-p1.json   one file per cell, every token
                                            sequence
    logs/                         the four run logs of the measured pass
    logs-attempt1/                the pass in which both arms ran ROCM_ATTN
    nondet-attn-ab.log            the driver's own log, both passes' order and
                                  timings
