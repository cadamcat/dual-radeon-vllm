# Seven rented machines in one night — 2026-09-03

Every projection in this repository was built from two Radeons under a desk and
whatever Colab offered. This is what $29.94 of rented time added: **six machine
configurations on the decode/prefill ladder, eight on the collective, 586
measurements, 0 errors**, and the first context past 32 000 anywhere here — four
of six models to **128 000**.

Same `vllm 0.28.0 / torch 2.13.0+cu130 / CUDA 13.0` as every CUDA row already
in the repository, the same six checkpoints from one Volume at pinned commit
revisions, the same eleven-rung ladder cut from Gutenberg #1228.

    cuda-h100/campaign-2026-09-03       5 models x 16 rungs, TP=1     75 points
    cuda-h100/campaign-2026-09-03b      Muse-Glimmer + three controls 49
    cuda-h100/campaign-2026-09-03-tp2   5 models, TP=2                75
    cuda-h100/campaign-2026-09-03-tp4   Qwen3-8B, TP=4                11
    cuda-h200/campaign-2026-09-03       6 models, TP=1                91
    cuda-b300/campaign-2026-09-03       6 models, TP=1                91
    cuda-pro6000/campaign-2026-09-03    6 models, TP=1                91
    cuda-pro6000/campaign-2026-09-03-tp2 5 models, TP=2               75
    cuda-a100/campaign-2026-09-03       platform control              12
    cuda-l4/campaign-2026-09-03         platform control              16
    allreduce-2026-09-03                7 collective configurations  385 cells

## The two controls that let any of this be read

Every ratio below divides a Modal card by an A100 or an L4 that **Colab**
measured. That gap was assumed until it was measured:

| gemma-4-12B, A100-80GB, `mml` 33 000 | 500 | 8 000 | 16 000 | 32 000 |
|---|--:|--:|--:|--:|
| Colab 2026-08-29 | 115.05 | 94.17 | 83.26 | 71.26 |
| Colab 2026-08-30 | 114.90 | 94.02 | 83.09 | 71.25 |
| **Modal 2026-09-03** | **114.95** | **94.00** | **83.05** | **71.30** |
| Modal against Colab 08-30 | +0.04 % | −0.02 % | −0.06 % | +0.07 % |

Four rungs inside **0.07 %** — closer than Colab's own two sittings agree with
each other. The L4 control says the same on a card that appears in no
conclusion: 25.29 tok/s at 32 000 against Colab's 25.07 and 25.17.

The band was not chosen after seeing the answer. Colab measured that A100 twice
and that L4 twice, in August, and those four rows have been committed since.

## `mem_busy` predicts, ordinally, in five independent settings

`campaign-2026-09-02d` found that what a second Radeon is worth is decided by
the memory controller's busy fraction, not by the interconnect. That was five
cells on one box. It now orders the answer in five settings that share no
hardware:

| setting | most memory-bound (`B8`, 87 %) | least (`G26A4B`, 38 %) |
|---|--:|--:|
| second Radeon (2026-09-02d) | 1.696× | 1.198× (12B, 53 %) |
| **H200 / H100**, bandwidth alone | **1.254×** | **1.044×** |
| **B300 / H100** | 1.660× | **0.995×** |
| **RTX PRO 6000 / H100** (slower) | **0.554×** | **0.827×** |
| **second H100**, NVLink | 1.484× | **1.029×** |

Read the last row twice: **38 % `mem_busy`, and the second H100 is worth 2.9 %.**
A $3.95/h card added to another one, for nothing. And the PRO 6000 row runs the
other way — the model that gains most from more bandwidth loses most from less.

**It does not predict magnitude.** `cuda-h200/campaign-2026-09-03/PREDICTION.md`
was committed before that run's data, applying `1/((1-f) + f/r)` with `f` =
`mem_busy` and `r` = the two cards' measured memory clocks. The ordering came
back right at both ends and the spread came back nearly double the prediction —
0.210 measured against 0.114 — with `B8` at 1.254× exceeding `r` itself, which
that form cannot produce. Ordinal, not cardinal, and now on the record as such.

## Four cards, three attention backends, nobody asked for any of them

| | H100 | H200 | B300 | RTX PRO 6000 |
|---|---|---|---|---|
| gemma-4 ×3 | FLASH_ATTN | FLASH_ATTN | **FLASHINFER** | **TRITON_ATTN** |
| Qwen ×2, Muse-Glimmer | FLASH_ATTN | FLASH_ATTN | **FLASHINFER** | FLASH_ATTN |
| Muse-Glimmer's W4A16 kernel | **Machete** | **Machete** | Marlin | Marlin |

One vLLM, one command line, no backend flag anywhere in this campaign. Every
cross-machine comparison here therefore carries a backend term, and the only
reason that is known is that the runner reads the backend out of each serve log
instead of trusting a table — `build_prefill.py` records the disagreement
against its own `ARMS_CUDA` fallback rather than resolving it.

**Within the PRO 6000 the backends separate cleanly**, same card, same run:

| PRO 6000, 500 → 128 000 | backend | change |
|---|---|--:|
| gemma-4-12B | TRITON_ATTN | **−52 %** |
| gemma-4-26B-A4B | TRITON_ATTN | **−51 %** |
| gemma-4-31B | TRITON_ATTN | −39 % |
| Qwen3.8-27B | FLASH_ATTN | −28 % |
| Muse-Glimmer-30B | FLASH_ATTN | **−13 %** |

Three Triton arms collapse at depth and two FlashAttention arms do not. The
backend is confounded with the model, so this is not attribution — but it is
the same card in the same sitting, which the cross-machine version is not.

## The newest card is not the fastest card

| 500-token decode | B8 (87 %) | G12 (53 %) | G26A4B (38 %) |
|---|--:|--:|--:|
| H100 80GB, $3.95/h | 150.7 | 160.6 | 230.6 |
| B300 SXM6, $7.10/h | **250.2** | 166.5 | **229.3** |
| B300 / H100 | 1.66× | 1.04× | **0.995×** |

**A B300 loses to an H100 on gemma-4-26B-A4B and beats it by 4 % on
gemma-4-12B**, at 1.8× the price, while winning by 66 % on Qwen3-8B. It also
changed attention backend, so machine and backend move together in that
column; the control that would separate them is not available — `vllm serve`
in 0.28 has no flag for the decoder's backend, which was checked in the
installed package rather than assumed.

## The collective has 62× of dynamic range and inference uses none of it

`allreduce-2026-09-03`, hidden 4096 (Qwen3-8B), `t_graph_us`:

| | n=1, a decode step | n=16 384 | ratio |
|---|--:|--:|--:|
| H100 ×2, NVLink | **12.06** | 474 | 39× |
| H100 ×4, NVLink | 14.76 | 617 | 42× |
| B300 ×2 | 14.24 | **290** | 20× |
| A100 ×2 | 17.74 | 759 | 43× |
| RTX PRO 6000 ×2, **no NVLink** | 14.44 | 3 775 | 261× |
| RTX PRO 6000 ×4, **no NVLink** | **39.10** | 11 294 | 289× |
| RX 7900 XT ×2, no P2P at all | 16.65 | **18 050** | 1 084× |

All of these are hidden 4096, one model, so the two ends are comparable. Across
the seven configurations the collective's **bandwidth end spans 62×** and its
**latency end spans 3.2×**; among the pairs alone the latency end spans
**1.5×** against the same 62×. Batch-1 decode reduces one row and lands on the
latency end. That is why the second card's value tracks `mem_busy` and not the
interconnect: the interconnect's range is real and inference never enters it.

*(Two corrections before publication, both from the gate that recomputes this.
The latency range was first written as 1.8× for the whole set — that is the
pairs only, and the RTX PRO 6000 ×4 at 39.10 µs puts the full set at 3.2×. The
pairs figure was then written as 1.8× as well, which came from reading 21.49 µs
off the Radeon's hidden-2816 row while every other number here is hidden 4096;
at 4096 the pairs span 1.5×.)*

**What NVLink is for is the fourth card, not the second:**

| adding cards three and four | decode | bandwidth |
|---|--:|--:|
| H100, NVLink | ×1.22 | ×1.30 |
| RTX PRO 6000, no NVLink | **×2.71** | **×2.99** |

Two cards without NVLink cost 20 % over two with it. Four cost 171 %.

## Context past 32 000, and what makes a curve flat

| 500 → 128 000, decode | H100 | structure |
|---|--:|---|
| Muse-Glimmer-30B | **−4.8 %** | attention through a 2 048 window |
| gemma-4-12B | −11.9 % | sliding 1 024, 1 global layer in 6 |
| gemma-4-26B-A4B | −17.9 % | as above, MoE |
| Qwen3.8-27B | −21.8 % | **hybrid SSM** |
| gemma-4-31B | −22.0 % | sliding 1 024 |

**The recurrent-state model is not the flat one.** Qwen3.8-27B falls as far as
the dense 31B. What flattens a curve is a bounded attention window, not a
constant-size state — the SSM still carries attention layers whose cost grows.

The same ordering appears in what a faster machine is worth: `MG30` gains
1.37× from an A100→H100 move at 32 000 where the gemma models gain 1.90–2.16×
and the SSM 1.45×. Two independent quantities, one ordering.

This falsified a prediction written into `campaign-2026-09-03b/run.py` before
the run: Muse-Glimmer was expected to land **between** the attention models and
the SSM. It landed below both. `MG30` is also the one model whose W4A16 kernel
differs across machines (Machete on the Hoppers, Marlin elsewhere), which makes
its ratio machine-and-kernel — and Machete is the faster of the two on Hopper,
so the confound runs against the conclusion rather than toward it.

## What is not here, and why

* **A backend control on the B300.** `attention_backend` exists only in
  `config/speculative.py` in vLLM 0.28 — there is no flag for the decoder's,
  and `vllm serve --help` exits non-zero without a GPU, so this was read from
  the installed source. Not measured.
* **gemma-4-26B-A4B at TP=4.** `CompressedTensors WNA16 MoE with static group
  scales requires the MoE intermediate size per tensor-parallel partition to be
  divisible by group_size (32)` — an arithmetic property of the checkpoint, not
  a failure to retry. TP=2 works; TP=4 cannot.
* **The other four models at TP=4**, and Muse-Glimmer at TP=2. Budget.
* **Achieved memory bandwidth.** `harness/telemetry.py` records it as absent on
  both platforms and says why. `mem_busy` is a busy fraction; 87 % at one
  absolute rate is not 87 % at another, and nothing here separates them.

## Two mistakes, both corrected in place

**`gpu="H100:4"` returned four H200s** on one collective sweep while the same
string returned four H100s for a ladder an hour earlier. Caught by
`nvidia-smi nvlink -s` printing the card name, confirmed by `vram_total_b`
(150 754 820 096 against 85 520 809 984 per card), and the file was renamed and
every row given `mislabelled_as` and `relabel_reason`. The data is unchanged;
only the label was wrong. `allreduce_app.py` now reads the device and asserts,
and prints both what was asked for and what arrived.

**Two runs shared a work directory** on the Volume because both were launched
with the same `--machine`, so one file held 118 rows from two machines. Split at
the `ar_meta` boundary and identified by `vram_total_b`; both halves carry the
note.

## Cost

| | |
|---|--:|
| probes, checkpoint fetch, diagnosis | $0.26 |
| six machine configurations of ladders | $22.6 |
| eight collective configurations | $0.17 |
| two platform controls | $0.62 |
| failed starts and retries | $0.78 |
| controls, Muse-Glimmer, re-runs | $5.5 |
| **total** | **$29.94 of $30.00** |

The checkpoints were fetched once on a CPU (`modal-2026-09-02/fetch_models.py`,
102.9 GiB, 169 s) and the Volumes were deleted when the sweep ended;
`volume.json` holds the six commit revisions that rebuild them.
