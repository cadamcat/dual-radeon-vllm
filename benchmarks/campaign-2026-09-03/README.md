# The pair to 128 000 tokens — 2026-09-03

Every ladder this machine had run stopped at 32 000 tokens, and until this
campaign that read as the machine's limit. It was the campaign's: every arm set
`max_model_len` 33 000 while its KV pool held 2.5× to 10.8× more (the TP=2 arms
whose serve logs are committed: 31B 2.49×, Qwen3-8B 3.90×, Qwen3.8-27B 3.74×,
12B 10.76×). This is the same six models the rented sweep of the same day
measured, on the machine this repository is about, with the ladder carried as
far as each configuration holds it — 352 measurements, 0 errors, telemetry
on every row.

| decode, TP=2 | 500 | 32 000 | deepest | 500 → deepest |
|---|--:|--:|--:|--:|
| gemma-4-12B | 59.34 | 40.97 | **28.21** @ 128 000 | −52.5 % |
| gemma-4-26B-A4B | 104.28 | 71.34 | **47.75** @ 128 000 | −54.2 % |
| gemma-4-31B | 42.41 | 29.38 | **23.27** @ 80 000 | −45.1 % |
| Qwen3-8B | 79.12 | 61.27 | **61.27** @ 32 000 | −22.6 % |
| Muse-Glimmer-30B | 43.09 | 36.99 | **35.64** @ 128 000 | −17.3 % |
| Qwen3.8-27B | 12.33 | 10.79 | **8.93** @ 96 000 | −27.6 % |

## Past 32 000 the curves keep their shape, and the shape is the window

Between 500 and 32 000 every arm here repeats its 2026-08-24 line to within
3 % (below). Past 32 000 nothing bends: the three gemma arms, whose attention
is a 1 024-token sliding window with a global layer every sixth, go on losing
at the rate they were losing and reach 128 000 at roughly half their
500-token rate; Muse-Glimmer, whose every layer attends through a 2 048-token
window, loses 17 % over the same span; the hybrid-SSM 27B loses 28 % to
96 000. That is the ordering the H100 gave the same six checkpoints on the
same day — Muse flattest, the hybrid no flatter than dense — and it is the
ordering [`cuda-modal/`](../cuda-modal/README.md) reads as *a bounded window
flattens a curve, a recurrent state does not*. The pair adds a second reading
of the same order on a third vendor's kernels, and adds what the H100 could not
say: why. The telemetry table below has all three gemma arms at the power cap
with `mem_busy` at 24–25 % by 128 000 — attention has become compute — while
Muse sits at 74 %, still reading weights, because a 2 048 window is the same
amount of attention at 128 000 as at 4 000.

Two things this table does not say. The 27B's level is not the pair's best
for that model: this is the asymmetric-AWQ checkpoint on the 0.23 container,
which misses the native W4A16 kernel and decodes at 12.33 tok/s where Figure 1's
line for the same model, on vLLM 0.27, starts at 49.8 — its *shape* is what
the row is for, and its `mem_busy` at 500 tokens, **15 %**, says the step is
the dequantisation kernel and not memory. And the 31B stops at 80 000 and the
27B at 96 000 for reasons the next section gives, so their "deepest" is not
the same depth as the others'.

## Where the ladder stops, and why

The 31B holds **83 118 tokens of KV** at 85 % utilisation on two 20 GB cards,
so the runner's capacity retry cut its `max_model_len` from 132 000 to 82 558
and the ladder ended at 80 000 — 14 rungs of 16. The 12B and the MoE hold
581 229 and 512 730 and ran all sixteen; Muse-Glimmer holds 484 921 and ran
all sixteen under its own cap of 131 072. The 27B holds 123 399 at
`max_model_len` 122 633, which admits fifteen rungs and not the 128 000 one;
Qwen3-8B holds 122 352 but is capped by its own `max_position_embeddings`,
40 960, not by the pool, so it ran the eleven rungs to 32 000, because 48 000
is past its cap.

The retry is the runner's, not vLLM's: vLLM refuses to start when the pool is
smaller than `max_model_len`, prints the pool size, and the runner reads it
back and restarts 1 % below it (`kv_max_len=83392, mml->82558` in
`results.jsonl`).

## Prefill at depth is quadratic, and the 31B is power-capped from the first rung

Whole-ladder fits of `T(S) = a + b·S + c·S²` to time-to-first-token:

| prefill, TP=2 | rungs | `b` µs/token | `c` ns/token² | `b/c` tokens | `c` share at deepest | TTFT at deepest |
|---|--:|--:|--:|--:|--:|--:|
| gemma-4-12B | 14/16 | 298.3 | 11.48 | 25 974 | 83.1 % @ 128 000 | 226.6 s |
| gemma-4-26B-A4B | 15/16 | 242.2 | 7.18 | 33 733 | 79.1 % @ 128 000 | 148.7 s |
| gemma-4-31B | 14/14 | 738.5 | 28.35 | 26 049 | 75.4 % @ 80 000 | 240.7 s |
| Qwen3-8B | 11/11 | 197.1 | 9.07 | 21 734 | 59.4 % @ 32 000 | 15.6 s |
| Muse-Glimmer-30B | 15/16 | 590.5 | 12.69 | 46 544 | 73.3 % @ 128 000 | 283.8 s |
| Qwen3.8-27B | 14/15 | 821.0 | 8.76 | 93 667 | 50.6 % @ 96 000 | 160.6 s |

R² is 1.0000 on all six. On the four arms that reach 128 000 the quadratic
term owns three quarters of the time at the deepest rung, and the crossover
`b/c` sits at 25 974–46 544 tokens — past where every earlier ladder stopped,
which is why the earlier fits could not see it. The 27B's crossover is at
93 667 because its `b` is the largest here, the dequantisation kernel again,
not because its attention is cheap. Five rungs are below chart grade on
two-round range (the 12B at 500 and 4 000, 14.2 % and 13.9 %; the MoE at
4 000, 16.4 %; Muse at 2 000, 13.1 %; the 27B at 500, 52.4 %, a first request
at 504.7 tok/s against a second at 864.4) and are carried unlit.

Telemetry, decode rows, per card against the 265 W cap:

| | at 500 | at deepest |
|---|--:|--:|
| gemma-4-12B | 171.2 W (65 %), `mem_busy` 34 %, sclk 2 899 | 264.8 W (99.9 %), `mem_busy` 24 %, sclk 2 680, 78 °C |
| gemma-4-26B-A4B | 148.4 W (56 %), `mem_busy` 30 %, sclk 2 936 | 264.7 W (99.9 %), `mem_busy` 24 %, sclk 2 759, 78 °C |
| gemma-4-31B | **264.4 W (99.8 %)**, `mem_busy` 57.5 %, sclk 2 930 | 264.6 W (99.8 %), `mem_busy` 25 %, sclk 2 567, 78 °C |
| Qwen3-8B | 206.1 W (78 %), `mem_busy` 76 %, sclk 2 927 | 263.4 W (99.4 %), `mem_busy` 76.5 %, sclk 2 906, 75 °C |
| Muse-Glimmer-30B | 248.2 W (94 %), `mem_busy` 43.5 %, sclk 2 918 | 264.6 W (99.8 %), `mem_busy` 74 %, sclk 2 537, 78 °C |
| Qwen3.8-27B | **264.3 W (99.7 %)**, `mem_busy` 15 %, sclk 2 645 | 264.7 W (99.9 %), `mem_busy` 61 %, sclk 2 726, 78 °C |

The 31B and the 27B are at the power cap at 500 tokens already; the others
reach it on the way down. The bf16 8B is the one arm whose `mem_busy` stays
high at depth, 76 % at both ends, and Muse-Glimmer is the one whose `mem_busy`
*rises* with depth, 43.5 % to 74 %: its attention is a 2 048-token window, so
the step at 128 000 is still a weight-read, not an attention scan, and its curve
is the flattest on the pair (−17.3 %) as it was on the H100 (−4.8 %). Those two
and the 27B are the three arms on ROCM_ATTN — no serve script here passes a
backend, and vLLM's own `Overriding with ROCM_ATTN` line chose it for the bf16
8B, the int4 Muse and the AWQ 27B as on every earlier sitting, where the three
w4a16 gemma arms report `Using TRITON_ATTN backend`. At the deepest rung the
three gemma arms sit at the cap with `mem_busy` at 24–25 % and the shader clock
150–350 MHz below its 500-token value: at depth the pair is compute-bound and
power-limited, not bandwidth-bound. The ordinal `mem_busy` rule of
campaign-2026-09-02d was about a second card's value at 500 tokens; at
128 000 the quantity it orders has moved from 34 % to 24 % on the 12B, and
what that does to the second card's value is not measured here — TP=1 did not
run on this ladder.

## The overlap with the eleven-rung ladder is a measurable difference

The first eleven rungs repeat 2026-08-24's ladder on the same arms. Decode
agrees: worst rung 1.8 % (12B), 3.1 % (MoE, every rung 2–3 % slower), 1.0 %
(31B), 0.4 % (8B), 1.6 % (Muse, every rung 1–1.6 % slower), 1.0 % (27B).
Prefill agrees at depth and not at the MoE's short rungs:

| gemma-4-26B-A4B prefill, tok/s | 500 | 1 000 | 2 000 | 32 000 |
|---|--:|--:|--:|--:|
| 2026-08-24, two rounds | 1 694 / 1 772 | 2 997 / 2 945 | 3 190 / 3 157 | 2 114 / 2 110 |
| 2026-09-03, two rounds | **3 350 / 3 248** | 3 495 / 3 492 | 3 548 / 3 554 | 2 117 / 2 114 |

+90 % at 500, +18 % at 1 000, +12 % at 2 000, +0.2 % at 32 000, with both
rounds agreeing on both days. The 12B, 31B, 8B, Muse and 27B prefill rows agree
within 3.6 % at every rung. Whatever moved is MoE-specific and
short-prompt-specific; the two containers differ by the 2026-09-02d patch set
(vllm#45450's 3D admission, vllm#45916's split-KV, the window block-skip) and
by prefix caching being on here, and this campaign does not separate them. The
rows are recorded as they are; `build_prefill.py` fits each date on its own.

## The prompts are a new cut, and most of the old rungs reproduce

`prompts-v2/` holds four manifests of a sixteen-rung ladder (500 → 128 000)
cut in one pass on 2026-09-03 in the `vllm-tp2` container from Gutenberg
#1228, fetched the same morning. The committed eleven-rung manifests were not
extended in place, because the cut could not be reproduced for every
tokenizer:

| tokenizer | rungs of 11 with the same text | what moved |
|---|--:|---|
| Muse-Glimmer | 11 | nothing |
| Qwen | 10 | 12 000: 12 044 → 12 000 tokens, −198 chars |
| gemma-4-26B | 6 | 12 000 – 32 000 recut, −18 to +78 tokens |
| gemma-4 (12B, 31B) | 2 | every rung but 1 000 and 8 000 recut, +6 to +56 tokens; those two share their text and count **ten tokens fewer** |

Same text, fewer tokens, is the tokenizer moving, not the text — the container
carries transformers 5.14.0 and tokenizers 0.22.2, and the committed cut does
not record the versions it was made with, so which side counts 481 and which
498 at rung 500 is not pinned here. Every row in `results.jsonl` carries
`prompt_tokens` as the server counted it, and the projections use that, not
the manifest.

## The container, read before the run

vllm 0.23.1.dev1+g9ddef7117.d20260715, ROCm 7.14, kernel 7.0.0-30,
`chunked_prefill_paged_decode.py` 63f0505d (vllm#45916 split-KV and the window
block-skip, three `first_block` sites), `triton_attn.py` 7e275cdc (vllm#45450).
The same state as campaign-2026-09-02d. The runner does not assert those
md5s — they were read out of `vllm-tp2` before the run, and the proof that
every arm ran in that container is the version line each serve log opens
with, `0.23.1.dev1+g9ddef7117.d20260715`, which the other container
(`vllm-027`) cannot print. `host_link.json` is the preflight's reading, both
root ports x16 — taken at 09:25:42 UTC, eight minutes after the run started at
09:17:50 rather than before it, which departs from the rule and is recorded as
such; the width does not change without a link retrain, and the 08-24 overlap
above is the same evidence the rule was written from.

## Three models rev2 could not start, and what happened to them

`max_model_len` was 132 000 for every arm. Two checkpoints cap their own
positions below that — Qwen3-8B at **40 960** and Muse-Glimmer-30B at
**131 072**, read from each `config.json` (top level for the 8B, `text_config`
for Muse; `position_caps.json` holds all six readings with each file's md5) —
and vLLM 0.23 refuses a `max_model_len` above `max_position_embeddings` at
configuration time (`vllm/config/model.py`, the derived-maximum check) unless
`VLLM_ALLOW_LONG_MAX_MODEL_LEN` is set, which would measure positions the
model was never trained on. Both died before loading a weight, 36 s after the
previous arm finished, and are `config_failed` rows in `results.jsonl`; the
first three attempts of this campaign had died the same way in the other
container (`results-failed-attempts.jsonl`), and the runner had not learned
from it.

The third is Qwen3.8-27B, which passed the position check (262 144) and the
capacity retry (132 000 → 122 633) and then died at CUDA-graph capture,
twenty minutes in and ten after the retry: `max_num_seqs (256) exceeds
available Mamba cache blocks (161)`. A hybrid-SSM model reserves one Mamba
block per decode sequence, and at a long `max_model_len` the state pool holds
fewer of them than the default `max_num_seqs`; the H100 runner met the same
wall on 2026-09-03 (969 blocks against 1 024) and learned to retry with
`--max-num-seqs`. The Radeon runner had not. Rev2's serve log for that
attempt is gone — the runner writes one serve log per arm, and rev3's attempts
overwrote rev2's before the failure logs were copied off the box, a gap in the
harvest rule recorded as one — so the 161 stands on `runner.py`'s comment and
on rev3 reading the same count at the same `max_model_len` (below).

`runner.py` here is rev3 (md5 8ddba508…): per-configuration `max_model_len`
(40 960 and 131 072 for the two capped checkpoints), the Mamba-blocks message
parsed into a `--max-num-seqs` retry, and up to four attempts per arm. It was
started as unit `c0903c` at 12:54:49 UTC, after `c0903b` (rev2, md5 eee03899…,
kept as `runner-rev2.py`) had finished, and it skipped the three complete arms
by their `config_complete` rows. Rev3 measured the 8B to 32 000 (the 40 960
cap admits 32 000 and not 48 000; 44 measurements) and Muse-Glimmer to 128 000
(131 072 admits every rung; 64 measurements, KV pool 484 921 tokens), and for
Qwen3.8-27B it took all four attempts: at 132 000 the Mamba pool held **198**
blocks and it restarted with `--max-num-seqs 198`; with 198 sequences the KV
pool no longer held 132 000 and the capacity retry cut `max_model_len` to
122 633; at 122 633 the Mamba pool held **161** — rev2's number — and the
fourth start, `--max-num-seqs 161`, was the one that served (the four `note`
rows in `results.jsonl`, 14:21 to 14:40 UTC; ready at 14:52). It then measured
fifteen rungs to 96 000, 60 measurements, 0 errors.

## Files

    runner.py                     rev3; runner-rev2.py is what c0903b ran
    results.jsonl                 352 measurements, 0 errors, plus the three config_failed rows
    results-failed-attempts.jsonl the three attempts in the wrong container
    logs/                         one serve log per arm (the runner writes serve-logs/, which is not tracked)
    serve-*.sh                    the exact commands, as the runner wrote them
    prompts-v2/                   the four manifests of the 2026-09-03 cut
    host_link.json                both root ports x16, read eight minutes after the start
    position_caps.json            max_position_embeddings of all six checkpoints, read
                                  from their config.json in the container, with md5s
    PROGRESS.txt                  the runner's own timeline, both units
