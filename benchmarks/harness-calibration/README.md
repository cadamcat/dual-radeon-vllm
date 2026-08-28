# Do this repository's two decode harnesses measure the same thing?

They do, to within half a percent — but only once the machine is warm, and
finding that out took four runs where the first read 31% low.

## Why it was asked

Two harnesses produce decode figures here. The campaign
(`../bench_runner.py`) drives an OpenAI server, streams, samples at temperature
0.8, and reports `(completion_tokens - 1) / (last_token - first_token)` over a
512-token generation. The probes drive `LLM.generate` offline, greedy, and
difference a 64-token run against an 8-token one. Charts that pool the two need
that difference to be a stated number rather than an assumption.

`Qwen3-8B` is the calibration model because it is immune to every other
variable in play: `head_dim` 128, so vllm#45916's split-KV (which requires 256)
cannot apply; no sliding window, so the block-skip patch cannot apply; bf16, so
no W4A16 kernel selection; and `gqa_ratio` 4, which the gfx11 gate admits
either way. A stock 0.23.1 container is therefore comparable to the campaign's
patched one **for this model**, which is the whole reason for choosing it.

Engine knobs are the campaign's: `max_model_len` 33000, `gpu_memory_utilization`
0.85, TP=2, and the same three environment variables. Depths are the campaign's
own `prompt_tokens` at three targets — 511, 8009, 32012 — so the two sides sit
at the same context rather than merely near it.

## The answer

Probe against the campaign's committed rows for `B-8B-tp2` on 2026-08-24, both
rounds:

| campaign target | campaign | probe, converged | |
|---:|---:|---:|---:|
| 500 | 79.47 | 79.12 | **-0.44%** |
| 8 000 | 73.36 | 73.36 | **-0.01%** |
| 32 000 | 61.39 | 61.34 | **-0.07%** |

Matching the campaign's 512-token generation span instead of the probe's 56
changes nothing that matters: -0.97%, -0.27%, -0.19%. So the span is not where
the difference lives, and neither is the server, the streaming, or the
sampling temperature. **The two harnesses are the same measurement.**

## The part that cost three extra runs

Four identical runs, back to back, same container image, same script:

| campaign target | r1 | r2 | r3 | r4 |
|---:|---:|---:|---:|---:|
| 500 | 54.81 | 75.05 | 79.09 | 79.15 |
| 8 000 | 60.70 | 68.80 | 73.22 | 73.49 |
| 32 000 | 55.70 | 61.66 | 61.39 | 61.30 |

Against the converged pair, **the first run reads -30.7%, -17.2% and -9.2%**,
and the second is still -5.1%, -6.2% and +0.5%. r3 and r4 agree to 0.07-0.36%.

The first run was the first container after `systemctl stop ollama
llamacpp-hub`, and it was slow for its whole four minutes rather than only at
its first depth, so this is not the probe's own warm-up generate failing to
warm. The mechanism is not established here. What is established is the size
of the effect and the shape of it: it converges by the third run, and it hurts
the fastest points most, which is why it looked like a harness difference at
first — the 500-token point is where the machine is fastest and the deficit was
largest.

**A single run is not a measurement on this box.** The campaign already does two
rounds per cell and its two rounds agree to 0.2%, because both run after the
server is up and warm. Probe runs that start from a cold machine need the same
treatment, and the first one should be discarded rather than averaged in.

This does not retroactively damage the campaign-style results in this
repository: each of those cells starts a fresh container whose engine
initialisation — model load, torch.compile, CUDA graph capture — takes minutes
before anything is timed, and the arm-order-reversed stage 3 pair agrees to
0.3% at 1K and 0.2% at 32K, which a first-cell penalty of this size would have
broken.

## Files

- `probe_harness_cal.py` — the probe side; `run_harness_cal.sh` — one container,
  one engine, three depths
- `harness-cal-r{1,2,3,4}.jsonl` — every round, including the two that
  were wrong
