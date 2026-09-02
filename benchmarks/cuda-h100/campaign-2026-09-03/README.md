# One rented H100, five models, and the first context past 32 000 — 2026-09-03

Every ladder in this repository stopped at 32 000 tokens because that is what
the machines could hold. This one goes to **128 000** on four of five models,
and it is also gemma-4-31B's first TP=1 arm anywhere here: the 31B has only
ever been measured across two Radeons, or as a partial ladder on an L4 that
could not hold it.

**Five models, 75 measurements, 0 errors, telemetry on every row.** Same
vllm 0.28.0 / torch 2.13.0+cu130 / CUDA 13.0 as every other CUDA row here, so
the machine is the only thing that changed.

| | 500 | 8 000 | 32 000 | 128 000 | 500 → 32 K |
|---|--:|--:|--:|--:|--:|
| `G26A4B` gemma-4-26B-A4B int4 MoE | 230.6 | 226.6 | 217.9 | 189.4 | −5.5 % |
| `G12` gemma-4-12B w4a16 | 160.6 | 157.9 | 154.2 | 141.5 | **−4.0 %** |
| `B8` Qwen3-8B bf16 | 150.7 | 142.1 | 121.8 | — | **−19.2 %** |
| `Q38` Qwen3.8-27B int4 hybrid SSM | 98.5 | 97.0 | 91.9 | 77.1 | −6.8 % |
| `G31` gemma-4-31B w4a16 | 85.9 | 85.1 | 80.7 | 67.0 | −6.0 % |

decode tok/s, median of two rounds. `B8` stops at 32 000 because its own
config.json caps `max_position_embeddings` at 40 960 — for that model the long
rungs are not a budget decision, they are impossible.

## The advantage over an A100 is not a constant, and one model says why

| model | 500 | 8 000 | 32 000 |
|---|--:|--:|--:|
| gemma-4-12B | 1.40× | 1.68× | **2.16×** |
| gemma-4-26B-A4B | 1.43× | 1.67× | **2.08×** |
| gemma-4-31B | 1.47× | 1.65× | 1.90× |
| **Qwen3.8-27B hybrid SSM** | 1.41× | 1.43× | **1.45×** |

Three attention models roughly double their lead between a short prompt and a
32 000-token one. The hybrid SSM does not move at all. Its recurrent state is
a fixed size and does not grow with context, so it has no term that grows with
depth for the faster machine to be faster at — **the same property this
repository's hybrid-SSM article is about, seen from the other side.**

The prefill fits say it in coefficients. `T(S) = a + b·S + c·S²`:

| cfg | machine | backend | `b` µs/tok | `c` ns/tok² |
|---|---|---|--:|--:|
| `G31` | RX 7900 XT ×2 | TRITON_ATTN | 868.7 | 29.06 |
| `G31` | A100-80GB | TRITON_ATTN | 375.4 | 9.13 |
| `G31` | **H100** | FLASH_ATTN | **184.9** | **1.11** |
| `G12` | A100-80GB | TRITON_ATTN | 145.7 | 3.62 |
| `G12` | **H100** | FLASH_ATTN | **72.5** | **0.38** |
| `Q38` | A100-80GB | FLASH_ATTN | 303.2 | 1.45 |
| `Q38` | **H100** | FLASH_ATTN | **160.4** | **0.32** |

The linear term improves about 2×; the quadratic term improves 4.5–9.5×. A
quadratic term that shrinks eight times faster than the linear one is exactly
a machine whose lead grows with depth.

**One caveat, and it is not small: the three gemma arms changed backend as
well as machine.** vLLM routes gemma-4 to `TRITON_ATTN` on the A100 and to
`FLASH_ATTN` here, by its own default and not by any flag this campaign sets —
which is why the runner reads the backend out of the serve log instead of
trusting a table, and why `build_prefill.py` records a `BACKEND_MISMATCH`
against its own `ARMS_CUDA` entry for `G31`. So for gemma the two columns are
machine **and** backend. The only clean single-variable comparison in the
table is `Q38`, FlashAttention on both: `b` 1.89×, `c` 4.5×.

## What limits this card at depth is its power cap

The sampler ran on every measurement. At the 500 rung and at each
configuration's deepest rung:

| cfg | `mem_busy` 500 → deep | `power_w` 500 → deep | `sclk` deep |
|---|--:|--:|--:|
| `B8` bf16 | **87 % → 91 %** | 441 → 646 W | 1 980 MHz |
| `G31` w4a16 | 67 % → 54 % | 493 → **704 W** | 1 755 MHz |
| `Q38` int4 SSM | 65 % → 75 % | 479 → **700 W** | 1 980 MHz |
| `G12` w4a16 | **53 % → 28 %** | 420 → **700 W** | 1 845 MHz |
| `G26A4B` int4 MoE | 38 % → 23 % | 315 → **701 W** | 1 755 MHz |

`gpu_busy` is 100 % everywhere. The card's cap is 700 W and its SM clock
ceiling 1 980 MHz, both read from the machine before the run.

**All four configurations that reach 128 000 sit at the 700 W cap** — 700,
700, 701 and 704 W — and three of the five clock down at their deepest rung:
1 755 MHz twice and 1 845 MHz once, against a 1 980 MHz ceiling, so up to
11 %. `B8`, which stops at 32 000, reaches only 646 W and does not clock down;
`Q38` reaches the cap and does not either.

That is the opposite regime from `campaign-2026-09-02d`, where decode on the
Radeon pair sat at 51–52 % of its power cap at every depth and only prefill
reached it. Same harness, same counters, different answer — and it is recorded
here rather than explained, because a clock that falls 11 % while `mem_busy`
also falls does not by itself say which of the two is cause and which effect.

## `mem_busy` is a property of the model, not of the card

At the 500 rung, on one card:

| model | H100, 2026-09-03 | RX 7900 XT, 2026-09-02d |
|---|--:|--:|
| Qwen3-8B bf16 | **87 %** | **90 %** |
| gemma-4-12B w4a16 | **53 %** | **56 %** |

Two vendors, two architectures, four years apart, and the same pair of models
lands within 3 points on both. `campaign-2026-09-02d` used exactly this
quantity to explain why the second Radeon is worth 1.70× to the 8B and 1.20×
to the 12B; that reading was a claim about one box until today.

It also survives being extended: the ordering across all five models here is
bf16 dense 87 %, w4a16 dense 67 % and 53 %, int4 SSM 65 %, int4 MoE 38 %.
The MoE reads least — it activates 4B of 26B per step.

## What broke, and what it cost

**Attempt 1 — `results-attempt1.jsonl`.** G31 died 187 s into engine start:
`Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`,
from FlashInfer's JIT. Not a gemma-4 problem and not a G31 problem — every
configuration would have gone the same way. `debian_slim` plus `pip install
vllm` has no CUDA toolkit; the nvcc that arrives as a pip dependency is 13.3
and flashinfer 0.6.16's bundled cccl rejects it ("CUDA compiler and CUDA
toolkit headers are incompatible"). The image is now
`nvidia/cuda:13.0.3-devel-ubuntu24.04`, on which
`gen_sampling_module().build()` was watched succeeding in 111 s. 13.0 is also
the toolkit every existing CUDA row here was measured against. **$0.32.**

**Q38's first attempt — `logs/serve-Q38-mamba-crash.log`.** `max_num_seqs
(1024) exceeds available Mamba cache blocks (969)`. A hybrid-SSM model
reserves one Mamba block per decode sequence, and at `mml` 132 000 the state
pool no longer holds 1 024 of them — at 33 000 it does, which is why the A100
ran this model at the default and this campaign could not. The runner now
retries on it the way it has always retried on a short KV pool, and the retry
fired first time: `Q38: Mamba cache holds 969 blocks -> retry mns 969`. The
successful arm is in `results-q38.jsonl`, and its rows carry `mns: 969`.

**The long ladder was nearly refused by arithmetic, not by cost.** The wrapper
prices the remaining configurations from the first one's measured wall clock,
and G31's included a 111 s FlashInfer compile that is cached after the first
container and never paid again. Pricing four more configurations as if they
would each pay it produced "need 53 min + 7 min reserve, have 57 min -> long
OFF" — three minutes short, on a run with $26 of credit left. Corrected, the
same decision on the same ceiling reads "need 33 min, have 70 -> long ON".

## Configuration, and one difference that is on purpose

Every row carries it, but plainly: the four configurations with long rungs ran
at `mml` 132 000 and `B8` at 33 000, against 33 000 for every A100 and Radeon
row they are compared with. At batch 1 `mml` sizes the KV pool and caps
request length; on an 80 GiB card neither binds at either value, and the KV
pools here hold 272 813 to 1 136 951 tokens against ladders that ask for at
most 128 512. It is still a difference, and the control — these same eleven
rungs at `mml` 33 000 on this same card — has not been bought.

`Q38` additionally ran at `mns` 969 rather than the default 1 024, for the
reason above. No other flag differs from the A100 campaign: util 0.90,
`--no-enable-prefix-caching`, batch 1, 512 generated tokens, two rounds.

## Cost

| | |
|---|--:|
| attempt 1, crashed | $0.32 |
| G31 container | $0.86 |
| G12 + G26A4B + B8 container | $1.23 |
| Q38 rerun | $0.74 |
| diagnosis probes (L4, CPU) | ~$0.15 |
| **H100 total** | **$3.30** |

The 82.2 GiB of checkpoints were fetched once on a CPU into a Modal Volume
(`modal-2026-09-02/fetch_models.py`, 169 s) and cost $0.243/day to keep, which
is less than four minutes of the card.

## Files

    run.py                    copied from harness/runner_cuda.py, config table changed
    app.py                    the Modal wrapper: budget rule, JIT cache, harvest
    results.jsonl             G31, G12, G26A4B, B8 -- 59 measurements
    results-q38.jsonl         Q38 after the Mamba retry -- 16 measurements
    results-attempt1.jsonl    the nvcc crash, kept
    serve-*.sh                the exact commands, as the runner wrote them
    logs/                     one serve log per arm, both crashes included
    PROGRESS.txt
