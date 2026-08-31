# Does the BLOCK_M crossover move with KV depth on CUDA? — 2026-08-31

**It moves, and it moves the other way.**

[vllm#52684](https://github.com/vllm-project/vllm/pull/52684)'s author measured
on one RX 7900 GRE that fixing `kv_len` at 4096 pulls the `BLOCK_M=16` →
`BLOCK_M=64` crossover *left*, from `q_len` 96–128 down to 32–64. On an A100 the
same fixed depth does the opposite: the short-`q_len` region collapses to
0.54–0.66, which is `BLOCK_M=64` running **1.5–1.9× slower**.

Median over the two head patterns, `production_ms / bm64_ms`, so **> 1 means
`BLOCK_M=64` is faster**:

| `q_len` | `kv=q_len` | `kv=4096` | `kv=16384` |
|--:|--:|--:|--:|
| 16 | 1.000 | **0.599** | **0.546** |
| 32 | 0.930 | **0.537** | **0.543** |
| 64 | 0.981 | **0.601** | **0.561** |
| 96 | 0.988 | **0.622** | **0.586** |
| 128 | 0.976 | **0.663** | **0.651** |
| 256 | 0.978 | 1.138 | 1.183 |
| 512 | 1.009 | 0.934 | 0.938 |
| 1024 | 1.028 | 1.051 | 1.049 |

His gfx1100 curve at the same fixed depth runs 0.84, 0.82, 0.90, **1.83**, 1.31,
1.85 across `q_len` 1 → 128: already 1.8× ahead at 64, where this card is 0.60.

The mechanism is not mysterious. At `q_len=16` a 64-wide query block is
three-quarters padding, and the kernel still scans every one of the 4096 or
16384 keys. On gfx1100 the wider block wins anyway; on this card the wasted work
dominates.

## Why this run exists

Our earlier pass, [`../52684-blockm`](../52684-blockm), sets `kv_len = q_len` on
all 140 of its rows, so it cannot speak to this axis at all. That was said in the
thread, the author agreed the KV-depth result was his alone and rested on one
board, and asked for this:
[issuecomment-5473180002](https://github.com/vllm-project/vllm/pull/52684#issuecomment-5473180002).

## Two deliberate departures from `../52684-blockm`

* **`kv_len` is a parameter** rather than `kv_len = q_len`.
* **Both arms are forced past the PR's own gate.** That gate is `q_len >= 512`,
  and the crossover under test sits at `q_len` 32–128, entirely below it —
  `52684-blockm`'s `select_for()` collapses every arm onto base there, which is
  right for asking "what does the PR do as written" and useless for asking where
  the crossover is. The author forced the same way, through a launch proxy.

Arms, head patterns and the speedup convention are all his, so the two sets land
on one axis: `production` is `BLOCK_M=16` with Triton's default warp count,
`bm64` is `BLOCK_M=64` with `num_warps=4`.

## The constant that had to be explained before the short rows could be used

Pass 1 timed one call between one pair of CUDA events, and every short-`q_len`
row came out near 0.2 ms whatever the shape — which looks exactly like an
instrument floor, and would have made the rows that matter worthless.

Pass 2 times **50 calls between one pair of events** and divides. The constant
survives, so it is a real per-call cost — the Python wrapper around the kernel —
and not a floor. Four cells were measured by both passes on purpose, far above
any such cost:

| | `q_len` | pass 1 | pass 2 | |
|---|--:|--:|--:|--:|
| `kv=4096` | 256 | 1.128 | 1.138 | 0.88 % |
| | 1024 | 1.051 | 1.051 | 0.03 % |
| `kv=16384` | 256 | 1.182 | 1.183 | 0.08 % |
| | 1024 | 1.048 | 1.049 | 0.08 % |

**It cuts the other way, though.** Both arms pay that cost, so at small shapes
the ratio is pulled toward 1 — the kernel-level difference is *larger* than the
table shows, in the same direction. `production_ms` and `bm64_ms` are in every
row so this can be checked rather than taken.

## What it says about the PR

The author's argument is that a `max_seqlen_q`-only gate cannot express a
crossover that moves with KV depth. These rows agree that the crossover moves —
and show that **which way it moves is a property of the vendor**. That is a
reason for the `_is_gfx1100()` guard to stay rather than a reason to widen the
gate: one global threshold tuned on either card would be wrong on the other, and
here it would be wrong in sign, not just in magnitude.

## What is here

    kv-depth-summary.json   the grid, the overlap check, and the limits
    kv_depth.jsonl          pass 1, 48 rows, one call per event pair
    kv_depth2.jsonl         pass 2, 28 rows, batched
    probe_kv_depth.py       pass 1
    probe_kv_depth2.py      pass 2
    logs/                   both runs, including setup's md5 assertions

    A100-SXM4-80GB · cap 8.0 · vLLM 0.28.0 · torch 2.13.0+cu130 · triton 3.7.1
    kernel md5 49fab3b643bf5a88eb65303ce377996b -> f1d7a7e3c6656303fa63b6a4c1b8aef5

## Not established

bf16 and `head_size=128` only, two head patterns, one board, single-sequence
prefill, no model-level TTFT claim. The `kv_len = q_len` column below
`q_len=1024` is dominated by the per-call cost on both arms and is a control
rather than a measurement — which is also why it sits flat at 0.93–1.03 where
his shows a real crossover: his baseline is an unpatched tree on a card where
that region is not overhead-bound.
