# The gfx1100 backend matrix — 2026-08-30

Four arms of one model on one machine on one day, asking a question the ledger
could not answer from either backend alone: **on gfx1100, is the attention
backend the selector picks the right one?**

    2x RX 7900 XT (gfx1100, RDNA3) · TP=2 · ROCm 10.0
    vLLM 0.27.1.dev5+gf46a9dfe2.d20260827
    Qwen3.8-27B-AWQ-INT4 — head_dim 256, 24 query heads over 4 KV heads,
    64 layers = 48 linear_attention + 16 full_attention

`PROVENANCE.md` has the md5 of every file state and how the arms were built.
`TABLES.txt` is the full aggregation. `results.jsonl` is every round.

## The four arms

| arm | `--attention-backend` | `triton_unified_attention.py` | `chunked_prefill_paged_decode.py` |
|---|---|---|---|
| A | *(none; the selector picks `ROCM_ATTN`)* | image, **= upstream main** | image |
| B | *(none; the selector picks `ROCM_ATTN`)* | image, **= upstream main** | + vllm#45916 |
| C | `TRITON_ATTN` | image, **= upstream main** | + vllm#45916 |
| D | `TRITON_ATTN` | + vllm#52684 | + vllm#45916 |

Within each pair the serve command is identical. What varies is which file is in
site-packages, asserted by md5 before every arm by `tools/setstate2.sh`. 44
measurements per arm, 0 errors, decode 11 of 11 rungs chart-grade on all four.

## What it found

**The backend the selector rejects is faster at decode at every depth.**

| ctx | A | B | C | D | C/A | D/B |
|---|---:|---:|---:|---:|---:|---:|
| 500 | 38.45 | 49.62 | 49.23 | 49.33 | 1.28x | 0.994 |
| 8 000 | 12.38 | 45.82 | 47.25 | 47.28 | 3.82x | 1.032 |
| 16 000 | 7.17 | 42.18 | 45.27 | 45.37 | 6.31x | 1.076 |
| 32 000 | 3.90 | 36.43 | 41.91 | 41.95 | **10.74x** | **1.151** |

decode tok/s, mean of two rounds. `C/A` is the cost on the image as it ships;
`D/B` is what is left once both open PRs are in.

**`ROCM_ATTN` is the better backend at prefill**, which is the cost side of the
trade rather than the case for switching:

| arm | rungs | a ms | b µs/tok | c ns/tok² | r² | prefill @32K |
|---|---:|---:|---:|---:|---:|---:|
| A | 11/11 | 52.5 | 942.4 | 2.79 | 1.0000 | 964.8 |
| B | 10/11 | 219.6 | 919.8 | 3.23 | 0.9999 | 965.8 |
| C | 10/11 | 358.7 | 848.2 | **18.17** | 1.0000 | 692.0 |
| D | 10/11 | 260.8 | 886.0 | **5.84** | 1.0000 | 923.4 |

## The mechanism

`ROCM_ATTN` is ranked first by `_get_backend_priorities` for its custom paged
attention HIP kernel. It does not call that kernel directly:
`rocm_attn.py` calls `chunked_prefill_paged_decode`, which asks
`use_rocm_custom_paged_attention` and takes a **Triton** branch when the answer
is no — and the RDNA branch of that predicate requires `head_size == 128`.

At `head_size` 256 both candidates are therefore Triton kernels, and the
selector prefers one of them for a property neither has here. The priority
function never sees `head_size`; it takes `use_mla`, `use_sparse` and
`use_kv_connector`.

**This is recorded, not inferred.** Both `ROCM_ATTN` arms log
`Cannot use ROCm custom paged attention kernel, falling back to Triton
implementation.` from `chunked_prefill_paged_decode.py` itself, at the branch;
neither `TRITON_ATTN` arm does. `check.py` asserts that, and asserts that each
of the four arms carries **exactly one** backend verdict in its log.

That last check exists because the speculative arms do not.
`campaign-2026-08-29/logs/Q38-mtp-triton-tp2.log` carries
`Using TRITON_ATTN backend (selected via --attention-backend)` for the target
at 20:00:00 and `Overriding with ROCM_ATTN` on **both** TP ranks at 20:00:17
for the drafter, and the kernels it then compiles are
`kernel_paged_attention_2d{,_splitkv,_splitkv_reduce}`. So `--attention-backend`
does not reach the draft model, that arm is a mixture rather than a
`TRITON_ATTN` arm, and none of the four arms here uses speculation for exactly
that reason.

One caveat on reading these logs: the JIT monitor only warns about kernels
compiled **during inference**, and Triton's cache lives in the container across
engine restarts. A kernel an earlier arm already compiled produces no line, so
the absence of a JIT line is not evidence the kernel did not run. The fallback
warning above is the load-bearing record; the JIT lines corroborate.

## Two things this round measured rather than assumed

* **vllm#52684 works on a model class it was never tested against.** Its four
  validated models are all `head_size` <= 128 and its data stops at 8 192. Here,
  at `head_size` 256 and out to 32 K: prefill +33.4 %, the fitted quadratic
  coefficient 18.17 -> 5.84, and decode unchanged at all 11 rungs to within
  0.25 %. The expectation before the run was an `OutOfResources` — its gate has
  no `head_size` condition and `BLOCK_M(64) x 256 x 2` plus
  `TILE(32) x 256 x 2 x 2` is 64 KiB exactly, gfx1100's LDS ceiling. **It does
  not happen**, 44/44, zero `OutOfResources`.
* **The two Triton files are not the same file.** `triton_unified_attention.py`
  in this image is byte-identical to upstream `main`
  (`49fab3b643bf5a88eb65303ce377996b`, 1189 lines, zero differing lines), so
  arms C and D are measurements of main's own kernel.
  `chunked_prefill_paged_decode.py` differs from main by 56 lines, all inside
  `kernel_paged_attention_2d`, where main drops the per-token load mask on
  non-final tiles. That is a speedup to exactly the kernel **arm A** runs, so
  arm A understates upstream main by an amount this round did not measure.

## Cross-day control

Arm C reproduces the 2026-08-29 campaign's `Q38-triton-tp2`, same file state and
serve command a day earlier, to within **1.73 %** on prefill and **1.31 %** on
decode at every rung, with fitted `c` 18.17 against 18.44. Arm B reproduces that
campaign's `Q38-tp2` to **0.1 %** on decode at 32 K and 0.3 % on prefill.
