# Qwen3-8B on one RX 7900 XT, vLLM 0.27, fully stock — 2026-08-30

Run to lift the July ladder's 6 000 ceiling by raising `--gpu-memory-utilization`
to 0.95. **It did not lift it**, and what it produced instead is worth more than
the ceiling would have been.

Five rungs, two rounds each: **20 measurements, 0 errors**, every rung
chart-grade. `results.jsonl` is the raw data, `runner.py` produced it,
`logs/` holds the engine's own log.

    RX 7900 XT · 19.98 GiB · gfx1100 · TP=1 · util 0.95
    vLLM 0.27.1.dev5+gf46a9dfe2.d20260827 · image rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0
    ROCM_ATTN · bfloat16 · enable_prefix_caching=True, hit rate 0.0 % throughout

**No patches.** The container's `triton_unified_attention.py`, `triton_attn.py`
and `chunked_prefill_paged_decode.py` were restored to the image's own bytes and
asserted by md5 before the run, so this row carries none. `patches=[]` in
`analyze/build_prefill.py` means it.

## The ceiling is arithmetic, and 0.95 does not move it

    predicted   ~3.9 GiB of KV, ~27 000 tokens, 10 rungs
    measured     1.13 GiB,        8 236 tokens,  5 rungs

The runner stepped `max_model_len` from 33 000 to 8 157. July's 0.23 run reached
8 442 tokens at a lower utilisation — **the same ceiling**. Two errors in the
prediction, and they compound:

* **weights** are 15.27 GiB on 0.27, not the 14.02 GiB July's 0.23 reported for
  the same checkpoint. `Model loading took N GiB` is also per rank, so at TP=2
  the whole model is 2N.
* **activation overhead** is 2.58 GiB for this model, not the 1.09 GiB measured
  on an int4 MoE. It is per-model: 1.13 (w4a16 dense), 1.09–2.16 (int4 MoE),
  2.58 (bf16 dense). Sizing a run from another model's overhead is worth up to
  1.5 GiB of error, which on a 20 GiB card is five rungs.

## Decode does not move across three stacks and two utilisations

| ctx | 0.23, 2026-07-25 | 0.23.1+patches, 2026-08-24 | **0.27 stock, util 0.95** |
|---|--:|--:|--:|
| 500 | 46.69 | 46.695 | **46.640** |
| 1 000 | 46.425 | 46.42 | **46.330** |
| 2 000 | 45.81 | 45.825 | **45.820** |
| 4 000 | 44.995 | 44.995 | **44.990** |
| 6 000 | 44.12 | 44.12 | **44.145** |

Worst deviation across all five rungs and all three configurations: **0.21 %**.

## Prefill moves, and the decomposition says where

Fitting `T(S) = a + bS + cS²` on the chart-grade rungs of each:

| | rungs | a ms | b µs/tok | c ns/tok² | r² |
|---|--:|--:|--:|--:|--:|
| 0.23, 2026-07-25 | 5/5 | 19.7 | 255.9 | 16.11 | 0.9999 |
| 0.23.1+patches, 2026-08-24 | 4/5 | 28.9 | 244.5 | 17.83 | 0.9999 |
| **0.27 stock, util 0.95** | 5/5 | 15.6 | **206.7** | **8.87** | 1.0000 |

**b improves 1.24× and c improves 1.82×.** The gain is mostly in the quadratic
term — the attention half — not in the GEMMs. A TTFT ratio (1.18× at 500 rising
to 1.36× at 6 000) shows the gain growing with depth without saying which term
grew; the fit says it.

`a` is not read as a measurement here. It goes 19.7 → 28.9 → 15.6 across three
campaigns of the same arm, and on other arms it reaches −22.1 ms; the ladder
does not determine it.

## What this arm is, in the selector's terms

Qwen3-8B is `head_dim` 128 with `gqa_ratio` 4, so it satisfies
`use_rocm_custom_paged_attention` on RDNA and `ROCM_ATTN` here means the actual
HIP kernel rather than a second Triton one. **This is the case
[vllm#54438](https://github.com/vllm-project/vllm/issues/54438) deliberately
leaves alone**: that report is about `head_size` 256, where both candidates are
Triton kernels and the selector prefers the slower one.

The backend is read from the serve log's

    Found incompatible backend(s) [TURBOQUANT] with AttentionType.DECODER.
      Overriding with ROCM_ATTN out of potential backends: ['ROCM_ATTN', 'TRITON_ATTN'].

which is a form `build_prefill.py`'s regex did not match until 2026-08-30.

## One thing this comparison cannot claim

The 2026-07-25 and 2026-08-24 rows have **no recorded backend** — those campaigns
kept no serve log. So "the same arm on a newer stack" is not "known to be the
same kernel"; it is "no contradiction recorded". The 1.24×/1.82× above is a
stack-to-stack figure, not a kernel-to-kernel one.
