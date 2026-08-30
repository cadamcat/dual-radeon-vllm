# Qwen3-8B and Qwen3.8-27B-INT4-sym on one NVIDIA L4 — 2026-08-30

The L4's second pass, after `cuda_run.py` gained the capacity retry the Radeon
runner has had since rev2. The first pass lost **four** configurations to its
absence: vLLM raises a `ValueError` when the KV pool cannot hold one request at
`--max-model-len`, the message names the length that would fit, and without the
retry that traceback was recorded as a crash. `B8` missed by **0.13 GiB** —
33 000 tokens need 4.53 GiB against 4.40 available, and the message said
"estimated maximum model length is 32000".

    NVIDIA L4 · 23034 MiB · sm89 · driver 580.82.07 · vLLM 0.28.0
    torch 2.13.0+cu130 · CUDA 13.0 · util 0.95 · max_num_seqs 1
    FLASH_ATTN on both · enable_prefix_caching=False

| cfg | checkpoint | KV | mml | rungs | prefill grade | decode grade |
|---|---|--:|--:|--:|--:|--:|
| `B8` | `Qwen3-8B`, bf16 | 4.4 GiB / 32 000 tok | 31 680 | 10 to 24 000 | 10/10 | 10/10 |
| `Q38S` | `RedHatAI/Qwen3.8-27B-INT4`, sym CT g128 | 0.78 GiB / 10 090 tok | 10 090 | 6 to 8 000 | 5/6 | 6/6 |

`B8` stops at 24 000 because the retry stepped `max_model_len` to 31 680 and the
32 000 rung plus 512 generated tokens no longer fits. That is the retry working,
not a ceiling.

`G31` is a `config_failed` row in the same file — `startup retries exhausted`.
It is measured in `../campaign-2026-08-30c` instead, and only with
`--enforce-eager`.

## `Q38S` is a different checkpoint, not another arm of `Q38`

`Q38` elsewhere in this repository is **cyankiwi's AWQ** build of Qwen3.8-27B.
`Q38S` is **RedHatAI/Qwen3.8-27B**, compressed-tensors with symmetric int4 at
group 128. On gfx1100 the two land on different mixed-precision kernels and
differ by **1.27–3.24× on decode** (`../../w4a16-symmetry/w4a16-ab.jsonl`).

They must not appear in the same table row anywhere, and `Q38S` has no
counterpart on any other machine in this repository — there is nothing to
compare it against yet.

## The one ungraded rung is the shortest one, again

`Q38S`'s 500 rung spreads 9.74 % across its two rounds and is the only ungraded
prefill point in either configuration; every other rung of both is under 3.8 %.
The runner's discarded warm-up request is in place here, so this is not the cold
engine the 2026-08-30 L4 campaign's 500 rungs were — it is a 0.585 s measurement,
where a 57 ms difference is 9.7 %. Decode is unaffected: 22 of 22 rungs across
both configurations are chart-grade, the worst at 0.157 %.

## Qwen3-8B against one RX 7900 XT, and the same split as gemma-4-12B

Both fitted on `target` over the chart-grade rungs:

| | b µs/tok | c ns/tok² | backend |
|---|--:|--:|---|
| RX 7900 XT, util 0.95 (`campaign-2026-08-30b`) | **206.7** | 8.87 | `ROCM_ATTN` |
| L4 | 288.3 | **5.38** | `FLASH_ATTN` |

**The Radeon wins the linear term by 1.39× and loses the quadratic by 1.65×** —
the same direction gemma-4-12B shows (Radeon 446–479 against the L4's 534.7 on
`b`; 24.2–25.2 against 8.03 on `c`), with very different magnitudes.

**This pair is a kernel difference as well as a card difference.** The Radeon
arm runs `ROCM_ATTN`'s HIP kernel and the L4 arm runs `FLASH_ATTN`; nothing here
holds the backend fixed across the two machines. The gemma-4-12B pair does not
hold it fixed either, for a different reason — the Radeon's 2026-07-25 and
2026-08-24 campaigns kept no serve log, so their backend is unrecorded rather
than known.

The one clean cross-machine prefill comparison in this repository is still
`gemma-4-31B`, which is `TRITON_ATTN` on both machines it has been measured on.
