# 2026-08-30c — the gfx1100 backend matrix, one harness, one day

Four arms of one model on one machine, run back to back without letting
`ollama` back onto the cards between them. The serve command is identical
within each backend pair; **what varies is which files are in site-packages**,
asserted by md5 before every arm by `setstate2.sh`.

    machine   2x RX 7900 XT (gfx1100), TP=2
    image     rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0
    vllm      0.27.1.dev5+gf46a9dfe2.d20260827  (= ROCm/vllm@f46a9dfe2)
    model     /models/Qwen3.8-27B-AWQ-INT4, read from its own config.json:
              model_type qwen3_5_text, head_dim 256, num_attention_heads 24,
              num_key_value_heads 4 -> num_queries_per_kv = 6, hidden_size 5120,
              64 layers = 48 linear_attention + 16 full_attention (hybrid SSM)
    serve     --max-model-len 33000 --max-num-seqs 16 --tensor-parallel-size 2
              --gpu-memory-utilization 0.92 ; enable_prefix_caching=False
    ladder    11 rungs 500..32000, two rounds each, RANGE_CUT 8 %

## The two axes, and the md5 of every state

`triton_unified_attention.py` is the **TRITON_ATTN** path;
`chunked_prefill_paged_decode.py` is the **ROCM_ATTN** path. They are
independent, which is why the matrix is a matrix.

| file | state | md5 | lines |
|---|---|---|---|
| `triton_unified_attention.py` | image | `49fab3b643bf5a88eb65303ce377996b` | 1189 |
| `triton_unified_attention.py` | + vllm#52684 | `f1d7a7e3c6656303fa63b6a4c1b8aef5` | 1220 |
| `chunked_prefill_paged_decode.py` | image | `86f68d47c7bdc390ced4c6d0c18025fa` | 493 |
| `chunked_prefill_paged_decode.py` | + vllm#45916 | `84c6d4f9b2dfe2714b3a8f43ee832b02` | 1083 |
| `triton_attn.py` | image | `f0a1379d724c870fa2703330524100f9` | |

The image md5s were taken from the image itself
(`docker run --rm --entrypoint bash IMG -c 'md5sum ...'`), not from the
container, which arrived carrying vllm#45450 in both Triton files from the
08-29 campaign. **That is why every arm here restores `triton_attn.py`**: the
08-29 leftovers are not part of this experiment.

The two image md5s independently match records written before this session:
`49fab3b6` is the stock prefix in `campaign-2026-08-29/provenance.json`, and
`86f68d47c7bdc390ced4c6d0c18025fa` / 493 lines are `MD5_STOCK` and
`cppd_lines_stock` in `hybrid-splitkv-027/provenance.json`.

## The arms

| id | `--attention-backend` | tua | cppd | what it stands for |
|---|---|---|---|---|
| `Q38-triton-52684-tp2` | TRITON_ATTN | +52684 | +45916 | the Triton path with its open fix |
| `Q38-triton-stock-tp2` | TRITON_ATTN | image | +45916 | the Triton path today |
| `Q38-rocm-45916-tp2` | *(none — selector picks)* | image | +45916 | the ROCm path with its open fix |
| `Q38-rocm-nopatch-tp2` | *(none — selector picks)* | image | image | **what upstream ships today** |

`vllm#52684`'s gate is
`max_seqlen_q >= 512 and num_queries_per_kv <= 16 and _is_gfx1100()`. On this
checkpoint `num_queries_per_kv` is 6 and `on_gfx1100()` returns True, so the
gate fires from the 1000 rung up: `BLOCK_M` 16 -> 64, `BLOCK_Q` 2 -> 8,
`num_warps` pinned to 4. The 500 rung is below the gate and is the within-arm
control for it.

## Files

    setstate2.sh      <stock|patched> <stock|p45916>, asserts all three md5s
    runner52684.py    campaign-2026-08-29/runner.py with D/D_IN_CONTAINER moved
                      and CFGS reduced to one arm chosen by $ARM_ID
    results.jsonl     every round of every arm
    serve-logs/       one per arm
