# The head_size this pre-flight never recorded — 2026-08-30

The pre-flight beside this directory established that `gemma-4-12B-it-qat-w4a16-ct`
cannot be served on sm75, and blamed a head dimension of 256. It recorded no
head size anywhere: not in `preflight.jsonl`, not in any of its three serve logs.
That mattered because **vllm#39018, the open fix for this failure, gates on
`head_size_padded >= 512`** — so 256 against 512 is the difference between that
PR covering this case and missing it.

Same Tesla T4, same vLLM 0.28.0 (`triton_unified_attention.py` md5
`49fab3b643bf5a88eb65303ce377996b`, the same file upstream `main` and the ROCm
image carry). `check_head.py` reads all 35 numbers below back out of
`headsize.jsonl`.

## What vLLM resolves, before any kernel

    model_arch_config.head_size            512
    per-layer head sizes                   {256, 512}
    config.json   head_dim 256 · global_head_dim 512
                  16 attention heads · 8 KV heads
                  48 layers = 40 sliding_attention + 8 full_attention

**512, not 256.** 256 is the sliding layers' local value.

## What the kernel is actually called with

A recorder installed immediately before `kernel_unified_attention[grid](...)`,
so it writes even though the launch that follows is the one that fails:

    head_size=256  padded=256  nq_per_kv=2   kv_heads=8  BLOCK_M=16  TILE_PREFILL=32
    head_size=512  padded=512  nq_per_kv=16  kv_heads=1  BLOCK_M=16  TILE_PREFILL=32

The heterogeneity is not only in the config; `unified_attention` is entered
twice with two different head sizes. The engine then dies exactly as the
pre-flight recorded: `TRITON_ATTN`, `Required: 98304, Hardware limit: 65536`,
with `MarlinLinearKernel` for W4A16.

## vllm#39018 fixes it, and the fix is correctly scoped

The PR applies to 0.28.0 with no rejected hunks. With it in place the recorder
shows the tile dropping on the 512 layers **and only on those**:

    head_size=256  ...  TILE_PREFILL=32     unchanged
    head_size=512  ...  TILE_PREFILL=16     halved by the patch

and the engine gets past the launch that used to kill it:

    Available KV cache memory: 5.17 GiB
    GPU KV cache size: 82,383 tokens · concurrency 2.50x for 33,000
    Application startup complete.

**Starting is not serving**, and #38918 reports this backend as "server starts
but crashes on first request", so both rungs were then actually generated:

| prompt tokens | completion tokens | TTFT | wall |
|---:|---:|---:|---:|
| 738 | 16 | 1.39 s | 2.10 s |
| 30 018 | 32 | 273.73 s | 277.05 s |

Both returned coherent text. So on this card, with this checkpoint, vllm#39018
is the difference between an engine that cannot start and one that serves 30 K
of context.

## Files

    headsize.jsonl            every row the probes emitted
    kernel-args.txt           the recorder, stock 0.28.0
    kernel-args-39018.txt     the recorder, with vllm#39018
    logs/                     one serve log per arm
    headprobe.py              installs the recorder, reads ModelConfig
    serveprobe.py             the stock arm
    test39018.py              applies the PR and starts the engine
    infer39018.py             the two generation rungs
    check_head.py             reads this README back out of headsize.jsonl
