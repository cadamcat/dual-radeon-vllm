# Why hybrid-SSM models collapse at long context under vLLM on RDNA3

`Qwen3.6-27B` decodes at 12.1 tok/s with a short prompt and 4.2 tok/s at 32K on
two 7900 XT. For a long time this repository explained that as a problem with the
linear-attention layers. **That explanation was wrong**, and a kernel-level
profile taken 2026-07-29 says so directly.

The short version:

- The 48 gated-delta-net layers are **fine**. Their decode kernels cost the same
  at 1K and at 32K context, which is exactly what a recurrent state should do.
- The whole slowdown is in the model's **16 ordinary full-attention layers**,
  which fall back to a Triton paged-attention kernel.
- They fall back because vLLM's ROCm custom paged-attention kernel has no
  `head_size=256` instantiation at all, and this model's `head_dim` is 256.
- The Triton fallback is slow **here specifically** because this model has only
  4 KV heads, so at batch 1 there is almost nothing to spread over 84 CUs.

None of that is about SSM. A dense model with `head_dim=256` takes the same
fallback; it just has more KV heads to hide it with.

---

## 1. The profile

Torch profiler, CUDA graphs left **on**, kernel calls attributed to the engine
step whose window they fall in so that chunked-prefill calls do not contaminate
the decode figures — [`benchmarks/analyze/split_decode_trace.py`](../benchmarks/analyze/split_decode_trace.py)
does that attribution. The raw traces are kept off-repo, see §7. Per-call GPU time
on rank 0:

| kernel | layers it serves | ctx=1024 | ctx=32768 | ratio |
|---|---|---:|---:|---:|
| `fused_recurrent_gated_delta_rule_packed_decode` | 48 linear | 8.466 µs | 8.038 µs | 0.95× |
| `_causal_conv1d_update_kernel` | 48 linear | 2.260 µs | 2.139 µs | 0.95× |
| `triton_w4a16_gemm_kernel` | all | 275.792 µs | 251.447 µs | 0.91× |
| **`kernel_paged_attention_2d`** | **16 full** | **356.664 µs** | **10 095.188 µs** | **28.3×** |

Call counts are identical across the two contexts — 528 for the linear-attention
kernels (48 layers × 11 steps), 176 for paged attention (16 × 11) — so these are
per-call costs, not extra work being issued.

The arithmetic closes. At 16 calls per step, paged attention goes from
5.71 ms/step to 161.5 ms/step, up 155.8 ms. The decode step itself, measured from
the CUDA-graph replay, goes from 85.85 ms to 236.9 ms, up 151 ms. **Paged
attention accounts for the entire increase**; everything else is flat or slightly
faster at the longer context.

That end-to-end step timing also matches the campaign in
[benchmarks.md](benchmarks.md), which measured 82.51 → 235.29 ms/token. The
profile is describing the same phenomenon, not an artefact of the profiling run.

## 2. Why the custom kernel is skipped

`vllm/platforms/rocm.py`, `use_rocm_custom_paged_attention`, RDNA branch:

```python
return (
    _ON_GFX1X
    and (sliding_window == 0 or sliding_window == (-1, -1))
    and (qtype == torch.half or qtype == torch.bfloat16)
    and head_size == 128
    and block_size == 16
    and (gqa_ratio >= 3 and gqa_ratio <= 16)
    ...
)
```

`Qwen3.6-27B` has `head_dim=256`, so the fourth condition fails. Its KV block size
is not 16 either — `chunked_prefill_paged_decode.py` explains that in a comment
right next to the fallback:

```
# Cap at 128 to avoid exceeding GPU shared memory limits
# (e.g. hybrid Mamba models inflate block_size to 2048).
```

`gqa_ratio` is 6 and does pass. Two independent conditions each disqualify the
model, and the warning at `chunked_prefill_paged_decode.py:420` fires.

**The `head_size` check is not conservative gating — it is the kernel's real
limit.** `CALL_CUSTOM_LAUNCHER_BLK_HEAD` in `csrc/rocm/attention.cu` dispatches
only `case 64` and `case 128`, and `TORCH_CHECK`s on anything else. Relaxing the
Python-side check would move the failure, not remove it. This matters because
"just widen the condition" is the first fix anyone reaches for, including us.

## 3. Why the fallback is slow on *this* model

Dense models take the same Triton path and do not collapse. The variable is how
much parallelism the kernel has at decode, where the batch is 1:

| model | head_dim | KV heads | query heads | GQA | µs per context token |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B, BF16 dense | 128 | 8 | 32 | 4:1 | 0.118 |
| gemma-4-31B, w4a16 dense | 256 | 16 | 32 | 2:1 | 0.339 |
| **Qwen3.6-27B, hybrid** | **256** | **4** | 24 | 6:1 | **4.840** |

Note the middle row: gemma-4-31B also has `head_dim=256`, so it is **also** on the
Triton fallback. With 16 KV heads it pays about 2.9× the slope of a
`head_dim=128` model and stays usable. Qwen3.6 has 4 and pays 41×.

Two independent signals agree that this is an occupancy problem:

**Power falls as context grows.** 265 W + 265 W at 518 tokens, 232 W + 227 W at
24K, while every dense model in the same sweep holds 265 W flat. Idle silicon,
not saturated silicon.

**Normalising by KV traffic makes the gap wider, not narrower.** Only 16 of the
64 layers carry a KV cache, and the GQA ratio is aggressive, so this model reads
the *least* KV of the three — 64 KiB per token, against 144 KiB for Qwen3-8B and
160 KiB for gemma-4-31B. Per KiB read it is 92× the cost of Qwen3-8B and 36× that
of gemma-4-31B. It cannot be "this model is simply heavier".

## 4. The control: the same model under llama.cpp

Same machine, same day, same model, only the inference engine changed.
`llama-bench -p 0 -n 128 -d <depth> -r 2`, Q4_K_M GGUF, both cards, layer split.
Raw output in [`benchmarks/llamacpp-depth-sweep-rocm.json`](../benchmarks/llamacpp-depth-sweep-rocm.json)
and [`-vulkan.json`](../benchmarks/llamacpp-depth-sweep-vulkan.json).

| engine | 512 depth | 32K depth | retained | slope µs/ctx-token |
|---|---:|---:|---:|---:|
| llama.cpp, ROCm backend | 24.89 tok/s | 21.84 tok/s | 87.7 % | 0.174 |
| llama.cpp, Vulkan backend | 28.61 tok/s | 26.04 tok/s | 91.0 % | 0.107 |
| **vLLM** | **12.1 tok/s** | **4.2 tok/s** | **34.7 %** | **4.840** |

vLLM's slope is 28× llama.cpp's on the same ROCm userspace and 45× the Vulkan
one. Both llama.cpp backends stay flat, which rules out the driver as well: the
ROCm run uses the same stack vLLM does.

The sharper point is that llama.cpp's slope lands **inside** the range the dense
models produce under vLLM, 0.118 to 0.339. Two independent sources now agree on
what a normal slope looks like on this hardware, and one configuration sits an
order of magnitude outside it.

This is a 2×2: change the engine and it is normal, change the model and it is
normal, only vLLM × Qwen3.6 is not.

Two caveats. Absolute throughput is not comparable — llama.cpp is running Q4_K_M
against vLLM's AWQ-INT4 — only the slope is. And the ROCm 16384 point came out at
21.35 tok/s with a standard deviation of 2.635 against 0.05–0.09 everywhere else,
while the deeper 24576 point is faster; that one measurement was disturbed and
should not be quoted alone.

## 5. Two different problems, and only one of them is this one

It is worth separating them, because this document replaced an explanation that
conflated them:

**Baseline speed.** At short context vLLM does 12.1 tok/s where llama.cpp does
24.89. That gap is *not* explained by anything above — it is present at 512
tokens, where paged attention costs 5.71 ms/step out of 85.85. The older
observations still stand as candidates for it: the gated-delta-net path is
Triton-only on ROCm, those kernels pick tile sizes from an NVIDIA-only
shared-memory table (gfx1100's 64 KB LDS falls below the 102400-byte default), and
the recurrent decode kernel is inherently `num_warps=1`. See
[architecture-notes.md](architecture-notes.md).

**The slope.** Everything in §1–§4. This is the paged-attention fallback, and it
has nothing to do with SSM.

Conflating the two is what produced the wrong conclusion the first time.

## 6. What we got wrong, and how

The previous version of this finding said the decode path "is not taking an
incremental recurrent path" — i.e. that the linear-attention layers were
re-scanning the sequence. It was a reasonable hypothesis: cost growing linearly in
sequence length is exactly what a re-scan looks like.

Two things disproved it. Reading
`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` shows
`_forward_core_decode_non_spec` calling
`fused_recurrent_gated_delta_rule_packed_decode` — the recurrent form, O(1) per
token, and not dependent on AITER. Then the profile showed that kernel costing
8.466 µs at 1K and 8.038 µs at 32K, with identical call counts.

A methodological note worth keeping. The first profiling attempt used
`enforce_eager`, because torch profiler cannot see inside a captured CUDA graph.
The eager decode slope came out ~18× the graph one, which meant the eager profile
was describing a different, amplified phenomenon; it was discarded and the run
redone with graphs on, using `ProfilerConfig`'s iteration schedule to skip prefill
instead. `benchmarks.md`'s standing rule about never concluding from
`--enforce-eager` numbers applies to profiles too.

## 7. Not established

- **CDNA is untested.** The `_ON_GFX9` branch of `use_rocm_custom_paged_attention`
  has looser conditions, so this may not reproduce on MI-series.
- **No NVIDIA comparison.** We cannot say whether the same model decodes flat on
  CUDA, where the custom-kernel question does not arise in this form.
- **No fix is proposed.** Widening the gating is not one, since there is no
  `head_size=256` kernel to fall through to. A real fix means adding that
  instantiation, or giving the Triton path a decode split that does not depend on
  KV-head count. We are not in a position to judge the cost or the numerical
  correctness of either.
- **Raw profiler traces are not in this repository.** They are ~5 MB of
  `.pt.trace.json.gz`, kept with the rest of the raw experiment data rather than
  committed here.
