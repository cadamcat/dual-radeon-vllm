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

> **Refined 2026-07-30.** The 2048 above is vLLM's own comment giving an example,
> not our value. Measured, `cache_config.block_size` is **784**. And it is three
> conditions, not two: `is_pow2(784)` is `False`, which disqualifies `use_custom`
> independently. Counterfactual checks in §6.5 show no single one is the blocker.

> **The complementary case, 2026-08-24.** `Muse-Glimmer-30B` clears every clause
> of that predicate except the `sliding_window` one: `head_dim` 128,
> `block_size` 16, GQA ratio 16, `kv_cache_dtype` auto, bf16. Its 13
> full-attention layers therefore *do* take the custom kernel and its 39
> sliding-window layers do not, in the same forward pass — the first time the
> custom HIP kernel has run at all on this machine. That also makes the
> `sliding_window` clause the sole blocker for a real model, which
> `Qwen3.6-27B` could not show because three clauses fail for it at once. See
> [sliding-window-block-skip.md](sliding-window-block-skip.md).

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
24K, while no other model in the sweep loses power with depth — the others climb
to about 265 W and stay there, and this is the only one that comes back down.
(An earlier version of this sentence said every dense model "holds 265 W flat",
which is not what the data shows: at 518 tokens the 8B is at 245 W, the 12B at
206 W and the 26B MoE at 193 W. They get there with depth. The contrast that
matters is the direction, not the starting value.) Idle silicon,
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
| **vLLM** | **12.1 tok/s** | **4.25 tok/s** | **35.1 %** | **4.840** |

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

## 6.5 Fixed upstream: PR #45916, verified here 2026-07-30

Filing this as [vllm#50264](https://github.com/vllm-project/vllm/issues/50264)
turned up the answer within hours. @thegoldenflow pointed out that
[vllm#45916](https://github.com/vllm-project/vllm/pull/45916) already adds
`kernel_paged_attention_2d_splitkv` — the exact kernel profiled above — but gates
it to `on_gfx12x()`, so it is inert on RDNA3. That PR had sat untouched for six
weeks. @Lafunamor then verified it on gfx1151 at kernel level, and we ran the
gfx1100 end-to-end half that nobody else had hardware for.

Changing only `on_gfx12x()` → `on_gfx1x()`, nothing else:

| | before | after |
|---|---:|---:|
| PR's own test suite | — | **69/69** |
| `kernel_paged_attention_2d` @32K, per call | 10 095.188 µs | — |
| `kernel_paged_attention_2d_splitkv` @32K, per call | — | 635.779 µs |
| `..._splitkv_reduce` | — | 4.117 µs |

**15.8× at the kernel.** The profile contains only the `_splitkv` kernels; the old
one is absent, which is the only available confirmation since nothing logs the
choice.

End to end, decode isolated from prefill. The two sides are separate runs, and
not quite the same measurement either: "before" is the campaign, which generated
512 tokens and timed decode first token to last, averaged over two rounds;
"after" differences a 64-token generation against an 8-token one at the same
depth, once per depth with no repeat. The depths differ too, the "after" context
being longer in every row, which makes the speedups conservative. Working the
generation-length difference through the two slopes puts it under 0.5 %, and
that part is arithmetic rather than a measurement:

| before (ctx) | ms/tok | after (ctx) | ms/tok | speedup |
|---:|---:|---:|---:|---:|
| 518 | 82.51 | 1 024 | 79.68 | 1.04× |
| 8 026 | 117.23 | 8 192 | 82.60 | 1.42× |
| 16 058 | 157.60 | 16 384 | 85.97 | 1.83× |
| 32 084 | 235.29 | 32 768 | **93.30** | **2.52×** |

At 32K the campaign reported 4.2 tok/s; this run measures **10.72**. The decode
slope falls from 4.840 to **0.430 µs per context token**, an 11.3× reduction.
That is still above the 0.118–0.339 band this machine's dense models produce, by
about 27 % over the highest of them, but it is the same order of magnitude
rather than fourteen times the top of the band. **The collapse this document is
about is gone.**

Output quality holds: greedy decode at 21 012 tokens of context returns a correct,
coherent answer, so the split kernel's index arithmetic is sound at our block size.

### Two things the verification corrected

**Our KV block size is 784, measured** — `cache_config.block_size`, 16 × 49. §2
previously leaned on the 2048 in vLLM's own source comment, which was that
comment's example rather than our value. Same conclusion, real number. It is the
same family as @Lafunamor's 528 = 16 × 33: divisible by 16, not by 32.

**No single condition is the blocker.** Calling `use_rocm_custom_paged_attention`
directly with counterfactuals, `head_size` forced to 128 still returns `False` and
`block_size` forced to 16 still returns `False`. `is_pow2(784)` is also `False`,
which disqualifies `use_custom` on its own path. The custom HIP kernel is
unreachable here three times over — which is why making the *fallback* fast is the
only route that helps this class of model.

### The split-count heuristic, measured

The last of the four things @thegoldenflow asked us to check, and the only one
the first run left open. The heuristic sizes its target occupancy from
`multi_processor_count`, and the kernel's own docstring is careful to claim only
one architecture: "On gfx12 torch reports WGPs while rocprof reports CUs, so
target two workgroups per reported processor." Whether gfx11 reports the same
thing was an assumption. The `2 ×` is a correction, not headroom: if torch
reported CUs here, `target_workgroups` would come out at twice the hardware and
every split decision would be sized against a GPU that does not exist.

Measured, both cards:

```
gcnArchName           = gfx1100
multi_processor_count = 42        # the 7900 XT has 84 CUs, i.e. 42 WGPs
warp_size             = 32
```

RDNA3 reports WGPs as well, so `2 * num_sms` lands on 84, the real CU count. The
compensation is right here for the same reason it is on RDNA4, where @yanghoeg
measured 32 against the R9700's 64 CUs.

Replaying `_get_num_splits` at our configuration — `head_size=256`, 4 KV heads,
physical block 784, which `_choose_compute_block_size` maps to a compute block
of 16:

| max_seq_len | 512 | 1 024 | 2 048 | 4 096 | 8 192 | 16 384 | 32 768 |
|---|---:|---:|---:|---:|---:|---:|---:|
| splits | 1 | 1 | 15 | 14 | 14 | 14 | 14 |

14 across our whole range, which is what he predicted on paper. The 1 at short
context is the `num_n_blocks < 2 * num_sms` early return: splitting does not
begin until `max_seq_len ≥ 84 × 16 = 1344`, so below that the path switches
itself off rather than costing anything.

**Where the split count falls with batch depends on the card.** @yanghoeg found
it collapsing to 1 at batch 16 on gfx1201 and read that as this being a
low-concurrency optimisation. It is, but the threshold moves with the part:

| batch | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| gfx1100, `num_sms=42` | 14 | 9 | 5 | 5 | **5** |
| gfx1201, `num_sms=32` | 14 | 7 | 4 | 2 | **1** |

Read out of the source rather than inferred from the numbers: the guard is
`batch_nheads >= 0.8 * (2 * num_sms)`, and `batch_nheads` is
`batch_size * num_kv_heads`. At batch 16 both cards sit at 64, but the threshold
is 51.2 on a 32-WGP part and 67.2 on a 42-WGP one. Same batch, collapsed there,
not here. His reading holds; the concurrency at which it stops paying scales
with the card.

### What it does not fix

Paged attention falls from roughly 68 % of the decode step to **11 %**. The
remainder is dominated by `triton_w4a16_gemm_kernel`. The absolute rate is still
10.7 tok/s at 32K, against llama.cpp's 21.8 on the same cards, so **llama.cpp
remains the better choice for this model**. The *slope* is fixed; the baseline is
[open question 9](open-questions.md).

## 7. Not established

- **CDNA is untested.** The `_ON_GFX9` branch of `use_rocm_custom_paged_attention`
  has looser conditions, so this may not reproduce on MI-series.
- **No NVIDIA comparison.** We cannot say whether the same model decodes flat on
  CUDA, where the custom-kernel question does not arise in this form.
- **A fix now exists, and it is not ours — see §6.5.** This item used to say
  none was in sight: widening the gating is not one, since there is no
  `head_size=256` kernel to fall through to, and a real fix would mean adding
  that instantiation or giving the Triton path a decode split that does not
  depend on KV-head count. PR #45916 is the second of those, and §6.5 measures
  it working here. The first half still holds: widening
  `use_rocm_custom_paged_attention` is a dead end, three conditions over.
- **Raw profiler traces are not in this repository.** They are ~5 MB of
  `.pt.trace.json.gz`, kept with the rest of the raw experiment data rather than
  committed here.
