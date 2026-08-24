# The Triton paged-decode kernel reads the whole sequence, then masks the window away

On `Muse-Glimmer-30B` at 32 768 tokens of context that costs **70.67 % of decode
CUDA time**. Removing it is worth **3.15×** there and **2.75× on `gemma-3-27b`**.
Eleven lines. Upstream `main` still does it as of 2026-08-24.

**Someone proposed the same eleven lines a month before we found them.**
[vllm#49588](https://github.com/vllm-project/vllm/pull/49588), opened 2026-07-23,
is identical once the identifier and the argument order of `tl.maximum` are
normalised. It has been a draft since 2026-07-25 with no review comments. This
page is not a proposal; §9 says what the two sets of evidence are and why they
do not overlap.

Correctness rests on two kernel-level results, §6: upstream's own test file with
no case changing outcome, and 15 boundary cases bit-identical under
`torch.equal`. **An earlier version of this page used end-to-end token identity
instead. That test does not work on this machine and the claim has been
withdrawn** — see §7.

`gemma-3-27b` unpatched decodes at **8.06 tok/s** at 32 K, against 30.36 for
`gemma-4-31B`, a larger and newer model. §5 explains why the two land on
different kernels.

---

## 1. What the kernel does

`vllm/v1/attention/ops/chunked_prefill_paged_decode.py`, the decode kernel behind
the `ROCM_ATTN` backend:

```python
num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)
for j in range(0, num_blocks):
    ...
    context_len = seq_len - 1
    if SLIDING_WINDOW > 0:
        S = tl.where((context_len - seq_offset) < SLIDING_WINDOW, S, -10000)
```

The loop covers the whole sequence. The window is applied afterwards, as a mask
on the scores. Every block older than the window is loaded, multiplied, masked to
`-10000`, and contributes `exp(-10000 - m)`, which is zero.

At 32 768 context with a 2 048 window and `block_size` 16 that is **2 048 block
iterations where 128 are needed**.

**What those iterations actually read is one block, over and over.**
`SlidingWindowManager.remove_skipped_blocks` frees the blocks that fall out of
the window and writes `self._null_block` into their slots
(`single_type_kv_cache_manager.py`), so the block table stays full length and
every skipped position points at the same shared null block. The cost is
therefore the loop and the matmul against that block, not KV bandwidth — and the
block table being full length is also why starting the loop later cannot index
out of bounds. Both of these were read out of the source rather than inferred
from the outputs matching.

## 2. The change

```diff
     num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)
 
+    if SLIDING_WINDOW > 0:
+        first_block = tl.maximum(0, seq_len - SLIDING_WINDOW) // BLOCK_SIZE
+    else:
+        first_block = 0
+
     offs_n = tl.arange(0, BLOCK_SIZE)
     offs_d = tl.arange(0, HEAD_SIZE_PADDED)
     # iterate through tiles
-    for j in range(0, num_blocks):
+    for j in range(first_block, num_blocks):
```

The mask keeps a token when `context_len - seq_offset < SLIDING_WINDOW` with
`context_len = seq_len - 1`, so the lowest surviving absolute index is
`seq_len - SLIDING_WINDOW`. Every block below `(seq_len - SLIDING_WINDOW) //
BLOCK_SIZE` is masked in full. **Starting the loop there is an identity, not an
approximation** — which is why the test below is token equality rather than a
tolerance.

## 3. What it is worth

TP=2, 2× RX 7900 XT, one process per cell, greedy, 64 generated tokens
differenced against 8 at the same depth. **Three independent processes per
cell**; the median is shown with the observed range.

`gemma-3-27b-it` w4a16, window 1 024:

| context | before | after | speed-up |
|---:|---:|---:|---:|
| 512 | 21.32 (21.30–21.43) | 21.33 (21.31–21.49) | 1.00× |
| 1 024 | 23.78 | 23.64 | 1.01× |
| 2 048 | 26.96 | 24.32 | 1.11× |
| 4 096 | 33.34 | 25.70 | 1.30× |
| 8 192 | 46.28 | 28.46 | 1.63× |
| 16 384 | 72.30 | 33.99 | 2.13× |
| 32 768 | 124.29 (124.24–124.33) | 45.26 (45.21–45.37) | **2.75×** |

`Muse-Glimmer-30B` int4, window 2 048:

| context | before | after | speed-up |
|---:|---:|---:|---:|
| 512 | 21.92 | 21.97 | 1.00× |
| 1 024 | 23.55 | 23.53 | 1.00× |
| 2 048 | 26.41 | 26.26 | 1.01× |
| 4 096 | 30.16 | 26.27 | 1.15× |
| 8 192 | 37.84 | 26.31 | 1.44× |
| 16 384 | 53.21 | 26.52 | 2.01× |
| 32 768 | 83.99 (83.81–84.00) | 26.63 (26.56–26.73) | **3.15×** |

Milliseconds per generated token.

![sliding-window block skip](assets/sliding-window-block-skip.svg)

**The shape is the mechanism check.** Below each model's own window the change is
worth 1.00×, because there is nothing to skip. From there it grows monotonically.
A curve of any other shape would have said the explanation was wrong even if the
headline number was right. Patched, `Muse-Glimmer` is nearly context-independent:
21.97 ms/tok at 512 against 26.63 at 32 768.

At the kernel, same profiler settings as the existing traces:

| | before | after | |
|---|---:|---:|---:|
| `kernel_paged_attention_2d`, per call | 1.589 ms | **154.796 µs** | 10.3× |
| custom HIP `paged_attention_ll4mi_QKV`, per call | 30.482 µs | 30.954 µs | unchanged |
| decode step | 85.567 ms | **28.668 ms** | 2.98× |

10.3× rather than the 16× the block count suggests: the per-call overhead does
not shrink with the block count.

**Decode stops being attention-bound.** Attention falls from 74 % of the step to
22 %, and the 4-bit weight GEMM becomes the largest single term at 44.7 %
(260 calls per step, which is 52 layers × 5 matmuls). For a quantised dense model
that is the expected shape; the state before the change was not.

## 4. What `Muse-Glimmer` shows about the custom HIP kernel

`Muse-Glimmer-30B` is 52 layers, 39 of them sliding attention with a 2 048 window
and 13 full attention with `rope_theta = 0`. `head_dim` is 128, `block_size` 16,
GQA ratio 16, `kv_cache_dtype` auto, bf16, no alibi, no sinks.

That clears **every clause** of the RDNA branch of
`use_rocm_custom_paged_attention` except one:

```python
and (sliding_window == 0 or sliding_window == (-1, -1))
```

So the 13 full-attention layers take the custom HIP kernel and the 39 sliding
ones fall back to Triton. The call counts say so exactly: 2 496 = 39 × 64 decode
steps for the Triton kernel, 832 = 13 × 64 for the custom one. This is the first
model in this repository to reach the custom kernel at all, and the first case
where the `sliding_window` clause is the sole blocker. `gemma-3-27b` does not
get there for a different reason, a GQA ratio of 2 against the required 3, so
all of its layers stay on the Triton path —
[hybrid-decode-on-rdna.md §2](hybrid-decode-on-rdna.md) records the complementary
case, where `head_size` and `block_size` each disqualify independently.

**The exclusion of those 39 layers is untouched by this change.** After it they
still run at roughly 6.3 GiB/s of effective KV bandwidth against 412 GiB/s for
the custom kernel in the same run, about 65× apart. Closing that would now be
worth about 19 % rather than another 3×, because the weight GEMM has become the
larger term.

## 5. Scope: what decides whether a model is affected

The change only matters if the model reaches `ROCM_ATTN`. Two members of the same
family land on opposite sides of that, which is the clearest way to see the rule.

**`gemma-4` is not affected**, because vLLM forces it elsewhere:

```
Gemma4 model has heterogeneous head dimensions (head_dim=256,
global_head_dim=512). Forcing TRITON_ATTN backend to prevent
mixed-backend numerical divergence.
```

That is `Gemma4Config.verify_and_update_config`, and it fires only when
`head_dim` and `global_head_dim` both exist, differ, and the larger exceeds 256.
`TRITON_ATTN`'s decode kernel already bounds its loop by the window
(`triton_unified_attention.py`, `for j in range(loop_lo, loop_hi)`), so gemma-4
never paid this cost. Measured rather than assumed, two runs per state: `gemma-4-31B` goes
24.27 → 24.44 ms/tok at 8 192 and 33.10 → 33.16 at 32 768, within 0.7 %.

**`gemma-3` is affected.** It has a single uniform `head_dim` of 128 and no
`global_head_dim`, so that rule does not fire, and it takes `ROCM_ATTN` by the
platform's ordinary priority order. Its GQA ratio is 2, below the custom HIP
kernel's minimum of 3, so all 62 layers run on the Triton path.

**The gap between the two is the finding.** Unpatched, `gemma-3-27b` decodes at
8.06 tok/s at 32 K while `gemma-4-31B` — larger, newer, same family — does 30.36.
Patched, gemma-3 reaches 22.12 and what remains is explicable by the models
rather than by a kernel doing 32× the block iterations it needs.

**The no-window control** is `Qwen3.8-27B`, same `ROCM_ATTN` backend with
`sliding_window` unset. Three runs per state: 8 192 goes 84.09 → 84.08 and
32 768 goes 95.42 → 96.69, and the before and after ranges overlap at both
depths. An earlier single run showed +2.6 % at 32 768 and this settles it as
spread rather than a regression.

**Who else might qualify.** A small window relative to context turns out to be
uncommon among 2026 flagships. Checked and found not to qualify:
`Mistral-Small-3.2` and `Mistral-7B-v0.3` (no window), `Ministral-8B` (window
32 768, which covers the whole context at these depths), `Phi-4-reasoning` (no
window), `Phi-4-mini-reasoning` (window 262 144), `GLM-5.2`,
`DeepSeek-V4-Flash`, `Ornith-1.5` and `LFM2.5`. Gemma is where small windows
live, and gemma-2 was not checked because its config is gated.

## 6. Correctness

Two kernel-level results, both with fixed inputs, so neither is touched by the
non-determinism in §7.

**Upstream's own test file, unmodified.** The container carries a full vLLM
source tree at `/app/vllm` dated 2026-07-15, matching the installed package, so
`tests/kernels/attention/test_prefix_prefill.py` is version-matched rather than
borrowed from `main`. Run whole, unpatched then patched, compared per case from
junit XML rather than by summary counts:

| | cases | passed | skipped | failed | wall |
|---|---:|---:|---:|---:|---:|
| unpatched | 388 | 164 | 224 | 0 | 902 s |
| patched | 388 | 164 | 224 | 0 | **342 s** |

**Zero cases changed outcome and zero appeared in only one run.** The
sliding-window cases did run rather than skip: 41 passed at `sliding_window=0`,
16 at `16`, 16 at `2048`, across head sizes 24 and 128, `num_queries_per_kv` 1
and 64, both devices, `kv_cache_dtype` auto and fp8. The 224 skips are
`fp8_e5m2`, which ROCm's custom paged attention does not support, and are the
same 224 in both runs. The suite itself ran 2.6× faster patched, which is
independent corroboration of the change.

**Fifteen boundary cases, bit-identical.** Inputs built the way `tests/kernels`
builds them, run through the kernel directly in two separate processes, outputs
compared with `torch.equal` rather than `allclose` — skipped blocks contribute
`exp(-10000 - m)`, which is zero, so bit equality is the right bar and anything
less would mean the mechanism is not what §2 claims.

| case | what it pins |
|---|---|
| `mixed-w256-bs16` | `first_block` 32..96 across ten sequences in one batch |
| `seq-below-window` | `seq_len` 101 against window 2 048, `first_block` 0 |
| `seq-equal-window` | `seq_len` exactly the window, `first_block` 0 |
| `seq-window-plus1` | one token past, still 0 since `(257-256)//16 = 0` |
| `seq-window-plus-blk` | one block past, `first_block` becomes 1 |
| `seq-unaligned-window` | window 100 with `block_size` 16, `first_block` 118 of 126 |
| `mixed-w17-bs16` | almost everything skipped, 47..110 of 49..112 |
| `nowindow-bs16`, `-bs32` | no window: must be bit-identical, and is |

All 15 agree bit for bit. They ran in separate processes and still agreed, which
places §7 outside this kernel.

## 7. Greedy decoding here is not reproducible, and it is not this change

The original correctness argument on this page was that the generated tokens were
identical before and after. **That test does not work on this machine.**

Running three independent processes per cell with *no code change between them*,
10 of 36 cells produced more than one greedy output. It happens at any depth,
including 512 tokens against a 2 048 window where this change provably does
nothing, and the split is symmetric between the two kernel states, 5 and 5. The
first divergence is usually early, index 0 to 9 of 64, after which greedy
decoding amplifies it.

`gemma-4-31B`, on the `TRITON_ATTN` backend, was deterministic in all four of its
cells. The affected models are the ones on `ROCM_ATTN`.

**It is inside a single process, and a warm-up does not fix it.** One engine, one
prompt, a warm-up call, then eight identical greedy generations back to back:

Distinct outputs from eight identical calls, two processes per cell, **both
kernel states measured** so the change is controlled for rather than argued
around:

| model | context | unpatched | patched |
|---|---:|---:|---:|
| `Muse-Glimmer-30B` | 512 | **5, 5** | **7, 8** |
| `Muse-Glimmer-30B` | 8 192 | 2, 2 | 2, 2 |
| `gemma-3-27b` | 512 | 1, 1 | 1, 1 |
| `gemma-3-27b` | 8 192 | 2, 3 | 2, 2 |

**Present in both states and in the same shape**, so this change is not the
cause. The worst cell, `Muse-Glimmer` at 512, is *below* its own 2 048 window,
where the change is bit-identical by §6; the gap between 5 and 7 there is the
spread of a count of distinct draws, not an effect. Model and depth dependent
rather than uniform, and the processes agree on the pattern, so it is a property
of the configuration rather than luck.

This resembles [vllm#50603](https://github.com/vllm-project/vllm/issues/50603),
open since 2026-07-31, which reports first-call non-determinism from the same
Triton fallback on gfx1100 and names `gqa_ratio=2` as what gates the CK kernel
out — `gemma-3-27b` has exactly that. **One detail is now measured rather than
suspected and it does not match: that report says a warm-up call fixes it, and it
does not here.** Whether this is the same defect from a different angle or a
second one is unsettled, and nothing has been posted there. Data:
[`benchmarks/gfx1100-greedy-nondeterminism.json`](../benchmarks/gfx1100-greedy-nondeterminism.json).

## 8. What this does not establish

Two models on one machine, gfx1100 only. Whether the block table is genuinely
full-length — rather than the skip merely happening to be safe here — has not
been read out of the KV cache manager.

The model itself runs here through a downstream adaptation: vLLM support for
`Muse-Glimmer` merged upstream on 2026-08-14, after this container was built, so
the model file is the merge-commit version with `load_weights` rewritten to this
version's `stacked_params_mapping` convention and one inlined
`is_vit_use_data_parallel` fallback. The weight loader was checked by coverage —
1 600 of 1 912 parameters loaded, the 312 remainder being exactly 52 layers × 6
KV-cache quantisation scales that this checkpoint does not carry — and by plain
completions, which come out coherent.

Raw data: [`benchmarks/sliding-window-block-skip.json`](../benchmarks/sliding-window-block-skip.json).

## 9. Someone got here first, and with different evidence

[vllm#49588](https://github.com/vllm-project/vllm/pull/49588) by @hec-ovi,
opened 2026-07-23, draft since 2026-07-25, no review comments. Its five
executable lines are the same as §2's after normalising `first_block` against
`start_block` and the argument order of `tl.maximum`; the comments differ in
wording, not in reasoning.

We found this by running vLLM's own duplicate check from `AGENTS.md` before
drafting anything, which is the point of that check. **Nothing here has been
opened as a second PR and nothing has been posted to that one.**

The two bodies of evidence do not overlap:

| | #49588 | here |
|---|---|---|
| hardware | one gfx1151 board | gfx1100 |
| level | kernel microbenchmark, 100 iterations of one decode call | two real models end to end |
| repeats | 100 calls in one process | three independent processes per cell, seven depths |
| correctness | hand-written PyTorch reference, 0.002 tolerance | upstream's own test file, no case changing outcome, plus 15 boundary cases bit-identical |
| scope | not discussed | which models reach the path and which are routed away |

Its motivating example, `poolside/Laguna-S-2.1`, is real and its description
checks out — 48 layers, 36 of them sliding with a 512-token window, `head_dim`
128 — but it is cited as motivation rather than tested; the measurements are
synthetic parameters shaped after it.
