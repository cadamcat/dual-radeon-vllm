# The Triton paged-decode kernel reads the whole sequence, then masks the window away

On `Muse-Glimmer-30B` at 32 768 tokens of context that costs **70.67 % of decode
CUDA time**, and removing it is worth **3.11×** end to end with the generated
tokens identical. Eleven lines. Upstream `main` still does it as of 2026-08-24.

The scope is narrower than it first looks, and §5 says why.

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

At 32 768 context with a 2 048 window and `block_size` 16 that is **2 048 blocks
read where 128 are needed**.

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

`Muse-Glimmer-30B` int4, TP=2, 2× RX 7900 XT, one process per column, greedy,
64 generated tokens differenced against 8 at the same depth.

| context | before | after | speed-up | tokens identical |
|---:|---:|---:|---:|---|
| 8 192 | 35.80 ms/tok (27.94 tok/s) | 24.46 ms/tok (**40.88 tok/s**) | 1.46× | 64 / 64 |
| 32 768 | 83.83 ms/tok (11.93 tok/s) | 26.93 ms/tok (**37.14 tok/s**) | **3.11×** | 64 / 64 |

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

**Decode also becomes nearly flat with context** — 40.88 tok/s at 8 192 against
37.14 at 32 768. A capped window should behave that way, and no other model
measured in this repository does; they all start fast and fall.

## 4. Why this model reached the path at all

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
where the `sliding_window` clause is the sole blocker —
[hybrid-decode-on-rdna.md §2](hybrid-decode-on-rdna.md) records the complementary
case, where `head_size` and `block_size` each disqualify independently.

**The exclusion of those 39 layers is untouched by this change.** After it they
still run at roughly 6.3 GiB/s of effective KV bandwidth against 412 GiB/s for
the custom kernel in the same run, about 65× apart. Closing that would now be
worth about 19 % rather than another 3×, because the weight GEMM has become the
larger term.

## 5. Scope, which is narrower than "sliding-window models"

The change requires the model to land on `ROCM_ATTN`. **Gemma-4, the largest
sliding-window family available here, is routed away from it**:

```
Gemma4 model has heterogeneous head dimensions (head_dim=256,
global_head_dim=512). Forcing TRITON_ATTN backend to prevent
mixed-backend numerical divergence.
```

`TRITON_ATTN`'s decode kernel already bounds its loop by the window
(`triton_unified_attention.py`, `for j in range(loop_lo, loop_hi)`), so Gemma-4
never had this cost. Measured rather than assumed: `gemma-4-31B` before and after
is 24.36 → 24.24 ms/tok at 8 192 and 32.94 → 33.12 at 32 768, ±0.5 %, with 64 of
64 tokens identical at both depths.

The no-window control is `Qwen3.8-27B`, which uses the same `ROCM_ATTN` backend
with `sliding_window` unset: patched it reads 84.09 and 97.17 ms/tok against
83.32 and 94.70 measured unpatched earlier the same day, +0.9 % and +2.6 %. One
run each side, so **that does not separate a regression from process-to-process
spread** and should be repeated before the change is trusted on shared paths.

`Muse-Glimmer-30B` is therefore the only model on this machine confirmed to reach
the affected path. Mistral, Ministral and Phi-3 style models have uniform head
dimensions and a sliding window and are the obvious next candidates; none has
been tested.

## 6. What this does not establish

Every cell is n=1. Correctness rests on token identity for one synthetic prompt
at two depths, not a test suite. Whether the block table is genuinely full-length
— rather than the skip merely happening to be safe here — was inferred from the
identical output, not read out of the KV cache manager.

The model itself runs here through a downstream adaptation: vLLM support for
`Muse-Glimmer` merged upstream on 2026-08-14, after this container was built, so
the model file is the merge-commit version with `load_weights` rewritten to this
version's `stacked_params_mapping` convention and one inlined
`is_vit_use_data_parallel` fallback. The weight loader was checked by coverage —
1 600 of 1 912 parameters loaded, the 312 remainder being exactly 52 layers × 6
KV-cache quantisation scales that this checkpoint does not carry — and by plain
completions, which come out coherent.

Raw data: [`benchmarks/sliding-window-block-skip.json`](../benchmarks/sliding-window-block-skip.json).
