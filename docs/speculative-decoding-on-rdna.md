# Speculative decoding on RDNA3: fast at short context, 3.4x slower at 32K

`gemma-4-31B` with Google's official MTP assistant checkpoint runs **36.9 %
faster at 1K context and 70.8 % slower at 32K** on 2× RX 7900 XT. vLLM's own
documentation lists the 31B assistant as supported and gives a command line,
with no mention of context length.

The cause is one line in the Triton attention backend. Enabling speculation sets
`max_seqlen_q = 2`, which disables the segmented-softmax path that long-context
decode depends on.

- Turn MTP **on** for short prompts. It is a real 1.37× there.
- Turn it **off** for long context. By 8K it is already a net loss, and at 32K
  it costs more than it ever gave.
- The crossover was not located. It lies between 1K and 8K.

---

## 1. What was measured

`gemma-4-31B-it-qat-w4a16-ct`, TP=2, batch 1, greedy, `--speculative-config
'{"method":"mtp","model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":1}'`.
Decode isolated from prefill by differencing a 64-token generation against an
8-token one at the same depth. Rates are tok/s.

| context | no speculation | MTP | change |
|---:|---:|---:|---:|
| 1 024 | 43.42 | **59.43** | **+36.9 %** |
| 8 192 | 37.97 | 32.64 | −14.0 % |
| 16 384 | 35.28 | 23.86 | −32.4 % |
| 32 768 | 30.31 | **8.85** | **−70.8 %** |

Monotonic, not a stray point.

vLLM's own counter at 32K:

```
Mean acceptance length: 1.12   Accepted: 9   Drafted: 78
Per-position acceptance rate: 0.115
```

Back-solving the 1K speedup implies roughly 0.46 acceptance there, so acceptance
degrades with depth as well. Low acceptance is a real cost in itself — a
rejected draft has still paid for the assistant's forward pass and the target's
verification — but it cannot account for −70.8 % when the assistant is four
layers wide. §5 shows what can: restoring the segmented path recovers 3.66× with
acceptance unchanged.

## 2. The cause

`vllm/v1/attention/ops/triton_unified_attention.py`:

```python
use_3d = not (
    seq_threshold_3D is None
    or ...
    or max_seqlen_q > 1          # speculation makes this 2
    or num_seqs > seq_threshold_3D
    or is_batch_invariant
)
```

The 3D kernel splits a long KV sequence into segments and reduces them in
parallel. It is what
[vllm#45916](https://github.com/vllm-project/vllm/pull/45916)'s split-KV does
for the *other* Triton attention path, and it is what long-context decode lives
on. Losing it was reported upstream before we ran into it, through a different
trigger; §6 has the attribution. Launch grids:

```
2D:  (total_num_q_blocks, num_kv_heads)                            = 1 × 8
3D:  (total_num_q_blocks, num_kv_heads, num_par_softmax_segments)  = 1 × 8 × 16
```

**8 workgroups against 128.** Speculation gives that up.

The real constraint is not what the comment says. It reads "the batch includes
at least one prefill request", but the binding limit is buffer shape:

```
allocated (triton_attn.py):  [seq_threshold_3D, num_heads_q, segments, headdim]
                              per SEQUENCE
indexed (kernel):            segm_output[token, head, segm_idx, :]
                              per TOKEN
```

At `query_len = 1` these coincide. At 2 the buffer would overflow, so the guard
excludes it — and `max_seqlen_q > 1` catches speculation along with prefill.

## 3. Why the mean is misleading

Torch profiler, rank 0, same model and depth. Per-call time of
`kernel_unified_attention`, over 3 780 calls without speculation and 3 456 with:

| | no speculation | MTP | ratio |
|---|---:|---:|---:|
| median | 106.6 µs | 126.9 µs | **1.19×** |
| p75 | 108.7 | 132.1 | 1.22× |
| **max** | **647.4** | **9950.3** | **15.4×** |
| mean | 192.7 | 1770.3 | 9.2× |

Most calls barely move. A minority explode. That is gemma-4's alternating
attention: sliding-window layers see a short KV and do not care, global layers
scan the full 32K and lose the segmentation entirely. Quoting the 9.2× mean
describes neither population.

`hipGraphLaunch` goes 63 → 324, so CUDA graph replay is intact. That was checked
because it is the obvious suspect.

## 4. Three explanations that are wrong

Each was believed, then measured, then dropped. They are recorded because they
are the three things anyone would try first.

**Not KV-head starvation.** The 27B's collapse in
[hybrid-decode-on-rdna.md](hybrid-decode-on-rdna.md) comes from having only 4 KV
heads. The 31B has 16, enough to stay usable without speculation, and still
collapses under it, so head count is not the variable.

**Not `query_len` itself.** Calling the kernel directly with 31B's shapes, no
model and no speculation, sweeping query rows:

| kv_len | q=1 | q=2 | q=4 | q=8 |
|---:|---:|---:|---:|---:|
| 8 192 | 1362.1 | 1361.7 | 1357.7 | 1374.5 |
| 32 768 | 5036.0 | **5039.8** | 5047.4 | 5102.2 |

**0.08 % at 32K.** Two independent constructions agreed. Going from one query
row to two is nearly free; losing the segmented path is not.

That sweep also showed halving the head count leaves the time unchanged (5045 →
5036 µs) while halving the KV bytes, i.e. 53–106 GB/s against the card's 800
GB/s. At batch 1 this kernel is occupancy-bound, not bandwidth-bound.

**Not the tiling parameters.** Upstream carries a `tuned_large_head` block for
`head_size == 256 and max_seqlen_q > 1`, gated to NVIDIA Blackwell. Every other
condition matches this hardware. Porting its `BLOCK_M = 32` made 32K **8.7 %
slower**, so the B200-tuned constants are not transferable and under-tiling is
not the story.

## 5. Forcing the path on

Patching the buffers to `seq_threshold_3D × 2` and raising the cutoff to 2:

| 32K | tok/s |
|---|---:|
| no speculation | 30.31 |
| MTP, 3D disabled | 8.85 |
| **MTP, 3D forced on** | **32.42** |

**3.66× recovered**, and speculation finally beats the no-speculation baseline
by 7 %. This confirms where the bottleneck is.

**It is not a fix.** Greedy output at a 30K prompt diverges from stock at token
35, though both remain correct text and short-context output matches token for
token. Three causes were not separated: benign float reassociation in the
segmented reduction, an error in the crude buffer widening, or genuinely
undefined behaviour at `query_len = 2`. Separating them needs numerical
comparison against a reference, not token comparison.

Prefill also still has to be excluded, and `max_seqlen_q` alone cannot express
"speculation's 2 but not prefill's 4096". A real fix means making the 3D path
support multiple query rows, not widening the condition.

## 6. Upstream

**The same 3D-to-2D drop was reported first**, in
[vllm#48076](https://github.com/vllm-project/vllm/issues/48076) (2026-07-09), on
a single B200 running `Gemma-4-31B` in NVFP4 — same architecture as ours, a
different quantisation. That report's trigger is `num_seqs > seq_threshold_3D`,
a batch-size threshold it places at roughly 12, and its proposed fix
[vllm#47520](https://github.com/vllm-project/vllm/pull/47520) derives that
threshold from SM count. Both are open as of 2026-08-01.

**That fix does not reach this trigger.** `max_seqlen_q > 1` sits in the same
`or` chain but is independent of the threshold, so speculation keeps falling to
2D however the threshold is computed. What is new here is the second trigger,
its different cause, and RDNA3 measurements.

## 7. Not established

- **Crossover point.** Somewhere between 1K and 8K, not measured.
- **Sampling.** All numbers are `temperature=0`, where draft and target agree
  most often. Sampling should lower acceptance, but that was not measured.
- **One model, one machine.** gfx1100, TP=2, VFIO guest without P2P.
- **Whether the divergence in §5 is benign.** Unresolved.
- **CUDA.** No NVIDIA hardware here. The mechanism is platform-independent, but
  the numbers are not.

## 8. Reproducing

Scripts are on the test machine rather than in this repository, since they need
a 31 GiB checkpoint and a 0.9 GiB assistant to say anything:

- end-to-end sweep: differencing method, four depths
- kernel sweep: `unified_attention` called directly, no model
- correctness: `apply_chat_template`, greedy, compares `token_ids`

Their output is in
[`benchmarks/speculative-decoding/`](../benchmarks/speculative-decoding/). Two
sets of figures above are not: §3's per-call statistics are derived from
torch-profiler traces that stay on the machine at ~2 MB each, and
`trace-unified-attention.json` there carries what was derived from them plus the
trace filenames. The acceptance counters in §1 are vLLM's own log line, quoted
as printed and not otherwise recorded.

The assistant checkpoint is `google/gemma-4-31B-it-assistant`, 0.90 GiB, not
gated. Its `head_dim` and `global_head_dim` match the target, which is what lets
the two share a KV cache.
