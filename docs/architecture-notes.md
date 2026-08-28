# Which model architectures run well on RDNA3 — and why

Once tensor parallelism works, the next surprise is that **model architecture
matters far more than model size** on consumer Radeon. A 26B MoE outruns a
*larger* 31B dense by 2.513× on the same two cards.

> **Corrected 2026-08-27.** The opening example used to be "a 27B model can be
> 3.6× slower than a 31B model", and the hybrid-SSM row below carried that gap
> as an architecture result. Most of it is not architecture: that 27B's
> checkpoint is asymmetric int4, which misses vLLM's native gfx1100 W4A16 kernel
> and sends every quantised linear to Triton. Holding the model fixed and
> changing only the checkpoint is worth up to 3.24×, a flat ~60 ms per decode
> step ([benchmarks/w4a16-symmetry](../benchmarks/w4a16-symmetry/)). The
> context-*slope* findings in this document are unaffected.

Short version:

| Architecture | On gfx1100 | Why |
|---|---|---|
| **Dense transformer** (gemma, Llama, Qwen dense) | 🟢 **Good** | Mature generic GEMM + standard attention |
| **Hybrid SSM / linear attention** (Qwen3.5, Qwen3.6) | 🔴 **Poor, and worse the longer the context** | Two separate causes, and neither is the linear attention. The *collapse* is architectural: the model's few full-attention layers fall off vLLM's ROCm paged-attention fast path ([details](hybrid-decode-on-rdna.md)). The low *baseline* is not architectural at all — the checkpoints measured here are asymmetric int4 and miss the native W4A16 kernel ([w4a16-symmetry](../benchmarks/w4a16-symmetry/)). The gated-delta-net kernels are 0.56 % of a decode step and were never the problem |
| **MoE** (128/256 experts) | 🟢 **Good — fastest measured here** *(revised 2026-07-25)* | Once compiled it leads the field; the obstacle is a 26-minute single-threaded compile, not the kernels |

**Practical advice: dense is the safe default, MoE is the fast one if you can pay
the compile once, hybrid-SSM is the one to avoid.** Everything below is the
evidence. On stock vLLM that advice still holds as measured, but half of what
makes the hybrid-SSM entry bad is its checkpoint: pick a *symmetric* int4 build
and the baseline half goes away ([w4a16-symmetry](../benchmarks/w4a16-symmetry/)).

> **Revised 2026-07-25.** The MoE verdict here used to read "🟡 mediocre, ~15 tok/s".
> That number came from an `--enforce-eager` run, because an earlier attempt gave the
> compiler 20 minutes and it needed 26. Compiled, the same model does **107.8 tok/s**
> — the fastest of five architectures measured. See [benchmarks.md](benchmarks.md).

> Findings verified against vLLM `main` source at the time of writing. Kernel
> code moves; re-check before relying on a specific line.

---

## Hybrid SSM / linear attention: 12.1 tok/s vs 43.2 tok/s

`Qwen3.6-27B` (`model_type=qwen3_5`) is 64 layers: **48 gated-delta-net layers**
+ 16 full-attention + an MTP head. On our dual 7900 XT under vLLM TP=2 it decodes
at **12.1 tok/s**. Most of the distance to the *larger* dense gemma-4-31B is
the checkpoint rather than the architecture, see the correction at the top.

The symptom is diagnostic: **both GPUs sit at 265 W** (busy, not idle) while
**memory-bandwidth utilisation is far below the dense models'** (an earlier probe at
`--gpu-memory-utilization 0.92` put it near 12 %, against ~51 % for the 31B; the campaign
in [benchmarks.md](benchmarks.md) measures dense TP=2 at 38–75 % and does not cover the 27B). The cards
are working hard and achieving little — classic signature of many small,
low-occupancy kernels.

**Why:**

1. **The gated-delta-net path is Triton-only on ROCm.** vLLM's backend selector
   returns `"triton"` for any non-CUDA platform, so the FlashInfer and CuteDSL
   fast paths are unreachable by construction.
2. **Those Triton kernels are tuned for NVIDIA shared-memory sizes.** vLLM
   vendors flash-linear-attention, whose backend table contains **only NVIDIA
   entries** (ADA 101376, AMPERE 166912, HOPPER 232448, DEFAULT 102400 bytes).
   The chunk kernel picks its tile size from that threshold:

   ```python
   BKV_LIST = [64, 128] if check_shared_mem() else [32, 64]
   ```

   gfx1100 has **64 KB (65536) of LDS**, below the 102400 default, so it takes
   the small-tile branch — more iterations, lower arithmetic intensity.
3. **The decode kernel is inherently `num_warps=1`** (a sequential recurrence).
   One warp per instance is very low occupancy on RDNA3, and each token runs
   *48* such layers plus their conv updates.
4. **No AITER fallback**: AITER's gated-delta-net path is gated to gfx9.

> **Qualified 2026-07-29.** All four remain true as statements about the code, but
> the profile says they cannot carry much of the baseline gap. At 1 K context the
> two gated-delta-net kernels together account for **0.56 % of decode time**
> (0.203 + 0.054 ms out of 45.6 ms per step). The largest single item is
> `triton_w4a16_gemm_kernel` at **77 %**, which is the quantised GEMM and has
> nothing to do with SSM — though gemma-4-31B is also w4a16 and decodes at
> 43.2 tok/s, so that alone does not explain it either. **Why the baseline is
> 12.1 rather than ~25 tok/s is still open.** What is settled is the *slope*, and
> that is [hybrid-decode-on-rdna.md](hybrid-decode-on-rdna.md).

There is also this log line:

```
Cannot use ROCm custom paged attention kernel, falling back to Triton
```

That is the **16 full-attention layers**, not the SSM layers.

> **Corrected 2026-07-29 — this line used to be dismissed here as "minor: 16
> layers out of 64".** That was wrong, and it is the single biggest mistake this
> document has made. Those 16 layers are where *all* of the context-dependent cost
> lives: their paged-attention kernel goes from 356.664 µs to 10 095.188 µs per
> call between 1 K and 32 K context, while the 48 SSM layers stay flat at ~8 µs.
> Layer count is not cost share. The fallback happens because the model's
> `head_dim` is 256 and the ROCm custom kernel is only instantiated for 64 and
> 128 — the `block_size` condition is a second, independent disqualifier. Full
> analysis in [hybrid-decode-on-rdna.md](hybrid-decode-on-rdna.md).

**And it gets worse with context — the opposite of the architecture's promise**
*(measured 2026-07-25)*. A linear-attention layer carries a fixed-size recurrent
state, so decode should cost the same at any context length. Measured, the 27B's
decode time is a straight line in context:

```
ctx    518:  82.51 ms/token          ctx  16058: 157.60 ms/token
ctx   8026: 117.23 ms/token          ctx  32084: 235.29 ms/token
```

That is **4.84 µs of decode time per token of context — 41× the dense 8B's 0.118 µs**
and 14× the dense 31B's 0.339 µs. O(1) was promised; O(S) was measured. Power
*falls* at long context (232 + 227 W at 24 K vs 265 + 265 W short): the GPUs are
waiting. At 32 K it delivers **4.2 tok/s**, which is unusable.

**Fixed upstream 2026-07-30, pending merge.** [vllm#45916](https://github.com/vllm-project/vllm/pull/45916)
adds a split-KV decode kernel gated to `on_gfx12x()`; widening that gate to
`on_gfx1x()` gives 2.52× at 32K on this machine and takes the slope from 4.840 to
0.430 µs per context token. See [hybrid-decode-on-rdna.md §6.5](hybrid-decode-on-rdna.md).

The cause is the paged-attention fallback above, not the recurrence. The SSM
layers do take the incremental path — `_forward_core_decode_non_spec` calls
`fused_recurrent_gated_delta_rule_packed_decode`, and it profiles flat across
context. Running the same model under llama.cpp on the same two cards retains
87.7 % of its short-context rate at 32 K (ROCm backend) against vLLM's 35.1 %,
which puts the problem in vLLM's attention path rather than in ROCm, the hardware
or the architecture.

**Prefill, however, keeps the promise.** The 27B's prefill gets *faster* with
length (805 → 883 tok/s, +9.6 %) while dense models lose 8–61 %; the 26B MoE
also gains, +24 %, which is more. Linear attention does deliver O(S) prefill
here.

**What to do:** use **llama.cpp** for Qwen3.5/3.6. Measured at matched context
depth on the same two cards, plain Q4_K_M with no speculative decoding, it is
**2.1× stock vLLM at 512 tokens (24.89 vs 12.1) and 5.1× at 32 K (21.84 vs
4.2)** — the advantage widens with context precisely because llama.cpp does not
take the path in question. Both are against the stock gate; with #45916's gate
widened the 32 K gap narrows to 2.0× (21.84 vs 10.72), so that PR closes the
slope but not the baseline. Adding MTP speculative decoding on top measured
**34.5 tok/s** at a ~200-token prompt, though that figure is not depth-matched
to anything here.
vLLM only becomes
interesting for these models under heavy concurrency, where the low-occupancy
recurrent kernels parallelise across the batch (SSM layers use little KV cache,
so batch headroom is large: in the five-model campaign the 27B reaches 3.52× concurrency
against 1.74× for the dense 31B at the same utilisation) — but
keep the context short, or the linear degradation above will eat the benefit.

---

## MoE: one bad path and one very good one *(revised 2026-07-25)*

`gemma-4-26B-A4B` — 128 experts, standard attention, no SSM.

**Path 1, compiled — the fastest model we have measured.** `torch.compile` of the
128-expert fused-MoE graph takes **26 minutes** on this Zen 1 host (`init engine
1569 s`). An earlier attempt gave it 20 minutes, concluded "impractical", and that
verdict propagated into this document. It was wrong. Once compiled:

| | value |
|---|---|
| decode, short context | **107.8 tok/s** — 1.35× the next-fastest model of any size |
| decode at 32 K context | **72.8 tok/s** |
| power | **193 W / 192 W** at the 107.8 tok/s point; it only reaches 265 W / 265 W from 12 K on, where decode is 88 tok/s |
| concurrency | 9.50× (313 631 KV tokens) |
| compile cost | `init_engine_s` **1569 s** once *(corrected 2026-08-29:
this row used to add "then cached, a warm start is seconds". The campaign
never started this configuration a second time, so the warm cost is
unmeasured here — see below)* |

**Why the compile is so slow — and the one-line lever.** It is host-side
Inductor/Triton codegen, not GPU autotune (off by default) and not graph capture
(that uses the GPU). **vLLM hardcodes**

```python
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
```

unconditionally in `env_override.py:105` (an assignment, not `setdefault`, so an
external export will not override it). Verified still present in vLLM 0.23:
importing vllm leaves `torch._inductor.config.compile_threads == 1`. **Every
compile in our benchmark campaign was single-threaded** — `init_engine_s`
**1537.92 s** for the 12B dense graph at TP=2 and **1569.01 s** for the
128-expert MoE, the two slowest of the campaign's eight engine starts. Patching
that value is the single biggest lever on a slow host; a faster CPU only buys
its single-core ratio.

> **Corrected 2026-08-29.** These read "23 min for a 12B dense graph, 26 min for
> the 128-expert MoE", and `README.md` had the 12B at 24 min *at TP=1*. Neither
> matched `results.jsonl`: the 12B's long start is at **TP=2**, and its TP=1
> starts took 59.67 s and 33.36 s. `init_engine_s` is also engine
> initialisation rather than compile time — it covers loading, profiling, KV
> allocation, compile and capture — so these are upper bounds on the compile.
> Weight loading is 0.36 % of the 12B's start and 3.43 % of the MoE's, which is
> what rules the loader out. Every engine start in the campaign is now pinned by
> `verify_doc_figures.py`.
>
> The same correction retires the "then cached, a warm start is seconds" claim
> above. The campaign's only repeated start is `A-12B-tp1`, 59.67 s cold and
> 33.36 s warm — a configuration that never took a minute either way. Neither
> 25-minute start was ever repeated here, so the warm cost of the expensive
> case is not measured in this repository.

**Path 2, eager — runs, and badly misleads.** `--enforce-eager` starts in 39 s and
decodes at **~15 tok/s**. That is **7.2× below** the compiled figure, because every
expert-routing operation is launched individually. The same trap caught our 12B
dense sweep (eager 15.8 vs compiled 59.9 tok/s, 3.8×).
**Do not draw architecture conclusions from eager numbers.**

**And the kernels are untuned anyway.** vLLM ships tuned `fused_moe` configs for
many NVIDIA GPUs and **none for any AMD GPU**. Missing config is not fatal — the
default is correct, just suboptimal. You can generate your own:

```bash
# from a vLLM checkout, not this repository
python benchmarks/kernels/benchmark_moe.py --model <path> --tp-size 2 --tune
```

**A note on the asymmetric power draw.** We saw 131 W on one card and 65 W on the
other and assumed expert imbalance. That was wrong: under **pure TP**, vLLM keeps
*all* experts on *every* GPU and shards each expert's weights along the
intermediate dimension, so compute is symmetric regardless of routing. Routing
imbalance only matters under **expert parallelism**. The asymmetry is an
AllReduce/eager artefact, not something to fix by balancing experts —
**and it disappears entirely once compiled** (265 W / 265 W, measured 2026-07-25),
which confirms the diagnosis.

**Is expert parallelism better?** Probably not on two cards without P2P.
`TP+EP` still uses AllReduce (all-to-all needs `dp_size > 1`), and `DP+EP` would
use all-to-all — the worst pattern on a no-P2P link. DeepEP needs RDMA NICs.

---

## The common cause: RDNA3 is a second-class citizen in ROCm's kernel ecosystem

Both problems are the same structural gap. AMD's high-performance kernel
libraries target CDNA:

- **AITER** (assembly kernels, tuned MoE, MLA, GDN fast paths) — its supported
  hardware list is Instinct only. vLLM gates it behind `is MI3XX`, so gfx1100
  silently falls back to Triton.
- **Composable Kernel** — fused attention on RDNA3 is forward-only.
- **fused_moe tuned configs** — none for any AMD GPU.
- **FP8** — MI300+ only.

Dense transformers are fine because the *generic* Triton/GEMM path is mature
enough. MoE and hybrid-SSM depend on specialised fused kernels that on gfx1100
are either absent, gated out, or tuned for someone else's hardware. This gap
extends to RDNA4.

**None of this is fixable by configuration.** It is where the ecosystem
currently is. Which is why the honest recommendation is: **run dense models**,
and use llama.cpp where it is faster.
