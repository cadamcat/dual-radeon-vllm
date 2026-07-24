# Which model architectures run well on RDNA3 — and why

Once tensor parallelism works, the next surprise is that **model architecture
matters far more than model size** on consumer Radeon. A 27B model can be 3.5×
slower than a 31B model on the same two cards.

Short version:

| Architecture | On gfx1100 | Why |
|---|---|---|
| **Dense transformer** (gemma, Llama, Qwen dense) | 🟢 **Good** | Mature generic GEMM + standard attention |
| **Hybrid SSM / linear attention** (Qwen3.5, Qwen3.6) | 🔴 **Poor** — use llama.cpp | NVIDIA-tuned Triton kernels, degraded tile size on gfx1100 |
| **MoE** (128/256 experts) | 🟡 **Mediocre** | No tuned configs for any AMD GPU; slow graph compile |

**Practical advice: for now, bet on dense.** Everything below is the evidence.

> Findings verified against vLLM `main` source at the time of writing. Kernel
> code moves; re-check before relying on a specific line.

---

## Hybrid SSM / linear attention: 11.8 tok/s vs 42 tok/s

`Qwen3.6-27B` (`model_type=qwen3_5`) is 64 layers: **48 gated-delta-net layers**
+ 16 full-attention + an MTP head. On our dual 7900 XT under vLLM TP=2 it decodes
at **11.8 tok/s** — 3.5× slower than the *larger* dense gemma-4-31B.

The symptom is diagnostic: **both GPUs sit at 265 W** (busy, not idle) while
**memory-bandwidth utilisation is only ~12%** (versus 51% for dense). The cards
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

There is also a log line that looks alarming but is *not* the main cost:

```
Cannot use ROCm custom paged attention kernel, falling back to Triton
```

That is the **16 full-attention layers**, not the SSM layers. Hybrid Mamba models
inflate the KV `block_size` to 2048, and the ROCm custom paged-attention kernel
requires `block_size == 16`. Minor: 16 layers out of 64.

**What to do:** use **llama.cpp** for Qwen3.5/3.6. Same two cards, same model,
**34.5 tok/s** with MTP speculative decoding — ~3× vLLM. vLLM only becomes
interesting for these models under heavy concurrency, where the low-occupancy
recurrent kernels parallelise across the batch (SSM layers use little KV cache,
so batch headroom is large: we measured 9.2× the KV concurrency of dense).

---

## MoE: two bad paths

`gemma-4-26B-A4B` — 128 experts, standard attention, no SSM.

**Path 1, compiled — blocked at startup.** `torch.compile` of a 128-expert
fused-MoE graph ran **20+ minutes without finishing** on a Zen 1 host, with one
core pinned at 100% and the GPU idle at 11 W. That signature is the diagnosis:
it is neither GPU autotune (off by default in vLLM) nor graph capture (that uses
the GPU). It is host-side Inductor/Triton codegen — and **vLLM hardcodes**

```python
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
```

unconditionally in `env_override.py` (an assignment, not `setdefault`, so an
external export will not override it). Patching that value is the single biggest
lever if you want the compiled path on a slow CPU.

**Path 2, eager — runs, but slow.** `--enforce-eager` starts in 39 s and decodes
at **~15 tok/s**. Without CUDA graphs, every expert-routing operation is launched
individually, so this *understates* what MoE could do. **We never measured the
compiled number** — the compile wall blocked it.

**And the kernels are untuned anyway.** vLLM ships tuned `fused_moe` configs for
many NVIDIA GPUs and **none for any AMD GPU**. Missing config is not fatal — the
default is correct, just suboptimal. You can generate your own:

```bash
python benchmarks/kernels/benchmark_moe.py --model <path> --tp-size 2 --tune
```

**A note on the asymmetric power draw.** We saw 131 W on one card and 65 W on the
other and assumed expert imbalance. That was wrong: under **pure TP**, vLLM keeps
*all* experts on *every* GPU and sharded each expert's weights along the
intermediate dimension, so compute is symmetric regardless of routing. Routing
imbalance only matters under **expert parallelism**. The asymmetry is an
AllReduce/eager artefact, not something to fix by balancing experts.

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
