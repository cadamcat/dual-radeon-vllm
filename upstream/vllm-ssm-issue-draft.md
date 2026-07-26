# [Performance] Hybrid-SSM decode cost grows linearly with context on ROCm — 41x steeper than a dense model, unusable at 32K

**Draft — not yet submitted.** Target: `vllm-project/vllm`. Review before filing.

---

## Summary

For `Qwen3.6-27B` (`model_type=qwen3_5`: 48 gated-delta-net / linear-attention layers
+ 16 full-attention layers), decode time per generated token grows **linearly with
context length** on ROCm/gfx1100. A linear-attention layer carries a fixed-size
recurrent state, so its decode cost should be **independent** of context — that is the
architecture's central claim.

Measured, one token of context costs **4.84 µs of decode time**, against 0.118 µs for
a dense BF16 model and 0.340 µs for a *larger* dense model on the same two GPUs —
**41x and 14x steeper**. At 32K context the model delivers **4.2 tok/s**, which is not
usable.

Prefill, by contrast, behaves exactly as the architecture promises (see below), so
this looks specific to the decode path rather than to the model or the port as a whole.

## Data

Same machine, same prompts (incremental truncations of one public-domain text), TP=2,
CUDA graph enabled, 512-token outputs, mean of two runs.

Decode time per generated token, `Qwen3.6-27B-AWQ`:

| context | ms / token |
|---:|---:|
| 518 | 82.51 |
| 4 060 | 99.60 |
| 8 026 | 117.23 |
| 12 078 | 138.12 |
| 16 058 | 156.99 |
| 20 073 | 176.37 |
| 24 040 | 196.85 |
| 32 084 | 235.29 |

A straight line: ~4.8 µs added per token of context, constant slope end to end.

Against the dense models measured the same way:

| model | µs of decode time per context token | decode 500 → 32K |
|---|---:|---:|
| Qwen3-8B, BF16 dense | 0.118 | 79.6 → 61.4 tok/s (−22.8 %) |
| gemma-4-26B-A4B, MoE | 0.142 | 107.8 → 72.8 (−32.5 %) |
| gemma-4-12B, w4a16 dense | 0.228 | 59.9 → 41.9 (−30.1 %) |
| gemma-4-31B, w4a16 dense | 0.339 | 43.2 → 29.5 (−31.5 %) |
| **Qwen3.6-27B, hybrid SSM** | **4.840** | **12.1 → 4.2 (−64.9 %)** |

The 16 full-attention layers should contribute roughly a quarter of what a
64-layer full-attention model does; instead the model degrades **14x faster** than a
60-layer dense one.

## Corroborating signal: the GPUs go idle as context grows

Per-card power during decode, same model:

| context | card 0 + card 1 |
|---:|---|
| 518 | 265 W + 265 W |
| 16 058 | 253 W + 251 W |
| 24 040 | 232 W + 227 W |

Power *falls* as the work per token rises — the pattern of stalling, not of compute.
Every dense model in the same sweep holds 265 W flat across the whole range.

## Prefill is fine — the linear-attention advantage does show up there

`Qwen3.6-27B` is the only model in the sweep whose prefill throughput **improves**
with length (805 → 880 tok/s, +9 % from 500 to 32K tokens) while dense models lose
8–44 %. So the O(S) prefill behaviour is delivered. It is decode that inverts.

## Environment

- 2× Radeon RX 7900 XT (gfx1100, RDNA3), TP=2
- vLLM 0.23, PyTorch 2.11, ROCm 7.14
- `--max-num-seqs 128` (the default 256 exceeds the available Mamba cache blocks and
  crashes CUDA graph capture — separate matter)
- Startup logs: `Cannot use ROCm custom paged attention kernel, falling back to Triton`
  and `Using TRITON_ATTN backend`

Raw data (292 measurements, 5 models × 11 context lengths) and the runner:
https://github.com/2462381442/dual-radeon-vllm/tree/main/benchmarks

## Questions

1. **Is the decode path taking the recurrent/incremental form at all?** A per-token
   cost that scales with sequence length is what you would expect if the state update
   re-scans the sequence instead of advancing the recurrent state.
2. **Is this ROCm-specific?** We have no NVIDIA hardware to compare against. If the
   same sweep on CUDA shows a flat line, the problem is in the ROCm path — vLLM's
   backend selector returns `"triton"` for any non-CUDA platform, so the FlashInfer
   and CuteDSL fast paths are unreachable here by construction. If CUDA shows the same
   slope, it is the `qwen3_5` implementation generally.
3. Would you like the sweep re-run with any specific flags, or on another model in
   the Qwen3.5/3.6 family? The harness is scripted and one configuration takes about
   40 minutes.

## Not asking you to take our word for it

Everything above is reproducible from the linked repository:
`bench_runner.py` drives the sweep, `decode_slope.py` computes the per-context-token
cost from the raw `results.jsonl`. If the numbers are wrong we would rather find out
that way.
