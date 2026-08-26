# vllm#45450 validated: 3D flash-decoding under speculation is bit-exact

[vllm#45450](https://github.com/vllm-project/vllm/pull/45450) admits
speculative-decode verify steps (query_len = 1 + num_spec) into the
Triton unified-attention 3D flash-decoding path — the fix for the
collapse documented in
[speculative-decoding-on-rdna.md](../../../docs/speculative-decoding-on-rdna.md)
and measured across this annex. The PR has been stalled with conflicts
since 2026-07-03, its numbers were taken on B300 + vLLM v0.22.1, and
nobody had tested the one thing that gates a merge: whether the 3D path
under speculation computes the same thing the 2D path does.

This directory is that test. `inject_45450.py` ports the PR's
spec-admission mechanism onto released vLLM 0.28.0 — the launcher gate
(`max_seqlen_q > 1` → `> decode_query_len`) plus per-token sizing of the
three `softmax_segm_*` buffers, threaded from the speculative config.
The window-relative segmentation (`WINDOW_SEG_3D`) is deliberately not
ported: sliding layers keep today's window-blind full-sequence
segmentation, the same one they use for q=1 3D, so the port isolates
the admission question. A probe-only instrumentation line prints
`PROBE_3D_SPEC_ACTIVE` once when the 3D path actually serves a
`max_seqlen_q > 1` step; every patched log contains it exactly once,
every stock log zero times.

One Colab session (A100-SXM4-80GB, vLLM 0.28.0 as released,
`gemma-4-31B-it-qat-w4a16-ct` + the official MTP assistant, k=1,
gemma-4's config path forcing TRITON_ATTN throughout).

## Correctness: 8/8 bit-identical

`probe_ids.py`: fixed token-id prompt (8192 tokens), greedy, 64
generated tokens, two generations per engine, two engines per state.

- Machine determinism first: the four stock generations (2 in-process
  x 2 processes) are identical — this A100 is bit-deterministic for
  this workload, unlike the gfx1100 box (vllm#50603), so identity is a
  valid test here.
- Then the four patched generations are identical to each other **and
  to the stock four**: 8/8 equal token-id sequences across
  {2D, 3D-under-spec} x {in-process x2, cross-process x2}
  (`logs/A1,A2,B1,B2.log`; recomputed by
  `../analyze/verify_doc_figures.py`).

The segmented-softmax reduction is exactly associative-safe here or its
reordering lands below greedy's decision threshold for all 64 steps;
either way, admission is output-transparent for this configuration.

## Performance: the collapse is recovered on TRITON itself

Same session, decode tok/s by 64-vs-8 differencing, MTP on:

| routing under speculation | 1K | 8K | 16K | 30K | 50K |
|---|---:|---:|---:|---:|---:|
| stock 0.28.0 (2D path)   | 88.67 | 52.51 | 42.23 | 29.75 | 14.10 |
| ported #45450 (3D path)  | 110.71 | 75.63 | 72.13 | 61.03 | 37.91 |
| ratio | 1.25x | 1.44x | 1.71x | 2.05x | 2.69x |

The 1K/8K/16K columns were added in a third session on the same stack
(logs `C1K/C8K/C16K/D1K/D8K/D16K.log`); every column's pair is measured
within one VM. The shape mirrors the ROCm ladder: the admission's gain
grows monotonically with depth, and even at 1K — where the 2D path is
still healthy — it is a 1.25x win, so there is no depth at which the
admission costs anything.

For scale: explicit FLASHINFER + MTP on the same stack measured 62.43
(30K, session A) and 41.35 (50K, session B) — the 3D admission brings
TRITON within 2.2% and 8.3% of FlashInfer without leaving the backend.
On ROCm, where FlashInfer does not exist and TRITON_ATTN is the default
for these models, #45450 is the only fix path; the same collapse
measures -71% at 32K there.

Same-session baselines were rerun rather than reused: the cross-VM
spread against the earlier sessions' 31.50/15.75 is 5.6%/10.5%, which
is why every ratio above is within one VM.

## Files

- `inject_45450.py` — the port, anchored string surgery with asserts;
  apply to a stock 0.28.0 install, revert with
  `pip install --force-reinstall --no-deps vllm==0.28.0`.
- `probe_ids.py` — the bit-exactness probe.
- `logs/` — full engine logs: A1/A2 (stock ids), B1/B2 (patched ids),
  C30/C50 (stock perf), D30/D50 (patched perf).
- Perf probe: [`../probe_matrix.py`](../probe_matrix.py) (`AUTO mtp
  30000|50000`).

## k-sweep (same day, second session)

The k=1 limit above is closed by a second session sweeping
`num_speculative_tokens` — #45450's own config is k=4. Correctness
holds at every k: for k=2 and k=4, all four generations (two per
engine, stock 2D and patched 3D) are identical token sequences, with
the 3D marker present exactly once in every patched leg and never in a
stock leg (`logs-ksweep/`). Performance at 30K:

| k | stock 2D | patched 3D | ratio |
|--:|---:|---:|---:|
| 1 | 29.75 | 61.03 | 2.05x |
| 2 | 29.02 | 56.11 | 1.93x |
| 4 | 27.54 | 53.40 | 1.94x |

Two readings. On the 2D path, deeper speculation only digs deeper
(29.75 -> 27.54): each extra draft token raises the per-step verify
cost on the crippled kernel while the emitted-token gain cannot keep
up. The admission ratio is stable near 2x across k. And k=1 is the
throughput optimum on both paths for this model and prompt — the
drafter's own per-step cost grows linearly in k while acceptance gains
saturate. Acceptance-rate counters were not captured in these runs.

Limits: single-run probes; one GPU; the window-relative segmentation
half of #45450 is untested here; bit-exactness is demonstrated for
this model/depth/config at k in {1, 2, 4}, not proven in general.
