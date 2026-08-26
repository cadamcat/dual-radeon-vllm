# CUDA A100 annex — the gemma-4 MTP collapse is the kernel, not the GPUs

One day of measurements on a rented A100 80G (Colab), run as the control
experiment for [speculative-decoding-on-rdna.md](../../docs/speculative-decoding-on-rdna.md):
if the MTP collapse we root-caused on RDNA lives in the Triton
unified-attention kernel's 3D-to-2D drop, it must reproduce on any CUDA
machine that routes gemma-4 onto that kernel — and must vanish on a
backend without the guard. Both held. These numbers back our comments on
[vllm#52049](https://github.com/vllm-project/vllm/issues/52049),
[vllm#47547](https://github.com/vllm-project/vllm/pull/47547) and
[vllm#50891](https://github.com/vllm-project/vllm/issues/50891).

## The matrix

A100-SXM4-80GB, vLLM 0.28.0, flashinfer 0.6.16.post3,
`gemma-4-31B-it-qat-w4a16-ct` + the official MTP assistant (k=1), batch 1,
text-only serving, decode tok/s by 64-vs-8 differencing:

| routing                                   | 30K MTP | 30K no spec | 50K MTP | 50K no spec |
|-------------------------------------------|--------:|------------:|--------:|------------:|
| TRITON_ATTN (today's forced default)      |   31.50 |       43.85 |   15.75 |       40.47 |
| FLASHINFER, explicit                      |   62.43 |       47.42 |   41.35 |       45.27 |
| auto selector with vllm#47547 (FA2+FI)    |   66.09 |       48.95 |       — |           — |

Readings, each recomputed by `../analyze/verify_doc_figures.py`:

- On the default routing, MTP is a net **loss** that deepens with
  context: **-28.2%** at 30K, **-61.1%** at 50K. Turning MTP off at 50K
  is **2.57x** faster — vllm#52049's "3x or higher at 50K+" seen from
  its lower edge. The ROCm boxes measured -70.8% at 32K: same curve,
  harder-hit hardware.
- On a healthy backend the same MTP is **+35.0%** at 30K. FlashInfer
  with MTP at 50K runs **2.63x** the default. The residual -8.7% vs
  MTP-off at 50K is drafter economics, not the pathological drop.
- The mixed assignment vllm#47547's selector produces (FA2 on the
  sliding groups, FlashInfer on the 512-head full-attention group) is
  the fastest configuration measured.

![gemma-4 MTP backend matrix](../../docs/assets/gemma4-mtp-backend-matrix-a100.svg)

## Two findings beyond the matrix

1. **FlashInfer is walled off from multimodal-enabled gemma-4.** The
   checkpoint registers image, video and audio; unless every one of them
   is `limit_mm_per_prompt` 0, `use_mm_prefix=True` and backend
   validation rejects FlashInfer even when selected explicitly
   (`'partial multimodal token full attention not supported'`). The
   text-only door logs `running in text-only mode`. vllm#46558 is
   adding the missing mm-prefix support.
2. **A live case of vllm#50891.** The AOT compile cache key ignores
   `limit_mm_per_prompt`, and on this model flipping it also flips what
   the selector picks. Two materially different configurations keyed the
   same hash, the stale artifact loaded, and engine init died at dynamo
   `call_size` (`AttributeError: 'NoneType' object has no attribute
   'size'`). A fresh VM reproduced the identical hashes and, on a cold
   cache, both configurations ran — deterministic collision, crash
   caused by the artifact alone. Hashes and logs under `logs/`.

## Pitfalls this bench paid for

- The 40G A100 cannot hold 31B + MTP at 33000 `max_model_len` (KV needs
  15.81 GiB). Use the 80G runtime.
- Colab preinstalls a cu12.8 `torchaudio` that clashes with the cu13
  torch vLLM brings; uninstall it. The pip resolver's ERROR wall during
  install is preinstall noise.
- Zero **all three** modalities for text-only serving; `image=0` alone
  leaves multimodal enabled (video, audio) and changes nothing.
- Clear `~/.cache/vllm/torch_compile_cache` when switching backend or
  modality configuration (vllm#50891, finding 2).
- T4 runtimes are sm75, below FlashInfer's floor; A100 only.
- Capture probe output unconditionally and write full logs to disk. A
  filtered cell that prints nothing on an unexpected code path cost this
  bench one blind session.

## Files

- `gemma4-mtp-backend-matrix.json` — every number above, machine-readable,
  with method and crash findings.
- `probe_matrix.py` — the measurement cell: `<backend> <mtp|nospec> <ctx>`.
- `gemma4_backend_matrix_a100_v3.ipynb` — the Colab notebook that ran
  session B end to end (install, config state machine, cold-cache runner,
  all legs, summary). Sessions' executed notebooks stay out of the repo;
  their outputs are preserved as text below.
- `logs/session-b/` — full engine logs of the seven session-B legs.
- `logs/session-a/` — session A survives as notebook-output extracts
  (provenance header in each file): the 30K TRITON pair, the explicit
  FlashInfer MTP leg, both stale-cache crashes with the ERROR tails, and
  the cache-fingerprint table.

Chart: `../analyze/gen_cuda_annex_chart.py` regenerates the SVG
byte-identically from the JSON. Figure checks live in
`../analyze/verify_doc_figures.py`.
