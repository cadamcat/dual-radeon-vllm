# One MI300X: a declaring kernel is not the one that runs there either, the attention ranking is right where its kernel applies, and W4A16 has no native kernel on CDNA — 2026-09-17

One MI300X (gfx942, 192 GiB, TP=1) rented for two and a half hours, in the
container the Radeon pair's 2026-08-28 and 08-29 rows were measured in:
`rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0`,
`sha256:b8a082f346d0…`, vLLM `0.27.1.dev5+gf46a9dfe2.d20260827`, torch
2.12.0+rocm10.0.0, HIP 7.15.26333. The version string is identical to those
rows', so what differs from the pair here is the architecture, not the stack.
`PROVENANCE.json` carries the rest; `results.jsonl` is every measurement.

The card is a virtual function. `hipgate3` compiled for gfx942 launches both its
kernels and `[device] HOSTCALL_MARKER reached the host` prints, so the platform
supplies hostcall and the refusal this repository is about cannot be reproduced
here. What can be measured here is the declaration side.

## Declared is not selected on CDNA either, with the families exchanged

`vllm/_rocm_C`'s gfx942 images hold **3 040 kernels, of which 628 declare**
`hidden_hostcall_buffer` — all of them int4 skinny-GEMM
(`wvSplitK_int4_hf_`, `wvSplitK_int4_hf_sml_`, `wvSplitKrc_`).
**`paged_attention` has 1 920 instantiations and none of them declares.**

On gfx1100 the same library declares 348, of which 256 are
`paged_attention_ll4mi_QKV_mfma4_kernel`: CDNA template instantiations whose
gfx11 body is `assert(false)`, and the assert is what pulls the declaration in.
Compiled for gfx942 that family has a real body and declares nothing. **The
declaration follows the body a family compiles to on an architecture, not the
family.**

`launchtrace.so` names every kernel a process launches. Two models, each a
probe that drives the paged-attention op directly and a serve answering two
concurrent 4 000-token requests:

| serve | distinct kernels | launches | declaring launched |
|---|--:|--:|--:|
| Qwen3-8B, bf16 | 85 | 46 617 | **0** |
| Muse-Glimmer-30B-INT4 | 144 | 146 710 | **0** |

Qwen3-8B runs `paged_attention_ll4mi_QKV_mfma4_kernel<__hip_bfloat16, …, 16,
128, 256, false, 4>`; the int4 model runs the `mfma16` instantiation and the
bf16 `wvSplitK_hf_sml_` GEMMs. None of the 628 declaring kernels is dispatched
in either. Scope: two models and two shapes say that these do not dispatch one,
not that nothing does.

## gemma-4 fails to load here by the same two frames

Serving `gemma-4-12B-it-qat-w4a16-ct` on this image stops at configuration with
`head_dim = getattr(self.hf_text_config, "head_dim", 0)` in vLLM's Gemma 4
converter and `AmbiguousGlobalPerLayerAttributeError` from transformers'
per-layer accessor — the two frames of the gfx1100 traceback, on another
architecture, card and host (`logs/gemma4-load-failure-gfx942.log`). The fix is
upstream in vllm#49797, released in 0.28.0, which no ROCm container ships yet.
All three gemma-4 configurations fail this way, twelve seconds each, before a
weight is read.

## The ladder: bf16 is an H200's speed, W4A16 is an order of magnitude off

Sixteen rungs, two rounds, batch 1, same ladder as every other machine here.

| ctx | Qwen3-8B here | H200 | B300 | H100 ×2 |
|---:|--:|--:|--:|--:|
| 500 | **199.25** | 189.0 | 250.2 | 223.6 |
| 8 000 | **186.55** | 179.1 | 235.1 | 212.8 |
| 32 000 | **154.41** | 155.7 | 211.1 | 188.3 |

The two quantised checkpoints decode at 10.26 (Qwen3.8-27B-AWQ-INT4 at 2 000)
and 8.35 (Muse-Glimmer-30B-INT4 at 8 000) tokens a second, against 105–139 and
82–118 on the rented CUDA cards. The cause is in vLLM's kernel chooser.
`model_executor/kernels/linear/__init__.py` lists candidates per platform:

    CUDA: Cutlass W4A8, Machete, AllSpark, Marlin, Conch, Exllama, Triton, Humming
    ROCm: RDNA3W4A16, RDNAHybridW4A16, Triton, Conch, Exllama

Both ROCm natives refuse anything but gfx1100 —
`if not on_gfx1100(): return False, "RDNA3 W4A16 kernel requires gfx1100"` —
so on gfx942 the chooser falls through to `TritonW4A16LinearKernel`, which both
serve logs report. There is no CDNA entry in the ROCm list, and the image's own
int4 GEMMs are not reachable through it. AMD's AITER kernels do not change this:

| rung | `VLLM_ROCM_USE_AITER=1` | `=0` |
|---:|--:|--:|
| 500 | 10.66 | 10.58 |
| 8 000 | 9.19 | 9.14 |

Both arms logged `Using TritonW4A16LinearKernel`. Muse-Glimmer additionally has
no vLLM implementation in this version and runs through the Transformers
fallback, so its rows are not a kernel comparison at all; its ladder was stopped
at 64 000 once that was read out of the serve log.

That path also puts several tokens in one stream chunk, and the runner counts
chunks. At the 8 000 rung it counted 511 of the 512 tokens the request asked
for, which is why the rate above is a token rate; the shallow rungs counted as
few as 7 and report 0.098 tokens a second. Dividing the 512 by each row's own
`wall_s` minus its `ttft` gives 8.29 to 8.38 across the whole ladder, and the
engine's logger reports a median 8.30 while it ran, so this model decodes at one
speed from 500 to 64 000 and the shallow rows measure the stream rather than the
card. They are not decode measurements: `decode.jsonl` takes this model's
prefill and not its decode, and `build_decode.py` holds the reason.

## The attention ranking is right where the kernel applies

The 2026-09-07 campaign measured, on gfx1100 with a head_size 256 model, that
TRITON_ATTN decodes 1.48× faster than ROCM_ATTN at 128 000 tokens because the
custom kernel the ranking exists for is not instantiated at that head size.
gfx942 is the architecture that kernel was written for. Same harness, three arms
in one sitting, only `--attention-backend` differing, Qwen3-8B (head_size 128):

| ctx | ROCM_ATTN | TRITON_ATTN | decode ratio | prefill ratio |
|---:|--:|--:|--:|--:|
| 8 000 | 186.26 | 164.08 | 1.135 | 1.412 |
| 16 000 | 174.42 | 138.89 | 1.256 | 1.755 |
| 32 000 | **154.08** | **105.55** | **1.460** | **2.094** |

ROCM_ATTN wins on both halves and the gap widens monotonically with context.
A third arm repeated ROCM_ATTN after TRITON_ATTN at three rungs: −0.38 %,
−0.22 % and +0.58 % against the first arm, so the 46 % is not session drift.

Two architectures, two head sizes, one rule: where the custom kernel is
instantiated the ranking picks the faster backend, and where it is not it picks
the slower one. The two ratios are nearly symmetric, 1.48 there against 1.46
here.

## Batch, and what a cell's throughput means

Three depths, batch 1/2/4/8, five repeats each, one serve per model at
`--max-num-seqs 16`, so batch 1 is measured in the same session as 2/4/8.
Aggregate decode throughput, prefill excluded by subtracting each cell's slowest
time to first token:

| ctx | batch 1 | 2 | 4 | 8 | 8 against 1 |
|---:|--:|--:|--:|--:|--:|
| 500 | 199.8 | 340.4 | 590.1 | 1 255.2 | 6.28× |
| 8 000 | 188.2 | 307.2 | 504.2 | 914.6 | 4.86× |
| 32 000 | 156.1 | 233.5 | 343.7 | 507.9 | 3.25× |

Batch scales, and the scaling decays with depth as each step reads more KV. The
`decode_tps_cell` field on a `kind: batch` row divides by the whole wall clock
including prefill, so at 32 000 it reads 62.4 at batch 1 and 51.6 at batch 8 —
the same rows, a different question. Every row carries `wall_s`, `ttft_median`
and `ttft_max`, so either reading recomputes from it. The KV pool held
1 125 072 tokens (27.47× concurrency at 40 960) and no request was preempted.

Greedy decode at 500 tokens of context, eight repeats: one distinct completion,
so it is deterministic here for this bf16 model. The quantised models were not
repeated this way.

## What this does not buy

One card: no TP=2, no collective, nothing about what a second card is worth. The
platform has AtomicOps, so the refusal itself is not reproducible here and only
the declaration side is measured. The PCIe telemetry counters the Radeon rows
carry are absent on this virtual function, and `power_w` reads 0 through sysfs
while `rocm-smi` reports 391 W, so no energy figure is derived from these rows.
Muse-Glimmer's numbers describe a Transformers fallback, not a kernel.

## Files

`results.jsonl` (583 rows), `PROVENANCE.json`, `PROGRESS.txt`;
`rocm_C-gfx942-kernels.tsv` and `scan-gfx942.jsonl` (the device-code scan),
`trace-join-*.json` with `pa-probe-*.jsonl` (the launch traces),
`hipgate3.out`; `logs/` holds every serve log, the two traced serves, the probe
traces and the gemma-4 failure. `runner.py` is the ladder; `batch_greedy.py`,
`ab_attn.py` and `aiter_probe.py` drive the other three passes through it;
`00-preflight.sh` … `90-harvest.sh` are the steps in order.
