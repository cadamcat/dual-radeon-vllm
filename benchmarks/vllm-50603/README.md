# The gfx11 `gqa_ratio >= 3` gate costs 2-7x, and buys nothing

[vllm#50603](https://github.com/vllm-project/vllm/issues/50603) reports garbled
long-context output on gfx1100 from the Triton paged-attention fallback, and
traces the fallback correctly: `use_rocm_custom_paged_attention` in
`vllm/platforms/rocm.py` requires `gqa_ratio >= 3` on gfx11, their model has 2,
so the ROCm CK custom kernel is gated out.

The PR that introduced that bound ([vllm#17004](https://github.com/vllm-project/vllm/pull/17004),
merged 2025-05-21) says why:

> It supports gqa_ratio up to 16, and shows performance gains over the existing
> kernel when gqa_ratio is 3 or higher. Therefore, it is enabled for gqa_ratio
> values between 3 and 16.

So the bound is a **performance heuristic, not a capability limit**, and the
CDNA branch of the same function runs the same kernel at `gqa_ratio >= 1`.
Both are still true on `main` today.

This directory tests the heuristic. Two findings:

1. **It is inverted on this hardware.** At `gqa_ratio` 1 and 2 — exactly the
   excluded range — the CK kernel is **2.06x to 7.28x faster** than the Triton
   fallback it is being passed over for, and numerically equivalent to it.
2. **Neither path loses accuracy with context length.** Across 90 cells on two
   architectures, from 1K to 32K, relative error stays in a flat band
   (2.12e-3 to 4.72e-3, median 3.02e-3). A 16x context increase moves the
   median by **1.06x**. So "the Triton kernel degrades at long context" is not
   supported here, and the report's symptom B needs a different explanation.

## Method

Both paths are driven through vLLM's own `chunked_prefill_paged_decode`
dispatch with only `use_rocm_custom_paged_attention` forced, so nothing about
argument marshalling differs between arms. Pure decode: one query token per
sequence over a fully cached context, which is the shape the report describes
(warmed, deterministic, wrong). Reference is fp32, computed from the
pre-paging KV rather than from the cache, so a paging bug cannot cancel itself
out. Scored as relative L-inf against the reference's own scale, which is what
makes lengths comparable.

| | Stage 1 + 1b | Stage 2 |
|---|---|---|
| machine | RX 7900 XT (gfx1100) | A100-SXM4-40GB |
| vLLM | 0.23.1.dev1+g9ddef7117 | 0.23.0 |
| kernel source md5 | `854daa8f5d878449266519a9206db677` | same, asserted |
| paths | Triton 2D **and** CK custom | Triton 2D only |

The CUDA side compiles the same Triton text: `kernel_paged_attention_2d` is
lifted verbatim into `kernel_lifted.py` (vLLM 0.23.0's compiled `_C` wants
libcudart.so.13 on that host, so the module cannot simply be imported), and the
launch mirrors the dispatch's argument list. Inputs are same-distribution, not
bit-identical, across the two machines.

`SHAPES` covers `gqa_ratio` 1, 2, 3, 4 at head_size 128 / block_size 16, plus
`32/16`, which is `gemma-3-27b-it`'s real shape and satisfies every condition of
the gfx11 gate except `gqa_ratio >= 3`.

## Speed: the excluded ratios are where CK helps most

Triton time / CK time, so above 1.0 means the gated-out kernel is faster:

| shape | gqa | gated in? | 1K | 2K | 4K | 8K | 16K | 32K |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 8/8 | 1 | no | 2.06x | 2.28x | 2.87x | 2.51x | 4.11x | **4.83x** |
| 8/4 | 2 | no | 2.31x | 2.42x | 3.21x | 3.86x | 6.53x | **7.28x** |
| 32/16 | 2 | no | 2.14x | 2.24x | 1.84x | 1.88x | 2.71x | **2.86x** |
| 12/4 | 3 | yes | 2.40x | 2.57x | 3.20x | 3.64x | 6.36x | 7.23x |
| 16/4 | 4 | yes | 2.40x | 2.35x | 3.20x | 3.71x | 6.42x | 7.40x |

The gated-in rows are the control: they show the advantage the gate was written
to capture. The gated-out rows show the same advantage, at the same magnitude,
being declined.

## Accuracy: flat in context, on both architectures

Median relative L-inf over all shapes at each length:

| ctx | gfx1100 Triton | gfx1100 CK | A100 Triton |
|---|---|---|---|
| 1024 | 2.62e-03 | 3.56e-03 | 2.86e-03 |
| 2048 | 2.66e-03 | 3.67e-03 | 2.86e-03 |
| 4096 | 2.94e-03 | 3.38e-03 | 2.71e-03 |
| 8192 | 2.77e-03 | 3.78e-03 | 3.25e-03 |
| 16384 | 2.89e-03 | 3.40e-03 | 3.10e-03 |
| 32768 | 2.80e-03 | 3.42e-03 | 2.64e-03 |

CK sits consistently a little above Triton, which is a different accumulation
order, not a defect: both are in the band bf16 gives you for a sum this long,
and the A100 column running the identical Triton source lands in the same band.
That last column is what makes this a bounded negative rather than an absence
of evidence.

## The positive control: the harness does detect corruption

A clean sweep is worth nothing if the measurement cannot see a real fault. Two
things Stage 1 did not exercise: every length was a multiple of the block size,
so the final KV tile never straddled `seq_len`; and the cache was zero-filled,
so slots past `seq_len` held 0 rather than whatever the last sequence left.

vLLM 0.25.0 added a per-token mask on the final tile's K/V loads for exactly
this, with the reason in the source:

> Slots >= seq_len are unwritten KV cache that may hold NaN/garbage; they are
> score-masked below, but `0 * NaN = NaN` would still poison the output

This container runs 0.23.1.dev, which predates that mask. `probe_50603b.py`
fills the cache with NaN and uses lengths that straddle the final tile:

| fill | ctx | block-aligned | tail slots | Triton | CK |
|---|---:|---|---:|---|---|
| nan | 1024 | yes | 0 | finite | finite |
| nan | 4096 | yes | 0 | finite | finite |
| nan | 1000 | no | 8 | **all NaN** | **all NaN** |
| nan | 1015 | no | 9 | **all NaN** | **all NaN** |
| nan | 4095 | no | 1 | **all NaN** | **all NaN** |
| nan | 4090 | no | 6 | **all NaN** | **all NaN** |
| garbage (1e4) | all | either | any | finite, correct | finite, correct |

So the harness sees corruption when corruption is there, and Stage 1's flat
band means something. Three side notes worth keeping:

- only NaN propagates; finite garbage at 1e4 is handled correctly by the score
  mask alone, which is why this is subtle;
- **the CK path is poisoned identically** on this version. 0.25.0's fix is in
  the Triton kernel. Whether the CK path was ever fixed is not something this
  measurement can answer, and is worth asking upstream;
- this is **not** the reported bug. vllm#50603 is on 0.25.1, which has the mask.

## Stage 1c: gfx11 runtime evidence for vllm#53856

[vllm#53856](https://github.com/vllm-project/vllm/pull/53856) (open, assigned,
no human comments as of 2026-08-27) fixes the CK side of exactly this: the
kernel masks logits past `seq_len` but its vectorized **V**-cache load still
consumes the whole final block. Its test plan says gfx11 and gfx12 received
**compile validation only, because those devices were not available**.

Stage 1b above demonstrated the fault at `gqa_ratio=2`, which the gfx11 gate
excludes, so it showed a path stock gfx11 never takes. Stage 1c fixes that and
two other holes: it runs at `gqa_ratio=4`, which the gate admits; it proves the
CK kernel ran by wrapping `ops.paged_attention_rocm` rather than inferring it
from timing; and it poisons K and V separately, because #53856 sanitises V
only.

`gqa_ratio=4` (16/4), gate as shipped admits CK, `used_ck_kernel` true in 8/8:

| poison | ctx 4096 (tail 0) | ctx 4090 (tail 6) |
|---|---|---|
| none | ok, rel 3.0e-03 | ok, rel 3.1e-03 |
| K only | ok, rel 3.0e-03 | **ok, rel 3.1e-03** |
| V only | ok, rel 3.0e-03 | **all 4096 outputs NaN** |
| both | ok, rel 3.0e-03 | all 4096 outputs NaN |

Three things that make this the shape #53856 describes: **V alone is
sufficient**; **K alone does nothing**, because the logit mask already covers
it; and the boundary is exact, only the length whose final tile straddles
`seq_len`. The `gqa_ratio=2` rows behave identically and are recorded as
force-enabled.

One gfx11-specific note. #53856 is framed around FP8 NaN, but the gfx11 branch
of `use_rocm_custom_paged_attention` requires `kv_cache_dtype == "auto"`, so an
FP8 KV cache never reaches the CK kernel on gfx11 at all. The poisoning above
is bf16. The fix matters on gfx11, for a reason the PR's description does not
cover.

How a NaN gets into padding is not asserted here. `allocate_kv_cache`
(`vllm/v1/worker/utils.py`) zero-fills the backing allocation, so it is not
fresh-allocation garbage; #53856 attributes it to sleep mode and allocator
behaviour. This probe injects it deliberately and only measures what the kernel
then does.

## What this does not settle

Symptom B is not explained. We do not have the reporter's model
(`tencent/HunyuanOCR`, a VLM) or their 4x W7900D, and this measures the kernel,
not their output. What it removes is one hypothesis: that the Triton 2D kernel
loses accuracy as context grows. Sliding window, fp8 KV, prefill shapes and
TP>1 are all untested here.

## Files

- `probe_50603.py` — Stage 1, both ROCm paths against fp32
- `probe_50603b.py` — Stage 1b, the NaN-tail positive control
- `probe_53856.py` — Stage 1c, gfx11 runtime evidence for vllm#53856
- `probe_50603_cuda.py` + `kernel_lifted.py` — Stage 2, same kernel text on CUDA
- `setup_50603.py` — the CUDA-side install, with the kernel hash assertion
- `stage*.jsonl` — every measured cell; `logs/` — the runs that produced them

Figures above are recomputed from the JSONL by
[`verify_doc_figures.py`](../analyze/verify_doc_figures.py).
