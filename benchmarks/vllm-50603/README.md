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

1. **It is inverted on this hardware.** At `gqa_ratio` 1 and 2, exactly the
   excluded range, the CK kernel is **1.84x to 7.28x faster** than the Triton
   fallback it is being passed over for, and numerically equivalent to it. The
   floor is the 32/16 shape at 4K; the per-shape table below is the detail.
   (Corrected 2026-08-27: this line first said 2.06x, which was the minimum
   over `gqa_ratio` 1 alone and understated the floor.)
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

## Stage 3: what widening the gate is worth end to end

Stage 1 measured kernels. This measures a model, which is the number a PR has
to justify. `gemma-3-27b-it-w4a16`, TP=2, CUDA graphs on, decode tok/s by
64-vs-8 differencing, one fresh container per cell so the source edit cannot
leak between arms. The `widened` arm applies the one-line change at source
(`gqa_ratio >= 3` -> `>= 1`, gfx11 branch only).

| ctx | stock | widened | |
|---|---:|---:|---:|
| 1024 | 41.44 | 42.57 | **1.027x** |
| 8192 | 21.36 | 23.88 | **1.118x** |
| 32768 | 8.00 | 9.55 | **1.194x** |

The gain grows with context because of what moves. gemma-3-27b is 62 layers
with `sliding_window_pattern: 6`, so only the ~10 full-attention layers can
reach CK; the ~52 sliding layers fail the gate's `sliding_window == 0`
condition in **both** arms. At 1K the sliding layers dominate and the effect is
small. At 32K a full layer scans the whole context while a sliding layer scans
1024, so the layers that moved carry most of the KV traffic, and 19.4% is what
that is worth.

Routing is recorded from inside the TP workers, not inferred, in every cell
(`logs/stage3-routes/`):

| layer | stock | widened |
|---|---|---|
| full-attention, `window=0` | `use_custom=False` | `use_custom=True` |
| sliding, `window=1023` | `use_custom=False` | `use_custom=False` |

on both ranks, at all three depths. Those records also show the full-attention
layers arriving with `head_size=128, block_size=16`, native KV layout and a
power-of-two block, so **`gqa_ratio` is the only gate condition they fail**.

Two things this measurement is not. Output equality between arms is not the
correctness test: greedy decoding on this box is not reproducible across
processes for this model (`../gfx1100-greedy-nondeterminism.json`), so
comparing token ids would measure our own nondeterminism. The numerical case
is Stage 1's, against an fp32 reference. And the parent-process kernel counter
used in the first attempt was blind, because TP=2 runs attention in spawned
workers; `diag_route2.py` is the instrumentation that reaches them, and it is
why the routing table above exists.

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

## Stage 1d: runtime validation of vllm#53856 at its current head, on request

Stage 1c showed the fault on gfx1100 against the shipped kernel. The PR author
then asked for the other half: validation of **the fix**, at head
`f9bdfc886727c2590a8fbd0f82131a770383d05b`, on hardware they do not have.
Their nodes are MI350X (gfx950), where they had already run an `auto`-dtype NaN
matrix to 96 passes; what was missing was gfx11.

Building it needs a C++ rebuild, which turned out to be cheap here for three
reasons: the PR's two gfx11 hunks sit inside `#elif defined(__GFX11__)` (the
other four are gfx9, and the `mfma` names are misleading -- that block uses
wmma); the 0.27.1.dev5 image ships its own source at `/app/vllm` at the same
commit its `_rocm_C` was built from, so the ABI matches; and
`csrc/rocm/attention.cu` has not changed upstream since 2026-07-31, so the PR's
diff against its own base applies to that tree unchanged. Compiled for gfx1100
only, `build_53856.sh`.

Same probe as Stage 1c with FP16 added alongside BF16, run twice: once on the
image as shipped, once with the rebuilt `_rocm_C`. **Both arms are needed** --
a clean patched arm proves nothing unless the stock arm reproduces the fault.

64 rows, 32 per arm (2 dtypes x 2 gqa ratios x 4 poison modes x 2 lengths):

| | |
|---|---:|
| cells poisoned on the stock arm | **4** |
| of those, fixed by the PR | **4** |
| cells the PR newly breaks | **0** |
| gqa=4 cells that really ran the CK kernel | **32 / 32** |

The four are exactly the shape the fix is aimed at:

| dtype | gqa | poison | ctx | tail | stock | patched |
|---|---:|---|---:|---:|---|---|
| bf16 | 4 | v_only | 4090 | 6 | **NaN x4096** | ok, rel 3.1e-03 |
| bf16 | 4 | both | 4090 | 6 | **NaN x4096** | ok, rel 3.1e-03 |
| fp16 | 4 | v_only | 4090 | 6 | **NaN x4096** | ok, rel 5.5e-04 |
| fp16 | 4 | both | 4090 | 6 | **NaN x4096** | ok, rel 5.5e-04 |

Everything else is clean on both arms, and the pattern is the discriminating
part rather than the pass count:

- **K-only never poisons**, on either arm. The logit mask already covers K, so
  a fix that sanitised K as well would be treating a symptom that is not there.
- **only the straddling length**. 4096 fills the final tile exactly and is
  clean; 4090 leaves 6 unwritten slots and is not.
- **only gqa=4**. gqa=2 is excluded by the gfx11 gate, so both arms bypass CK
  entirely there. That is what makes this the path stock gfx11 actually takes
  rather than a forced one.
- **BF16 and FP16 behave identically**, reproducing and then fixing alike.

Post-fix relative error matches the un-poisoned baseline to the digit (3.1e-03
bf16, 5.5e-04 fp16), so the mask zeroes padding without touching live data.

Which kernel ran is counted by wrapping `ops.paged_attention_rocm`, not
inferred from timing: 32 of 32 gqa=4 cells went through CK on both arms.

Each cell also records a forced-CK column, which answers whether the fix is
tied to the shape the gate admits today. Forced past the gate, the stock arm
poisons **8** cells rather than 4: gqa=2 fails identically at 4090, 2048 NaN
outputs, both dtypes, V-only and K+V. All 8 are clean on the patched arm, so
the sanitisation is not specific to `gqa_ratio >= 3`. That bears on Stage 3,
which measures what widening that same gate is worth: widening it admits gqa=2
to this kernel, and this is the fault it would admit with it. The Triton
column is clean in all 64 cells on both arms, `kv_load_mask` doing its job.

Measured on vLLM `0.27.1.dev5+gf46a9dfe2`, ROCm 10.0, torch 2.12, 2x RX 7900 XT.
Files: `probe_53856_027.py`, `53856-027-{stock,patched}.jsonl`,
`build_53856.sh`, `53856-attn.diff`, `logs/v53856-*.log`.

Reported to the PR author on request, 2026-08-28:
[pull/53856#issuecomment-5451557090](https://github.com/vllm-project/vllm/pull/53856#issuecomment-5451557090).

## Stage 4: the same two questions, re-asked on 0.27

Everything above was measured on the 0.23.1 container. That is a premise, not a
constant, and this repository has already been wrong twice this week by reading
a 0.23.1 result as a statement about `main`. Both stages were therefore re-run
on `rocm/vllm:rocm10.0.0_..._vllm_0.27.0` (0.27.1.dev5+gf46a9dfe2, torch 2.12,
ROCm 10.0). The probes are byte-identical to the ones that produced the 0.23.1
rows, so the two sweeps compare directly.

**Kernel level, `probe_50603.py`, 30 cells twice.** No cell has CK slower than
Triton, at any ratio:

| gqa_ratio | gate | 0.23.1 | **0.27** |
|---:|:--|:--|:--|
| 1 | excluded | 2.06-4.83x | **2.03-4.09x** |
| 2 | excluded | 1.84-7.28x | **1.70-6.05x** |
| 3 | admitted | 2.40-7.23x | 2.20-6.05x |
| 4 | admitted | 2.35-7.40x | 2.35-6.04x |

The excluded band and the admitted band overlap on both versions, which is the
whole argument: `>= 3` does not separate a region where CK wins from one where
it loses. 0.27 is uniformly a little lower because the Triton fallback got
faster (32K, the 32/16 shape: 2.584 -> 2.231 ms), not because CK got
slower. Round to
round the spread is at most 5.8% and usually under 3%, and the numerical
columns are unchanged: 20 of 30 cells are bit-identical across versions and the
other ten move in the fourth significant digit, on the Triton side too.

**End to end, `probe_stage3.py`, run twice with the arms in opposite orders.**
Run A measured stock before widened at every context, so any drift over the
forty minutes would land on the widened arm each time; run B reverses that.

| ctx | A | B | pooled | 0.23.1 |
|---|---:|---:|---:|---:|
| 1024 | 1.026x | 1.023x | **1.024x** | 1.027x |
| 8192 | 1.065x | 1.103x | **1.084x** | 1.118x |
| 32768 | 1.168x | 1.167x | **1.167x** | 1.194x |

Same arm, same context, across the two passes: 0.0-0.7% apart in five of six
cells and 2.8% in the sixth. The 1024 gain is small but it is not noise; it
reproduces in both orders at a repeatability an order of magnitude tighter than
the effect.

**The routing proof needed its own run, and the reason is worth recording.**
Stage 3's recorder writes from inside the TP workers by rewriting
`chunked_prefill_paged_decode.py` before the engine starts. On 0.27 it produced
nothing. The diagnosis is in the line numbers: the worker logged the fallback
warning at `chunked_prefill_paged_decode.py:419`, which is where that warning
sits in the **pristine** file, while the file on disk in that same container had
it at 433 after the 13-line insert. **The workers ran unmodified bytecode even
though the file on disk was modified.** Either they inherited the parent's
already-imported module or they loaded a stale `.pyc`; the evidence here does
not separate those, and `route_027.py` closes both by editing before any vLLM
import and clearing `__pycache__`.

`rocm.py` is not affected by this, which is why the timing cells stand:
`probe_stage3.py` reloads that module explicitly after editing it, so the
patched module is the one the workers get. Re-run standalone, the routing comes
out the same shape as on 0.23.1, on both ranks:

| layer | stock | widened |
|---|---|---|
| full attention, `window=0` | `use_custom=False` | `use_custom=True` |
| sliding, `window=1023` | `use_custom=False` | `use_custom=False` |

Recorded as `(gqa_ratio, head_size, block_size, sliding_window, use_custom)`, so
those layers arrive at 128 and 16 and fail on `gqa_ratio` alone -- the same
conclusion Stage 3 reached on 0.23.1, now established on the version that
`main` actually resembles.

Files: `stage1-027-r{1,2}.jsonl`, `stage3-027.jsonl`, `stage3-027b.jsonl`,
`route-027.jsonl`, `route_027.py`, `run_stage1_027.sh`, `run_stage3_027{,b}.sh`,
`run_route_027.sh`, `logs/s1-027-*.log`, `logs/s3-027*-driver.log`,
`logs/r027-driver.log`, `logs/stage3-027-routes/`.

Proposed upstream on 2026-08-28 as
[vllm#54210](https://github.com/vllm-project/vllm/pull/54210), one line in the
gfx11 branch of `use_rocm_custom_paged_attention`. The PR body carries the
tables above; this directory carries the rows they are computed from.

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
- `probe_stage3.py` + `diag_route2.py` — Stage 3, end-to-end and the worker-side routing proof
- `route_027.py` — Stage 4's routing proof, edited in before any vLLM import
- `probe_50603_cuda.py` + `kernel_lifted.py` — Stage 2, same kernel text on CUDA
- `setup_50603.py` — the CUDA-side install, with the kernel hash assertion
- `stage*.jsonl` — every measured cell; `logs/` — the runs that produced them

Figures above are recomputed from the JSONL by
[`verify_doc_figures.py`](../analyze/verify_doc_figures.py).
