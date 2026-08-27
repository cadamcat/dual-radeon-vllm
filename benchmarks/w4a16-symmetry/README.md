# An asymmetric int4 checkpoint costs a flat 60 ms per decode step on gfx1100

[vllm#50264](https://github.com/vllm-project/vllm/issues/50264) settled why
`Qwen3.8-27B` *collapses* with context on this box, and
[vllm#45916](https://github.com/vllm-project/vllm/pull/45916) fixes that half.
It left the other half open in its own words: after the fix paged attention is
about 11% of the decode step, the remainder is dominated by
`triton_w4a16_gemm_kernel`, and "this issue's subject — the *slope* — is
resolved; the baseline is a separate question."

[`open-questions.md`](../../docs/open-questions.md) §9 is that separate
question, and it names the right suspect before ruling it out:

> The dominant item is `triton_w4a16_gemm_kernel` at **77 %** — but gemma-4-31B
> is also w4a16 and decodes at 43.2 tok/s, so "the quantised GEMM is slow" is
> not an answer either.

The alibi does not hold, because **"also w4a16" is not "also the same
kernel."** gemma-4-31B's checkpoint is symmetric and runs a native HIP kernel;
this one is asymmetric and runs Triton. Two different kernels behind one word
in a config file. Holding the model fixed and changing only the checkpoint is
worth **1.27x to 3.24x**, and the 77% observation was right the whole time.

## Why the native kernel is skipped

vLLM has had a native gfx1100 W4A16 kernel since
[vllm#41394](https://github.com/vllm-project/vllm/pull/41394) (merged
2026-05-29, so it is in this container), registered for the same
mixed-precision path compressed-tensors uses. Its own docstring says it is
"[r]egistered ahead of TritonW4A16LinearKernel for the ROCm-RDNA3 path", so
being preferred over Triton is the design intent.

One line excludes this checkpoint
(`vllm/model_executor/kernels/linear/mixed_precision/rdna3_w4a16.py`):

```python
SUPPORTED_QUANT_TYPES = [scalar_types.uint4b8]
```

`uint4b8` is symmetric int4. `compressed_tensors_wNa16.py` picks the weight
type from the checkpoint's `symmetric` flag — `WNA16_ZP_SUPPORTED_TYPES_MAP`
(`uint4`) when false, `WNA16_SUPPORTED_TYPES_MAP` (`uint4b8`) when true — and
sets `zero_points = not symmetric`. An asymmetric checkpoint therefore arrives
as `uint4` and fails the type check. Asked directly, the registry says so:

    no        RDNA3W4A16LinearKernel   Quant type (uint4) not supported by
                                       RDNA3 W4A16 kernel; supported: [uint4b8]
    SELECTED  TritonW4A16LinearKernel

**Nothing warns.** The model loads, runs, and is slow.

## The A/B

The headline is not the ratio. Decode time per token is additive, so if the
cause is a per-step GEMM it has to show up as a **context-independent number of
milliseconds**, with the ratio falling only because the attention term grows
underneath both arms. That is a prediction the ratio alone cannot make:

| ctx | asym ms/token | sym ms/token | **difference** | ratio |
|---:|---:|---:|---:|---:|
| 1 024 | 87.02 | 26.85 | **60.17 ms** | 3.241x |
| 8 192 | 131.91 | 73.88 | **58.03 ms** | 1.786x |
| 32 768 | 298.47 | 235.68 | **62.79 ms** | 1.266x |

Over a 32x change in context the ratio falls by **60.9%** while the penalty
stays inside **58.03 to 62.79 ms, a spread of 8.2%**. That is what separates
"the asymmetric checkpoint costs a fixed amount per decode step" from the much
weaker "the two arms differ somehow". The 8.2% is not nothing, and these are
single measurements; the one repeat this experiment has (below) moves the sym
arm by up to 1.31%, so the spread is larger than the noise floor but the same
order as it.

Same numbers as throughput, which is how they were measured:

| ctx | asym | sym | |
|---:|---:|---:|---:|
| 1 024 | 11.49 | 37.24 | **3.241x** |
| 8 192 | 7.58 | 13.54 | **1.786x** |
| 32 768 | 3.35 | 4.24 | **1.266x** |

Two checkpoints of the same model — `Qwen3_5ForConditionalGeneration`,
compressed-tensors pack-quantized on both sides — same two cards, same image,
one fresh container per cell:

| arm | checkpoint | symmetric | group_size |
|---|---|---|---|
| asym | `cyankiwi/Qwen3.8-27B-AWQ-INT4` | false | 32 |
| sym | `RedHatAI/Qwen3.8-27B-INT4` | true | 128 |

Decode tok/s by 64-vs-8 differencing after a warm-up generation, TP=2, CUDA
graphs on, `max_num_seqs=128` pinned on **both** arms, speculative decoding off
on both although both checkpoints ship an MTP head. The warm-up earns its
place: every cell JIT-compiles exactly four Triton kernels and all four land
inside it, so neither timed call carries a compile, on either arm.

Which kernel each arm actually got is recorded from inside the TP workers, by
patching `choose_mp_linear_kernel` on disk rather than monkeypatching it in the
parent — under TP=2 the layers are built in spawned workers, where a
parent-process patch sees nothing (`logs/kernels-<arm>-<ctx>.txt`):

| arm | both ranks, all three cells |
|---|---|
| asym | `RDNA3W4A16LinearKernel` rejected (`uint4`), `TritonW4A16LinearKernel` selected |
| sym | `RDNA3W4A16LinearKernel` selected (`uint4b8`), Triton never appears |

The symmetric checkpoint was fetched through a mirror rather than from
huggingface.co, so its three LFS files are checked against the ETags the Hub
itself advertised: **3/3 match**, repo commit `2fb0debc`
(`verify_ckpt_sha.py`, `ckpt-sha256-sym.json`).

## The other two differences, and why neither carries the result

**Group size**, 32 against 128. No checkpoint exists that separates it from
symmetry, so this A/B cannot. Three things retire it anyway:

- *The rule itself.* `can_implement` requires only that the group size be
  positive and divide K. There is no group-size-128 condition in the source.
- *All four corners.* Asked for symmetry x group size, the registry follows
  the `symmetric` flag and ignores the group size — `sym g32` selects the
  native kernel, `asym g128` does not (`w4a16-selection-2x2.json`).
- *This repository's own campaign*, which is the argument that needs no new
  measurement at all. Reading each checkpoint's `quantization_config`, the
  **fastest model measured here** (`gemma-4-26B-A4B`, 107.7 tok/s) is
  **group_size 32**, the same as the asymmetric checkpoint. Among all the
  group-32 checkpoints on this box both kernels appear, native and Triton,
  split strictly by symmetry (`w4a16-campaign-selection.json`).

Group size is not neutral in general — it changes accuracy and how much scale
and zero-point traffic each GEMM issues — so it stays disclosed in the speed
numbers. It is simply not what decides which kernel runs.

**Which layers are quantized.** The two `ignore` lists are not identical, so
the quantized linears were counted from each `model.safetensors.index.json`
rather than assumed: **399 against 400**, differing by exactly
`model.language_model.layers.0.linear_attn.out_proj`, which only the symmetric
checkpoint quantizes. One layer in 400, and it gives the symmetric arm one
*more* W4A16 GEMM, so it cannot be what makes that arm faster
(`ckpt-layer-census.json`).

## The one repeat measurement

An earlier attempt at this A/B lost all three asymmetric cells to a default:
`max_num_seqs=256` exceeds the Mamba cache blocks that checkpoint leaves free,
and engine init aborts during graph capture
(`logs/ab-asym-32768-firstattempt-maxnumseqs256.log`, verbatim: *"max_num_seqs
(256) exceeds available Mamba cache blocks (214)"*). Its symmetric cells ran,
and they are kept because they are the only repeat this experiment has — three
cells, an independent container start, and `max_num_seqs` 256 against 128:

| ctx | first attempt | this run | |
|---:|---:|---:|---:|
| 1 024 | 37.224 | 37.240 | 0.04% |
| 8 192 | 13.639 | 13.536 | 0.76% |
| 32 768 | 4.298 | 4.243 | 1.31% |

So the pinned `max_num_seqs` rescued the asymmetric arm without moving the
symmetric one, and the sym arm reproduces to 1.31% across container starts.

## What this means for the campaign

The 2026-08-24 campaign contains a natural control that was not read as one.
Two 27B models, same box, same day: `gemma-3-27b-it-w4a16` at 44.8 tok/s and
`Qwen3.8-27B-AWQ` at 12.3, a factor of **3.64x** with parameter count held
constant. That gap was attributed to architecture, hybrid-SSM against dense.
Symmetry is the other difference between them, and it is worth up to 3.24x on
its own.

Every quantized checkpoint in the campaign is symmetric except the two Qwen3.x
AWQ ones — which are the two the documents single out as slow. The architecture
comparison and the confound coincide exactly where the conclusion was drawn.
The affected claims are corrected in place and dated, and
[`open-questions.md`](../../docs/open-questions.md) §9 is marked answered.

What is **not** affected: every slope conclusion, including all of
[`hybrid-decode-on-rdna.md`](../../docs/hybrid-decode-on-rdna.md), which is
about the collapse rather than the baseline; and the ordering among the four
models that are all on their best kernel path, 26B MoE > 8B dense > 12B > 31B.
Only the 27B's place in that ordering was a packaging artefact.

One practical warning. `gemma-4-26B-A4B-AWQ` is named AWQ and is
`symmetric: true`. The name does not tell you. The field that decides is
`quantization_config.config_groups.*.weights.symmetric`.

## What this does not settle

**Accuracy is not measured here.** These are two independent quantizations by
two publishers at different group sizes, and this measures speed only. "Prefer
a symmetric checkpoint on RDNA3" is a throughput statement; whether these two
are equally good models is a separate question this directory does not answer.

**Do not read these cells against the campaign table.** The README's
12.3 / 11.7 / 10.7 were measured in the patched container. Every cell here runs
a fresh container off the stock image, which carries none of our patches, so
both arms show the unpatched long-context regression #50264 is about. Only asym
against sym, within this run, is meaningful.

Kernel selection is *measured* for these two checkpoints only. For the rest of
the campaign it is the registry's verdict on each checkpoint's real
quantization parameters, not a worker-side record of that model running.

One run per cell, batch of 1, one model, one machine.

## What follows

For anyone on RDNA3: a symmetric w4a16 checkpoint selects a native kernel and
an asymmetric one does not, nothing in the logs says so, and the model name is
not a reliable guide. Symmetric builds of this model exist from several
publishers.

For vLLM, the general fix looks smaller than "write a kernel". The RDNA3 kernel
does not avoid zero points, it **requires** them: `apply` asserts the
zero-point tensor is present, and the symmetric path works by synthesizing a
constant `qzeros` of `weight_type.bias - 1` on the host. So
`ops.gptq_gemm_rdna3` is already a general asymmetric GPTQ dequant, and
compressed-tensors already registers a real `weight_zero_point` for asymmetric
checkpoints under the same config field the kernel reads. What stands between
them is the type gate and the zero-point *encoding* convention — the GPTQv1
"+1 quirk" the source documents — rather than new kernel math.

Whether the two conventions agree is **not tested here**, and that is the next
experiment: admit `uint4`, run the asymmetric checkpoint, and check coherence
as well as speed. Incoherent output would mean the encodings differ and the
work is real after all. It would also close the last gap in this one, by
removing the publisher as a variable: the same checkpoint, two kernels.

## Files

- `probe_w4a16_ab.py` + `run_ab.sh` — the A/B, one fresh container per cell
- `probe_w4a16.py` — the registry, four real checkpoints
- `probe_w4a16_2x2.py` — the registry, all four corners plus every campaign
  checkpoint
- `census_ckpt.py` — the quantized-layer census
- `dl_sym.py`, `verify_ckpt_sha.py` — how the symmetric checkpoint was fetched,
  and its sha256 against the Hub's own ETags
- `w4a16-ab.jsonl` — the six measured cells
- `w4a16-selection*.json`, `ckpt-layer-census.json`, `ckpt-sha256-sym.json` —
  the derived records
- `logs/` — the runs that produced them, including the per-cell worker-side
  kernel records

Figures above are recomputed from these files by
[`verify_doc_figures.py`](../analyze/verify_doc_figures.py).
