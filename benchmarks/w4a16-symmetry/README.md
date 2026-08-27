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

## Forcing the native kernel onto the asymmetric checkpoint

The A/B compares checkpoints from two publishers, so "symmetry is what matters"
still rests on the two arms being alike in every other way. This removes the
publisher entirely: **one checkpoint, two kernels**, differing only in whether
`uint4` is admitted to `SUPPORTED_QUANT_TYPES`.

    -    SUPPORTED_QUANT_TYPES = [scalar_types.uint4b8]
    +    SUPPORTED_QUANT_TYPES = [scalar_types.uint4b8, scalar_types.uint4]

Applied on disk before the engine starts, since TP=2 builds layers in spawned
workers that re-import the module. Nothing else was changed.

**The type gate is indeed the only thing standing in the way of selection.**
Both ranks record `('RDNA3W4A16LinearKernel', 'uint4', 32, True, True)` — the
trailing `True` is `can_implement` passing — and `TritonW4A16LinearKernel` no
longer appears at all. Weight loading then completes without complaint.

**It fails at the kernel call**, not before:

    RuntimeError: b_scales must have same group count as qzeros

raised from `torch.ops._rocm_C.gptq_gemm_rdna3` on the first forward. So the
answer to "is the type gate the only thing?" is: for *selection* yes, for
*working* no. The reason is a layout the symmetric path never has to think
about, read here straight from the safetensors headers (`zp_layout.py`,
`zp-layout.json`, one `down_proj`, N=5120, K=17408, group_size 32, 544 groups):

| tensor | shape | layout |
|---|---|---|
| `weight_scale` | 5120 x 544 | (N, groups) |
| `weight_zero_point` | **640 x 544** | (N/8, groups) |
| what the kernel expects | **544 x 640** | (groups, N/8) |

`process_weights_after_loading` runs `permute_param_layout_` on `w_q` and on
`w_s`, so scales arrive group-major. **It never touches `w_zp`** — and on the
symmetric path it does not need to, because that path fabricates the tensor
itself as `(groups, out_features)` and packs along dim 1, which is group-major
by construction. A real asymmetric checkpoint ships the transpose of that. The
entry check then compares `b_scales.size(0) = 544` against
`qzeros.size(0) = 640` and stops.

So the missing piece is a layout transform on the host, in the same function
that already transforms the other two tensors — not new kernel math. The HIP
kernel was never reached with wrong data; it refused the shapes at the door.

vLLM's existing tool does that transform, with no repacking, which is worth
checking rather than assuming because `permute_param_layout_` asserts rather
than repacks when the packed dimension does not land where it is asked for.
Driving the real function with the attributes `compressed_tensors_wNa16.py`
registers (`input_dim=1, output_dim=0, packed_dim=0`) and the shape the
checkpoint ships (`check_permute.py`, `logs/check_permute.log`):

    checkpoint ships   : (640, 544)  input_dim=1 output_dim=0 packed_dim=0
    after permute      : (544, 640)  input_dim=0 output_dim=1 packed_dim=1
    PERMUTE_OK=True

The transpose moves the packed dimension from 0 to 1, which is exactly where
`packed_dim=1` asks for it, so the assertion passes.

**And a second problem sits behind this one.** The GPTQv1 "+1 quirk" — the
kernel adds 1 to the stored zero, which is why the symmetric path encodes
`bias - 1` — is *behind* this shape check and was not exercised by this run.
Fixing the layout is what exposes it. The next section does exactly that, and
the second problem is real: skipping it produces a fast, confident stream of
garbage.

The `stock` arm of the same script is the control that makes the instrument
credible: same checkpoint, same prompt, Triton as shipped, **11.41 tok/s**
(against 11.49 in the A/B, an independent reproduction 0.7% apart) and a mean
logprob of **-0.1859** over 29 generated tokens, answering the question
correctly. Coherence was going to be scored, not eyeballed; the patched arm
simply never got far enough to be scored.

## The three-line fix, and the control proving all three are needed

Reading `csrc/rocm/q_gemm_rdna3.cu` turns the remaining work from "host-side
layout plus an unknown encoding problem" into three Python lines, because the
kernel already implements both zero-point conventions and the Python side
simply never uses the second one:

    q_gemm_rdna3.cu:668   const int zero_offset = use_v2_format ? 0 : 1;

`use_v2_format` is already a parameter of `gptq_gemm_rdna3`. v2 means "the
stored zero *is* the zero", which is exactly what compressed-tensors writes.
`apply_weights` hard-codes `False`. So no zero-point arithmetic is needed at
all:

```diff
-    SUPPORTED_QUANT_TYPES = [scalar_types.uint4b8]
+    SUPPORTED_QUANT_TYPES = [scalar_types.uint4b8, scalar_types.uint4]

+        if c.zero_points:
+            def transform_w_zp(x):
+                assert isinstance(x, BasevLLMParameter)
+                permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=1)
+                x.data = x.data.contiguous()
+                return x
+            self._transform_param(layer, self.w_zp_name, transform_w_zp)

-        output = ops.gptq_gemm_rdna3(x_2d, w_q, w_zp, w_s, w_g_idx, False)
+        output = ops.gptq_gemm_rdna3(x_2d, w_q, w_zp, w_s, w_g_idx, c.zero_points)
```

Measured on the same AWQ checkpoint and the same prompt as the `stock` control
above. `layout_only` applies the first two changes and leaves `use_v2_format`
at `False`; it exists so that the third change has to earn its place:

| arm | decode | mean logprob | answer |
|---|---:|---:|---|
| stock, Triton as shipped | 11.41 | -0.1859 | correct |
| **fixed**, all three | **35.50** | **-0.1835** | correct, token for token identical to stock |
| layout_only, no v2 | 35.45 | **-4.4321** | `terasterasterasteras...` |

**3.11x, and it is right.** The `fixed` arm produces the same 64 tokens as the
Triton control. Not bitwise-identical arithmetic — the mean logprob differs,
-0.1835 against -0.1859 — but the difference is small enough not to move any
argmax, which is what changing kernels should look like.

**The control is the point.** `layout_only` runs at the same speed, 35.45
against 35.50, so it really is executing the native kernel; it just reads every
zero point one too high. Its logprob is **24x worse** and its output is
repetition. A patch that stopped after the layout fix would have looked like a
success in every way except correctness.

**Both entry points were exercised, which was not obvious.**
`gptq_gemm_rdna3` hands off to `gptq_gemm_rdna3_wmma` when K and N are both
multiples of 16 and M is at least 16 for bf16. Every one of this checkpoint's
**399 quantised linears qualifies** (`wmma_reach.py`, `logs/wmma-reach.log`),
so the prefills in this run — 30 tokens for the question, 1024 for the timing
prompt — went through the WMMA kernel, while decode at M=1 went through the
scalar one. Both produced correct output on the `fixed` arm. The control makes
the same point in reverse: `layout_only`'s very first generated token is
already garbage, and that token comes out of a prefill, so the WMMA path
honours `use_v2_format` too.

Against the symmetric checkpoint's 37.24, the fixed asymmetric arm reaches
95.3%. The residual is the direction group size predicts: 32 against 128 means
four times as many scale and zero-point rows to read per output tile. So group
size does cost something — just nothing to do with which kernel is chosen.

**Why "subtract one" would have been the wrong fix**, had `use_v2_format` not
existed. Sampling this checkpoint's own zero points, 22.3M entries across 12
tensors (`zp_values.py`, `logs/zp-values.log`):

    histogram 0..15: [22, 714, 5667, 37396, 264602, 1276645, 3595641, 5957601,
                      5956628, 3594489, 1280982, 266719, 38435, 5787, 874, 38]
    SUBTRACT_ONE_IS_SAFE=False

22 entries are 0, and 0 has no representation under GPTQv1's `stored = real-1`.
At ~1e-6 that is exactly the sort of error that survives a coherence check.
The v2 route is not merely more convenient, it avoids a wrong answer.

**Why no existing test catches any of this.**
`tests/kernels/quantization/test_rdna3_w4a16.py` does exercise a zero-point
path, but it builds `zeros_gn` as `[K//G, N]` and packs along N — handing the
kernel the group-major layout it wants. A real checkpoint ships the transpose.
It also pairs `uint4b8`, a symmetric type, with explicit zero points, which no
checkpoint produces. The layout mismatch is unreachable from the suite as
written.

## Tests, and the regression they caught

The end-to-end run above says the patch works on one checkpoint. Upstream will
want tests, and writing them changed the patch.

Added to the two files vllm#41394 shipped, built the way
`compressed_tensors_wNa16.py` actually registers an asymmetric layer —
`weight_zero_point` as `[N//8, K//G]` with `input_dim=1, output_dim=0,
packed_dim=0`, which is the transpose of what the existing `with_zp` cases
construct:

- `test_selection_prefers_rdna3_asymmetric` — `uint4` + `zero_points=True`
  resolves to the RDNA3 kernel, fp16 and bf16
- `test_can_implement_accepts_both_quant_types` — both int4 types admitted,
  `uint8b128` still refused
- `test_rdna3_w4a16_asymmetric_matches_reference` — five M/K/N/group shapes
  spanning the scalar path and the WMMA path, against an fp32 reference that
  does *not* apply the "+1"
- `test_rdna3_w4a16_asymmetric_zero_point_of_zero` — a zero point of exactly 0,
  which GPTQv1 cannot encode and which real AWQ checkpoints do use

Run on gfx1100 before and after the patch (`run_tests.sh`, `apply_patch.py`,
`logs/tests-run.log`):

| | selection | numerical |
|---|---|---|
| stock | 1 failed (asym) | **14 failed** |
| patched | **12 passed** | **38 passed** |

**The regression the tests caught.** The first version of the patch chose the
convention with `use_v2_format=c.zero_points`. That passed the end-to-end run
and broke ten existing cases: `test_rdna3_w4a16_matches_reference[...with_zp]`
pairs `uint4b8` with explicit zero points, and that is not a synthetic
combination — **a GPTQ checkpoint stores its zero points explicitly and still
uses the v1 "+1"**. Keying on `zero_points` would have silently mis-dequantized
every asymmetric GPTQ checkpoint on RDNA3.

The right key is the one the symmetric branch already uses to decide whether it
can synthesize at all:

```python
output = ops.gptq_gemm_rdna3(x_2d, w_q, w_zp, w_s, w_g_idx,
                             not c.weight_type.has_bias())
```

`uint4b8` carries bias 8 and means v1; `uint4` has no bias and means v2. With
that key the ten regressions disappear and all 38 numerical cases pass. The
end-to-end result above is unaffected — on `uint4` both keys agree — but the
patch as first written would have been wrong for a checkpoint format this
directory never tested.

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

For vLLM, the general fix is smaller than "write a kernel", and the section
above measures how much smaller. The RDNA3 kernel does not avoid zero points,
it **requires** them: `apply_weights` asserts one is present, and the symmetric
path works by fabricating a constant `qzeros` of `weight_type.bias - 1` on the
host. `ops.gptq_gemm_rdna3` is already a general asymmetric GPTQ dequant, and
compressed-tensors already registers a real `weight_zero_point` for asymmetric
checkpoints under the same config field the kernel reads.

Forced onto an asymmetric checkpoint it selects, loads, and then refuses the
zero-point tensor's shape, because `process_weights_after_loading` transforms
`w_q` and `w_s` and leaves `w_zp` alone. Two things follow. The work is on the
host, in that function, alongside transforms that already exist. And it is
**two** problems rather than one: the layout, and then the GPTQv1 "+1"
encoding, which sits behind the shape check and has still never been
exercised. A patch that fixes only the first will run and may well be wrong,
so output quality has to be checked, not just that it starts.

That patch is written and measured above: three Python lines in one file, no
C++ change, **3.11x on the asymmetric checkpoint with output that matches the
Triton control token for token**, and a control arm showing the third line is
load-bearing rather than decorative.

What it still needs before it is a pull request. Only gfx1100 was tested, and
only one model, one group size and TP=2; the `partition_scales` and
channel-wise (`PackedColumnParameter`, which carries no `input_dim`) paths are
untouched here and the permute assumes the group-quantized layout. Only bf16
activations were run, though the kernel also serves fp16. And the upstream tests
would need a case built the way `compressed_tensors_wNa16.py` actually
registers the parameter, since the existing ones construct the layout the
kernel already wants and therefore cannot fail.

## Files

- `probe_w4a16_ab.py` + `run_ab.sh` — the A/B, one fresh container per cell
- `probe_w4a16.py` — the registry, four real checkpoints
- `probe_w4a16_2x2.py` — the registry, all four corners plus every campaign
  checkpoint
- `census_ckpt.py` — the quantized-layer census
- `probe_w4a16_forced.py` + `run_forced.sh` — one checkpoint, two kernels
- `zp_layout.py` — the zero-point layout census, from safetensors headers
- `check_permute.py` — whether vLLM's existing layout tool can do the fix
- `probe_w4a16_fix.py` + `run_fix.sh` — the three-line fix and its control arm
- `zp_values.py` — the checkpoint's actual zero-point distribution
- `upstream-tests/` + `apply_patch.py` + `run_tests.sh` — the test cases
  proposed upstream, and the before/after run that validates them
- `dl_sym.py`, `verify_ckpt_sha.py` — how the symmetric checkpoint was fetched,
  and its sha256 against the Hub's own ETags
- `w4a16-ab.jsonl` — the six measured cells; `w4a16-forced.jsonl` and
  `w4a16-fix.jsonl` — the forced-kernel control and the fix
- `w4a16-selection*.json`, `ckpt-layer-census.json`, `ckpt-sha256-sym.json` —
  the derived records
- `logs/` — the runs that produced them, including the per-cell worker-side
  kernel records

Figures above are recomputed from these files by
[`verify_doc_figures.py`](../analyze/verify_doc_figures.py).
