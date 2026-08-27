# vllm#52684 on A100: Ampere wants the wider query block too, but not at 512

[vllm#52585](https://github.com/vllm-project/vllm/issues/52585) measured
`BLOCK_M=64` (against the default 16) worth 1.26-2.18x on the gfx1100
Triton unified-attention kernel for long prefill.
[PR #52684](https://github.com/vllm-project/vllm/pull/52684) ships that as a
gfx1100-gated launch choice:

```python
tuned_gfx1100_prefill = (
    max_seqlen_q >= 512 and num_queries_per_kv <= 16 and _is_gfx1100()
)
```

The reporter asked whether the gate belongs there at all, and by 27 August the
PR had three independent gfx1100 confirmations and no CUDA number. This
directory is the CUDA number.

**Both answers are no.** Ampere wants the same tuning, so `_is_gfx1100()` is
not what separates the machines that benefit from the machines that don't. But
`>= 512` is the wrong threshold for CUDA: on A100 the wider block loses a few
percent from 512 to 1024 and only starts paying from about 1280.

## What was measured

Two Colab A100-SXM4-40GB sessions, vLLM 0.28.0, torch 2.13.0+cu130, Triton
3.7.1. Released 0.28.0's copy of `triton_unified_attention.py` is byte-identical
to `main` (md5 `49fab3b643bf5a88eb65303ce377996b`), so PR #52684 applies to the
release unmodified; `setup_vm.py` asserts both that hash and the patched one
(`f1d7a7e3c6656303fa63b6a4c1b8aef5`) before anything runs, and
`logs/setup-pass1.log` shows the assertion passing.

The probes drive `unified_attention` directly, with the input construction
lifted from upstream's own `tests/kernels/attention/test_triton_unified_attention.py`,
and flip the PR's own `_is_gfx1100` seam to choose the launch configuration.
`max_seqlen_q > 1` forces the 2D path, so every row is the 2D prefill kernel the
PR targets. Three arms, because the PR changes two things at once:

| arm | (BLOCK_M, BLOCK_Q) | num_warps |
|---|---|---|
| `base` | (16, 16/nq) | Triton's default |
| `pr` | (64, 64/pow2(nq)) | 4 — the PR verbatim |
| `bm64` | (64, 64/pow2(nq)) | Triton's default |

Pass 1 (`pass1-matrix.jsonl`, 140 rows): single-sequence prefill at
`q_len` 256 to 16384, `num_queries_per_kv` in {4, 7, 8, 16}, `head_size` in
{64, 128, 256}, sliding window on and off, bf16 and fp16.
Pass 2 (`pass2-numerics-crossover.jsonl`, 117 rows) resolves what pass 1 left
open: the ULP size of the numerical delta, and where the gain crosses 1.0.

`q_len=256` sits below the PR's own gate, so all three arms select the same
launch there. Those 20 rows are the control: they span 0.981 to 1.046, which
puts the noise floor at **±4.6%** and is the band every other figure here has to
clear.

## Speed: the gain is real on Ampere and grows with prefill

Pass 1, `pr` against `base`, median over the 20 configurations at each length:

| q_len | 256 (ctl) | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---:|---:|---:|---:|---:|---:|---:|
| median | 1.021 | **0.948** | 0.987 | 1.281 | 1.394 | 1.529 | **1.597** |
| min | 0.981 | 0.848 | 0.911 | 1.102 | 1.181 | 1.221 | 1.238 |
| max | 1.046 | 0.991 | 1.282 | 1.819 | 2.186 | 2.080 | 2.189 |

At 16384 every one of the 20 configurations wins, the smallest by 1.238x and
the largest by 2.189x (bf16, head_size 64, nq=16). That is the same order of
gain #52585 reported on gfx1100, on a machine three architectures away.

**The gain is not occupancy.** At `q_len=8192, nq=4` the two arms launch 16392
and 4104 workgroups against 108 SMs; neither arm is short of parallelism, so
this is not the workgroup starvation that drives the gfx1100 case. What changes
is how much KV traffic each program amortizes: a 4x wider `BLOCK_Q` means 4x
fewer programs walking the same KV range. That reading also predicts the shape
of the curve, since the amount to amortize grows with prefill length, and it is
consistent with the gain being architecture-independent.

**The `num_warps=4` pin does nothing here.** Across the 120 gated rows the
median `pr`/`bm64` ratio is 1.001. On A100 the entire effect is the wider query
block; the warp count is free either way.

## The threshold, not the architecture, is what's wrong for CUDA

Pass 1 showed all 20 `q_len=512` rows below 1.0. Pass 2 swept the crossover
region on a second VM, running `base` a second time at the end of each row so
ordering drift is measured rather than assumed (median drift 1.000):

| q_len | 512 | 640 | 768 | 896 | 1024 | 1280 | 1536 | 1792 | 2048 | 3072 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| median | 0.982 | 0.996 | 0.991 | 0.982 | 0.991 | **1.165** | 1.244 | 1.363 | 1.404 | 1.536 |

The sign flips between 1024 and 1280 and the gain is solid from 1536 up. The
losses below that are small — a few percent, inside or near the ±4.6% control
band — but they are one-directional across 20/20 pass-1 rows and 4 of 5 pass-2
lengths, which is not what noise looks like.

So a single global `>= 512` would apply the wider block across a band where
CUDA does not want it. Raising the threshold, or keeping one threshold per
architecture, is the choice the evidence actually supports.

## Numerics: one bf16 ULP, in one slice, on 0.0001% of elements

117 of pass 1's 140 rows are bitwise-equal between the two launches. All 23 that
are not sit in a single slice: bf16 with `head_size=64`. bf16 at 128 and 256 and
fp16 at 128 are bitwise-equal everywhere, at every length.

Pass 2 measured the size of that difference against the local ULP:

- **maximum distance 1.0000 ULP**, over every differing element in every row
- **zero elements more than 1 ULP apart**
- 507 differing elements out of 550 502 400, which is **0.0001%**

`pr` and `bm64` differ from `base` identically, so the wider query block causes
it and the warp pin does not. One ULP on a ten-thousandth of a percent of
elements is the same class of reordering our
[#45450 kernel check](../45450-validation/README.md) bounded, and it is worth
stating precisely rather than calling the change bitwise-equal in general: on
CUDA it is bitwise-equal for four of the five slices tested and one ULP on the
fifth.

## Reproducing

```bash
pip install vllm==0.28.0
patch -p1 -d "$(python -c 'import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))')" \
    -i 52684-kernel.diff
python probe_block_m.py  pass1.jsonl     # 140 rows, ~25 min on an A100
python probe_block_m2.py pass2.jsonl     # 117 rows, ~6 min
```

`setup_vm.py` is the exact provisioning used here, hash assertions included.
Every figure above is recomputed from the two JSONL files by
[`verify_doc_figures.py`](../../analyze/verify_doc_figures.py).

The two passes ran on different VMs. Cross-VM spread on this stack measures
5.6-10.5%, so pass 1 and pass 2 absolute times are not comparable with each
other; every ratio quoted here is between arms measured inside one VM.
