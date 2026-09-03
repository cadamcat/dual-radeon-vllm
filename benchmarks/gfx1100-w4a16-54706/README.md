# The greedy non-determinism is the W4A16 kernel — vllm#54706 built and A/B'd, 2026-09-03

**Yes, the kernel.** The same harness, models, depths and attention backend as
the published cells, and one variable: which `_rocm_C.abi3.so` the container
loads. Our own build of the container's vLLM commit varies in two of four
cells, as the shipped wheel does; the same build with
[vllm#54706](https://github.com/vllm-project/vllm/pull/54706)'s two kernel
files — FP32 partials and a fixed-order reduction in place of the CAS-atomic
split-K epilogue of `RDNA3W4A16LinearKernel` — returns **32 of 32** greedy
generations identical.

| cell, ROCM_ATTN | shipped wheel, 2026-09-02 | our build, unpatched | our build + #54706 |
|---|--:|--:|--:|
| Muse-Glimmer-30B, 512 | 6 distinct of 8 | **6 distinct of 8** | 1 of 8 |
| Muse-Glimmer-30B, 8 192 | 4 of 8 | 1 of 8 | 1 of 8 |
| gemma-3-27b w4a16, 512 | 1 of 8 | 1 of 8 | 1 of 8 |
| gemma-3-27b w4a16, 8 192 | 4 of 8 | **2 distinct of 8** | 1 of 8 |

Each cell is eight greedy generations of 64 tokens from one prompt in one
process; "distinct" counts the different sequences among the eight. The first
column is [`../gfx1100-greedy-attn-ab/`](../gfx1100-greedy-attn-ab/README.md)'s
ROCM_ATTN arm, the wheel the image ships. The middle column is the control
this directory adds: without it, a change between the wheel and the patched
build could be our toolchain rather than the patch.

## What was built, and from where

The container's wheel is `vllm 0.23.1.dev1+g9ddef7117.d20260715.rocm714`, and
`9ddef7117` is a commit of the **ROCm/vllm** fork, not of vllm-project's main:
the first build cloned upstream, found no such commit, and compiled main's
kernels for six minutes before the log showed it (`build-pr54706/` is the
second attempt; the first is not kept). Both builds here check out that
commit from the fork, assert `HEAD` is it, and compile only the `_rocm_C`
target for `gfx1100` with the container's own `hipcc`, `cmake` and `ninja` —
about thirteen minutes each, 30 MB against the wheel's 514 MB multi-arch
object.

| build | source | `q_gemm_rdna3.cu` | `q_gemm_rdna3_wmma.cu` | `_rocm_C.abi3.so` |
|---|---|---|---|---|
| baseline | 9ddef7117, unpatched | `ad465004…` | `96363bb0…` | `4f4c1b93…`, 29 977 912 B |
| pr54706 | 9ddef7117 + the PR's two files | `c4c7520e…` | `584960b9…` | `83f1d74c…`, 30 052 528 B |
| shipped | the wheel | — | — | `069f038d…`, 514 664 296 B |

`git apply --check` of the PR's diff against the two files at that commit
passes clean; the tests file is excluded, nothing else in the PR touches the
build. The patched object exports 103 `rdna3` symbols against the wheel's 85 —
the PR's new template instantiations.

## How the A/B ran

`run_c1_ab.sh`, on the box: the two GPU services stopped, the shipped
`_rocm_C.abi3.so` copied aside, then for each arm the built object copied
over the installed one (md5 printed from inside the container beside the
source's) and `nondet_attn.py <model> 1 ROCM_ATTN` run for both models — the
harness that produced the published cells, unchanged, with the backend passed
as the engine argument it takes on 0.23. Every run's log is grepped for the
backend and the quantisation kernel it actually used; all four say
`Using ROCM_ATTN backend` and `Using RDNA3W4A16LinearKernel`. The shipped
object was put back at the end (md5 `069f038d…` on both the installed file and
the copy), the services restarted, both cards back at the 27 971 584-byte
VRAM baseline. Start to finish, fifteen minutes. The `first_block` patch state
of the attention kernel was read before the run and left as found (3 sites).

## What this settles, and what it does not

The 36-cell set's confound is resolved. Every unstable cell there was
`RDNA3W4A16LinearKernel` **and** `ROCM_ATTN`; the attention A/B held the
kernel and moved the backend, and the variation stayed. This holds the
backend and moves the kernel, and the variation goes. The source is the
kernel's split-K epilogue, and #54706's replacement of it is the fix — on
this box, at TP=2, for these two checkpoints and depths.

Eight repeats is what the published cells used, and it is not many: the
baseline's Muse-Glimmer at 8 192 came back 1 of 8 where the wheel had 4 of 8,
which is the same kernel on a different day. The claim rests on the patched
arm's 32 of 32 against the baseline's two unstable cells in the same sitting,
not on the wheel's row. What would make it stronger is more repeats and TP=1,
which this box cannot run (both checkpoints exceed one card).

## Files

    pr54706.diff                    the PR as fetched, 878 lines; applied without the tests file
    c1_build.sh                     the build, run inside vllm-tp2; C1_NOPATCH=1 for the baseline
    build-pr54706/build.log         the patched build: HEAD, file md5s, the object's md5
    build-baseline/build.log        the unpatched build, "baseline: PR not applied"
    run_c1_ab.sh                    the A/B, with the restore in an EXIT trap
    c1-ab.log                       the A/B's console: arms, md5s, verdicts, restore
    nondet-c1-<arm>-<model>-ROCM_ATTN-p1.json   the eight sequences per cell
    logs/                           each run's full engine log, PROGRESS.txt
