# Upstream reports — drafts

Three findings from this repository that belong upstream rather than only here.
**One is filed; two remain drafts.** They are kept in the repo so the
evidence and the wording can be reviewed in one place, and so that anyone who wants
to file one first can see exactly what is claimed.

| draft | target | status of the claim |
|---|---|---|
| [`rccl-hostcall-issue-draft.md`](rccl-hostcall-issue-draft.md) + [`rccl-6074-comment.md`](rccl-6074-comment.md) | **filed as [ROCm/ROCm#6520](https://github.com/ROCm/ROCm/issues/6520)**, with a pointer comment on [#6074](https://github.com/ROCm/ROCm/issues/6074) | **Mechanism established.** Note that AMD had already identified the atomics dependency on 21 July; what this adds is that the declaration survives `NDEBUG` on 2.30.4, so removing the `assert()` is not sufficient |
| [`rocm-issue-draft.md`](rocm-issue-draft.md) | `ROCm/ROCm` or `ROCm/clr` | **Mechanism established, one variable untested.** Dependency-free reproducer; cannot rule out that VFIO/IOMMU is required |
| [`vllm-ssm-issue-draft.md`](vllm-ssm-issue-draft.md) | `vllm-project/vllm` | **Behaviour characterised, cause not identified.** Framed as a report + questions, not a diagnosis |

## Duplicate check — done 2026-07-26

| candidate | verdict |
|---|---|
| [safetensors#183](https://github.com/huggingface/safetensors/issues/183) · [diffusers#2507](https://github.com/huggingface/diffusers/issues/2507) — "loading directly to GPU is slower than to CPU then moving" | **Not the same.** Both are NVIDIA V100 / CUDA 11.6 (Feb 2023), and the gap is 1.2x (3.93 s vs 3.17 s). Our mechanism lives in `kfd_ioctl_svm`/`hmm_range_fault`, which does not exist on NVIDIA, and our gap is 4x–4400x. Same *shape* of symptom, unrelated cause. |
| [ROCm#2433](https://github.com/ROCm/ROCm/issues/2433) — SVM change in ROCm 5.6 slowed `hipHostRegister`, fixed by `HSA_USE_SVM=0` | **Not the same, and tested:** `HSA_USE_SVM=0` leaves the pathology untouched (16 036 ms vs 16 020 ms) and *degrades* the read-only fast path (8 905 → 844 MiB/s). Recorded in the draft so a maintainer does not have to suggest it. |
| [ROCm#5952](https://github.com/ROCm/ROCm/issues/5952) — SVM mapping failure during sequential model loads, RDNA3 / RX 7900 GRE | **Possibly the same underlying subsystem.** Theirs crashes, ours crawls, but both are `svm_range_*` in amdgpu during weight loading on RDNA3, and their log says "VRAM loading crawls extremely slowly". Notably **bare metal**, which weakens the "maybe it needs VFIO" worry. The draft references it. |

No existing report of the writable-mapping performance cliff was found.

## Before filing any of these

1. **Check the repository links resolve** — they point at
   `github.com/2462381442/dual-radeon-vllm`, which must be public by then.
2. **Re-run the reproducers on the day of filing.** One of these drafts previously
   claimed a root cause that our own minimal reproducer then failed to support; the
   claim was withdrawn and rewritten. Verify before asserting.
3. **Re-check the duplicate table above** — #5952 was still untriaged when we looked,
   and if AMD has since explained it, that answer may cover our case too.

## What is deliberately *not* here

- **A PyTorch issue about `UntypedStorage.from_file` mapping writable.** It is what
  exposes every PyTorch user to the mmap problem, but mapping a mutable storage
  writable is a defensible contract. If ROCm responds that applications should avoid
  the pattern, that is the point to open a PyTorch conversation, with their answer
  as the reason.
- **A safetensors issue.** Investigated and cleared: safetensors maps read-only
  (`map_copy_read_only`), verified in the v0.8.0 source and observable with
  `framework="np"`. The writable mapping comes from the PyTorch path it delegates to.

## Lesson recorded, 2026-07-26

The first version of the RCCL report announced the atomics dependency as a new root
cause. It was not: @harkgill-amd had said so in #6074 five days earlier. The
pre-submission check that was supposed to catch this used a summarising fetch of the
comment thread, which returned comments from April and none from July, and the
summary was taken at face value.

For any future filing: read the last comments as raw JSON, not as a summary. The same
mistake would have been caught by `curl .../comments | python3 -m json.tool | tail`.
