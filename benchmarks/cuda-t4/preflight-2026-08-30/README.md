# The T4 pre-flight — 2026-08-30

One engine start, asked one question: **can `gemma-4-12B-it-qat-w4a16-ct` be
served on sm75 at all?** The answer is no, and the reason is not the one the
plan expected.

    Tesla T4 · 15360 MiB · compute capability 7.5 · driver 580.82.07
    vLLM 0.28.0 · torch 2.13.0+cu130 · transformers 5.15.1 · --dtype float16

`preflight.jsonl` is what each attempt recorded; `logs/` holds a serve log per
attempt; `preflight.py` and `setup.py` produced them.

## What was expected to fail, and did not

The risk this pre-flight existed to test was whether compressed-tensors W4A16
loads on sm75 at all — on CUDA that path normally wants Marlin, which is
documented as Ampere and above. It loads:

    compressed_tensors_wNa16.py:137] Using MarlinLinearKernel for CompressedTensorsWNA16

Memory is not the wall either. At `--gpu-memory-utilization 0.90` the engine
had 0.65 GiB of KV against the 2.07 GiB that 33 000 tokens need, but the budget
is spent on activations and CUDA graphs sized for a `max_num_seqs` this harness
never uses: 4.57 GiB of the 13.50 GiB budget, against 2.82 GiB for the same
model on an RX 7900 XT. `--gpu-memory-utilization 0.95 --max-num-seqs 1` turns
0.65 GiB into **3.5 GiB, 55 809 tokens** — more than 32 K needs — without
touching the compute path.

## What actually fails

gemma-4's head dimensions are **heterogeneous**, and the one that matters is not
the one this file first named. Measured 2026-08-30 by constructing vLLM 0.28.0's
own `ModelConfig` against this checkpoint (`headsize.jsonl`, `check_head.py`):

    config.json     head_dim 256 (sliding) · global_head_dim 512 (full)
                    16 attention heads · 8 KV heads · 48 layers
                    = 40 sliding_attention + 8 full_attention
    model_arch_config.head_size            512
    per-layer head sizes                   {256, 512}

So the value the kernel is sized for is **512**, not the 256 this README
originally gave; 256 is the sliding layers' local value. Turing reports 49 152
bytes of shared memory per block and **65 536 per SM**.

This matters beyond pedantry: vllm#39018, the open fix for this failure, gates
on `head_size_padded >= 512`, so the corrected value is what puts this case
inside its scope rather than outside.

| backend | selector | outcome | log |
|---|---|---|---|
| `FLASH_ATTN` | **rejects** | `Reason: ['compute capability not supported']` | `serve-T4-G12-flash.log` |
| `TRITON_ATTN` | accepts | `OutOfResources: Required: 98304, Hardware limit: 65536` | `serve-T4-G12-triton.log` |
| `FLEX_ATTENTION` | accepts | `OutOfMemoryError: Required: 163840 Hardware limit:65536` | `serve-T4-G12-flex.log` |
| `FLASHINFER` | — | below its floor, recorded in `../README.md` | — |

**The only backend honest about sm75 is the one that is excluded.** vLLM 0.28
chooses by asking each candidate `validate_configuration(device_capability=…)`
and taking the first that does not object; that predicate models compute
capability for `FLASH_ATTN` and shared memory for nothing, so Turing is routed
onto kernels needing 96 KB and 160 KB against a 64 KB ceiling and fails at
kernel load — with an error naming Triton rather than the selector.

This is the complement of the 2026-08-29 campaign's finding. That one: *a patch
does nothing unless the kernel it patches is on the path.* This one: **the
selector will put a kernel on the path that the hardware cannot run.**

One further sm75 line from the same start, and a different subsystem:

    topk_topp_sampler.py:69] FlashInfer top-p/top-k sampling unavailable:
    unsupported compute capability 7.5; falling back.

## Why there is no T4 row anywhere else

There is no substitute model. A 15.0 GiB card holds only `gemma-4-12B` out of
this repository's set — 9.6 G on disk, 8.28 GiB resident — and that is the model
that cannot run. `head_dim` 128 models would need roughly half the shared
memory and should fit, but every one of them here is 16 G or larger. So the
fifth machine of the 2026-08-30 round is a documented architectural wall rather
than a gap in the measurements.
