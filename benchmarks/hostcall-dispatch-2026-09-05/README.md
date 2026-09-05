# The inference engine's own attention kernel is not refused without PCIe AtomicOps — because the kernels that declare the requirement are never dispatched here — 2026-09-05

[The C1 scan](../hostcall-abi-2026-09-04/README.md) counted 348 kernels in
`vllm/_rocm_C` that declare `hidden_hostcall_buffer`, and said that whether any
of them is dispatched on this box, and whether a platform without PCIe
AtomicOps refuses them, was *not measured*. Two facts already in this
repository made that a sharp question: Qwen3-8B (`head_dim` 128, `gqa_ratio`
4) satisfies `use_rocm_custom_paged_attention` on gfx1100, and this box's
Qwen3-8B serve logs select `ROCM_ATTN` with no Triton fallback — yet every
Qwen3 run here was after 2026-08-23, with atomics present. The mechanism in
[root-cause.md](../../docs/root-cause.md) predicted that without atomics,
Qwen3-8B at TP=1, with no collective library anywhere, would be refused at its
first decode step.

**It is not refused. The prediction was wrong, and the reason is exact.** The
platform capability was flipped both ways with the one VM-configuration line
of [vfio-atomics.md](../../docs/vfio-atomics.md), both cards, and in both
states the custom paged-attention path dispatches, computes correctly, and
serves:

    gfx1100 pair, VM 101, rocm/vllm:rocm7.14.0_rdna..._vllm_0.23.0, Qwen3-8B, TP=1

                          probe: as_shipped   ck_forced   triton_forced      serve: default (ROCM_ATTN)   --attention-backend TRITON_ATTN
    AtomicOps present     dispatched_ok       dispatched_ok   dispatched_ok    healthy, answered, ok        healthy, answered, ok
    AtomicOps absent      dispatched_ok       dispatched_ok   dispatched_ok    healthy, answered, ok        healthy, answered, ok

    hipDeviceAttributeHostNativeAtomicSupported, both cards:  present 1 / absent 0
    root ports advertising 32bit+ 64bit+ completion:           present 2 / absent 0
    amdgpu "PCIE atomic ops is not supported" at boot:          present 0 / absent 2

The runtime knew the capability was gone — the attribute it derives from
`pcie_atomics_` read 0 on both cards — and still dispatched, because **the
kernel it dispatched does not declare the requirement**. The 256
paged-attention kernels that do are the CDNA variant's template
instantiations, whose entire body on gfx11 is `assert(false)`, and the RDNA
launcher never selects them.

---

## What was dispatched, and what declares the requirement

The gfx1100 device image inside `vllm/_rocm_C.abi3.so` holds **2 350** kernels,
of which **348** declare `hidden_hostcall_buffer` — the same 348 C1 counted —
and they are four template families, not one:

| family | instantiations | declaring | reachable on gfx1100? |
|---|---:|---:|---|
| `paged_attention_ll4mi_QKV_mfma4_kernel` | 256 | **256** | never: the navi launcher selects `mfma16` for every `gqa_ratio` 1–16 |
| `paged_attention_ll4mi_QKV_mfma16_kernel` | 1 536 | 0 | **this is what runs** |
| `paged_attention_ll4mi_reduce_kernel` | 128 | 0 | runs after the QKV kernel |
| `wvSplitKQ_hf_`, `wvSplitKQ_hf_sml_` | 32 + 32 | **64** | only from the fp8 per-tensor scaled-mm path |
| `wvSplitKrc_` | 28 | **28** | gated `on_gfx950()` |
| `wvSplitK_hf_`, `_sml_`, `_big_` | 300 | 0 | the skinny GEMM this box does use (`on_gfx9() or on_gfx1x()`) |

`logs/rocm_C-gfx1100-kernels.tsv` is every kernel by name with its
declaration, produced by `list_hostcall_kernels.py` with the C1 scanner's
reader over all five offload bundles the `.so` carries.

**The 256 are one stub.** In `csrc/rocm/attention.cu` at this container's
vLLM commit (`9ddef7117`, byte-identical to `v0.23.0` for this file), the
kernels are written three times under `#if defined(__HIP__GFX9__)`,
`#elif defined(__GFX11__)`, `#elif defined(__GFX12__)`, with an `#else` of
stubs. In the GFX11 section the `mfma16` kernel is a real WMMA implementation
and the `mfma4` kernel's body is `UNREACHABLE_CODE` — which the file defines
as `assert(false)` **whether or not `NDEBUG` is set** (it `#undef`s `NDEBUG`
around the include to make sure). A device `assert` is a hostcall user, so
every instantiation of that stub declares the buffer. The template grid is
exactly the launcher's: `{half, bf16} × {same-type, fp8 uint8 cache} ×
block {16, 32} × head {64, 128} × alibi {on, off} × gqa {1..4}` = 256, and the
`mfma16` grid is the same with gqa 1..16 and two `MFMAType`s = 1 536, none
declaring.

**Which one runs is decided on the host, by architecture.** `paged_attention()`
calls `is_navi_gpu()` (`gfx11`/`gfx12` by `gcnArchName`) and takes
`paged_attention_custom_launcher_navi`, whose `switch (gqa_ratio)` launches
`MFMA16` for every case 1–16. The non-navi launcher launches `MFMA4` for
`gqa_ratio` 1–4 ("mfma4 kernel is faster than mfma16 for gqa_ratio <= 4") and
`MFMA16` above. So on CDNA the declaring kernel is the one a gqa-4 model
would run; on RDNA it is dead code compiled into the image.

**Measured, not inferred.** This wheel's HIP runtime prints no `ShaderName`
line at any `AMD_LOG_LEVEL` (its `:3:` lines carry no file names and the
kernel-dispatch log is compiled out), and `torch.profiler` records no GPU
events on it, so `launchtrace.so` — an `LD_PRELOAD` shim that resolves every
`hipLaunchKernel` through `hipKernelNameRefByPtr` — names what the probe
launched. In both platform states the CK arms launch

    paged_attention_ll4mi_QKV_mfma16_kernel<__hip_bfloat16, __hip_bfloat16, Fp8KVCacheDataType(0),
                                            __hip_bfloat16, 16, 128, 256, false, 4, MFMAType(0)>
    paged_attention_ll4mi_reduce_kernel<__hip_bfloat16, __hip_bfloat16, 128, 128, 256, 1>

and no `mfma4` kernel; the Triton arm launches no `paged_attention` kernel at
all. Both named kernels are in the TSV with declaration 0.

---

## Design

Two platform states × three probe arms, and two platform states × two served
backends, on one machine, one container, one sitting per state. The design is
[B2](../rccl-ndebug-ab-2026-09-04/README.md)'s: the state is flipped with
`hostpci0: 0000:0b:00.0` → `0000:0b:00` (and `hostpci1` likewise, **both
cards**), which makes QEMU pass the card with its audio function and stop
advertising AtomicOp completion on the emulated root port; every row reads
the state back from `lspci` and `dmesg` and refuses to run if the label
disagrees.

**The probe** (`pa_probe.py`, one arm per process) builds the shape Qwen3-8B
decodes with — 16 query heads over 4 KV heads, head 128, block 16, bf16, a
4 096-token context, batch 2 — and drives it through vLLM's own
`chunked_prefill_paged_decode`, exactly as
[`vllm-50603/probe_53856.py`](../vllm-50603/probe_53856.py) does, with
`use_rocm_custom_paged_attention` left as shipped, forced on, or forced off.
It reads three refusal signals, because a custom op's launch failure need
not raise ([hipgate3.cpp](../../diagnose/hipgate3.cpp) saw launch and sync
both succeed on bare metal): the exception text, `hipGetLastError` after a
synchronize, and an output buffer pre-filled with NaN. It records which
kernel ran by wrapping the C++ op `torch.ops._rocm_C.paged_attention` and
capturing vLLM's "falling back to Triton" warning, scores the output against
an fp32 reference, and reads the platform's own answer —
`hipDeviceAttributeHostNativeAtomicSupported` on every device — from
`hipattr.cpp`, compiled in the container so the compiler resolves the enum.

**The serve half** (`serve_row.sh`, `one_request.py`) starts `vllm serve
/models/Qwen3-8B` at TP=1 with the backend the engine picks or with
`--attention-backend TRITON_ATTN` (0.23 ignores the environment variable;
[greedy-attn-ab](../gfx1100-greedy-attn-ab/README.md) found that the hard
way), sends one request, and greps the log for the backend the engine
actually chose.

Every cell is correct to the same tolerance in both states:

| arm | max rel. error vs fp32 reference, present | absent |
|---|---|---|
| as_shipped (CK) | 3.013e-3 | 3.013e-3 |
| ck_forced | 3.013e-3 | 3.013e-3 |
| triton_forced | 2.944e-3 | 2.944e-3 |

which are the 3.0e-3 and 2.9e-3 that `probe_53856.py` measured for this shape
on 2026-08-27. Every served request answered with 64 tokens on the backend
it asked for; the server was up in 45–87 s (the first run of each model load
is the slow one).

## Runs, and the two harness defects the first run caught

The rows in `logs/pa-cells.jsonl` and `logs/serve-cells.jsonl` are every
accepted run; `analyze.py` keys by (state, arm) and reports the latest. Every
run of every cell gave the same outcome:

| state | probe row runs | serve runs | guest kernel |
|---|---:|---:|---|
| present | 3 | 2 | 7.0.0-30 (first), 7.0.0-31 |
| absent | 5 | 1 | 7.0.0-31 |

The guest booted a newer kernel on the first flip — `7.0.0-31-generic` had
been installed by unattended upgrades after B2 — so the present row was
repeated on `-31` after the revert. Outcomes and errors are identical on both.

Before any accepted run, the harness was reviewed twice by a second model
(`AGENTS/reviews/REVIEW-0007.md` in the workspace: 23 findings, then 13 more
on the revision) and then failed its own first run, which is what the
present row is for: `pa_probe.py` had loaded `libamdhip64` by bare name with
`ctypes` to read the attribute, and this container ships **three** copies of
it (`/usr/lib/.../libamdhip64.so.5` from ROCm 5.7, the SDK's `.so.7`, and a
`/rb/hipkit` copy); a query against the wrong one left a sticky
`hipErrorInvalidValue` that torch reported as `CUDA error: invalid argument`
on its first kernel. The row refused to write because the attribute reading
disagreed with the label. The fix is `hipattr.cpp` (a separate process,
compiled against the SDK's runtime) and a `ctypes` handle opened on the
library torch has already mapped.

## What this licenses, and what it does not

**Licensed.** On this platform, vLLM 0.23's custom paged-attention path for a
gqa-4, head-128, bf16 model dispatches and serves without PCIe AtomicOps; the
kernel it dispatches declares no hostcall buffer; the 256 paged-attention
kernels in the same image that do declare it are the CDNA template's gfx11
stubs, `assert(false)` and nothing else, and the RDNA launcher cannot reach
them. C1's "the inference engine declares the requirement" stands as a
statement about the image; as a statement about what runs on gfx1100 it does
not, and this directory is the measurement that separates the two.

**Not licensed.** That the same holds on CDNA: there the launcher runs the
`mfma4` kernel for `gqa_ratio` ≤ 4, which is a real kernel on gfx9 and whose
declaration was not scanned here (this container carries no gfx9 image). That
`wvSplitKQ` (fp8 scaled-mm) or `wvSplitKrc` (gfx950) behave either way —
neither path is taken by a bf16 model on gfx1100. That vLLM 0.27 or 0.28
select the same kernel; the launcher's `is_navi_gpu()` split is at this
commit. One host, one architecture, one model shape.

## Reproducing

    # guest side, takes the lease; ~15 min per state
    bash pa_row.sh atomics_present                      # or atomics_absent; the label is checked against lspci/dmesg
    bash serve_row.sh atomics_present default
    bash serve_row.sh atomics_present triton
    PA_TRACE=1 bash pa_row.sh atomics_present           # adds the launched kernel names to each row

    # host side, the flip and the revert, each polled to completion and verified from the guest
    bash tools/pve_flip.sh absent ; bash tools/pve_flip.sh present

    python3 analyze.py                                  # both tables from logs/, non-zero if a cell is missing

`hipattr.cpp` and `launchtrace.c` are compiled inside the container by
`pa_row.sh`; `list_hostcall_kernels.py` needs no GPU. `logs/` holds every
row, every per-arm log, the serve logs, the `lspci` lines each row recorded,
and the kernel-name traces.
