# Check at load, refuse by name: forty-five lines in the HIP runtime turn the opaque refusal into a named one — 2026-09-05

The letter's §VI proposes that the runtime reconcile a kernel's hostcall
requirement with the platform's capability **at load** and refuse **by name**,
because both halves already exist in it. Items 1 and 2 of that proposal are
now a patch against the exact runtime commit this box's container was built
from, and the patched runtime was run through the same two-state toggle as
[B2](../rccl-ndebug-ab-2026-09-04/README.md) and
[the dispatch experiment](../hostcall-dispatch-2026-09-05/README.md):

    gfx1100 pair, VM 101, the vLLM 0.23 container; runtime = rocm-systems 2b22ab01 (TheRock 7.14)

                          stock runtime                              patched runtime
                          probe        12 collectives, stock RCCL    probe                       12 collectives, stock RCCL
    AtomicOps present     ok           12/12                         ok                          12/12
    AtomicOps absent      refused,     refused, "the operation       refused, hipErrorHostcall-  refused, hipErrorHostcall-
                          generic      cannot be performed in the    Unsupported; named at load  Unsupported; 13 kernels per
                                       present state"                on both devices             rank named at load

With the capability present the patch changes nothing: the probe passes and
the twelve collectives compute under both runtimes. Without it, the stock
runtime fails as it always has, four layers above the cause, and the patched
one says what happened — once per kernel when the module is read, and again,
by name, when the launch is refused before any command exists:

    :1:rockernel.cpp :29 :  kernel _Z10k_hostcallPf declares hidden_hostcall_buffer (device printf/assert)
                            but device gfx1100 has no PCIe AtomicOps to the host
                            (hipDeviceAttributeHostNativeAtomicSupported=0); its launches will return
                            hipErrorHostcallUnsupported
      hostcall  REFUSED   launch:kernel declares a hostcall buffer (device printf/assert) that this
                          device cannot provide: no PCIe AtomicOps to the host

and, under stock RCCL 2.30.4 with two ranks, **thirteen kernels per rank**
named at load — `ncclDevKernel_Generic_{1,2,4}` and the ten
`ncclSymkDevKernel_ReduceScatter_RailA2A_LsaLD_{sum,avg}_{bf16,f16,f32,f8e4m3,f8e5m2}`,
which is [C1's](../hostcall-abi-2026-09-04/README.md) list to the name — then
the first launch refused:

    :1:hip_module.cpp :338 :  launch of _Z23ncclDevKernel_Generic_424ncclDevKernelArgsStorageILm4096EE refused:
                              it declares a hostcall buffer and device 0 has no PCIe atomics
    [rank0]: HIP failure 'kernel declares a hostcall buffer (device printf/assert) that this device
             cannot provide: no PCIe AtomicOps to the host' at .../rccl/build/hipify/src/enqueue.cc:2061

`ncclDevKernel_Generic_4` is the kernel [root-cause.md](../../docs/root-cause.md)'s
July crash log names, and `enqueue.cc:2061` is the line it names. The failure
is the same; the sentence is different.

The same eight cells, run again with the ROCm 10.0 image's runtime commit and
its own RCCL 2.30.4, give the same table cell for cell —
[below](#the-same-eight-cells-at-rocm-100).

And with a fifteen-line opt-in on top of the check, **stock RCCL 2.30.4 completes
all twelve collectives without atomics** — [item 3, measured](#item-3-measured-an-opt-in-null-buffer-and-stock-2304-runs-without-atomics).
Under the same opt-in, **vLLM serves Qwen3-8B at TP=2 on stock 2.30.4 without
atomics** — [end to end](#end-to-end-vllm-at-tp2-on-stock-2304-without-atomics).

---

## The patch

`clr-hostcall-load-check.patch`: five files, forty-five added lines including
comments, against `rocm-systems` at `2b22ab01` — the commit TheRock's
`therock-7.14` tag pins, which is what built the container's
`libamdhip64.so.7.14.60850` (the binary embeds its build paths; its git hash
is stripped to `0000000`).

| where | what |
|---|---|
| `projects/hip/include/hip/hip_runtime_api.h` | `hipErrorHostcallUnsupported = 1055`, a provisional code after `hipErrorInvalidClusterSize` |
| `projects/clr/hipamd/src/hip_error.cpp` | its name and its description, in both tables |
| `projects/clr/rocclr/device/devkernel.hpp` | one bit in the device kernel's `flags_`, `hostcallUnsatisfiable_`, with an accessor |
| `projects/clr/rocclr/device/rocm/rockernel.cpp` | `Kernel::init()`: after the metadata is parsed, if a hidden argument is `HiddenHostcallBuffer` and `device().info().pcie_atomics_` is false, set the bit and log the kernel and the device |
| `projects/clr/hipamd/src/hip_module.cpp` | `ihipLaunchKernel_validate()`: if the device kernel carries the bit, log the refusal and return the code — before a command is built |

That is the whole of items 1 and 2. The requirement was already in the
signature the loader builds (`GetAttrCodePropMetadata` → `createSignature`),
the capability was already in `pcie_atomics_` from `checkAtomicSupport()` at
device init, and the check already existed in `submitKernelInternal` — per
dispatch, on the worker, returning `false` into a path that surfaces as
`hipErrorIllegalState`. The patch moves the decision to the one place both
inputs first meet and gives it a name. Item 3's runtime half is implemented as an opt-in (below); its device-library
half and item 4 (the toolchain) are not implemented here.

## What was run

`clr_demo_row.sh <state>`, the [B2](../rccl-ndebug-ab-2026-09-04/capability_row.sh)
skeleton: the label is checked against `lspci` and `dmesg`, the lease is
taken, and four cells run — the 57-line probe and the twelve elementwise
collective cases under stock RCCL 2.30.4 (`librccl-stock2304.so`, installed
and md5-checked as in B2), each once under the SDK's runtime and once under
the patched one. The platform state was flipped with the one VM-configuration
line, both cards; the revert was verified line for line and the config
compared byte for byte with the 2026-09-04 backup.

**The patched runtime is put in place, not on a path.** `LD_LIBRARY_PATH`
reaches the probe binary but not torch: `rocm_sdk.preload_libraries()`
`dlopen`s the SDK's `libamdhip64.so.7` by absolute path at import, and the
first copy in wins. So a patched cell copies the built library over the two
SDK copies (`_rocm_sdk_core/lib`, `_rocm_sdk_devel/lib`, both md5
`701fd9c6…`), runs, and restores them, and the row records the library the
measuring process actually mapped with its md5 — `c6c7a3df…` in every patched
cell, `701fd9c6…` in every stock one. The first attempt (`logs/clr-demo.attempt1.jsonl`)
used `LD_LIBRARY_PATH` and its "patched" collective cells ran the stock
runtime; they are kept as the reason the row now records what was mapped.

**The build needs `ROCM_KPACK_ENABLED=ON`.** CLR's default is off; the SDK's
own build has it on, because torch's device code is kpack-split ([C1](../hostcall-abi-2026-09-04/README.md)
found the `NOBITS` fatbins) and a runtime without it cannot load a single
torch kernel. The second attempt (`logs/clr-demo.attempt2.jsonl`) was built
without it: the probe, a classic fatbin, worked, and every torch process
died with SIGSEGV on its first GPU op in both platform states — `logs/CLRDIAG.txt`
is the three-step bisection that found it. `clr_build.sh` records the
other accommodations the wheel layout needs (GL and zstd headers, the
`CppHeaderParser` module, a one-line `rocm-kpack-config.cmake` because the
SDK ships the targets file but no config); none of them touches the runtime's
behaviour.

## The same eight cells at ROCm 10.0

The 7.14 image is the one this box serves from; ROCm 10.0 is the current
release. The patch applies unchanged to the commit its runtime was built from
(`6b0e43f3`, TheRock's `therock-10.0` tag — the image's RCCL says so itself:
`RCCL version 2.30.4 compiled with ROCm "10.0.0.0-9999-6b0e43f3"`), builds in
that image against its wheel SDK with the same `clr_build.sh` (`CLR_TAG=rocm10`;
the 10.0 SDK ships its own `rocm-kpack-config.cmake`, so the stand-in is not
written), and `clr_demo_row.sh` with `CLR_SDK=rocm10` runs the same four cells
per state in a container of that image (`clr100`), with the image's own
`librccl.so.1` as the stock library — md5 `a3963038…`, the file
[C1](../hostcall-abi-2026-09-04/README.md)'s cross-architecture scan read,
checked unchanged at the end of each row.

    gfx1100 pair, VM 101, the vLLM 0.27 container; runtime = rocm-systems 6b0e43f3 (TheRock 10.0)

                          stock runtime                              patched runtime
                          probe        12 collectives, stock RCCL    probe                       12 collectives, stock RCCL
    AtomicOps present     ok           12/12                         ok                          12/12
    AtomicOps absent      refused,     refused, "the operation       refused, hipErrorHostcall-  refused, hipErrorHostcall-
                          generic      cannot be performed in the    Unsupported; named at load  Unsupported; 13 kernels per
                                       present state"                on both devices             rank named at load

Cell for cell the 7.14 table. The thirteen kernels named at load are the same
thirteen by name; `ncclDevKernel_Generic_4` is refused on both devices (the two
ranks' refusal lines land in the same microsecond and interleave in
`logs/clrdemo-rocm10-atomics_absent-patched-collective.log`; the gate counts
the sentence, not the line); RCCL's own error path carries it through at
`enqueue.cc:2119` — 10.0's RCCL is a different binary under the same version
string, and the line number moves with it. Under the patched runtime the
probe's device `printf` still reaches the host when the capability is present
(the marker prints on both devices), and the patched cells mapped `423fdad5…`,
the stock ones `7c440eb5…`, that image's SDK runtime. Logs: `logs/*rocm10*` and
`logs/clr-rocm10-build.log`; `python3 analyze.py` reads both files.

## PR A alone, at ROCm 10.0

For upstream the patch is split in two. **A** is the load-time check and the
launch refusal, and returns the existing `hipErrorNotSupported` (three files,
37 added lines, one of them blank: `clr-hostcall-load-check-a.patch`); **B** is the new status on
top of it (four files, +10/−2). A+B is byte for byte this directory's patch.
A was built and run on its own at the 10.0 commit (`CLR_TAG=rocm10a`,
`CLR_SDK=rocm10a`; library md5 `992c7ad8…`, which contains no
`hipErrorHostcallUnsupported` string). PR A alone gives the same eight cells:
nothing changes with atomics; without them the same thirteen kernels are named
at load, `ncclDevKernel_Generic_4` is refused on both devices, and the probe
is refused once per device. What differs is only the string an application
gets back without `AMD_LOG_LEVEL`: "operation not supported", the existing
status's, at `enqueue.cc:2119` from RCCL and at launch from the probe, where
B's names the buffer and the missing atomics. That difference is the case for
B. Logs: `logs/*rocm10a*`.

## Item 3, measured: an opt-in null buffer, and stock 2.30.4 runs without atomics

The letter's third item is a fallback: a kernel that declares the buffer but
never executes a hostcall should be able to run without one. Fifteen lines on
top of A (`clr-hostcall-load-check-ac.patch`: five files, 48 added lines in
all) add a runtime flag, `HIP_HOSTCALL_ALLOW_MISSING`, off by default. When it
is set, the load-time line records the choice, the launch validator lets the
launch through, and `submitKernelInternal` writes a null buffer into the hidden
argument instead of refusing. Built at the 10.0 commit (`CLR_TAG=rocm10c`,
library md5 `f739295c…`) and run through the toggle with the flag set in the
patched cells (`CLR_SDK=rocm10c`), collectives before the probe:

    gfx1100 pair, VM 101, the vLLM 0.27 container; runtime = 6b0e43f3 + A + the opt-in
    HIP_HOSTCALL_ALLOW_MISSING=1 in the patched cells

                          stock runtime                              patched runtime, opt-in
                          probe        12 collectives, stock RCCL    probe                       12 collectives, stock RCCL
    AtomicOps present     ok           12/12                         ok, marker prints           12/12
    AtomicOps absent      refused,     refused, "the operation       plain kernel ok; the        12/12 -- stock RCCL 2.30.4,
                          generic      cannot be performed in the    printf kernel faults on     no atomics, no rebuild;
                                       present state"                the device; SIGABRT         13 kernels per rank marked at load

**Stock RCCL 2.30.4 completes all twelve collective cases without PCIe atomics**
under the opt-in: 26 load-time lines say `HIP_HOSTCALL_ALLOW_MISSING=1: its
launches proceed with a null hostcall buffer`, no launch is refused, no `HIP
failure` is reported, and the thirteen kernels so marked are the same thirteen
by name. This is the first time a declaring kernel has been shown to run with
no buffer at all: 2.27.7's working build had removed the declaration, not
survived it. The cost is the documented one. The probe's `printf` kernel, which
does execute a hostcall, dereferences the null buffer — `Memory access fault by
GPU node-1 … on address (nil)` in the probe's output, and in the kernel log
`[gfxhub] page fault … in page starting at address 0x0000000000000000` for
process `hipgate3-rocm10` (`logs/clrdemo-rocm10c-atomics_absent-patched-probe.dmesg.txt`,
recovered from the previous boot's journal after the revert flip) — and the
process aborts; the plain kernel on the same device had run first and was
fine. With atomics present the flag changes nothing: 12/12, and the marker
prints under both runtimes.

What this means for 2.30.4: the declaration is a linker artifact and the
buffer is never used on the working path, so the runtime can serve a null one
and the library needs no rebuild. What it does not mean: a kernel that does
call hostcall — a real device `printf`, a failing `assert` — faults instead of
being refused, which is why the flag is off by default and the choice is the
user's. The device-library half of item 3, OCKL checking for a null buffer and
returning, would turn that fault into a no-op; it is not implemented here.
Logs: `logs/*rocm10c*`.

## End to end: vLLM at TP=2 on stock 2.30.4, without atomics

The collective cases are RCCL alone. The failure this repository exists for
is `--tensor-parallel-size 2` under vLLM, so the opt-in was also run the way
a user meets it: Qwen3-8B, bf16, TP=2, vLLM 0.27 in the ROCm 10.0 container,
one chat request (`serve_c_row.sh <state> <stock|patched>` with
`one_request_tp.py`; the patched cells run with `HIP_HOSTCALL_ALLOW_MISSING=1
AMD_LOG_LEVEL=1`, the runtime swapped in place and md5-verified as in the
other rows):

    Qwen3-8B, TP=2, vLLM 0.27, stock RCCL 2.30.4; one request, 64 tokens asked for

                          stock runtime                                   A + the opt-in
    AtomicOps present     healthy in 201 s, 64 tokens                     healthy in 138 s, 64 tokens; nothing marked
    AtomicOps absent      never healthy (600 s): a worker raises          healthy in 99 s, 64 tokens; 718 kernels marked
                          "NCCL error: unhandled cuda error", the         at load "proceed with a null hostcall buffer",
                          engine stays up and broken -- July's failure    0 refused, 0 faults

**vLLM serves at TP=2 on stock RCCL 2.30.4 without PCIe atomics under the
opt-in**, and the 718 load-time lines are the static scan made dynamic: 26
from RCCL (13 per worker), 512 from vLLM's paged-attention family (256 per
worker, the CDNA stubs whose body on gfx11 is `assert(false)`) and 178 from
its `wvSplitK` kernels — 361 distinct names per worker, [C1](../hostcall-abi-2026-09-04/README.md)'s
`_rocm_C` 348 plus RCCL's 13. Every one of them was given a null buffer, none
of them faulted, and the engine answered: a declaration is not a dispatch,
measured one more way. torch's own declaring kernels did not appear in the
log; this serve does not load them. The stock cells are the control: with
atomics the SDK runtime serves; without, it hangs where it hung in July. Load
times are one run each, page cache uncontrolled. Logs: `logs/serve-rocm10c-*`
and `logs/SERVE-rocm10c-*`.

## What this licenses, and what it does not

**Licensed.** On this platform, at the 7.14 and the 10.0 runtime commits, moving the check to
kernel init and naming the error is forty-five lines and changes nothing when the
capability is present; when it is absent the refusal names the kernel, the
device, the attribute and the reason, at load and at launch, and the
collective library's own error path carries the sentence through
(`enqueue.cc:2061` in the 7.14 image's RCCL, `enqueue.cc:2119` in 10.0's) with no
change to RCCL or torch. Under an explicit opt-in, a declaring kernel that
never executes a hostcall runs with a null buffer: stock 2.30.4, 12/12, no
atomics, no rebuild; vLLM serves Qwen3-8B at TP=2 on it. The thirteen kernels the
runtime names are the thirteen the static scan counted.

**Not licensed.** That a real submission would use error code 1055, or log
at `LOG_ERROR`, or refuse in `ihipLaunchKernel_validate` rather than at
`hipModuleGetFunction`; those are choices for the runtime's maintainers. That
the opt-in is safe: a kernel that does execute a hostcall faults on the device,
and the device-library half of item 3 that would make it a no-op is not
implemented. Item 4 (the toolchain) remains a proposal. The cost
of the check was not measured; it is one pass over a kernel's hidden
arguments at init. One host, one architecture, two runtime commits.

## The upstream record, read 2026-09-05

Neither half of this has been absent from AMD's own tracker. In January 2025
a user hit the same refusal by compiling with `-O0`
([clr#126](https://github.com/ROCm/clr/issues/126)); AMD's reply named the
mechanism — a lingering `__assert_fail` that needs hostcall, which needs
PCIe atomics — and in April 2025 the user asked for exactly item 2: *"HIPAMD
should check for it as 'HIP error: the operation cannot be performed in the
present state' is not a very useful error message"*. The answer was a BIOS
setting. In February 2026 an AMD engineer opened
[rocm-systems#3612](https://github.com/ROCm/rocm-systems/pull/3612),
*"Downgrade hostcall error to warning on missing PCIe atomics"*: `LogError`
to `LogWarning` and the `return false` removed, so the dispatch proceeds. Its
only review, automated, made item 3's point: with no buffer the hidden kernarg
slot is left uninitialised, so a kernel that does use hostcall reads garbage.
Marked WIP in May, closed unmerged for inactivity on 2026-06-16. In May 2026
[rocm-systems#6402](https://github.com/ROCm/rocm-systems/pull/6402) took the
RCCL side — `if constexpr (COLLTRACE)` so that `COLLTRACE=false`
instantiations stop declaring the buffer — citing two more reports,
[#6074](https://github.com/ROCm/legacy-rocm-build/issues/6074) (2× RX 7900
XTX, 42 comments, open) and
[#6148](https://github.com/ROCm/legacy-rocm-build/issues/6148) (2× gfx1201,
Radeon AI PRO 9700); closed unmerged for inactivity on 2026-06-26.

So the record has the refusal removed (unsafe without the device-library
half) and the declaration removed for one of its sources (the trace, not the
asserts that [B1](../rccl-ndebug-ab-2026-09-04/README.md) found load-bearing),
each stalled. What is here — keep the refusal, decide it at load, name it —
is the part neither attempted, and it is the part that composes with both.

**Proposed upstream 2026-09-06** as [rocm-systems#11277](https://github.com/ROCm/rocm-systems/pull/11277)
(the check and the opt-in, two commits; the dedicated status follows once it
lands), with a comment on #377 pointing at it. Links and times: [UPSTREAM.md](UPSTREAM.md).

## Reproducing

    # CPU only, ~15 min in a throwaway container of the vLLM 0.23 image; writes /rb/clr-build
    docker run --rm --entrypoint bash -v /data/rccl-build:/rb <image> /rb/clr_build.sh /rb/clr-hostcall-load-check.patch
    # each state takes the lease, ~1 min; the flips are tools/pve_flip.sh in the workspace
    bash clr_demo_row.sh atomics_present
    bash clr_demo_row.sh atomics_absent
    python3 analyze.py                     # the table from logs/clr-demo.jsonl, non-zero if a cell is missing
    # the same at ROCm 10.0: the source tarball is /rb/clr-rocm10-src.tgz (rocm-systems 6b0e43f3, projects/clr + projects/hip)
    docker run --rm --entrypoint bash -e CLR_TAG=rocm10 -v /data/rccl-build:/rb <10.0 image> /rb/clr_build.sh /rb/clr-hostcall-load-check.patch
    CLR_SDK=rocm10 bash clr_demo_row.sh atomics_present
    CLR_SDK=rocm10 bash clr_demo_row.sh atomics_absent
    # PR A alone at 10.0: link the tarball as /rb/clr-rocm10a-src.tgz, then CLR_TAG=rocm10a with clr-hostcall-load-check-a.patch, CLR_SDK=rocm10a for the rows
    # the opt-in (item 3) at 10.0: /rb/clr-rocm10c-src.tgz, CLR_TAG=rocm10c with clr-hostcall-load-check-ac.patch, CLR_SDK=rocm10c for the rows
    bash serve_c_row.sh atomics_present stock; bash serve_c_row.sh atomics_present patched   # and the same in the absent state: TP=2 under vLLM

`logs/` holds every row, every per-cell log with the runtime's `:1:` lines,
the build logs, the diagnostic, the two earlier attempts, the ROCm 10.0
rows (`*rocm10*`), the PR-A-alone rows (`*rocm10a*`) and the opt-in rows (`*rocm10c*`,
with the kernel-log excerpt of the fault).
