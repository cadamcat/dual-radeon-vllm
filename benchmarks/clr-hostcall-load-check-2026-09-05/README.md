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
inputs first meet and gives it a name. Item 3 (a null-buffer fallback) and
item 4 (the toolchain) are not implemented here.

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

## What this licenses, and what it does not

**Licensed.** On this platform, with this runtime commit, moving the check to
kernel init and naming the error is forty-five lines and changes nothing when the
capability is present; when it is absent the refusal names the kernel, the
device, the attribute and the reason, at load and at launch, and the
collective library's own error path carries the sentence through
(`enqueue.cc:2061`) with no change to RCCL or torch. The thirteen kernels the
runtime names are the thirteen the static scan counted.

**Not licensed.** That a real submission would use error code 1055, or log
at `LOG_ERROR`, or refuse in `ihipLaunchKernel_validate` rather than at
`hipModuleGetFunction`; those are choices for the runtime's maintainers. The
fallback (item 3) and the toolchain change (item 4) remain proposals. The cost
of the check was not measured; it is one pass over a kernel's hidden
arguments at init. One host, one architecture, one runtime commit.

## Reproducing

    # CPU only, ~15 min in a throwaway container of the vLLM 0.23 image; writes /rb/clr-build
    docker run --rm --entrypoint bash -v /data/rccl-build:/rb <image> /rb/clr_build.sh /rb/clr-hostcall-load-check.patch
    # each state takes the lease, ~1 min; the flips are tools/pve_flip.sh in the workspace
    bash clr_demo_row.sh atomics_present
    bash clr_demo_row.sh atomics_absent
    python3 analyze.py                     # the table from logs/clr-demo.jsonl, non-zero if a cell is missing

`logs/` holds every row, every per-cell log with the runtime's `:1:` lines,
the build log, the diagnostic, and the two earlier attempts.
