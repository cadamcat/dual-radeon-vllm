<!--
INTERNAL NOTE — not part of the issue text. HTML comments do not render on GitHub,
so the whole file can be pasted as-is.

Filed as ROCm/ROCm#6520 on 2026-07-26. Rewritten the same day after finding
harkgill-amd's 2026-07-21 comment in #6074: AMD had already identified the atomics
dependency, so this is framed around the mechanism and the 2.30.4 consequence rather
than as a root-cause announcement.

Suggested title:
[Issue]: "operation cannot be performed in the present state" — on RCCL 2.30.4 the
hostcall declaration survives NDEBUG and a device image with zero __ockl_* symbols
-->

Following up on #6074, where @harkgill-amd narrowed this down to a PCIe atomics
dependency introduced in RCCL with ROCm 7.2.1, and said the open question is whether
to keep atomics as an explicit dependency or remove it.

This report is about **where the dependency actually lives**, because it turns out not
to be in the code. On RCCL 2.30.4 the failure survives a build with zero
hostcall-calling instructions, which means removing the device-side `assert()` would
not be sufficient. That seemed worth its own issue rather than a comment.

## The part that bears on the keep-or-remove decision

We built RCCL 2.30.4 (`ROCm/rocm-systems`, `projects/rccl`) with
`add_compile_definitions(NDEBUG)`, for seven architectures. It still fails, at
`hipify/src/enqueue.cc:2118`. Static inspection of the linked device image:

| check | result |
|---|---|
| `__assert_fail` | 0 — NDEBUG did its job |
| `__ockl_fprintf` | 0 — COLLTRACE is gone from 2.30.4 entirely |
| `__ockl_*` of any kind | 0 — no hostcall-calling code remains at all |
| `hidden_hostcall_buffer` in kernel metadata | 3 — one per `ncclDevKernel_Generic_{1,2,4}` |

So the metadata declaration alone is enough for ROCr to refuse dispatch. Removing
every instruction that could use hostcall does not help while the kernel still
declares the implicit argument.

Bisected to the link step:

```
device_build/common.o             (before device link)  hostcall = 0
device_build/gfx1100/device.elf   (after device link)   hostcall = 3
```

2.30.4 replaced the ordinary HIP fat-binary path with its own device linker
(`tools/rccl-device-compile --link`, which also hand-patches SGPR/VGPR fields in the
kernel descriptor). That tool contains no `hostcall` or `hidden_` strings, so the
declaration appears to be emitted during the dispatcher link, conservatively, for
kernels that demonstrably never call hostcall. Ruled out with trivial-kernel probes
(all give hostcall = 0): plain whole-program compile, `-fgpu-rdc`,
`-mcode-object-version=4`, and `-fgpu-rdc` + COv4. So it is neither RDC as such nor
the code-object version.

If the decision is to drop the dependency, the device-link step looks like the place
it has to be dropped. One workaround we have not tried: strip the implicit-argument
entry from the kernel metadata after linking, since RCCL's own tool already rewrites
that region.

## What works today, for anyone arriving here from a search engine

RCCL **2.27.7** rebuilt with `NDEBUG` does remove the declaration, and multi-GPU
works: gemma-4-31B at 43.2 tok/s under vLLM TP=2 on 2× RX 7900 XT, both GPUs at 265 W
synchronised. 2.30.4 is not fixable this way, so 2.27.7 is the only route we have
verified. Build script, deployment steps and prebuilt libraries:
https://github.com/2462381442/dual-radeon-vllm

## A reproducer that needs no RCCL

Useful for triage, and for telling this failure apart from the several other things
that produce the same message. 30 lines of HIP, one GPU is enough. Two kernels with
identical launch shape; the only difference is that the second needs a hostcall
because it calls `printf`:

```
devices=2
--- dev0 ---
[plain    ] launch:no error               sync:no error
[hostcall ] launch:the operation cannot be performed in the present state sync:the operation cannot be performed in the present state
--- dev1 ---
[plain    ] launch:no error               sync:no error
[hostcall ] launch:the operation cannot be performed in the present state sync:the operation cannot be performed in the present state
```

Source: `diagnose/hipgate3.cpp` in the repository above. Driver-level confirmation
with `AMD_LOG_LEVEL=4`, at the moment RCCL fails:

```
ncclDevKernel_Generic
rocvirtual.cpp:4208  Pcie atomics not enabled, hostcall not supported
rocvirtual.cpp:4636  AQL dispatch failed!
                     -> hipErrorIllegalState
```

## Telling affected machines apart from unaffected ones

This may be why the failure has been awkward to reproduce on request: whether a
machine shows it is decided entirely by the root port, not by the GPU, the slot width
or the RCCL build. One command, no ROCm needed:

```bash
dmesg | grep "PCIE atomic"
# affected:     amdgpu 0000:01:00.0: PCIE atomic ops is not supported
# not affected: no output
```

Grep the exact phrase; a bare `grep -i atomic` also matches unrelated
`DMA: preallocated ... pool for atomic allocations` lines from early boot.

On the machine in this report:

```
[    9.044936] amdgpu 0000:01:00.0: PCIE atomic ops is not supported
[    9.568264] amdgpu 0000:02:00.0: PCIE atomic ops is not supported

0000:00:1c.0: AtomicOpsCap: Routing- 32bit- 64bit- 128bitCAS-
```

`Routing-` on the root port is the upstream cause of the amdgpu line. Every QEMU
`pcie-root-port` reports it that way regardless of the host CPU, so this also reaches
any VFIO passthrough guest, including virtualised Instinct. The bare-metal reports in
#6074 and vllm-project/vllm#38587 are single-NUMA desktops (7800X3D and 13900K) with
2× 7900 XTX.

## Two conclusions of ours that were wrong

Recorded because they cost us several days and may be costing others the same.

1. *"It is an RCCL regression between 2.27.7-b38 and -b43."* Plausible, since a
   downgrade does fix it, but building b38 from source on the newer runtime fails the
   same way. The version correlation is real, the causation is not.
2. *"ROCm ≥ 7.13's HIP runtime gates kernel submission in VFIO guests."* Disproved by
   a minimal probe: `hipExtLaunchKernel` with a completion event, 4 KB kernargs and
   cross-stream waits all pass on the same runtime. Submission is fine; only
   hostcall-carrying kernels are refused.

13 hypotheses were tested and 12 eliminated (kernel version, `iommu=pt`, event flags,
memory, NUMA, SDMA, P2P, `cuMem`, shared-memory size, transport layer, and others).
The matrix is in `docs/root-cause.md` in the repository.

## Two smaller asks

- On 2.27.7, we did not test `COLLTRACE=OFF` on its own, which is the supported knob
  rather than a patch; `NDEBUG` happens to remove both sources at once. If disabling
  COLLTRACE alone is sufficient there, that is the cleaner answer and we would be glad
  to hear it.
- `hipErrorIllegalState` at `enqueue.cc` sends people hunting through RCCL versions and
  environment variables for days. A message along the lines of "hostcall unavailable:
  PCIe atomics disabled" would end that search immediately, whatever is decided about
  the dependency itself.

## Environment

- 2× Radeon RX 7900 XT (gfx1100), cross-die, PCIe 3.0, no P2P
- Host: Threadripper 1950X / X399; Proxmox VE + QEMU 11.0.2, VFIO passthrough
- ROCm 7.14, PyTorch 2.11, vLLM 0.23. Also reproduced on ROCm 7.13 and 7.0.0
- Last working combination before the rebuild: the ROCm 7.0.0 image, whose RCCL 2.26.6
  device kernels predate the hostcall dependency
