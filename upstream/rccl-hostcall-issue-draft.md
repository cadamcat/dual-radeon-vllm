<!--
INTERNAL NOTE — not part of the issue text. HTML comments do not render on GitHub,
so the whole file can be pasted as-is.

Target: a NEW issue on ROCm/ROCm, plus a two-line pointer comment on #6074.
Status: not yet submitted. Re-verified on hardware 2026-07-26.
-->

# [Issue]: Root cause for "operation cannot be performed in the present state" on multi-GPU Radeon — RCCL device kernels declare hostcall, hostcall needs PCIe AtomicOps

Opening this separately from #6074 because it is a root-cause analysis rather than
another report of the same symptom, and because the title is what people hitting this
will search for. #6074 and vllm-project/vllm#38587 are, I believe, the same bug.

## Summary

Every multi-GPU collective fails on affected machines with some combination of:

```
RuntimeError: NCCL error: unhandled cuda error
HIP failure 'the operation cannot be performed in the present state' at enqueue.cc:2061
NCCL WARN cuMem support requires VMM RDMA support
```

This is not an RCCL version bug and not a virtualisation bug. The chain is:

```
Root complex does not route PCIe AtomicOps
   (consumer chipsets, and every QEMU emulated pcie-root-port)
        ▼
amdgpu disables PCIe atomics        dmesg: "PCIE atomic ops is not supported"
        ▼
ROCr cannot establish a hostcall buffer (its signalling needs atomics)
        ▼
any kernel that *declares* hidden_hostcall_buffer is refused at dispatch
        ▼
RCCL >= 2.27.7-b43 device kernels declare it, because of device-side assert(),
a debug facility that never runs on the happy path
        ▼
the first cross-GPU collective dies with hipErrorIllegalState
```

Rebuilding RCCL 2.27.7 with `-DNDEBUG` removes the declaration and the problem goes
away: gemma-4-31B at 43.2 tok/s under vLLM TP=2 on 2× RX 7900 XT, both GPUs at 265 W
synchronised. That is a workaround, not a fix, and it does not help on 2.30.4 — see
below.

## Minimal reproducer

30 lines of HIP. No RCCL, no PyTorch, no vLLM, one GPU is enough. Two kernels with
identical launch shape; the only difference is that the second needs a hostcall
because it calls `printf`. Run today on ROCm 7.14:

```
devices=2
--- dev0 ---
[plain    ] launch:no error               sync:no error
[hostcall ] launch:the operation cannot be performed in the present state sync:the operation cannot be performed in the present state
--- dev1 ---
[plain    ] launch:no error               sync:no error
[hostcall ] launch:the operation cannot be performed in the present state sync:the operation cannot be performed in the present state
```

Source: `diagnose/hipgate3.cpp` in https://github.com/2462381442/dual-radeon-vllm

Driver-level confirmation, `AMD_LOG_LEVEL=4`, at the moment RCCL fails:

```
ncclDevKernel_Generic
rocvirtual.cpp:4208  Pcie atomics not enabled, hostcall not supported
rocvirtual.cpp:4636  AQL dispatch failed!
                     -> hipErrorIllegalState
```

## Why this has been hard to reproduce

In #6074, @darren-amd could not reproduce it on the nightly container, and the thread
has since spent several months on PCIe slot layout, kernel versions, container images
and Resizable BAR. All of those are downstream of the one variable that decides it:
whether the root port routes AtomicOps. A machine whose root complex routes them will
never show this, whichever RCCL build it runs, which is what "cannot reproduce" looks
like from that side.

The slot discussion in that thread is close but aimed at the wrong property. It is not
bandwidth (x16 versus x4), it is AtomicOp routing.

One command tells the two apart, and it needs no ROCm:

```bash
dmesg | grep "PCIE atomic"
# affected:     amdgpu 0000:01:00.0: PCIE atomic ops is not supported
# not affected: no output
```

Grep the exact phrase; a bare `grep -i atomic` also matches unrelated
`DMA: preallocated ... pool for atomic allocations` lines from early boot.

On the machine in this report, re-checked today:

```
[    9.044936] amdgpu 0000:01:00.0: PCIE atomic ops is not supported
[    9.568264] amdgpu 0000:02:00.0: PCIE atomic ops is not supported

0000:00:1c.0: AtomicOpsCap: Routing- 32bit- 64bit- 128bitCAS-
```

`Routing-` on the root port is the upstream cause of the amdgpu line. Every QEMU
`pcie-root-port` reports it that way regardless of the host CPU.

## NDEBUG is not sufficient on 2.30.4, which is what ROCm 7.13/7.14 ship

We built RCCL 2.30.4 (`ROCm/rocm-systems`, `projects/rccl`) with the same `NDEBUG`
patch, for seven architectures. It still fails, at `hipify/src/enqueue.cc:2118`.
Static inspection of that build:

| check on the linked device image | result |
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

A workaround we have not tried: strip the implicit-argument entry from the kernel
metadata after linking. RCCL's own tool already rewrites that region.

## Two conclusions of ours that turned out to be wrong

Recorded because they cost us several days and may be costing others the same.

1. *"It is an RCCL regression between 2.27.7-b38 and -b43."* Plausible, since a
   downgrade does fix it, but building b38 from source on the newer runtime fails the
   same way. The version correlation is real, the causation is not: newer builds ship
   `assert()` in device code, which is what pulls in hostcall.
2. *"ROCm ≥ 7.13's HIP runtime gates kernel submission in VFIO guests."* Disproved by
   a minimal probe: `hipExtLaunchKernel` with a completion event, 4 KB kernargs and
   cross-stream waits all pass on the same runtime. Submission is fine; only
   hostcall-carrying kernels are refused.

13 hypotheses were tested and 12 eliminated (kernel version, `iommu=pt`, event flags,
memory, NUMA, SDMA, P2P, `cuMem`, shared-memory size, transport layer, and others).
The full matrix is in `docs/root-cause.md` in the repository above.

## Scope: not Radeon-only, not VM-only

The trigger is the root complex, so this reaches:

- consumer chipsets whose root ports do not route AtomicOps. The two bare-metal
  reports in #6074 and vllm#38587 are this case, on single-NUMA desktops (7800X3D and
  13900K) with 2× 7900 XTX;
- any VFIO/QEMU passthrough guest, because the emulated `pcie-root-port` reports
  `AtomicOpsCap: Routing-` regardless of what the host CPU supports. That includes
  virtualised Instinct.

## What would help

1. Confirmation of the mechanism. The `hipgate3` result takes minutes on any affected
   machine, and does not need RCCL.
2. On 2.27.7: should device `assert()` pull in hostcall in a release build? If the
   release artefacts were built with `-DNDEBUG`, or the asserts compiled out for
   device code, this class of hardware would work out of the box. We did not test
   `COLLTRACE=OFF` on its own, which is the supported knob rather than a patch;
   `NDEBUG` happens to remove both sources at once. If disabling COLLTRACE alone is
   sufficient, that would be the cleaner answer and we would be glad to hear it.
3. On 2.30.4: why does the device link declare an implicit argument that nothing in
   the image uses? This is the one that matters for current releases.
4. A clearer failure message. `hipErrorIllegalState` at `enqueue.cc` sends people
   hunting through RCCL versions and environment variables; "hostcall unavailable:
   PCIe atomics disabled" would end the search immediately.

## Environment

- 2× Radeon RX 7900 XT (gfx1100), cross-die, PCIe 3.0, no P2P
- Host: Threadripper 1950X / X399; Proxmox VE + QEMU 11.0.2, VFIO passthrough
- ROCm 7.14, PyTorch 2.11, vLLM 0.23. Also reproduced on ROCm 7.13 and 7.0.0
- Last working combination before the rebuild: the ROCm 7.0.0 image, whose RCCL 2.26.6
  device kernels predate the hostcall dependency

Reproducers, the rebuild script, the deployment steps and prebuilt libraries:
https://github.com/2462381442/dual-radeon-vllm
