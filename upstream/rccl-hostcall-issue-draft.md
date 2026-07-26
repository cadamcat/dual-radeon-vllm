# Root cause for "operation cannot be performed in the present state" on consumer Radeon: RCCL device kernels need hostcall, hostcall needs PCIe atomics

**Draft — not yet submitted.** Target: a comment updating
[ROCm/ROCm#6074](https://github.com/ROCm/ROCm/issues/6074), cross-posted to
[vllm-project/vllm#38587](https://github.com/vllm-project/vllm/issues/38587).
Review before filing.

---

## Summary

The failure that stops multi-GPU RCCL on consumer Radeon —

```
NCCL WARN cuMem support requires VMM RDMA support
HIP failure 'the operation cannot be performed in the present state' at enqueue.cc:2061
RuntimeError: NCCL error: unhandled cuda error
```

is **not** an RCCL version bug and **not** a virtualisation bug. The chain is:

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

**A workaround available today:** rebuild RCCL **2.27.7** with `-DNDEBUG` so the device
kernels no longer carry `hidden_hostcall_buffer`. Verified: `gemma-4-31B` w4a16 at
43.2 tok/s on 2× RX 7900 XT, TP=2, both GPUs at 265 W synchronised.

> ### ⚠️ That workaround does **not** carry over to 2.30.4 — the version shipping today
>
> We built **RCCL 2.30.4** (`ROCm/rocm-systems`, `projects/rccl`) with the same
> `NDEBUG` patch, for seven architectures. It still fails, at
> `hipify/src/enqueue.cc:2118`. Static inspection of that build:
>
> | check on the linked device image | result |
> |---|---|
> | `__assert_fail` | **0** — NDEBUG did its job |
> | `__ockl_fprintf` | **0** — COLLTRACE is gone from 2.30.4 entirely |
> | `__ockl_*` of any kind | **0** — *no hostcall-calling code remains at all* |
> | `hidden_hostcall_buffer` in kernel metadata | **3** — one per `ncclDevKernel_Generic_{1,2,4}` |
>
> **The metadata declaration alone is enough for ROCr to refuse dispatch.** Removing
> every instruction that could use hostcall does not help while the kernel still
> *declares* the implicit argument.
>
> Bisected to the link step:
>
> ```
> device_build/common.o             (before device link)  hostcall = 0
> device_build/gfx1100/device.elf   (after device link)   hostcall = 3
> ```
>
> 2.30.4 replaced the ordinary HIP fat-binary path with its own device linker
> (`tools/rccl-device-compile --link`, which also hand-patches SGPR/VGPR fields in
> the kernel descriptor). That tool contains no `hostcall` or `hidden_` strings, so
> the declaration appears to be emitted during the dispatcher link, conservatively,
> for kernels that demonstrably never call hostcall. Ruled out with trivial-kernel
> probes (all give `hostcall = 0`): plain whole-program compile, `-fgpu-rdc`,
> `-mcode-object-version=4`, and `-fgpu-rdc` + COv4. So it is neither RDC as such
> nor the code-object version.
>
> **The sharper form of this report, then:** *the device-linking step declares
> `hidden_hostcall_buffer` on all three `ncclDevKernel_Generic_*` kernels while the
> linked image contains zero `__ockl_*` symbols; on any platform without PCIe
> AtomicOps that makes every collective fail, and `NDEBUG` cannot fix it.*
>
> A workaround we have **not** tried: strip the implicit-argument entry from the
> kernel metadata after linking. RCCL's own tool already rewrites that region.

## The decisive experiment

30 lines of HIP, no RCCL, no PyTorch, no vLLM. Two kernels, identical launch shape;
the only difference is that the second one needs a hostcall (it calls `printf`).
Re-run today on ROCm 7.14:

```
devices=2
--- dev0 ---
[plain    ] launch:no error               sync:no error
[hostcall ] launch:the operation cannot be performed in the present state sync:the operation cannot be performed in the present state
--- dev1 ---
[plain    ] launch:no error               sync:no error
[hostcall ] launch:the operation cannot be performed in the present state sync:the operation cannot be performed in the present state
```

That is the whole bug, on both cards, with no collective and no second process
involved. Full source: `diagnose/hipgate3.cpp` in
https://github.com/2462381442/dual-radeon-vllm

### Why this has been hard to reproduce at AMD

`darren-amd` reported being unable to reproduce on the nightly container, and the
thread has since spent months on PCIe slot layout, kernel versions, container images
and Resizable BAR. Those are all downstream of the one variable that decides it:
**whether the root port routes AtomicOps**. A machine whose root complex routes them
will never show this, whichever RCCL build it runs, and that is exactly what "cannot
reproduce" looks like from the other side.

One command separates the two worlds, and it needs no ROCm at all:

```bash
dmesg | grep "PCIE atomic"
# affected:     amdgpu 0000:01:00.0: PCIE atomic ops is not supported
# not affected: no output
```

(Grep for the exact phrase; a bare `grep -i atomic` also matches unrelated
`DMA: preallocated ... pool for atomic allocations` lines from early boot.)

The PCIe slot discussion in this thread is close to the answer but aimed at the wrong
property. It is not bandwidth (x16 versus x4), it is AtomicOp routing.

Driver-level confirmation with `AMD_LOG_LEVEL=4`, at the moment RCCL fails:

```
ncclDevKernel_Generic
rocvirtual.cpp:4208  Pcie atomics not enabled, hostcall not supported
rocvirtual.cpp:4636  AQL dispatch failed!
                     -> hipErrorIllegalState
```

## Two earlier conclusions that were wrong

Recorded because they cost us days and may be costing others the same:

1. **"It is an RCCL regression between 2.27.7-b38 and -b43."** Plausible, since a
   downgrade does fix it, but building b38 from source on the newer runtime fails
   the same way. The version correlation is real, the causation is not: newer builds
   ship `assert()` in device code, which is what pulls in hostcall.
2. **"ROCm ≥ 7.13's HIP runtime gates kernel submission in VFIO guests."** Disproved
   by a minimal probe: `hipExtLaunchKernel` with a completion event, 4 KB kernargs,
   cross-stream waits — all pass on the same runtime. Submission is fine; only
   hostcall-carrying kernels are refused.

13 hypotheses were tested and 12 eliminated (kernel version, `iommu=pt`, event flags,
memory, NUMA, SDMA, P2P, `cuMem`, shared-memory size, transport layer, …). The full
matrix is in `docs/root-cause.md`.

## Scope: this is not Radeon-only and not VM-only

The trigger is the **root complex**, so this hits:

- consumer chipsets whose root ports do not route AtomicOps — the two bare-metal
  reports in #6074 and vllm#38587 are exactly this, on single-NUMA desktops (7800X3D and 13900K) with 2× 7900 XTX;
- **any VFIO/QEMU passthrough guest**, because the emulated `pcie-root-port` reports
  `AtomicOpsCap: Routing-` regardless of what the host CPU supports. That includes
  virtualised Instinct.

Check your own machine:

```bash
dmesg | grep "PCIE atomic"     # any output at all = affected
lspci -vvv -s <root port> | grep AtomicOpsCap
```

On this machine, re-checked today:

```
[    9.044936] amdgpu 0000:01:00.0: PCIE atomic ops is not supported
[    9.568264] amdgpu 0000:02:00.0: PCIE atomic ops is not supported

0000:00:1c.0: AtomicOpsCap: Routing- 32bit- 64bit- 128bitCAS-
```

`Routing-` on the root port is the upstream cause of the amdgpu line; every QEMU
`pcie-root-port` reports it that way regardless of the host CPU.

## What we are asking for

1. **Confirmation of the mechanism** — the `hipgate3` result should be reproducible
   in minutes on any affected machine.
2. **On 2.27.7: should device `assert()` pull in hostcall in a release build?** If
   the release artefacts were built with `-DNDEBUG` (or the asserts compiled out for
   device code), this class of hardware would work out of the box: a packaging
   decision with a large blast radius.
   *(We did not test `COLLTRACE=OFF` on its own, which is the supported knob rather
   than a patch; `NDEBUG` happens to remove both sources at once. If disabling
   COLLTRACE alone is sufficient on 2.27.7, that would be the cleaner answer and we
   would be glad to be told so.)*
3. **On 2.30.4: why does the device link declare an implicit argument nothing uses?**
   That is the box above, and it is the one that matters for current releases.
4. **A clearer failure message.** `hipErrorIllegalState` at `enqueue.cc` sends
   everyone hunting through RCCL versions and env vars. "hostcall unavailable: PCIe
   atomics disabled" would end the search immediately.

## Environment

- 2× Radeon RX 7900 XT (gfx1100), cross-die, PCIe 3.0, no P2P
- Host Threadripper 1950X / X399; Proxmox VE + QEMU 11.0.2, VFIO passthrough
- ROCm 7.14, PyTorch 2.11, vLLM 0.23; also reproduced on ROCm 7.13 and 7.0.0
- Working combination before the fix: ROCm 7.0.0 image (RCCL 2.26.6, whose device
  kernels predate the hostcall dependency), giving TP=2 at 1.80x on Qwen3-8B. That
  figure comes from the older stack and a different measurement method; the current
  five-model campaign measures 1.70x.

Reproducers, the rebuild script, and the deployment steps:
https://github.com/2462381442/dual-radeon-vllm
