# Root cause: no PCIe atomics → no hostcall → dispatch refused

This document is the evidence chain. Every link has an independent test, and the
sections at the end list what was **ruled out** and how.

---

## 1. The causal chain

| # | Link | Independent evidence |
|---|---|---|
| 1 | The root port above the GPU advertises no 32/64-bit AtomicOp **completer** support. *In a guest this is usually configuration rather than a limit — QEMU advertises it automatically since 8.1.0 but not for multifunction devices; see [vfio-atomics.md](vfio-atomics.md).* | guest `lspci -vvs 00:1c.0` → `AtomicOpsCap: Routing- 32bit- 64bit-`, the emulated `pcie-root-port`. Two bits matter here and are easy to conflate: *completion* on the root port, *routing* on any switch port in between. This repository's own host reports `Routing- 32bit+ 64bit+` on all eight root ports and does pass: `amdgpu` bound these same cards on the host across five boots and never emitted the message, while in the guest it emits it for both GPUs at every boot. The passthrough layer removes the capability, not the Zen 1 silicon |
| 2 | → amdgpu therefore disables PCIe atomics | guest `dmesg`: `amdgpu 0000:01:00.0: PCIE atomic ops is not supported` (every GPU). `pci_enable_atomic_ops_to_root()` requires `COMP32`+`COMP64` at the root port and AtomicOp routing on each switch port below it; the emulated root port carries neither completer bit |
| 3 | → ROCr cannot build a hostcall buffer | `AMD_LOG_LEVEL=4` at the exact failure: `rocvirtual.cpp:4208 Pcie atomics not enabled, hostcall not supported` → `4636 AQL dispatch failed!` → `hipErrorIllegalState`. Hostcall's ring signalling depends on atomics |
| 4 | → any kernel *needing* hostcall is refused | `diagnose/hipgate3.cpp`: identical launch path, two kernels. Plain kernel **passes**; kernel with device `printf` (which requires hostcall) **fails with the exact production error** |
| 5 | → RCCL ≥ 2.27.7-b43 device kernels need hostcall | `llvm-readelf --notes` on the device image: every `ncclDevKernel_Generic*` carries `hidden_hostcall_buffer`. Source: device-side `assert()` throughout `src/device/` links `__assert_fail`; `ENABLE_COLLTRACE` adds device `printf` → `__ockl_fprintf` |

Step 4 is the one that matters. It removes RCCL, PyTorch and vLLM from the
picture entirely and reduces the whole failure to **~30 lines of HIP**.

```
--- dev0 ---
[plain    ] launch:no error   sync:no error
[hostcall ] launch:no error   sync:the operation cannot be performed in the present state
```

Verbatim failure window from `AMD_LOG_LEVEL=4`, identical on both ranks
(full excerpt in [`diagnose/logs/amdlog-crash-excerpt.txt`](../diagnose/logs/amdlog-crash-excerpt.txt)):

```
rocvirtual.cpp:4151  ShaderName : ncclDevKernel_Generic_4(ncclDevKernelArgsStorage<4096ul>)
rocvirtual.cpp:4208  Pcie atomics not enabled, hostcall not supported
rocvirtual.cpp:4636  AQL dispatch failed!
hip_module.cpp:605   hipModuleLaunchKernel: Returned hipErrorIllegalState
```

---

## 2. Why downgrading appears to work

Community reports say "downgrade librccl to the ROCm 7.1.1 build and it works."
That is true, and it is *consistent with* — in fact evidence *for* — this root cause.

We dissected the shipped libraries:

| Shipped RCCL | `hidden_hostcall_buffer` count | Behaviour on a no-atomics platform |
|---|---|---|
| ROCm 7.0 (2.26.6) | 0 | works |
| ROCm 7.1.1 (2.27.7-b38) | **0** | works |
| ROCm 7.2+ (2.27.7-b43+) | N (every Generic kernel) | fails |
| ROCm 7.13 / 7.14 (2.30.4) | N | fails |

The platform is identical in all four cases. **What changes is whether the
library's kernels demand a hostcall.** So "downgrade fixes it" does not mean the
newer RCCL is broken in general — it means the newer RCCL requires a platform
capability that these machines never had.

This also unifies the bare-metal reports with the VFIO ones: if those bare-metal
platforms *did* route AtomicOps, the newer RCCL's hostcall requirement would be
satisfiable and downgrading would not be necessary. The fact that downgrading
helps implies those platforms also lack atomics.

---

## 3. What was ruled out

Each of these was a live hypothesis, tested and discarded. Listing them saves
the next person from re-running them.

| Hypothesis | Verdict | How it was ruled out |
|---|---|---|
| Guest RAM too small | ❌ | Raised 16 G → 22 G, identical signature; no OOM in dmesg, headroom at crash |
| NUMA asymmetry | ❌ | Two public bare-metal reports are single-NUMA desktops (7800X3D, 13900K) and reproduce |
| VFIO/virtualization itself | ❌ **not necessary** | independently reproduced on bare metal with IOMMU entirely disabled, by @adderek in [ROCm#6520](https://github.com/ROCm/ROCm/issues/6520): 2× RX 7900 XTX on a B550 board, same failure, same fix. Their machine has one affected GPU (chipset-attached) and one healthy one (CPU-direct) |
| vLLM version | ❌ | 0.19.1 and 0.23 fail, 0.11.2 passes — it is only the caller |
| RCCL version regression (b38 ↔ b81) | ❌ *as the primary cause* | Our own **b38 built from source** still failed — because our build did not get `NDEBUG` to the device pass either. The variable is hostcall, not version |
| Guest kernel too old (6.8) | ❌ | Upgraded to HWE 7.0.0-28, identical signature |
| Missing `iommu=pt` | ❌ | Read the source: string check only, warns but does not gate |
| Event flags / `ReleaseToSystem` | ❌ | Read the source: both generations use an ordinary `DisableTiming` event |
| HIP runtime ≥7.13 gating kernel submission | ❌ **explicitly disproven** | `diagnose/hipgate.cpp` issues the exact `hipExtLaunchKernel` + completion-event form, 4 KB kernarg, cross-stream wait, fine-grained host memory — **all pass** on 7.14 |
| Dynamic shared-memory size | ❌ | `diagnose/hipgate2.cpp`, 0 → 64 KB, all pass (over-limit returns a correctly *different* error) |
| Transport / env tuning | ❌ | `diagnose/sweep.sh`: 11 combinations of `NCCL_P2P_DISABLE`, `HSA_ENABLE_SDMA`, `NCCL_SHM_DISABLE`, `NCCL_CUMEM_ENABLE`, `HSA_FORCE_FINE_GRAIN_PCIE`, `GPU_MAX_HW_QUEUES`, `AMD_SERIALIZE_KERNEL` — signature unchanged in all |
| Pipeline parallel instead of tensor parallel | ❌ **cannot dodge it** | `diagnose/ar_p2p.py`: `ncclSend/Recv` kernels need hostcall too, same crash |
| **No PCIe atomics → hostcall unavailable** | ✅ **root cause** | The whole of §1, and decisively `hipgate3.cpp` |

---

## 4. Why removing the hostcall requirement is safe

Both sources of the requirement are **pure debug facilities that never execute on
the working path**:

- **Device-side `assert()`** — scattered through `src/device/` (`all_gather.h`,
  `reduce_scatter.h`, `common_kernel.h`, `primitives.h`, `prims_simple.h`, …).
  Compiling with `NDEBUG` makes them no-ops, exactly as the C standard specifies.
- **`ENABLE_COLLTRACE`** — a collective-trace device `printf`, controlled by
  `option(COLLTRACE ... ON)` in RCCL's `CMakeLists.txt`.

Neither participates in collective correctness. Removing them changes no data
path — it only removes the ability to print a message from device code at the
moment a debug assertion would have fired.

The strongest argument that this is safe: **AMD's own shipped ROCm 7.1.1 RCCL has
hostcall count 0**, from the same source tree. A no-hostcall RCCL is a
configuration AMD has already shipped and supported.

---

## 5. Applying the fix

> **Scope, before you spend 85 minutes.** Everything above is the mechanism, and
> it holds wherever atomics are missing. This section is the fix for hardware
> that genuinely cannot deliver them. **If your GPUs are passed through to a VM,
> they probably can**, and the reason they appear not to is usually that the
> card's audio function was passed alongside it, which stops QEMU advertising
> completer support on the root port. That is a one-line change with no rebuild
> — see [vfio-atomics.md](vfio-atomics.md). Rule it out first.

> **Version scope.** Everything in this section applies to **RCCL 2.27.7**
> (`ROCm/rccl`, branch `release/rocm-rel-7.1.1.1`), which we verified end to end.
> It is **not sufficient for RCCL 2.30.4**: there `NDEBUG` removes the asserts and
> the linked image ends up with zero `__ockl_*` symbols, yet the device linker
> still declares `hidden_hostcall_buffer` and ROCr still refuses the dispatch.
> Tested on hardware — see [open-questions.md §0](open-questions.md).

The important subtlety, and the reason a naive `-DCMAKE_BUILD_TYPE=Release` is
not enough:

> RCCL's device compilation uses its own `-O3` target flags that **bypass
> `CMAKE_CXX_FLAGS_RELEASE`**, so `NDEBUG` never reaches the device pass.

Therefore the fix must be applied **globally**, not through the Release flags:

```cmake
project(rccl CXX)
add_compile_definitions(NDEBUG)   # must reach the device compilation pass
```

Acceptance test — this is the number that matters:

```bash
llvm-readelf --notes librccl.so.1.0.*gfx1100 | grep -ic hidden_hostcall_buffer
# MUST print 0
```

See [`build/build-rccl-nohostcall.sh`](../build/build-rccl-nohostcall.sh) and
[`build/verify-nohostcall.sh`](../build/verify-nohostcall.sh).

Note that removing the hostcall requirement is necessary but **not sufficient to
run vLLM** — see [deploy-vllm.md](deploy-vllm.md) for the two further traps
(`librocm_smi64` poisoning `amdsmi` enumeration, and the stub shadowing it back).

---

## 6. Result

Same hardware, same VFIO guest, same ROCm 7.14 container, RCCL rebuilt with no
hostcall:

- vLLM `--tensor-parallel-size 2` initialises, profiles, **captures 86 CUDA
  graphs through the exact code path that used to crash** (`enqueue.cc:2061`),
  allocates KV cache, serves.
- gemma-4-31B (w4a16) at **43.2 tok/s** decode, both GPUs at 265 W *synchronised* —
  i.e. genuine tensor parallelism, not one card waiting on the other.
- Achieved on a deliberately hostile topology: cross-die, PCIe 3.0, no P2P,
  `NCCL_P2P_DISABLE=1`, inside a VM, with no PCIe atomics at all.

PCIe atomics are irrelevant to inference performance — decode is
bandwidth-bound and prefill never touches them. They were only ever a
precondition for hostcall, which is a debug facility, and AMD shipped a build
without it. The step from there to "removing the dependency costs nothing" is
an argument, not a measurement: no A/B between the stock library with atomics
and this one without was run. [open-questions.md §6](open-questions.md) records
it as assumed.
