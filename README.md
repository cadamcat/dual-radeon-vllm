# rccl-nohostcall

**RCCL multi-GPU collectives crash on any platform without PCIe AtomicOps routing — here's why, how to prove it in 60 seconds, and how to fix it.**

Affects dual/multi-GPU RCCL on consumer Radeon (gfx1100 / RX 7900 XT / XTX) and **any VFIO / QEMU passthrough guest**, including virtualized Instinct. Verified on ROCm 7.2 → 7.14.

---

## Does this affect you?

You are in the right place if you see any of these:

```
RuntimeError: NCCL error: unhandled cuda error
NCCL WARN HIP failure 'the operation cannot be performed in the present state'
  at .../src/enqueue.cc:2061   (or :1750 on older RCCL)
NCCL WARN cuMem support requires VMM RDMA support
```

and, with `AMD_LOG_LEVEL=4`, the real cause one line above the failure:

```
rocvirtual.cpp:4151  ShaderName : ncclDevKernel_Generic_4(ncclDevKernelArgsStorage<4096ul>)
rocvirtual.cpp:4208  Pcie atomics not enabled, hostcall not supported
rocvirtual.cpp:4636  AQL dispatch failed!
hip_module.cpp:605   hipModuleLaunchKernel: Returned hipErrorIllegalState
```

Typical trigger: `vllm serve --tensor-parallel-size 2`, `torchrun` with `all_reduce`, or anything that runs its **first cross-GPU collective**. Single-GPU work is always fine — that is the tell.

Related upstream reports (all still open, none with a root cause):
[ROCm#6074](https://github.com/ROCm/ROCm/issues/6074) ·
[ROCm#2429](https://github.com/ROCm/ROCm/issues/2429) ·
[vllm#38587](https://github.com/vllm-project/vllm/issues/38587) ·
[rocm-systems#2779](https://github.com/ROCm/rocm-systems/issues/2779)

---

## 60-second triage

Two commands tell you whether this is your bug.

```bash
# 1. Does your platform route PCIe atomics to root?
./diagnose/check-platform.sh
```

```bash
# 2. The decisive probe: a trivial kernel vs. one that needs a hostcall.
hipcc --offload-arch=gfx1100 -O2 diagnose/hipgate3.cpp -o hipgate3 && ./hipgate3
```

If `plain` passes and `hostcall` fails with *"the operation cannot be performed in the present state"* — **that is this bug, confirmed end to end**, with no RCCL, no PyTorch and no vLLM in the picture.

```
--- dev0 ---
[plain    ] launch:no error   sync:no error
[hostcall ] launch:no error   sync:the operation cannot be performed in the present state
```

---

## What is actually wrong

```
Platform root complex does not advertise AtomicOp routing
        │   consumer desktop chipsets, and QEMU's emulated pcie-root-port
        ▼
amdgpu disables PCIe atomics       →  dmesg: "PCIE atomic ops is not supported"
        ▼
ROCr cannot establish a hostcall buffer (its ring signalling needs atomics)
        ▼
Any kernel carrying `hidden_hostcall_buffer` is REFUSED AT DISPATCH
        ▼
RCCL ≥ 2.27.7-b43 device kernels carry that flag, because device-side
`assert()` (and ENABLE_COLLTRACE's device printf) link in __assert_fail /
__ockl_fprintf — both pure debug facilities, never hit on the happy path
        ▼
First cross-GPU collective dies with hipErrorIllegalState = "present state"
```

**The fix is therefore not a workaround — it is removing a debug facility that should never have shipped enabled.** Rebuild RCCL so its device kernels carry no hostcall, and the dispatch succeeds on exactly the same hardware.

Full evidence chain, including the experiments that ruled out RCCL version, kernel version, `iommu=pt`, memory, and 12 environment-variable combinations: **[docs/root-cause.md](docs/root-cause.md)**.

### Why "it works on some machines" is consistent with this

The shipped RCCL from **ROCm 7.1.1** has `hidden_hostcall_buffer` count **0**; from 7.2+ every `Generic` kernel has it. That is why "downgrade librccl to the 7.1.1 build" is reported to fix it — the working library simply carries no hostcall requirement. The platform never changed. See [docs/root-cause.md#why-downgrading-appears-to-work](docs/root-cause.md).

---

## The fix

Three pieces, **all required** — the second and third are non-obvious and cost days to discover independently:

| # | Piece | Why |
|---|---|---|
| 1 | **RCCL rebuilt with `NDEBUG` reaching the device pass** | Kills device `assert()` → no `__assert_fail` → no `hidden_hostcall_buffer` |
| 2 | **`rsmi` stub + `patchelf`** | RCCL's `librocm_smi64` dependency makes ROCm-PyTorch's `amdsmi` enumeration return **0 devices**. Removing it leaves 9 undefined symbols. A stub satisfies them; RCCL falls back to `alt_rsmi` (sysfs). |
| 3 | **`sitecustomize.py`** | The stub's global `rsmi_*` symbols shadow the real ones inside `amdsmi`, so `amdsmi` reports `NOT_INIT` and vLLM fails platform detection. Pre-binding the real rsmi before torch loads fixes it. **Do not call `amdsmi_shut_down()`.** |

```bash
./build/build-rccl-nohostcall.sh     # rebuild (device LTO link is the slow part)
./build/verify-nohostcall.sh <lib>   # acceptance test: hostcall count MUST be 0
./deploy/deploy-tp2.sh               # inject into a ROCm/vLLM container
```

Step-by-step, including every "false failure" symptom: **[docs/deploy-vllm.md](docs/deploy-vllm.md)**.

---

## Verified vs. inferred

Be precise about this — it decides whether a bug report against us is useful.

| Environment | Status |
|---|---|
| VFIO/QEMU guest · 2× RX 7900 XT (gfx1100) · ROCm 7.14 · vLLM 0.23 · torch 2.11 | ✅ **Verified end to end.** vLLM TP=2 runs; gemma-4-31B-w4a16 at 42 tok/s, both GPUs 264 W synchronised |
| VFIO/QEMU guest · same hardware · ROCm 7.0 (RCCL 2.26.6) | ✅ **Verified working without any fix** — old kernels carry no hostcall |
| VFIO/QEMU guest · same hardware · ROCm 7.13 / 7.14 stock | ✅ **Verified failing** with the signature above |
| Bare metal · 2× RX 7900 XTX · consumer chipset | ⚠️ **Inferred.** Mechanism matches the public reports, but we have no such machine. **Reports welcome** — see below |
| Virtualized Instinct (MI2xx/MI3xx) via passthrough | ⚠️ **Inferred.** Same QEMU root-port limitation should apply |
| Bare metal with a root complex that *does* route AtomicOps | ➖ Should be unaffected; nothing to fix |

**If you run `diagnose/hipgate3.cpp` on any machine, please open an issue with the output and your platform.** That is the single most useful contribution — it turns the inferred rows into verified ones.

---

## Repository map

```
diagnose/   Minimal, dependency-free probes. Start here.
  hipgate3.cpp  ★ plain kernel vs hostcall kernel — the decisive test
  hipgate.cpp     hipExtLaunchKernel + completion event (rules out submission itself)
  hipgate2.cpp    dynamic shared memory 0–64KB (rules out LDS size)
  ar.py           30-second torchrun all_reduce reproducer
  ar_p2p.py       point-to-point send/recv — shows pipeline parallel cannot dodge it
  sweep.sh        12 environment-variable combinations, all ineffective
  check-platform.sh  one-shot: dmesg + AtomicOpsCap + hostcall count
  logs/           AMD_LOG_LEVEL=4 excerpt at the moment of failure

build/      Rebuild RCCL without hostcall, and verify it
deploy/     Inject into a ROCm/vLLM container (the 3 pieces above)
docs/       root-cause · diagnosis · deploy-vllm · open-questions
```

---

## Status and support policy

This is **a documented workaround with reproducible evidence, not a supported product.**

- It exists because the upstream issues have been open for months with no root cause published.
- The correct long-term fix belongs in ROCm's RCCL build. **When upstream ships RCCL with no hostcall in device kernels, this repository becomes obsolete and will be archived.**
- ROCm moves on a ~6-week release cadence, so version drift is expected. Each new ROCm may need a fresh rebuild.
- Issues about *this* mechanism, and platform reports from `hipgate3`, are very welcome. General ROCm/vLLM support questions are out of scope.

**[docs/open-questions.md](docs/open-questions.md)** lists what we deliberately have *not* proven — most importantly *which* upstream change flipped shipped binaries from 0 hostcall (7.1.1) to N (7.2+). If you know, that is the missing piece of the upstream report.

---

## Credits and licence

Original investigation on a Threadripper 1950X / PVE / dual RX 7900 XT home server.
Everything here — probes, build scripts, deployment glue, documentation — is original work, released under the **MIT Licence** ([LICENSE](LICENSE)).

**This repository contains no RCCL source code.** It contains a build recipe and diagnostics. If you distribute a *compiled* RCCL produced by these scripts, that binary is a derivative work of RCCL and carries its BSD-3-Clause obligations — see [NOTICE.md](NOTICE.md).

RCCL is © Advanced Micro Devices, Inc., with portions © NVIDIA CORPORATION, under BSD-3-Clause.
This project is not affiliated with, endorsed by, or supported by AMD.
