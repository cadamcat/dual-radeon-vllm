<!-- Cover image goes here once finalised:
     ![dual-radeon-vllm](docs/assets/cover.jpg) -->

# dual-radeon-vllm

**Running vLLM with tensor parallelism on two consumer Radeon cards — verified end to end, including the RCCL bug that stops most people before they start.**

`gemma-4-31B` (w4a16) at **42 tok/s** decode on 2× RX 7900 XT, both GPUs at 264 W *synchronised*, inside a VFIO virtual machine with **no P2P, no PCIe atomics, cross-die PCIe 3.0** — deliberately the least favourable topology.

<table>
<tr>
<td><b>42 tok/s</b><br><sub>gemma-4-31B w4a16, TP=2</sub></td>
<td><b>1.80×</b><br><sub>TP=2 speed-up, 90% efficiency</sub></td>
<td><b>51%</b><br><sub>memory-bandwidth utilisation at decode</sub></td>
<td><b>264 W × 2</b><br><sub>synchronised = real tensor parallel</sub></td>
</tr>
</table>

---

## Who this is for

You have **two AMD consumer GPUs** and want `--tensor-parallel-size 2` to actually work. You are probably here because of one of these:

- 🔴 **It crashes immediately**: `NCCL error: unhandled cuda error`, or
  `HIP failure 'the operation cannot be performed in the present state'`.
  → **[Jump to the 60-second triage](#am-i-hit-by-the-rccl-bug)**. This is the single biggest blocker on consumer Radeon, it has been open upstream for months with no published root cause, and this repository explains and fixes it.
- 🟡 **It runs, but you do not know what to expect** → [What performance to expect](#what-performance-to-expect)
- 🟢 **You are deciding whether to buy/build this** → [What does *not* work](#what-does-not-work) first, please

**Not a vLLM fork.** Nothing here patches vLLM. The fix lives in RCCL — one library, one rebuild — so there is no upstream to keep rebasing against.

---

## Verified configuration

Everything below was measured on this machine. Nothing is extrapolated.

| | |
|---|---|
| **GPUs** | 2× Radeon RX 7900 XT (gfx1100, RDNA3), 20 GB each = 40 GB |
| **Interconnect** | Cross-die, PCIe 3.0, **no P2P**, `NCCL_P2P_DISABLE=1` |
| **Host** | Threadripper 1950X (Zen 1), X399 |
| **Virtualisation** | Proxmox VE + QEMU, **VFIO passthrough** (no PCIe atomics — see below) |
| **Stack** | ROCm 7.14 · vLLM 0.23 · PyTorch 2.11 · **RCCL 2.27.7 rebuilt** |

> The topology is intentionally hostile. If it works here, a bare-metal box with
> P2P should do at least as well.

---

## Am I hit by the RCCL bug?

Two commands. The second is decisive and needs no RCCL, no PyTorch, no vLLM.

```bash
./diagnose/check-platform.sh          # dmesg + AtomicOpsCap + hostcall count
```

```bash
hipcc --offload-arch=gfx1100 -O2 diagnose/hipgate3.cpp -o hipgate3 && ./hipgate3
```

**You are affected if a trivial kernel passes and a hostcall-needing kernel fails:**

```
[plain    ] launch:no error   sync:no error
[hostcall ] launch:no error   sync:the operation cannot be performed in the present state
```

<details>
<summary><b>What is actually wrong</b> (click to expand)</summary>

```
Root complex does not route PCIe AtomicOps
   (consumer chipsets, and every QEMU emulated pcie-root-port)
        ▼
amdgpu disables PCIe atomics    →  dmesg: "PCIE atomic ops is not supported"
        ▼
ROCr cannot establish a hostcall buffer (its signalling needs atomics)
        ▼
Any kernel that *declares* hidden_hostcall_buffer is refused at dispatch
        ▼
RCCL ≥ 2.27.7-b43 device kernels declare it — because of device-side assert(),
a debug facility that never runs on the happy path
        ▼
The first cross-GPU collective dies with hipErrorIllegalState
```

Confirmed at the driver level with `AMD_LOG_LEVEL=4`:

```
rocvirtual.cpp:4208  Pcie atomics not enabled, hostcall not supported
rocvirtual.cpp:4636  AQL dispatch failed!
```

**This is not a Radeon-only problem.** The trigger is the root complex, so it
also hits **any VFIO/QEMU passthrough guest — including virtualised Instinct.**

Full evidence chain, and the 13 hypotheses we eliminated:
**[docs/root-cause.md](docs/root-cause.md)**

</details>

**→ Fix it: [docs/deploy-vllm.md](docs/deploy-vllm.md)** · **→ Not sure yet: [docs/diagnosis.md](docs/diagnosis.md)**

> ### ⚠️ Build RCCL **2.27.7**, not the newest source
>
> | RCCL source | `NDEBUG` patch | Status |
> |---|---|---|
> | **2.27.7** — `ROCm/rccl`, `release/rocm-rel-7.1.1.1` | hostcall → **0** | ✅ **Verified working** |
> | 2.30.4 — `ROCm/rocm-systems`, `projects/rccl` | hostcall still **3** | ❌ **Verified failing** |
>
> On 2.30.4 the device linker declares the hostcall buffer even though the linked
> image contains *zero* `__ockl_*` symbols. `NDEBUG` cannot fix that.
> [Details](docs/open-questions.md). Note RCCL **moved**: `ROCm/rccl` is now
> `develop_deprecated`; development continues in the `rocm-systems` monorepo.

---

## What performance to expect

Measured on the configuration above. Prompt is a fixed reasoning task; every run
uses a random prefix to defeat prefix caching. Decode figures are steady-state
over 1536-token outputs.

### vLLM, tensor parallel

| Model | Precision | Decode | Notes |
|---|---|---|---|
| **gemma-4-31B-it** | w4a16 QAT | **42.2 tok/s** | 🟢 the sweet spot. 264 W × 2 synchronised |
| Qwen3-8B | BF16 | **55.4 tok/s** | 🟢 TP=1 → TP=2 is 30.8 → 55.4 = **1.80×** |
| Qwen3.6-27B | AWQ int4 | 11.8 tok/s | 🔴 hybrid-SSM architecture, see below |
| gemma-4-26B-A4B | int4 | ~15 tok/s | 🟡 128-expert MoE, eager only, see below |

### llama.cpp, for comparison

| Model | Mode | Decode |
|---|---|---|
| gemma-4-12B | single card, Vulkan | **64.9 tok/s** |
| gemma-4-31B | dual card, Vulkan layer split | 27.0 tok/s |
| Qwen3.6-27B | dual card, Vulkan layer split | 27.7 tok/s |
| **Qwen3.6-27B** | **+ MTP speculative decoding** | **34.5 tok/s** 🟢 |

### How to read this

- **Dense models are the sweet spot.** vLLM TP=2 beats llama.cpp by ~55% on
  gemma-4-31B (42.2 vs 27.0).
- **For Qwen3.5/3.6, llama.cpp currently wins by ~3×** (34.5 vs 11.8). Those are
  hybrid SSM / gated-delta-net models, and vLLM's ROCm path for them is a
  NVIDIA-tuned Triton kernel that falls onto a degraded tile size on gfx1100.
- **Small models: pin to one card.** Layer-splitting a 12B across two cards is
  *slower* than one card (45.7 vs 64.9) — it serialises.
- Utilisation at decode: **~51% of theoretical memory bandwidth**, ~1.2% of FP16
  compute. Decode is bandwidth-bound; this is expected, not a defect.
- Prefill saturates at **~37% of FP16 peak** (~77 of 206 TFLOP/s).

More benchmarks are planned across models and context lengths.

---

## What does *not* work

Stating this plainly is the point of the repository.

| | Status |
|---|---|
| **FP8 weights/KV** | 🔴 Not available. FP8 is MI300+; RDNA3 has no FP8 path |
| **AITER kernels** | 🔴 Gated to `is MI3XX` in vLLM. gfx1100 silently falls back to Triton |
| **Tuned fused-MoE configs** | 🔴 vLLM ships none for *any* AMD GPU. MoE runs a generic default |
| **Hybrid SSM (Qwen3.5/3.6)** | 🔴 ~3× slower than llama.cpp. Use llama.cpp + MTP instead |
| **MoE `torch.compile`** | 🟡 vLLM hardcodes `TORCHINDUCTOR_COMPILE_THREADS=1`; a 128-expert graph took 20+ min on a slow CPU. Patch it or use `--enforce-eager` |
| **Multi-tenant serving** | 🟡 Untested. Everything here is single-stream or light concurrency |
| **P2P between cards** | 🔴 Not on this topology. Everything measured is *without* it |
| RCCL 2.30.4 | 🔴 See the warning above |

Background on the SSM and MoE findings, with source-level evidence:
**[docs/architecture-notes.md](docs/architecture-notes.md)**

---

## Hardware notes

**Why does a VM lack PCIe atomics?** PCIe atomic operations must be *completed*
by the root complex and *routed* by every bridge in between. QEMU's emulated
`pcie-root-port` does not implement AtomicOp routing at all, so the guest sees
`AtomicOpsCap: Routing-` and amdgpu switches atomics off. Many consumer desktop
root complexes do not advertise it either — which is why bare-metal users hit
the same bug.

**Do I need atomics for inference?** No. They are a precondition for *hostcall*,
which is a debug facility (device `printf`/`assert`). Removing that dependency
costs nothing at runtime — AMD's own ROCm 7.1.1 build shipped with zero hostcall.

**Two identical cards?** Strongly preferred. Mixed cards are limited by the
smaller/slower one, and layer-splitting makes the older card the thermal hotspot.

**Thermals — do not skip this.** Two cards in adjacent slots: the upper one
inhales the lower one's exhaust. We measured **junction 99–100 °C** on the upper
card at sustained load. One 120 mm fan aimed at the gap between the cards
dropped it to **90 °C** and *inverted* which card runs hotter — while its own
fan spun **slower**. Cheapest fix in the entire build.

**Slow host CPU?** It shows up at startup, not at decode: `torch.compile` is
CPU-bound and vLLM pins it to one thread. Weight loading is also
single-threaded.

**RAM ceiling.** vLLM `mmap`s the whole checkpoint. A 21.67 GiB file will not
map into a 21.43 GiB guest even with plenty free — the limit is `MemTotal`, not
available memory. Add swap or raise the VM's RAM.

---

## Repository map

```
diagnose/     Start here. Dependency-free probes
  hipgate3.cpp     ★ plain kernel vs hostcall kernel — decisive, ~30 lines
  check-platform.sh  one-shot triage: dmesg + AtomicOpsCap + hostcall count
  ar.py            30-second torchrun all_reduce reproducer
  sweep.sh         12 env-var combinations that do NOT help
  logs/            AMD_LOG_LEVEL=4 capture at the moment of failure

build/        Rebuild RCCL without hostcall, and verify it
  verify-nohostcall.sh   hostcall / assert / fprintf must all be 0
  check-symbols.sh       all 38 nccl symbols PyTorch needs

deploy/       Inject into a ROCm/vLLM container (3 pieces, all required)
docs/         root-cause · diagnosis · deploy-vllm · architecture-notes · open-questions
```

---

## Status and support policy

**A reproducible engineering record, not a supported product.**

- The RCCL fix exists because upstream has been silent for months. **When ROCm
  ships an RCCL whose device kernels declare no hostcall, that part becomes
  obsolete** and will be marked as such.
- ROCm releases roughly every 6 weeks; expect version drift. Binaries are tied to
  **both** an architecture and a ROCm version.
- Verified on **gfx1100 only**. Prebuilt binaries cover more architectures
  because they cost nothing extra to compile — they are **not verified**.
- Welcome: `hipgate3` output from any machine, benchmark numbers, corrections.
  Out of scope: general ROCm/vLLM support.

[**docs/open-questions.md**](docs/open-questions.md) lists what we deliberately
have *not* proven — including which upstream change flipped shipped binaries
from zero hostcall to three.

---

## Credits and licence

Original investigation on a home Proxmox server with dual RX 7900 XT.
Probes, build scripts, deployment glue and documentation are original work,
released under the **MIT Licence** ([LICENSE](LICENSE)).

**This repository contains no RCCL source code** — only a build recipe and
diagnostics. A *compiled* RCCL produced by these scripts is a derivative work of
RCCL and carries its BSD-3-Clause obligations; see [NOTICE.md](NOTICE.md).

RCCL is © Advanced Micro Devices, Inc., portions © NVIDIA CORPORATION, BSD-3-Clause.
Not affiliated with, endorsed by, or supported by AMD.
