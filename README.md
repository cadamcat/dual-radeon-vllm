# dual-radeon-vllm

English | [中文](README.zh.md)

**Tensor-parallel vLLM on two consumer Radeon cards (RX 7900 XT, gfx1100, ROCm 7.14), verified end to end — including the RCCL bug that stops most people before they start.**

`gemma-4-31B` (w4a16) decodes at **43 tok/s** on 2× RX 7900 XT with both cards drawing 265 W *at the same time*, and a 26B MoE reaches **108 tok/s** at short context. The machine is a VFIO virtual machine with **no P2P and cross-die PCIe 3.0**, and those figures were measured with **no PCIe atomics** either: deliberately the least favourable topology.

<table>
<tr>
<td><b>43 tok/s</b><br><sub>gemma-4-31B w4a16, TP=2</sub></td>
<td><b>108 tok/s</b><br><sub>gemma-4-26B-A4B MoE, TP=2</sub></td>
<td><b>1.70×</b><br><sub>TP=2 speed-up on BF16, 85% efficiency</sub></td>
<td><b>265 W × 2</b><br><sub>31B, both cards together = real tensor parallel</sub></td>
</tr>
</table>

### What is in here

Three things, each usable on its own:

| | |
|---|---|
| 🔧 **A fix** | The RCCL bug that makes `--tensor-parallel-size 2` fail on consumer Radeon, root-caused to PCIe AtomicOps, with a 30-line reproducer. **On bare metal the fix is one RCCL rebuild** (recipe and deployment script in here); **in a VM it is usually one line of VM configuration** ([here](docs/vfio-atomics.md)). [Start here](#am-i-hit-by-the-rccl-bug) |
| 📊 **The data** | **292 measurements**: five model architectures across eleven context lengths, single vs dual GPU, with the raw per-request records, the runner that produced them, and analysis scripts that need no GPU. [Charts and findings](#what-performance-to-expect) · [`benchmarks/`](benchmarks/) |
| 🔬 **A regression in the kernel Ubuntu shipped for months — now fixed** | Host→device copies collapse to **2 MiB/s** from a writable file mapping whose pages are resident — the path every PyTorch process takes to load a safetensors checkpoint. Traced to a half-applied backport in `7.0.0-28-generic`, **proven by applying the missing commit**, and **fixed in `7.0.0-30.30~24.04.1`**: the same reproducer binary on the same machine goes **16 019.3 ms → 15.3 ms** across the upgrade ([data](benchmarks/hmm-kernel-three-states.json)) — and the fix arrived through the normal stable route, not through this report. Filed as [ROCm#6523](https://github.com/ROCm/ROCm/issues/6523), where AMD confirmed the copy-on-write trigger and a third party reproduced it on bare metal, and with Ubuntu as [LP#2161985](https://bugs.launchpad.net/ubuntu/+source/linux-hwe-7.0/+bug/2161985); workaround at [vllm#49991](https://github.com/vllm-project/vllm/pull/49991). The writable-mapping penalty itself survives on current kernels: the loader flag is worth **1.5× to 2.0× while the checkpoint fits in RAM and 7.5× when it does not** ([data](benchmarks/loader-flag-kernel-30.json)); the **3.9× to 5.6× published here and upstream on 2026-07-28 came from a run with no control over page cache and does not reproduce.** The full chain — the half-pair of commits, the rebuild, the resident-set mechanism — is [open-questions.md §8](docs/open-questions.md) |

### Which GPUs this applies to

The bug is triggered by the **platform**, not by the GPU, so it hits any AMD GPU that
cannot get PCIe AtomicOps to its root complex: cards behind a consumer chipset switch,
and QEMU/VFIO passthrough guests, including virtualised Instinct.

**In a guest, check the VM configuration first** — passing a card's audio
function alongside the GPU is enough to remove AtomicOps on its own
([the one-line fix](docs/vfio-atomics.md)). The rebuild below is for hardware
that genuinely cannot deliver AtomicOps: chipset-fed slots on bare metal, root
ports without completer support, QEMU older than 8.1.0. We build it for seven
targets:

| Target | Cards | Status |
|---|---|---|
| **gfx1100** | RX 7900 XTX / XT | ✅ **verified end to end**: every number in this repository |
| gfx1030 | RX 6800 / 6800 XT / 6900 XT | 🟡 dispatch gate verified, collectives not. On a mixed gfx1030+gfx1100 pair the stock library fails on the gfx1030 rank with "the operation cannot be performed in the present state" and ours reaches Init COMPLETE on that same rank. No collective ran: the pair faults in `libamdhip64` under every env tried, and we cannot separate that from architecture mixing without a second gfx1030 |
| gfx1101 | RX 7800 XT / 7700 XT | ⚪ same |
| gfx1102 | RX 7600 / 7600 XT | ⚪ same |
| gfx1200 | RX 9060 | ⚪ same |
| gfx1201 | RX 9070 / 9070 XT | ⚪ same |
| gfx908 | MI100 | ⚪ same |

⚪ means the device image passes the static check that matters
(`hidden_hostcall_buffer` = 0) but has never been run on real silicon. 🟡 means the
dispatch gate was verified against a stock control but no collective completed. We
have only ever owned 7900 XTs and borrowed a 6800 XT for two days, so if you try one
of the others, a one-line report either way is genuinely useful.

The failure is **not** limited to virtual machines: @adderek independently reproduced
it, and the fix, on bare metal with IOMMU entirely disabled (2× RX 7900 XTX on a B550
board) in [ROCm#6520](https://github.com/ROCm/ROCm/issues/6520). Their machine is also
a useful shape to know about — one GPU affected because it sits behind the chipset,
one healthy because it is CPU-direct. On mainstream boards the second
full-length slot is often wired to the chipset rather than the CPU, so a
two-GPU build can land in exactly this shape with no VM involved.

> **The built library is not in this repository.** A 97 MB binary does not belong in
> git. Two ways to get one:
>
> - **[Releases](../../releases)** — `librccl-nohostcall-2.27.7-gfx1100.so` (19 MB,
>   what every number here was measured on) or the 97 MB multi-arch build, both with
>   SHA256 sums.
> - **Build it yourself** with [`build/build-rccl-nohostcall.sh`](build/build-rccl-nohostcall.sh):
>   about 85 minutes for one target on a slow host, and
>   [`build/verify-nohostcall.sh`](build/verify-nohostcall.sh) checks the result
>   independently of who compiled it.

---

## Who this is for

You have **two AMD consumer GPUs** and want `--tensor-parallel-size 2` to actually work. You are probably here because of one of these:

- 🔴 **It crashes immediately**: `NCCL error: unhandled cuda error`, or
  `HIP failure 'the operation cannot be performed in the present state'`.
  → **[Jump to the 60-second triage](#am-i-hit-by-the-rccl-bug)**. This is the single biggest blocker on consumer Radeon, it has been open upstream for months with no published root cause, and this repository explains and fixes it.
- 🟡 **It runs, but you do not know what to expect** → [What performance to expect](#what-performance-to-expect)
- 🟢 **You are deciding whether to buy/build this** → [What does *not* work](#what-does-not-work) first, please

**Not a vLLM fork.** The RCCL fix lives below vLLM — in the VM configuration if you are in a guest, otherwise in one RCCL rebuild — so there is no upstream to keep rebasing against. [`patches/`](patches/) does carry downstream vLLM changes, but only the ones the 2026-08-24 campaign needed; none of them is part of the fix this repository is about.

<details>
<summary><b>Did you get here from a search engine?</b> These are the exact messages this repository explains</summary>

```
RuntimeError: NCCL error: unhandled cuda error
HIP failure 'the operation cannot be performed in the present state' at .../rccl/src/enqueue.cc:2061
hipModuleLaunchKernel: Returned hipErrorIllegalState
NCCL WARN cuMem support requires VMM RDMA support
rocvirtual.cpp:4208  Pcie atomics not enabled, hostcall not supported
rocvirtual.cpp:4636  AQL dispatch failed!
amdgpu 0000:0b:00.0: amdgpu: PCIE atomic ops is not supported
```

If any of those look familiar, and you are running **two or more AMD GPUs** under
**vLLM, PyTorch DDP/FSDP, or anything else that calls RCCL**, on a consumer chipset
or inside a **VFIO/QEMU passthrough VM**, then this repository has the root cause, a
30-line reproducer and a working fix.

It applies to **RX 7900 XTX / XT / GRE, RX 7800 XT, RX 7600, RX 6800 / 6900 XT,
RX 9070 / 9060 and virtualised Instinct**, because the trigger is the PCIe path to
the card rather than the card itself. Verified end to end on gfx1100; the table above
says what is and is not tested for the rest.

One line in that list is the odd one out. `cuMem support requires VMM RDMA support`
is RCCL declining its own cuMem path because VMM RDMA is unavailable — benign, and
not the cause of anything here. It is listed because it appears in the same logs and
because people search for it. `NCCL_CUMEM_ENABLE=1` changes nothing on this machine;
it is one of the 11 combinations in `diagnose/sweep.sh`.

</details>

---

## Verified configuration

Everything below was measured on this machine. Nothing is extrapolated.

| | |
|---|---|
| **GPUs** | 2× Radeon RX 7900 XT (gfx1100, RDNA3), 20 GB each = 40 GB |
| **Interconnect** | Cross-die, PCIe 3.0, **no P2P**, `NCCL_P2P_DISABLE=1` |
| **Host** | Threadripper 1950X (Zen 1), X399 |
| **Virtualisation** | Proxmox VE + QEMU, **VFIO passthrough**. No PCIe atomics through 2026-08-23, single-function and with atomics since — [see below](#hardware-notes) |
| **Stack** | ROCm 7.14 · vLLM 0.23 · PyTorch 2.11 · **RCCL 2.27.7 rebuilt** |

> The topology is intentionally hostile. If it works here, a bare-metal box with
> P2P should do at least as well.

---

## Am I hit by the RCCL bug?

**If your GPUs are passed through to a VM, check one thing first.** Passing a
card's audio function alongside the GPU stops QEMU from advertising PCIe
AtomicOp completer support, and that alone produces every symptom below. On
Proxmox it is the difference between `hostpci0: 0000:0b:00` and
`hostpci0: 0000:0b:00.0`, and on this machine it took stock RCCL from failing to
working with nothing else changed. **[Read this before rebuilding
anything](docs/vfio-atomics.md)** — it is a one-line change and it costs
nothing to rule out.

If you are on bare metal, or the fix above does not apply, carry on: two
commands, and the second is decisive and needs no RCCL, no PyTorch, no vLLM.

```bash
./diagnose/check-platform.sh          # dmesg + PCIe bridge chain + hostcall count
```

```bash
hipcc --offload-arch=gfx1100 -O2 diagnose/hipgate3.cpp -o hipgate3 && ./hipgate3
```

**You are affected if the plain kernel passes and the hostcall kernel is refused,
or if its device marker never prints:**

```
--- device 0 (gfx1100) ---
  plain     ok        launch:no error | sync:no error | lastError:no error
  hostcall  REFUSED   launch:the operation cannot be performed in the present state | ...
```

The probe reads `hipGetLastError()` and the device `printf` as well as the return
codes, because on some machines launch and sync both report success while the
dispatch was refused and the `printf` silently never arrives. It also runs per
device: a machine can have one affected GPU behind the chipset and one healthy one
on CPU-direct lanes.

<details>
<summary><b>What is actually wrong</b> (click to expand)</summary>

```
PCIe AtomicOps cannot reach the GPU
   (a consumer chipset switch does not route them to slots behind
    it; a QEMU root port advertises no completer support unless the
    device is passed as a single function -- see docs/vfio-atomics.md)
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

**This is not a Radeon-only problem.** The trigger is the PCIe path, not the card,
so it reaches **any VFIO/QEMU passthrough guest whose emulated root port declines
to advertise completer support — including virtualised Instinct.** On the Proxmox
default that is because the card is passed multifunction, and passing the
function explicitly fixes it; `vfio_pci_enable_rp_atomics()` has six other ways
to decline, listed in [vfio-atomics.md](docs/vfio-atomics.md) §1.

Full evidence chain, and the 13 hypotheses we tested and the 12 we eliminated:
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

Five models × eleven context lengths, 292 measurements, zero errors, on stock
vLLM. Full write-up in **[docs/benchmarks.md](docs/benchmarks.md)**; raw data and
the runner in [`benchmarks/`](benchmarks/). Every run uses a random prefix to
defeat prefix caching; decode is measured first-token-to-last, so TTFT is
excluded.

**A second campaign on 2026-08-24** measured the same ladder on a patched
container: 372 measurements, nine configurations, six of them the July ones
rerun as controls. Four reproduce within 0.25 %, one is too noisy to say, and one
does not — which is discussed rather than dropped ([benchmarks.md §6](docs/benchmarks.md#6-the-same-machine-patched-a-second-campaign-on-2026-08-24)).

![decode throughput vs context length, best known configuration](docs/assets/decode-vs-context-best.svg)

One line per model, each from whichever stack measured it best, with what that
took written under the chart: solid needs nothing but a released vLLM, dashed
needs a patch that is not merged. It is drawn from
[`benchmarks/ledger.jsonl`](benchmarks/ledger.jsonl), which carries the date,
vLLM, ROCm and patch list of every point.

**The chart and the table below answer different questions, and Qwen3.8-27B is
where that shows.** The table is one campaign, run on one afternoon on one
stack; the chart is the best each model has been measured at. That model reads
10.7 tok/s at 32 K in the table and 36.1 on the chart, and the difference is
vLLM 0.27 with [#45916](https://github.com/vllm-project/vllm/pull/45916)
applied ([the A/B](docs/hybrid-decode-on-rdna.md)).

### vLLM, tensor parallel — decode tok/s (2026-08-24, patched container)

The six configurations in the chart, measured the same way. The four July
models were rerun as controls and come back within 0.25 %, except gemma-4-31B
at −0.85 %, which is recorded as unexplained. The stock 2026-07-25 campaign —
chart and table — lives in [benchmarks.md](docs/benchmarks.md).

| Model | Precision | 500 tok | 8 K | 32 K | |
|---|---|---:|---:|---:|---|
| **gemma-4-26B-A4B** | int4, 128-expert **MoE** | **107.7** | 92.6 | **72.9** | 🟢 fastest of the nine — *needs one 26-min compile, then cached* |
| Qwen3-8B | BF16 dense | 79.5 | 73.4 | 61.4 | 🟢 TP=1 → TP=2 is **1.70×** |
| gemma-4-12B-it | w4a16 QAT dense | 59.9 | 52.0 | 41.4 | 🟢 TP=1 → TP=2 only **1.19×** — see below |
| **Muse-Glimmer-30B** | int4, **sliding window 2048** | 43.7 | 37.8 | **37.4** | 🟢 flat from its window onward. 0.122 µs slope, second only to BF16 |
| **gemma-4-31B-it** | w4a16 QAT dense | 42.8 | 36.6 | 29.3 | 🟢 the workhorse. 265 W × 2 synchronised |
| **Qwen3.8-27B** | AWQ int4 (asymmetric), **hybrid SSM** | 12.3 | 11.7 | **10.7** | 🟢 **2.51×** the July Qwen3.6 at 32 K, slope 12.4× flatter. Slowest in this campaign; on vLLM 0.27 with #45916 it is not — see the chart above |

gemma-3-27b (44.8 / 34.6 / 22.1) is measured but not plotted: between 500 and
4 000 it runs within two tok/s of both Muse-Glimmer and gemma-4-31B and the
three lines read as one.

> **The Qwen3.8-27B row is checkpoint-bound, not architecture-bound**
> *(2026-08-27)*. That checkpoint is asymmetric int4, which misses vLLM's native
> gfx1100 W4A16 kernel and sends every quantised linear to Triton. The same
> model with a symmetric checkpoint is **3.24× faster at 1 K** under matched
> conditions, a flat ~60 ms per decode step
> ([benchmarks/w4a16-symmetry](benchmarks/w4a16-symmetry/)). The *slope*
> conclusions for this row are unaffected. Note also that gemma-3-27b, one row
> above in size, is symmetric — the two 27B models differ by 3.64× here.

### llama.cpp, for comparison

| Model | Mode | Decode |
|---|---|---|
| gemma-4-12B | single card, Vulkan | **64.9 tok/s** |
| gemma-4-31B | dual card, Vulkan layer split | 27.0 tok/s |
| Qwen3.6-27B | dual card, Vulkan layer split | 27.7 tok/s |
| **Qwen3.6-27B** | **+ MTP speculative decoding** | **34.5 tok/s** 🟢 |

### The charts worth the scroll

**What the second GPU actually buys.** Dashed is one card, solid is two. The blue pair
(BF16) separates; the green pair (4-bit) barely does. Same machine, same interconnect,
same RCCL, and a threefold difference in what the second card is worth. This is the
2026-08-24 rerun; the July original — within 0.25 % of it, and the closest thing here
to a repeatability check on the whole apparatus — is in
[benchmarks.md](docs/benchmarks.md):

![single card vs dual card, patched](docs/assets/tp1-vs-tp2-2026-08-24.svg)

**What one context token costs at decode time.** The slope is the number that
matters: it is milliseconds added per token of context, so a flat line is a
model whose decode does not care how long the conversation is. Every line here
is that model at its best known configuration.

![cost of one context token at decode time, best known configuration](docs/assets/decode-ms-per-token-best.svg)

**The one architecture that was unusable at long context, and what closes it.**
Qwen3.8-27B is a hybrid SSM: 48 linear-attention layers that promise O(1) per
token, and 16 full-attention layers that do not. On a released vLLM the full
layers dominate and the cost climbs a straight line to **261.9 ms per token at
32 K**. The same model on the same machine with
[#45916](https://github.com/vllm-project/vllm/pull/45916) applied is **27.7 ms
and flat** — 9.5× at 32 K, and the slope falls from 7.41 to 0.26 ms per
thousand tokens of context. That PR is not merged.

![the hybrid-SSM collapse and what closes it](docs/assets/hybrid-ssm-collapse.svg)

Both arms are two passes on one stack with the arm order reversed, and the
routing is recorded from inside the TP workers rather than inferred; the 8 K
point came out in two modes and the chart says so. Method and raw rows:
[hybrid-decode-on-rdna.md §6.6](docs/hybrid-decode-on-rdna.md).

The per-campaign versions of the first two charts, with every model pinned to
one stack, are in [benchmarks.md](docs/benchmarks.md).

**What eleven lines buy on a windowed model.** The sliding-window block skip
changes nothing below each model's own window (1.00×, there is nothing to
skip), then the gain grows monotonically: 2.75× on gemma-3 and 3.15× on
Muse-Glimmer at 32 K. The shape is the mechanism check —
[the patch and its correctness argument](docs/sliding-window-block-skip.md):

![sliding-window block skip](docs/assets/sliding-window-block-skip.svg)

**Prefill, for completeness.** Every model peaks early — 2 K for four of the six,
4 K for the MoE, 6 K for the hybrid-SSM — and then falls away. The ordering does
not survive the fall: the 8B leads at 500 by 2.1× over the MoE and the MoE has
passed it by 32 K. The derivation of where the peak sits (`S* = √(a/c)`, fitted
to every measured point) is in
[docs/benchmarks.md](docs/benchmarks.md#4-prefill-peaks-and-where-the-peak-sits).

![prefill throughput vs context length, patched](docs/assets/prefill-vs-context-2026-08-24.svg)

**The speculative-decoding collapse, taken to CUDA.** One rented A100 replaces
both Radeons and the story survives the port: on the routing gemma-4 gets by
default (TRITON_ATTN whenever FA4 is unavailable), turning MTP **on** costs
−28.2% at 30 K and −61.1% at 50 K of context — turning it off is 2.57× faster
there, which is [vllm#52049](https://github.com/vllm-project/vllm/issues/52049)
reproduced. Route the same model to FlashInfer and speculation is a gain again.
Same kernel guard, same collapse, different vendor — measured the same day, one
GPU, [the annex has the logs](benchmarks/cuda-a100/README.md):

![gemma-4 MTP backend matrix on A100](docs/assets/gemma4-mtp-backend-matrix-a100.svg)

**The collapse is a path choice.** Speculative decoding makes every decode
step carry two query tokens, and the Triton attention launcher treats
anything above one as prefill: decode silently leaves the segmented **3D
flash-decoding path** — the one that splits the KV scan across the whole
GPU — for the serial **2D path**, 8 workgroups per rank instead of 128.
Dashed below is that 2D detour, solid is the 3D path readmitted by
[vllm#45450](https://github.com/vllm-project/vllm/pull/45450), which we
validated on both vendors the day we found it (bit-exact output on the
A100, one-ULP-bounded at the kernel on the Radeons): **8.81 → 32.57
tok/s at 32 K** on this machine, and no depth at which 3D loses. The
case file: [45450-validation](benchmarks/cuda-a100/45450-validation/README.md)
and [speculative-decoding-on-rdna.md §5–§6](docs/speculative-decoding-on-rdna.md):

![speculative decode on the 2D vs the 3D path](docs/assets/spec-decode-45450-ladder.svg)

### Two Radeons against one A100

The ladder above permits a cross-vendor reading, and at this regime it is
not the blowout the price gap suggests. Batch-1 decode is
bandwidth-bound, and two 800 GB/s cards against one 2.0 TB/s card is only
a 1.27× difference in nominal ceiling. Measured on the healthy 3D path,
MTP on, matched depths:

| context | 2× RX 7900 XT | A100 80G | A100 advantage |
|--:|--:|--:|--:|
| 1K | 74.89 | 110.71 | 1.48× |
| 8K | 63.25 | 75.63 | 1.20× |
| 16K | 63.09 | 72.13 | **1.14×** |
| 32K vs 30K | 32.57 | 61.03 | 1.87× |

The gap is U-shaped, and each end has its own mechanism. At 1K the step
time is short and TP=2's fixed all-reduce floor weighs heaviest — the
single A100 pays no such tax. In the middle the two nearly converge:
fourteen percent apart at 16K, on hardware an order of magnitude apart
in price. Past 16K the KV scan dominates and the difference in realized
bandwidth (the Radeons reach 63 % of their 800 GB/s on this model)
compounds again.
*(Corrected 2026-08-29: this cited 38 %, which is the 12B's utilisation,
not this comparison's 31B. Recomputed from `results.jsonl` and the
per-GPU bytes/token in [benchmarks.md §3](docs/benchmarks.md), the 31B at
TP=2 reaches 62.8 %. All three utilisation figures are now pinned by
`verify_doc_figures.py`.)*

Two more readings from the same table:

- **The 2D path punishes tensor parallelism itself.** Splitting KV heads
  across two cards halves each card's 2D workgroups — 8 per rank against
  the A100's 16 — so the starved path starves harder: at 30K-32K the 2D
  path retains 15.8 % of its 1K rate here, against 33.6 % on the A100.
  The same collapse, same kernel, twice the damage.
- **TP erodes speculation economics too.** The MTP drafter is small, but
  every draft step still pays the all-reduce floor, which does not
  shrink with model size: speculation's net win over no-speculation is
  +39 % on the A100 at 30K and only +7.5 % here at 32K.

What this table does not cover is the territory the A100 wins outright:
prefill and batched throughput are compute-bound, where its tensor cores
run against RDNA3 WMMA that already realizes only ~37 % of nominal peak
here — expect multiples, not percentages. One model, one stack per
platform (the stack each side actually runs today), single-run probes;
the campaign-grade version of this comparison is future work.

### Want the raw numbers?

```bash
cd benchmarks/analyze
python3 summarize.py       # every configuration, exactly as measured
python3 decode_slope.py    # cost of one context token, per model
python3 analyze.py         # TP2/TP1 speed-up, bandwidth utilisation
```

No GPU, no dependencies beyond the standard library. They read
[`benchmarks/results.jsonl`](benchmarks/results.jsonl): 309 records, one per request,
each with its prompt length, TTFT, decode rate, per-card power and VRAM. That file is
the 2026-07-25 campaign; the 2026-08-24 one is
[`results-2026-08-24.jsonl`](benchmarks/results-2026-08-24.jsonl), and the separate
findings have their own files. What ties them together is
[`analyze/verify_doc_figures.py`](benchmarks/analyze/verify_doc_figures.py), which
recomputes the headline figures quoted in this README and in `docs/` from whichever
file each came from, and exits non-zero if one disagrees.

### How to read this

- **Architecture beats parameter count.** The fastest model here is the 26B MoE,
  ahead of the 8B dense by **1.355×** and of the *larger* 31B dense by **2.513×**.
  *(Corrected 2026-08-27: this used to read "the slowest is a 27B, beaten 3.6× by
  a larger 31B dense". That comparison is confounded — the 27B's checkpoint is
  asymmetric int4 and misses the native W4A16 kernel, worth up to 3.24× on its
  own, see [w4a16-symmetry](benchmarks/w4a16-symmetry/). The MoE-over-dense
  result is not confounded: every model in it is on its best kernel path.)*
- **Never benchmark with `--enforce-eager`.** It costs **3.8–7.2×** on this stack and
  invents artefacts (asymmetric power, context-independence). Two wrong conclusions
  in this repository came from exactly that, including "MoE is mediocre, ~15 tok/s",
  which was really 107.8.
- **What the second card buys depends on the model.** BF16 scales 1.70×; w4a16 only
  1.19×, because the quantised model was never bandwidth-bound in the first place.
  For quantised models the second card mostly buys *capacity*: the 12B's KV pool goes
  151 808 → 354 707 tokens, concurrency 4.60× → 10.75×.
- **Below ~1 K prompt tokens, one card prefills faster than two** (3460 vs 2270 tok/s
  at 512): TP adds a ~76 ms per-request communication floor, 72 all-reduces at
  ~1.05 ms each over host shared memory.
- **Long context: avoid hybrid-SSM *under vLLM*.** The 27B costs 4.84 µs of decode
  time per token of context, **41× the dense 8B**; dense and MoE lose only 23–32 %
  out to 32 K. The cause is not the SSM layers — it is the model's few
  full-attention layers falling off the ROCm paged-attention fast path
  ([why](docs/hybrid-decode-on-rdna.md)).
- **For Qwen3.5/3.6, llama.cpp wins, and by more the longer the context**:
  against stock vLLM, 2.1× at 512 tokens (24.89 vs 12.1) and 5.1× at 32 K
  (21.84 vs 4.2), same two cards, same model, ROCm backend both sides. With
  vllm#45916's gate widened the 32 K gap narrows to 2.0× (21.84 vs 10.72); that
  PR is not merged.
- Bandwidth utilisation at decode: 88 % (8B BF16, single card) down to 38 %
  (12B w4a16, TP=2). Prefill saturates at ~37 % of FP16 peak.

---

## What does *not* work

Stating this plainly is the point of the repository.

| | Status |
|---|---|
| **FP8 weights/KV** | 🔴 Not available. FP8 is MI300+; RDNA3 has no FP8 path |
| **AITER kernels** | 🔴 Gated to `is MI3XX` in vLLM. gfx1100 silently falls back to Triton |
| **Tuned fused-MoE configs** | 🔴 vLLM ships none for *any* AMD GPU. MoE runs a generic default |
| **Hybrid SSM (Qwen3.5/3.6/3.8)** | 🟡 **Fixed upstream, not yet merged.** Stock vLLM keeps 35.1 % of its short-context rate at 32 K; with [vllm#45916](https://github.com/vllm-project/vllm/pull/45916)'s split-KV gate widened to RDNA3 the same architecture holds **86.8 %**, a slope of 0.390 µs against 4.840 — verified at the kernel (69/69, 15.8×) and end to end over the eleven-point ladder ([benchmarks.md §6](docs/benchmarks.md#6-the-same-machine-patched-a-second-campaign-on-2026-08-24), [details](docs/hybrid-decode-on-rdna.md)). llama.cpp is ahead either way at 32K: 5.1× against stock vLLM, 2.0× with the gate widened |
| **Speculative decoding (MTP)** | 🟡 Context-dependent. `gemma-4-31B` with Google's official MTP assistant is **+36.9% at 1K** and **−70.8% at 32K** on this hardware: speculation sets `max_seqlen_q=2`, which disables the Triton backend's segmented-softmax path that long-context decode relies on. Enable it for short prompts, disable it by 8K, where it is already 14% down ([details](docs/speculative-decoding-on-rdna.md)) |
| **Sliding-window decode on `ROCM_ATTN`** | 🟡 **Ours to fix, 11 lines.** The Triton paged-decode kernel iterates the whole sequence and masks the window away afterwards, so a 1 024-token window at 32 K reads 2 048 blocks where 64 are needed — **`gemma-3-27b` pays it at 8.05 tok/s while the larger `gemma-4-31B`, routed to a backend that bounds its loop, does 30.21**. Skipping the masked blocks is an identity, not an approximation: **2.75× on gemma-3 and 3.15× on `Muse-Glimmer-30B`** at 32 K, 1.00× below each window; end to end on 2026-08-24, gemma-3 reaches 22.05 tok/s and `Muse-Glimmer-30B` runs flat at 37.4 from its window onward. Upstream's own kernel suite passes with no case changing outcome. **The same eleven lines were already proposed as [vllm#49588](https://github.com/vllm-project/vllm/pull/49588) on 2026-07-23 and have sat as a draft since**, so this is a second body of evidence rather than a second PR ([details](docs/sliding-window-block-skip.md)) |
| **MoE `torch.compile`** | 🟡 vLLM hardcodes `TORCHINDUCTOR_COMPILE_THREADS=1` in `env_override.py`, unconditionally and on every `import vllm`, so **setting that variable in the environment does not help — it is overwritten**. Inductor's own default would be one thread per core. A 128-expert graph took `init_engine_s` **1569 s** here and `gemma-4-12B` at **TP=2** took **1538 s**; both ran at one core out of eight. *(Corrected 2026-08-29: this said "26 min" and "TP=1 took 24". The 12B's long start is at TP=2 — its TP=1 starts were 59.67 s and 33.36 s — and `init_engine_s` bounds the compile rather than measuring it.)* Patch the line; `--enforce-eager` avoids the compile at 3.8–7.2× and invents artefacts, see [the article](docs/articles/moe-written-off-by-eager.html) |
| **Multi-tenant serving** | 🟡 Untested. Everything here is single-stream or light concurrency |
| **P2P between cards** | 🔴 Not on this topology. Everything measured is *without* it |
| RCCL 2.30.4 | 🔴 See the warning above |

Background on the SSM and MoE findings, with source-level evidence:
**[docs/architecture-notes.md](docs/architecture-notes.md)**

---

## Hardware notes

**What decides whether a GPU has atomics?** A PCIe atomic operation must be
*completed* by the root complex and *routed* by every switch in between, and
`amdgpu` checks exactly that through `pci_enable_atomic_ops_to_root()`: 32- and
64-bit completer support on the root port, AtomicOp routing on each switch port
below it.

**On bare metal it is the chipset.** Root ports on consumer boards normally
do advertise completer support — this project's own X399 host reports
`Routing- 32bit+ 64bit+` on all eight, and that passes, since a root port's
`Routing` bit refers to peer-to-peer between root ports and is not part of the
check. That `Routing-` is consistent with this machine having no GPU P2P at
all, which is a separate capability from atomics to system memory; the kernel
function says as much in a comment ("no peer-to-peer"). Confirmed on the
hardware: before these cards were handed to `vfio-pci`, `amdgpu` bound them on
the host across five boots and never reported atomics missing, while in the
guest it reports it for both GPUs every time. What breaks these machines is a
chipset downstream port that reports `Routing-`: it cuts off every slot behind
it, while a CPU-attached slot on the same board is fine. @adderek's B550 in
[ROCm#6520](https://github.com/ROCm/ROCm/issues/6520) has one GPU of each kind
and only the chipset-attached one is affected.

**Why does a VM lack them?** QEMU's emulated `pcie-root-port` reports
`32bit- 64bit-` **when the device is passed as multifunction**, which is the
Proxmox default. Passed as a single function it advertises completer support
automatically, and the guest gets atomics — QEMU has done this since 8.1.0.
[vfio-atomics.md](docs/vfio-atomics.md) has the A/B.

**This machine changed sides on 2026-08-23 at 14:17 UTC.** Everything measured
before that, including the 2026-07-25 campaign, ran multifunction and without
AtomicOps; everything after, including the 2026-08-24 campaign, the loader work
and the sliding-window measurements, ran single-function and with them. Where
that matters to a comparison it is called out at the comparison.

**Do I need atomics for inference?** No. They are a precondition for *hostcall*,
which is a debug facility (device `printf`/`assert`), and AMD's own ROCm 7.1.1
build shipped with zero hostcall. The runtime cost of removing it is **assumed
zero and has not been measured** — see
[open-questions.md §6](docs/open-questions.md).

**Two identical cards?** Strongly preferred. Mixed cards are limited by the
smaller/slower one, and layer-splitting makes the older card the thermal hotspot.

**Thermals — do not skip this.** Two cards in adjacent slots: the upper one
inhales the lower one's exhaust. We measured **junction 99–100 °C** on the upper
card at sustained load. One 120 mm fan aimed at the gap between the cards
dropped it to **90 °C** and *inverted* which card runs hotter — while its own
fan spun **slower**. Cheapest fix in the entire build.

**Slow host CPU?** It shows up at startup, not at decode: `torch.compile` is
CPU-bound and vLLM pins it to one thread.

**Weight loading is far slower than your disk, and the kernel decides how
much.** The disk sustains 1.5 GB/s; vLLM loads checkpoints at 30–76 MiB/s. The
mapping is the cause: safetensors' PyTorch path maps the checkpoint
**writable**, and a ROCm host→device copy from such a mapping breaks
copy-on-write on every resident page — catastrophic on kernel `7.0.0-28`
(2.0 MiB/s), a 4–8× tax everywhere else. Upgrading the kernel removes the
catastrophe. For the rest:
[vllm#49991](https://github.com/vllm-project/vllm/pull/49991)'s clone flag is
worth 1.5–2.0× while the checkpoint fits in RAM and 7.5× when it does not;
`safe_open(..., backend="pread")` holds the least memory; and
`--safetensors-load-strategy eager` peaks at about twice the shard, which is
how a 21.67 GiB single-shard checkpoint stops fitting on a 23.4 GiB host. All
of it is measured, four load paths across four checkpoints, in
[`loader-flag-kernel-30.json`](benchmarks/loader-flag-kernel-30.json); the full
chain — including what was disproven on the way — is
[open-questions.md §8](docs/open-questions.md).

**RAM ceiling.** vLLM `mmap`s the whole checkpoint, and the limit is `MemTotal`
rather than free memory. A 21.67 GiB file would not map into the 21.43 GiB this
guest had when that was first hit; the fix was to raise the VM's RAM, which is
now 23.40 GiB with 8 GiB of swap, and that checkpoint loads. Size for the
ceiling rather than expecting to hit it.

---

## Repository map

```
diagnose/     Start here. Dependency-free probes
  hipgate3.cpp     ★ plain kernel vs hostcall kernel — decisive, ~30 lines
  check-platform.sh  one-shot triage: dmesg + bridge chain + hostcall count
  ar.py            30-second torchrun all_reduce reproducer
  sweep.sh         11 env-var combinations that do NOT help
  logs/            AMD_LOG_LEVEL=4 capture at the moment of failure

build/        Rebuild RCCL without hostcall, and verify it
  verify-nohostcall.sh   hostcall / assert / fprintf must all be 0
  check-symbols.sh       all 38 nccl symbols PyTorch needs

deploy/       Inject into a ROCm/vLLM container (3 pieces, all required)

benchmarks/   The measurement data and everything that produced it
  results-2026-08-24.jsonl 372 measurements on the patched container, same ladder,
                       six of the nine configurations rerun as controls
  results.jsonl        ★ 292 measurements, one record per request; the source of
                         the 2026-07-25 tables and charts
  analyze/             turn that into the tables and charts; no GPU needed
  bench_runner.py      the campaign runner: serial, checkpointed, VRAM-safe
  prompts/             rebuild the prompt ladders from Gutenberg #1228, and check
                       them against the counts that were actually measured
  repro-mmap-prot.py   host→device copy from a writable mapping; kernel-sensitive
  repro-mmap-prot.hip.cpp  the same case in plain HIP, for machines with no
                       PyTorch — a hypervisor host, a rescue image, a bare ROCm
                       install
  cuda-a100/           one day on a rented A100: the gemma-4 MTP collapse
                       reproduced on CUDA and removed by rerouting — the
                       cross-vendor control for docs/speculative-decoding-on-rdna.md

patches/      Downstream changes to the installed vLLM, so the numbers above can
              be reproduced. None is a recommendation to run in production
  sliding-window-block-skip.patch  start the paged-decode loop at the window
  wintest.py           the before/after timing harness for it. It also records
                       token ids, which turned out not to be a correctness test
                       here: greedy decoding is irreproducible on this machine
                       with the patch absent
  adapt-muse-glimmer.py  back-adapt upstream's model file to a vLLM that
                       predates the model

docs/
  benchmarks.md        ★ the five-model study, with all four charts
  root-cause.md        the RCCL bug: evidence chain and 13 tested hypotheses
  open-questions.md    what we have NOT proven — including one root cause we
                       published, disproved ourselves, and rewrote
  architecture-notes.md  why MoE, dense and hybrid-SSM behave so differently here
  hybrid-decode-on-rdna.md  why the hybrid-SSM model collapses at long context —
                       kernel-level profile, and the wrong answer it replaced
  sliding-window-block-skip.md  the paged-decode kernel reads the whole sequence
                       and masks the window away; 11 lines, 2.75-3.15× at 32K on
                       two models, why gemma-3 pays it and gemma-4 does not, and
                       the correctness argument we had to withdraw and replace
  deploy-vllm.md       step-by-step deployment
  diagnosis.md         is this your bug?
  assets/              every chart above and in docs/, as standalone SVG

```

**Reported upstream:** the RCCL root cause is
[ROCm/ROCm#6520](https://github.com/ROCm/ROCm/issues/6520), with a pointer on
[#6074](https://github.com/ROCm/ROCm/issues/6074). The passthrough caveat behind
it went to `pve-devel` on 2026-08-24 as a two-patch `pve-docs` series. The SSM
behaviour is written up in `docs/` but not filed; see
[open-questions.md](docs/open-questions.md) for what is claimed and how strongly.

**If you only open one file:** [`docs/benchmarks.md`](docs/benchmarks.md) if you came
for numbers, [`docs/root-cause.md`](docs/root-cause.md) if you came for the bug.

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
