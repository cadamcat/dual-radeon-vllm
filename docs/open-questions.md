# Open questions — what we have *not* proven

---

## 0. ⚠️ RCCL 2.30.4: `NDEBUG` is necessary but **no longer sufficient**

**Tested on real hardware, 2026-07-23.** We built RCCL **2.30.4** (from
`ROCm/rocm-systems`, `projects/rccl` — the monorepo RCCL migrated to; the old
`ROCm/rccl` repo is now `develop_deprecated`) with the same
`add_compile_definitions(NDEBUG)` patch that fixes 2.27.7, for seven
architectures. Result on 2× RX 7900 XT in a VFIO guest:

```
device_count: 2                     ← library loads, both hacks removed OK
torchrun all_reduce  →  HIP failure 'the operation cannot be performed in the
                        present state' at .../rccl/build/hipify/src/enqueue.cc:2118
```

Static inspection of that build:

| Check | Result |
|---|---|
| `__assert_fail` in device image | **0** — NDEBUG worked |
| `__ockl_fprintf` in device image | **0** — no device printf (COLLTRACE is gone from 2.30.4 entirely) |
| `__ockl_*` symbols of any kind | **0** — there is *no hostcall-calling code at all* |
| `hidden_hostcall_buffer` in metadata | **3** — one per `ncclDevKernel_Generic_{1,2,4}` |

**So the metadata declaration alone is enough for ROCr to refuse dispatch.**
Removing the *code* that would use hostcall does not help if the kernel still
*declares* the implicit argument.

**Where the declaration comes from — narrowed by bisection:**

```
device_build/common.o      (before device link)  hostcall = 0
device_build/gfx1100/device.elf (after link)     hostcall = 3
        …and the linked image contains 0 __ockl symbols
```

**The device-linking step introduces it.** RCCL 2.30.4 replaced the ordinary HIP
fat-binary path with its own device linker (`tools/rccl-device-compile --link`,
which also hand-patches SGPR/VGPR fields in the kernel descriptor). That tool
contains no `hostcall` or `hidden_` strings itself, so the declaration is being
emitted by the compiler during the dispatcher link, conservatively, for kernels
that demonstrably never call hostcall.

Ruled out as causes (trivial-kernel probes, all give `hostcall = 0`):
plain whole-program compile · `-fgpu-rdc` · `-mcode-object-version=4` ·
`-fgpu-rdc` + COv4. So it is **not** RDC per se and **not** the code-object
version.

**Consequences:**

- **2.27.7 remains the only verified working route** (see the main README).
- The upstream report gets much sharper: *"your device-linking step declares
  `hidden_hostcall_buffer` on all three `ncclDevKernel_Generic_*` kernels while
  the linked image contains zero `__ockl_*` symbols; on any platform without
  PCIe AtomicOps this makes every collective fail, and `NDEBUG` cannot fix it."*
- A plausible workaround not yet attempted: **strip the implicit-argument entry
  from the kernel metadata after linking.** RCCL's own tool already rewrites that
  metadata block (`_patch_amdgpu_metadata`), so there is a natural place to do it.
  Untested, and metadata surgery may break kernarg offsets.

**Confirmed at the driver level.** We reinstalled the failing 2.30.4 build and
re-ran under `AMD_LOG_LEVEL=4`. The rejection is verbatim the documented one,
on both ranks:

```
rocvirtual.cpp:4151  ShaderName : ncclDevKernel_Generic_4(ncclDevKernelArgsStorage<4096ul>)
rocvirtual.cpp:4208  Pcie atomics not enabled, hostcall not supported
rocvirtual.cpp:4636  AQL dispatch failed!
hip_module.cpp:605   hipModuleLaunchKernel: Returned hipErrorIllegalState
```

So this is now a hard fact rather than an inference: **a kernel that contains no
hostcall-calling code whatsoever (zero `__ockl_*` symbols) is still refused at
dispatch purely because its metadata declares `hidden_hostcall_buffer`.**

### Metadata surgery: works on the device image, blocked at repacking

We tested the obvious workaround — rewrite the implicit-argument entry after the
fact. Exploiting the fact that `hidden_hostcall_buffer` and
`hidden_global_offset_x` are both exactly 22 bytes, the substitution keeps every
msgpack length and every kernarg `.offset` byte-identical.

**On the unpacked device image this works perfectly:**

| | before | after |
|---|---|---|
| `hidden_hostcall_buffer` declarations | 3 | **0** |
| image size | 160,432,296 | **identical** |
| metadata still parses | — | **yes, 180 fields, no errors** |

**But it cannot be put back into the library.** The `.hip_fatbin` section is a
*compressed* offload bundle (magic `CCOB`), which is why the string is not
visible in the `.so` at all. Round-tripping it needs
`clang-offload-bundler`, and that fails on this bundle:

```
clang-offload-bundler --type=o --input=fat.bin --list
error: Failed to decompress input: Could not decompress embedded file contents: Src size is incorrect
```

We stopped here rather than reverse-engineering the container format. **Anyone
wanting to finish this needs to handle the compressed-bundle round-trip** —
either by making the bundler accept it, by rebuilding with compression disabled
(look for an offload-compression flag) so the string stays patchable in place,
or by patching inside RCCL's own `_patch_amdgpu_metadata`, which already rewrites
this metadata block *before* bundling and is therefore the natural place to do
it.

**Practical takeaway:** the fix is understood and demonstrably correct at the
image level, but there is no quick post-hoc patch for a shipped `.so`. For now,
**2.27.7 remains the only verified working route**, and chasing 2.30.4 would
trade three well-understood hacks for a fragile build-time metadata patch — not
a good trade for production.

---

Keeping this list honest is the point. Everything in
[root-cause.md](root-cause.md) has a test behind it; everything here does not.
If you can close one of these, it is a genuinely useful contribution — and #1 is
the missing piece of a good upstream bug report.

---

## 1. Which upstream change flipped shipped binaries from 0 → N hostcall? ⭐

**What we know for certain:** AMD's shipped RCCL from ROCm 7.1.1 has
`hidden_hostcall_buffer` count **0**; from ROCm 7.2 onwards every `Generic`
kernel has it. Same source project, same architecture.

**What we do not know:** *why*. Two hypotheses have now been **eliminated** by
comparing the public release branches
`release/rocm-rel-7.1.1.1` (ships hostcall 0) and `release/rocm-rel-7.2`
(ships hostcall N):

- ❌ **Not a CMake change.** Both branches' `CMakeLists.txt` are equivalent in
  every relevant respect: neither sets `CMAKE_CXX_FLAGS_RELEASE`, neither
  mentions `NDEBUG`, and both carry the same `option(COLLTRACE ... ON)` with the
  same `if(COLLTRACE) target_compile_definitions(rccl PRIVATE ENABLE_COLLTRACE)`.
  (The `set(CMAKE_CXX_FLAGS_RELEASE "-O3" ... FORCE)` line does exist — but only
  on `develop`, i.e. it was added *after* 7.2, so it cannot explain this
  regression. Do not cite it as the cause.)
- ❌ **Not obviously new device asserts.** Sampling the device headers
  (`all_gather.h`, `reduce_scatter.h`, `common_kernel.h`, `primitives.h`,
  `prims_simple.h`, `all_reduce.h`, `broadcast.h`, `reduce.h`) gives the **same
  `assert(` count in both branches**, and `common.h` has the same number of
  `ENABLE_COLLTRACE` sites. This is a sample, not an exhaustive diff, so treat it
  as strong evidence rather than proof.

**Therefore the leading hypothesis is now:** the RCCL *source* did not
meaningfully change between 7.1.1 and 7.2 — **AMD's release build invocation
did**. Something that used to deliver `NDEBUG` to the device compilation pass
stopped doing so. That is not visible from the repository, because it lives in
AMD's packaging/CI, not in `CMakeLists.txt`.

**Why this matters for the upstream report:** the ask is not "please support
platforms without atomics" and not "please change your CMake". It is
**"your source is unchanged; your shipped 7.1.1 binary has hostcall 0 and your
7.2 binary does not — please check what your build pipeline stopped passing."**
That is a one-line fix in AMD's own build configuration.

**How to close it definitively:** build `release/rocm-rel-7.1.1.1` and
`release/rocm-rel-7.2` in an identical environment with identical flags and count
hostcall in each device image. If both come out the same, the difference is
conclusively in AMD's pipeline, not the source. The counting one-liner is in
[`build/verify-nohostcall.sh`](../build/verify-nohostcall.sh).

---

## 2. Does this reproduce on bare metal?

We have no bare-metal multi-GPU Radeon machine. Our claim that the public
bare-metal reports share this root cause is an **inference** from:

- the mechanism requiring only "no AtomicOp routing", which is common on
  consumer chipsets, and
- the reported fix (downgrade to the 7.1.1 build) being exactly "use a library
  with no hostcall requirement".

Neither reporter posted `AMD_LOG_LEVEL=4` output, so the
`Pcie atomics not enabled` line has not been observed on those machines.

**How to close it:** run `diagnose/check-platform.sh` and `diagnose/hipgate3.cpp`
on a bare-metal dual-Radeon box and open an issue with the output. This is the
single highest-value contribution to this repository.

---

## 3. Is `COLLTRACE=OFF` alone sufficient? — **ANSWERED: no**

We applied `NDEBUG` globally, which removes both hostcall sources at once (device
`assert()` and, indirectly, the trace path), and never tested `-DCOLLTRACE=OFF` on its
own. @adderek did, on RCCL 2.27.7 at tag `rocm-7.2.4`, and posted the counts in
[ROCm#6520](https://github.com/ROCm/ROCm/issues/6520):

| build | `__assert_fail` | `__ockl_fprintf` | `hidden_hostcall_buffer` | collectives |
|---|---|---|---|---|
| distro package as shipped (a Release build) | 5 | 3 | 6 | fail |
| `-DCOLLTRACE=OFF` | 4 | 3 | **3** | fail |
| `-DCOLLTRACE=OFF -DCMAKE_CXX_FLAGS=-DNDEBUG` | 0 | 0 | **0** | **pass** |

`COLLTRACE=OFF` halves the declarations and does not remove them, because HIP's device
`assert()` is itself a hostcall user: it routes through `__ockl_fprintf` to print
before aborting. The asserts, not the trace path, are load-bearing.

**The corollary is the part worth acting on.** That distro package is a
`CMAKE_BUILD_TYPE=Release` build and still ships 5 `__assert_fail` and 6 declarations,
so **Release does not imply `NDEBUG` for RCCL's device compile**. If that holds for
other distro and vendor packages, every 2.27.7-era RCCL binary is affected out of the
box on any machine without atomics, and adding `NDEBUG` to the device compile would
fix that whole class at the source.

(Neither we nor @adderek have tested `NDEBUG` *alone*; we used it globally, they used
it with `COLLTRACE=OFF`. The counts above explain why the global patch works.)

---

## 4. Does virtualized Instinct hit this?

The mechanism depends on the *root port*, not the GPU. A QEMU `pcie-root-port`
advertises no AtomicOp completer support (`32bit- 64bit-`) regardless of what is
behind it, so passthrough Instinct should be affected too. We have no Instinct
hardware to confirm. If true, this materially raises the severity of the upstream issue,
since it would mean RCCL collectives are broken in virtualized datacentre
deployments and not merely on consumer desktops.

---

## 5. Can the guest be given PCIe atomics instead? — **one reason we ruled it out was wrong**

QEMU 11.0.2's `pcie-root-port` advertises no AtomicOp completer support
(`32bit- 64bit-`) and there is no PVE-level switch, so the guest cannot have
atomics as things stand. Patching QEMU was out of scope and remains so.

The second half of the original answer was a mistake we should record. It read:
*"our host's own Zen 1 root port reports `Routing-`, so even a fixed QEMU would
not have helped us."* That conflates two different capability bits.
`pci_enable_atomic_ops_to_root()` in `drivers/pci/pci.c` checks the **completer**
bits at the root port (`COMP32|COMP64`, which is what amdgpu asks for) and the
**routing** bit only on switch ports strictly between the device and that root
port; a root port's own routing bit concerns peer-to-peer between root ports and
is never read here. All eight root ports on this X399 host report
`Routing- 32bit+ 64bit+`, and the only bridges between a card and its root port
are the Navi switch on the card itself (`Routing+`, `EgressBlck-`).

**The host half is no longer an inference.** `amdgpu` did bind these cards on the
host, before they were handed to `vfio-pci`, across five boots in July 2026
(`0000:0b:00.0` on four of them, `0000:44:00.0` on the fifth). In none of those
boots did it print `PCIE atomic ops is not supported`, which the driver emits
exactly when `have_atomics_support` is false, and none of the short-circuit
branches around that assignment apply to a discrete gfx1100 card. Every later
init stage in those logs succeeded, so the check certainly ran. In the guest, the
same driver generation prints the message for both GPUs at every boot.

Same cards, same silicon, opposite answer. The Threadripper 1950X is not what
takes atomics away here; the passthrough layer is.

That makes a QEMU-side fix a real avenue for *us*, not merely for someone on
newer hardware, and it is worth being precise about why it is plausible: the
physical GPU is still wired to a physical root port that completes AtomicOps, so
a guest driver that believed the capability existed would be issuing requests
onto a path that can carry them. Whether QEMU can advertise completer support on
an emulated root port, and whether the request survives the VFIO/IOMMU path
unchanged, we have not tested. Patching QEMU remains out of scope for this
repository, but the reason we gave for dismissing it was wrong.

---

## 6. Performance impact of removing hostcall: assumed zero, not measured

We argue it is zero because `assert()` and trace `printf` never execute on the
working path, and our post-fix throughput matches expectations for the hardware.
We did not benchmark a with-hostcall vs without-hostcall build side by side on a
platform where both can run (which would require a machine that *has* atomics).

---

## 7. How far forward does the fix hold?

Verified on ROCm 7.13 and 7.14 (technology-preview stream) and against RCCL
2.27.7 / 2.30.4. ROCm ships roughly every six weeks. We do not know when this
will break, nor when upstream will make it unnecessary. Treat the version table
in the README as a snapshot, not a guarantee.

---

## 8. Why is weight loading 20–50× slower than the disk? — **effect characterised and worked around; mechanism NOT established**

> **Read this section as: we have a reliable reproduction, a measured shape, and a
> working fix — but we cannot yet say why it happens, and an earlier version of this
> section claimed a root cause that we have since disproved ourselves.**
>
> ### What is solid
>
> **The workaround.** Materialising each tensor into anonymous memory before the
> host→device copy — one `.clone()` per tensor, transient cost of one tensor —
> removes essentially all of the loss:
>
> | | before | after |
> |---|---|---|
> | Qwen3-8B, 15.26 GiB BF16 | 206 s | **18.7 s** (11×) |
> | gemma-4-12B, 9.56 GiB w4a16 | 328 s | **10.5 s** (31×) |
> | gemma-4-31B, 21.67 GiB w4a16 | 569 s | **25.1 s** (23×) |
>
> **The shape of the cost.** Timing each tensor of one real checkpoint shard
> (Qwen3-8B, 81 tensors, `safe_open` → `.to("cuda")`):
>
> | tensor size | time per copy | effective rate |
> |---|---:|---:|
> | < 4 MiB (28 tensors) | **0.3 ms** | fine |
> | 8 MiB | **1005 ms** | 8 MiB/s |
> | 96 MiB | **1042 ms** | 92 MiB/s |
> | 1187 MiB | **1522 ms** | 780 MiB/s |
>
> So this is **not a bandwidth problem**: it is a roughly **fixed ~1 s penalty per
> copy above a threshold somewhere between 4 and 8 MiB**. 22 large tensors × ~1 s
> accounts for the whole 17.7 s shard. It also explains two things that looked odd:
> a 128-expert MoE checkpoint with 35 743 *tiny* tensors is barely affected, and the
> 12B is slower than the 8B because of **how many tensors clear the threshold**, not
> because of quantisation repack (repack is nearly free — the 12B now loads in 10.5 s
> *including* it).
>
> **Where the time goes.** `perf` on a loading worker: **98.7 % in
> `kfd_ioctl_svm → svm_range_validate_and_map → hmm_range_fault`**; `strace`: **54
> ioctls in a 12-second window, ~189 ms each**. Reproduces on **both ROCm 7.0 and
> 7.14**, so it is long-standing behaviour rather than a regression.
>
> ### What we got wrong
>
> This section previously stated the cause as *"the source is a file-backed mmap page,
> and ROCm does not take the pinned-staging path for those."* **That is disproved.**
> A dependency-free reproducer — plain `mmap` of a warm file, `.to("cuda")` — runs at
> full speed, and every variable we could think of fails to trigger it:
>
> | hypothesis | result |
> |---|---|
> | file-backed mmap vs anonymous memory | both ~13 GiB/s ❌ |
> | pages not yet faulted into the process | 7–12 GiB/s ❌ |
> | a fresh source range per copy (no reuse) | 11.6 GiB/s ❌ |
> | overlayfs vs ext4 backing | both fast ❌ |
> | a new device allocation per copy vs one reused buffer | **both slow** — not this either ❌ |
> | size sweep 1–256 MiB on a plain ext4 mapping | no threshold, 12 GiB/s throughout ❌ |
> | page-unaligned range start (safetensors starts at +1144) | no effect ❌ |
> | bf16 view vs uint8 | no effect ❌ |
>
> ### One factor that *is* confirmed: the mapping is writable
>
> `/proc/self/maps` while safetensors holds a checkpoint open, next to our own
> mapping of the same file:
>
> ```
> 75d757200000-75d845520000 rw-p ... model-00001-of-00005.safetensors   <- safetensors
> 75d468a00000-75d556d20000 r--p ... model-00001-of-00005.safetensors   <- ours
> ```
>
> **safetensors maps a read-only checkpoint `rw-p`** (MAP_PRIVATE, writable).
> Isolating that one variable — same file, same bytes, plain `mmap` +
> `torch.frombuffer`, no safetensors and no vLLM — is a **dependency-free
> reproducer**:
>
> | mapping | 64 MiB copy | rate |
> |---|---:|---:|
> | `MAP_PRIVATE \| PROT_READ` | 6.5 ms | **9 874 MiB/s** |
> | `MAP_PRIVATE \| PROT_READ\|PROT_WRITE` | 28.8 ms | **2 223 MiB/s** |
>
> A **4.3× penalty**, stable from 4 MiB to 512 MiB, independent of alignment and
> dtype — and that turned out to be only the mild half of it.
>
> ### The second variable: whether the pages are already resident
>
> Adding one more dimension — does the CPU touch the mapping before the copy? —
> produces a third regime, and this is the one real loads fall into:
>
> | source mapping | pages resident first | 32 MiB copy | rate |
> |---|---|---:|---:|
> | `r--p` | yes | 3.7 ms | **8 573 MiB/s** |
> | `rw-p` | no | 14.4 ms | 2 229 MiB/s |
> | **`rw-p`** | **yes** | **16 021 ms** | **2.0 MiB/s** |
>
> Roughly **4 400×**. Reproduced back to back (16 020.5 / 16 020.1 ms) and at 256 MiB
> (128 s, also 2.0 MiB/s), so the cost tracks the number of resident pages rather
> than being a fixed penalty. Reproducer:
> [`benchmarks/repro-mmap-prot.py`](../benchmarks/repro-mmap-prot.py).
>
> This is what the real load sits in — 211 MiB/s is between the two writable
> regimes, which is what you would expect when only part of the mapping has been
> faulted in by the time each tensor is copied. It also matches the per-tensor shape
> above: ~1 s for anything past the threshold, near-free below it.
>
> Breaking copy-on-write page by page when the range is mapped for DMA is the
> obvious reading — but that is our inference, not a verified explanation.
>
> ### Where the writable mapping comes from — not safetensors
>
> safetensors maps read-only (`map_copy_read_only`, confirmed in the v0.8.0 source
> and observable with `framework="np"`, which yields `r--p`). Its **PyTorch** path
> does not use that mapping at all; it calls
> `torch.UntypedStorage.from_file(filename, shared=False, nbytes=size)`, and PyTorch
> maps the file **writable** because a torch storage must be mutable. So every tensor
> any framework loads from a safetensors checkpoint on the PyTorch path is backed by
> `rw-p`.
>
> ### Still open
>
> - **`HSA_USE_SVM=0` is not the lever.** Suggested by
>   [ROCm#2433](https://github.com/ROCm/ROCm/issues/2433); measured here it leaves the
>   pathological case unchanged (16 036 vs 16 020 ms) and makes the read-only fast
>   path *worse* (8 905 → 844 MiB/s).
> - **A 17× asymmetry between the two TP ranks** in a single load (~190 ms vs ~11 ms
>   per ioctl), which we could not reproduce with concurrent processes, memory
>   pressure, or torchrun + RCCL.
> - **Whether a passthrough guest is required.** We have no bare-metal machine.
>   [ROCm#5952](https://github.com/ROCm/ROCm/issues/5952) reports `svm_range_*` hogging
>   CPU during model loading on bare-metal RDNA3, which is suggestive but is a crash
>   report, not this.
>
> **Not yet reported upstream.** Our first attempt at a root cause for this section
> was disproved by our own minimal reproducer, so the bar for the second one is that
> a stranger can run it and see the same thing. The reproducer is
> [`benchmarks/repro-mmap-prot.py`](../benchmarks/repro-mmap-prot.py); if you can
> account for the remaining gap between it and a real load, that is the missing piece.
>
> The workaround is what made the [five-model benchmark campaign](benchmarks.md)
> practical: model swap cost fell from 5–10 minutes to 2.5–52 seconds.
> The original investigation notes are kept below for the record.

### Historical record — superseded, kept for the trail

*Everything below this line is the original 2026-07-24 investigation, before the
per-tensor shape and the writable-mapping finding above. It contradicts the
current conclusions in places (it still calls the 20× gap unexplained, lists
candidates that have since been tested, and proposes a `py-spy` step that was
since done with `perf`). It is kept because the record of getting it wrong is
part of what this section is for — not because it is current.*

**Measured on the verified configuration (2026-07-24):**

| | value | effective rate |
|---|---|---|
| Disk, raw sequential read (`dd iflag=direct`) | 1.5 GB/s | — |
| Qwen3-8B, 15.6 GiB, **BF16, no quantisation** | 206 s | **77 MB/s** |
| gemma-4-12B, 9.56 GiB, **w4a16** | 328 s | **29 MB/s** |

A 9.56 GiB file at disk speed would take ~6.4 s. It takes 328 s.

**Disproven:**

- ❌ **Disk throughput** — 1.5 GB/s measured directly on the same file.
- ❌ **The disabled auto-prefetch.** vLLM logs
  `Auto-prefetch is disabled because the filesystem (EXT4) is not a recognized
  network FS (NFS/Lustre)` and suggests `--safetensors-load-strategy=prefetch`.
  We set it, confirmed it took effect (`safetensors_load_strategy: 'prefetch'`
  in the startup args), and the load time was **328 s vs 326 s — unchanged**.
  That log line is a red herring for this bottleneck.
- ❌ **Serving the weights over NFS** to satisfy the filesystem check — pointless
  given the above, and it would add network overhead to a local disk.
- ❌ **Swapping / memory pressure** — swap usage was 0 throughout.

**One confirmed contributing factor:** quantised weights are repacked at load
time. The log shows `Using RDNA3W4A16LinearKernel for CompressedTensorsWNA16`,
and the effect is visible: the 12B w4a16 checkpoint is **40% smaller** than the
8B BF16 one yet takes **60% longer** to load. This is additive, not the root
cause — it does not explain the BF16 case at 77 MB/s.

**Still unexplained:** the ~20× gap that remains even without quantisation.
Candidates, none tested: per-tensor synchronous host→device copies with no
pipelining; TP=2 weight-sharding overhead; Python-side safetensors
deserialisation cost.

**How to close it:** attach `py-spy dump` / `py-spy record` to a worker during
the load phase and see where the time actually goes. That is the obvious next
step and has not been done.

**Note on RAM.** Adding host RAM is a *separate* fix for a *different* problem:
vLLM `mmap`s the whole checkpoint, so a 21.67 GiB file cannot map into a
21.43 GiB guest regardless of free memory (the limit is `MemTotal`). That
ceiling is real, but raising it is not expected to fix the throughput gap above.


---

## What we looked for before reporting any of this

Checked 2026-07-26, so that nobody repeats the search:

| candidate | verdict |
|---|---|
| [safetensors#183](https://github.com/huggingface/safetensors/issues/183) and [diffusers#2507](https://github.com/huggingface/diffusers/issues/2507), "loading directly to GPU is slower than to CPU then moving" | Not the same. Both are NVIDIA V100 / CUDA 11.6 from 2023, and the gap is 1.2x. Our mechanism is in `kfd_ioctl_svm`/`hmm_range_fault`, which does not exist on NVIDIA, and our gap is 4x to 4400x. The same shape of symptom, an unrelated cause. |
| [ROCm#2433](https://github.com/ROCm/ROCm/issues/2433), an SVM change in ROCm 5.6 that slowed `hipHostRegister`, fixed by `HSA_USE_SVM=0` | Not the same, and tested: `HSA_USE_SVM=0` leaves the pathological case untouched (16 036 ms against 16 020 ms) and makes the read-only fast path *worse* (8 905 → 844 MiB/s). |
| [ROCm#5952](https://github.com/ROCm/ROCm/issues/5952), SVM mapping failure during sequential model loads on RDNA3 | Possibly the same subsystem. Theirs crashes, ours crawls, but both are `svm_range_*` in amdgpu during weight loading on RDNA3, and their log says "VRAM loading crawls extremely slowly". Notably bare metal. |

No existing report of the writable-mapping performance cliff was found.

### Two things we decided *not* to report, and why

- **PyTorch, about `UntypedStorage.from_file(shared=False)` mapping writable.** That
  is what exposes every PyTorch user to the mmap problem, but mapping a mutable
  storage writable is a defensible contract. If ROCm answers that applications should
  avoid the pattern, that is the moment to open a PyTorch conversation, with their
  answer as the reason.
- **safetensors.** Investigated and cleared: it maps read-only
  (`map_copy_read_only`, verified in the v0.8.0 source and observable by opening the
  same checkpoint with `framework="np"`, which yields `r--p`). The writable mapping
  comes from the PyTorch path it delegates to.
