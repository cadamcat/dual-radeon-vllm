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

## 2. Does this reproduce on bare metal? — **ANSWERED: yes**

> **Closed 2026-08-25.** @adderek reproduced it on bare metal with IOMMU
> entirely disabled, in [ROCm#6520](https://github.com/ROCm/legacy-rocm-build/issues/6520);
> [root-cause.md](root-cause.md) has carried that since, listing
> VFIO/virtualisation as **not necessary** to the mechanism. This section still
> asked a reader to go and settle it, which it should not have. The reasoning
> below is left as the inference it was before that report arrived.

We have no bare-metal multi-GPU Radeon machine. Our claim that the public
bare-metal reports share this root cause is an **inference** from:

- the mechanism requiring only "no AtomicOp routing", which is common on
  consumer chipsets, and
- the reported fix (downgrade to the 7.1.1 build) being exactly "use a library
  with no hostcall requirement".

Neither reporter posted `AMD_LOG_LEVEL=4` output, so the
`Pcie atomics not enabled` line has not been observed on those machines.

What would have closed it, and did: `AMD_LOG_LEVEL=4` output from a bare-metal
dual-Radeon box. @adderek's is in
[ROCm#6520](https://github.com/ROCm/legacy-rocm-build/issues/6520).

---

## 3. Is `COLLTRACE=OFF` alone sufficient? — **ANSWERED: no**

We applied `NDEBUG` globally, which removes both hostcall sources at once (device
`assert()` and, indirectly, the trace path), and never tested `-DCOLLTRACE=OFF` on its
own. @adderek did, on RCCL 2.27.7 at tag `rocm-7.2.4`, and posted the counts in
[ROCm#6520](https://github.com/ROCm/legacy-rocm-build/issues/6520):

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

## 4. Does virtualized Instinct hit this? — **premise corrected, question still open**

> **Corrected 2026-08-25, same error as §5.** "Regardless of what is behind it"
> is false. QEMU advertises completer support on an emulated root port
> automatically for a single-function device below a root port supporting
> DEVCAP2, and has since 8.1.0; it declines for a multifunction device, which is
> what the Proxmox default produces. So a passthrough Instinct is affected when
> it is passed multifunction and not when it is passed as `.0`, exactly as here.
> That narrows the question rather than answering it: whether datacentre
> deployments pass multifunction is not something this repository knows.

The mechanism depends on the *root port*, not the GPU. A QEMU `pcie-root-port`
advertises no AtomicOp completer support (`32bit- 64bit-`) when the device below
it is passed as multifunction, so passthrough Instinct should be affected too. We have no Instinct
hardware to confirm. If true, this materially raises the severity of the upstream issue,
since it would mean RCCL collectives are broken in virtualized datacentre
deployments and not merely on consumer desktops.

---

## 5. Can the guest be given PCIe atomics instead? — **one reason we ruled it out was wrong**

QEMU 11.0.2's `pcie-root-port` advertises no AtomicOp completer support
(`32bit- 64bit-`) and there is no PVE-level switch, so the guest cannot have
atomics as things stand. Patching QEMU was out of scope and remains so.

> **Wrong, and expensively so. Corrected 2026-08-23.** The observation was right
> — the root port did report `32bit- 64bit-` — but the diagnosis was not.
> Nothing needed patching. QEMU has advertised AtomicOp completer support on an
> emulated root port automatically since v8.1.0, in
> `vfio_pci_enable_rp_atomics()`, and 11.0.2 has that code. It declines to do it
> for a *multifunction* device, and this VM passed each card's HDMI audio
> function alongside the GPU, so the function never ran. Changing
> `hostpci0: 0000:0b:00` to `hostpci0: 0000:0b:00.0` gives the guest
> `32bit+ 64bit+`, and stock RCCL 2.30.4 then completes collectives that fail
> without it. The A/B, including the revert, is in
> [vfio-atomics.md](vfio-atomics.md).
>
> The two questions the paragraph below poses as untested are both answered yes:
> QEMU can advertise completer support on an emulated root port, and the request
> does survive the VFIO path. This section had already worked out that a
> QEMU-side fix was "a real avenue for *us*" and then stopped one step short of
> trying it. That step was a config edit and a reboot.

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

## 6. Performance impact of removing hostcall: measured 2026-09-04, and it is segmented

This section said *assumed zero, not measured* from the day the fix was found
until 2026-09-04. The reason was platform, not inclination: until 2026-08-23
this box had no PCIe AtomicOps, so a library that declares a hostcall could not
be dispatched here at all and there was nothing to compare against. It has
atomics now, and [the ±NDEBUG A/B](../benchmarks/rccl-ndebug-ab-2026-09-04/README.md)
measured the fix on one RCCL 2.27.7 source tree, two builds one line apart, with
the capability present so that both run:

- graph-replayed all-reduce latency, 55 cells, three interleaved sweeps per
  arm: the unfixed arm is **faster at 8 KB** (0.987, slower in 0 of 5 cells),
  **2.7–4.6 % slower from 16 KB to 512 KB** (29 of 30 cells), and **no different
  from 2 MB up** (median 1.0001, 11 of 20, p = 0.82);
- end to end, 60 served requests: decode differs by ≤ 0.6 % in every cell, five
  of six nominally favouring the unfixed arm;
- twelve correctness cases pass under both libraries, in both rounds.

So the answer is neither "zero" nor "free": free where a batch-1 decode step
runs, a few percent in a band above it, free again where the collective is
bandwidth-bound. The argument this section used to make — that `assert()` and
trace `printf` never execute on the working path — still holds, and it is not
what the band measures: two builds differ in code layout as well as in a flag,
and the 8 KB reversal is unexplained.

---

## 7. How far forward does the fix hold?

Verified on ROCm 7.13 and 7.14 (technology-preview stream) and against RCCL
2.27.7 / 2.30.4. ROCm ships roughly every six weeks. We do not know when this
will break, nor when upstream will make it unnecessary. Treat the version table
in the README as a snapshot, not a guarantee.

---

## 8. Why is weight loading 19–48× slower than the disk? — **two effects, both now established**

> **Read this section as: there are two things here.** A 4× to 8× penalty whose
> mechanism AMD confirmed down to the kernel line, and on top of it a collapse to
> whole seconds per copy on guest kernel `7.0.0-28-generic` that is a separate
> regression. The second one is a backport that took `c08972f55594` without its
> follow-up `342981fff328`, which is why every timing lands on an exact multiple of
> the 1000 ms `HMM_RANGE_DEFAULT_TIMEOUT`. **That is now proven by applying it**: the
> same kernel rebuilt with `342981fff328` applied does the copy in 17.0 ms instead
> of 16 019.7, with the reproducer's other cases unmoved — see "Proven by applying the missing commit"
> below, which also states what the test does not settle. An earlier version of this
> section claimed a root cause that we disproved ourselves, and a later one called
> the whole thing long-standing rather than a regression, which the kernel
> comparison overturned. Both are kept below.
>
> ### Measured on the shipped fix, and two of our own numbers do not survive — 2026-08-23
>
> Everything below this subsection was measured on `7.0.0-28`, which nobody runs
> now. Re-run on Canonical's `7.0.0-30` with page cache controlled per cell
> ([`loader-flag-kernel-30.json`](../benchmarks/loader-flag-kernel-30.json), 89
> cells, four load paths, four checkpoints, each cell a fresh process):
>
> | checkpoint | cache | default | `eager` | clone | `pread` |
> |---|---|---:|---:|---:|---:|
> | gemma-4-12B, 9.56 GiB, 1 shard | warm | 4.96 | 10.23 | **2.51** | 4.79 |
> | gemma-4-12B | cold | 8.15 | 11.46 | **4.71** | 7.16 |
> | Qwen3-8B, 15.26 GiB, 5 shards | warm | 6.78 | 15.93 | **4.49** | 10.48 |
> | Qwen3-8B | cold | 13.36 | 18.26 | **7.43** | 13.03 |
> | gemma-4-31B, 21.67 GiB, 1 shard | cold | 88.52 | did not fit | **11.75** | 17.08 |
> | gemma-4-26B-A4B MoE, 35 743 tensors | cold | **19.17** | not run | 19.95 | 15.87 |
>
> **The clone is worth 1.5× to 2.0× while the checkpoint fits in RAM, not the
> 3.9× to 5.6× this repository published on 2026-07-28.** That figure came from a
> run with no control over page cache which loaded three models in sequence, so
> every cell inherited its predecessor's cache. It does not reproduce.
>
> **The 4×–8× is not a constant ratio, and the resident set says why.** `VmHWM`
> cannot separate an anonymous page from a mapped file page. Sampling `RssAnon`
> and `RssFile` *during* the load — safetensors unmaps a shard the moment the
> iterator leaves it, so an end-of-loop sample sees nothing — gives the mechanism
> directly for the first time here. On the 31B the default path peaks at
> **21 390 MiB `RssAnon` against 782 MiB `RssFile`**; with the clone those swap
> places, 843 against 21 479. Breaking copy-on-write converts the whole
> checkpoint into private dirty memory, which is no longer evictable, so a
> checkpoint that is a large fraction of host RAM drives the loader into swap.
> That is where the 7.5× comes from, and it is why sharding bounds the damage:
> Qwen3-8B is the same 15.26 GiB in five shards and peaks at 4 535 MiB of
> `RssAnon`, roughly one shard.
>
> **`safe_open(..., backend="pread")` has shipped since safetensors 0.8.0 and we
> missed it for a month.** It is absent from the Python docstring and named in
> the v0.8.0 release notes, "useful for specific archs/platforms". It never maps
> the file, so the trigger cannot arise; it returned byte-identical tensors for
> all 1334 tensors of a 12B shard. It is slower than the clone everywhere here
> except the MoE checkpoint, where it wins, and its peak resident set is 2.4 to
> 3.9 GiB against 4.9 to 21.8 GiB for every other path.
>
> **The clone is not free.** On the MoE checkpoint it is no better than doing
> nothing and possibly slightly worse, 19.95 against 19.17 with overlapping
> ranges: almost nothing there is large enough to pay the copy-on-write cost, so
> the extra host copy is pure overhead.
>
> **What this does not settle.** The 2026-07-28 "healthy kernel" column does not
> reproduce on `-30` even when its ordering is replayed with no cache control
> (8B baseline 13.87 s against the 36.4 s reported then; 31B 126.61 s against a
> >900 s timeout). Page cache is therefore not the whole explanation, and two
> things changed between the sessions: the kernel state, and this VM's PCIe
> atomics, off in July and on now. The 32 MiB reproducer moved only 17.0 → 15.3 ms
> across the same change, so the per-copy path is not where the difference lives,
> but nothing here says where it does. Treat the July table as superseded rather
> than reconciled.
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
> **vLLM already ships something that helps, and it is worse than the clone.**
> `--safetensors-load-strategy eager` reads each shard with `f.read()` into
> anonymous memory instead of mapping it, so it sidesteps the bad path too.
> Measured on 2026-07-27 with one script that iterates the checkpoint and copies
> every tensor to the device, on kernel `7.0.0-28`:
>
> | model | baseline | `eager` | clone |
> |---|---:|---:|---:|
> | Qwen3-8B, 5 shards | 87.0 s / 4.95 GiB RSS | 18.0 s / **12.01 GiB** | **5.9 s** / 4.98 GiB |
> | gemma-4-12B, 1 shard | 146.0 s / 10.31 GiB | 10.3 s / **20.31 GiB** | **4.4 s** / 10.83 GiB |
> | gemma-4-31B, 1 shard | 338.3 s / 21.61 GiB | *not runnable* | **11.6 s** / 21.71 GiB |
>
> `eager`'s peak RSS is about twice the shard: the `bytes` buffer plus the tensors
> deserialised out of it. On a single-shard checkpoint that is twice the model, so
> the 31B does not fit in this machine's 20.3 GiB of available RAM at all, while
> the clone adds 0.1 GiB. Cloning is also 2.3–2.6× faster than `eager` where
> `eager` runs, because `f.read()` finishes the whole shard before the first copy
> starts. These are loader-path numbers from an isolated script, not end-to-end
> `Loading weights took` figures; the table above is the end-to-end one.
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
> **Why ~1 s and not something proportional to size — 2026-07-30.** Each timeout
> window covers 2 MiB of *resident* source. That falls out of the reproducer,
> 32 MiB in 16 windows and 256 MiB in 128, and @shineday999 confirmed it from
> outside on a different board and a different ROCm: 8 MiB in exactly four
> windows, with the residual real work after subtracting the windows landing at
> 1575-1695 MiB/s against our 1624. So window count is `resident_bytes / 2 MiB`,
> not tensor size. The tensors above are barely resident when the copy starts,
> since nothing reads them first, which is why an 8 MiB and a 96 MiB tensor both
> come out at one window. **That last step is inference:** it follows from the
> granularity, but we never sampled residency per tensor to check it directly.
>
> **Where the time goes.** `perf` on a loading worker: **98.7 % in
> `kfd_ioctl_svm → svm_range_validate_and_map → hmm_range_fault`**; `strace`: **54
> ioctls in a 12-second window, ~189 ms each**. Reproduces on **both ROCm 7.0 and
> 7.14** — but that varies the userspace only. It **is** a regression, in the guest
> kernel; see the 2026-07-27 section below.
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
> This is what the real load sits in. The three models above work out to 76, 30 and
> 39 MiB/s (15.26 GiB / 206 s, 9.56 / 328, 21.67 / 569), between the two writable
> regimes, which is what you would expect when only part of the mapping has been
> faulted in by the time each tensor is copied. It also matches the per-tensor shape
> above: ~1 s for anything past the threshold, near-free below it.
>
> ### The COW reading was right, and AMD gave the line — 2026-07-27
>
> @ashetaia-amd confirmed the mechanism in
> [ROCm#6523](https://github.com/ROCm/legacy-rocm-build/issues/6523) and named the site. We
> traced the whole chain in the upstream tree afterwards; all three hops are real:
>
> ```c
> // drivers/gpu/drm/amd/amdkfd/kfd_svm.c:1777, inside svm_range_validate_and_map
> readonly = !(vma->vm_flags & VM_WRITE);
> // :1807 passes it on
> r = amdgpu_hmm_range_get_pages(&prange->notifier, addr, npages, readonly, owner, range);
> // drivers/gpu/drm/amd/amdgpu/amdgpu_hmm.c:188
> if (!readonly)
>         hmm_range->default_flags |= HMM_PFN_REQ_WRITE;
> ```
>
> **The permission is taken from the VMA rather than from what the GPU is about to
> do with the range.** A host→device copy only *reads* the source, but because the
> VMA carries `VM_WRITE`, KFD asks `hmm_range_fault()` for write access, which
> breaks copy-on-write on every resident page. That is the 3–4× penalty, it is on
> both kernels, and the line predates them both.
>
> It also explains the two-regime shape recorded above without any extra
> assumption: pages the CPU has not touched are not COW-shared yet, so the
> not-resident case never pays for the break.
>
> AMD's reading of the 700× is that it is a *separate* kernel regression
> amplifying this same fault path, with the linear ~2 ms per page and the flat
> per-filesystem offset pointing at a fixed per-page synchronous cost, and
> VFIO/IOMMU as a prime suspect since it is the only environment measured here.
>
> One avenue is closed. Forcing the staged path would sidestep this entirely — with
> pinning disabled the source is never SVM-registered and a CPU *read* into a
> staging buffer does not break COW — but @ashetaia-amd states that the long-term
> fix taken in ROCm/rocm-systems#6676 does not help this copy path, so
> `ROC_FORCE_STAGED_D2H` is not the remedy to wait for.
>
> ### It is a kernel regression inside Ubuntu's 7.0.0 series — 2026-07-27
>
> Four configurations, one reproducer, and only one of them is slow:
>
> | environment | kernel | `rw-p`, resident, 32 MiB |
> |---|---|---|
> | VFIO guest | **`7.0.0-28-generic`** | **16 019.3 / 16 019.6 / 16 019.9 / 16 020.1 ms** |
> | VFIO guest | `7.0.0-14-generic` | 18.6 / 18.9 / 20.2 ms |
> | VFIO guest | `6.8.0-136-generic` | 22.1 / 23.3 / 26.0 ms |
> | bare metal | `7.0.14-4-pve` | 24.1 / 24.3 / 25.1 / 28.6 ms |
>
> The load path follows: **86.7 s on `-28` against 14.4 s on `-14`** for the same
> 15.26 GiB checkpoint, and 14.6 s on `6.8.0-136`.
>
> **The first two rows are the ones that matter.** Same guest, same ROCm 7.14
> userspace, same container image, same reproducer, only the kernel ABI swapped —
> so **passthrough is not the ingredient**, which was the leading suspicion both
> here and at AMD. It also reproduces @loreggia's boundary in
> [ROCm#5952](https://github.com/ROCm/legacy-rocm-build/issues/5952) from a different distro on
> a different card, and [ROCm#6508](https://github.com/ROCm/legacy-rocm-build/issues/6508)
> reports a KFD work queue deadlock specific to the same `-28`.
>
> **This is the kernel Ubuntu ships today.** `linux-image-generic-hwe-24.04`
> resolves to `7.0.0-28.28~24.04.1` out of `noble-updates` and `noble-security`,
> so it is not a version anyone has moved past — it is what a 24.04 machine gets
> by letting updates run, on the distro ROCm supports first.
>
> **No longer true, corrected 2026-08-23.** The paragraph above was accurate when
> written and is left standing because the argument it supports — that this was
> not an exotic kernel — depended on it. `linux-hwe-7.0` is now
> `7.0.0-30.30~24.04.1` in both `noble-updates` and `noble-security` (published
> 2026-08-20, with `-31` in proposed), and that changelog carries
> `drm/amdgpu: drop retry loop in amdgpu_hmm_range_get_pages`, i.e. the follow-up
> commit this section is about. Checked against the Launchpad archive and the
> published changelog, not against a summary. **So the remedy on 24.04 is now an
> upgrade, and the rebuilt module below is evidence rather than advice.** Two
> things do not change: the residual penalty on *writable* mappings, which that
> commit does not touch and which measured 4.8× on `-30` itself (15.3 ms against
> 3.2 for 32 MiB — much smaller in absolute terms than the sixteen seconds it
> used to sit next to), and the fact that LP#2161985 is still `New` and
> untriaged, so this report did not cause the fix — it came through the ordinary
> stable route.
>
> It does not localise the change further than ABI `-14` to `-28`, and the archive
> carries no generic kernel between them. The bare-metal row is a Proxmox `7.0.14`
> build, so it is not evidence that a newer *Ubuntu* kernel is clear; we have no
> data on one.
>
> ### The commit — 2026-07-27
>
> `-28` picked up **`c08972f55594`** (`drm/amdgpu: fix amdgpu_hmm_range_get_pages`,
> 2026-02-18) in that upload itself; it is under `-28` in the changelog and absent
> from the `-26` and `-14` sections. It does **not** carry the follow-up that
> repairs the consequence, **`342981fff328`** (`drm/amdgpu: drop retry loop in
> amdgpu_hmm_range_get_pages`, 2026-05-29), which has never been applied. That
> commit's own message describes what we measured:
>
> > the captured notifier_seq is no longer refreshed across retries ... the "goto
> > retry" therefore degenerates into a busy spin that simply burns CPU for the
> > full HMM_RANGE_DEFAULT_TIMEOUT (~1s) window before finally bailing out with
> > -EAGAIN ... it actively hurts the KFD userptr stack
>
> `HMM_RANGE_DEFAULT_TIMEOUT` is **1000** (`include/linux/hmm.h`). Every timing in
> this section is an integer multiple of it plus ~20 ms of real work:
>
> | measured | windows |
> |---|---|
> | 16 019.3 / 16 019.6 / 16 019.9 / 16 020.1 ms | 16.0 |
> | 17 020.5 / 17 020.4 ms (overlayfs, tmpfs) | 17.0 |
> | per-tensor 1005 / 1042 / 1522 ms | 1.0 / 1.0 / 1.5 |
>
> So the sub-millisecond reproducibility that looked like a fixed per-page cost is
> a `jiffies` timeout, and the flat one-second gap between filesystems is one extra
> retry. **This also joins the two halves.** Breaking copy-on-write is what
> advances the notifier sequence, so a read-only mapping never invalidates, never
> gets `-EBUSY`, and never enters the futile retry. The VMA-permission bug is the
> trigger; `c08972f55594` turned each trigger into a full second.
>
> **Why the pair got split.** `342981fff328` carries neither a `Fixes:` tag nor
> `Cc: stable@vger.kernel.org`, so it was never queued for the stable trees, while
> `c08972f55594` did reach `-28`. A fix that is not tagged does not follow the
> commit it repairs. That makes this not an Ubuntu packaging slip but an upstream
> tagging gap, and any series tracking 7.0.y that took the first commit is in the
> same state. `Documentation/process/stable-kernel-rules.rst` lets anyone request
> an already-mainlined commit by mail to `stable@vger.kernel.org`.
>
> For the record, `342981fff328` is Honglei Huang's patch, reviewed by Christian
> König, committed by Alex Deucher; `c08972f55594` is Christian König's, signed
> off by Alex Deucher.
>
> ### Proven by applying the missing commit — 2026-07-28
>
> **Renamed 2026-08-29.** This section, and the sentence above pointing at it,
> said "proven by revert". That is the opposite of what was done: the missing
> commit `342981fff328` was **applied** to Ubuntu's `-28` source, not removed.
> The published article carries the same correction, dated the same day; the
> phrase also stood in `README.md` and is corrected there. The one place a
> revert is still the right word is §7's VFIO configuration A/B, where the
> `hostpci0` line really was put back.
>
> `amdgpu.ko` built from the `linux-hwe-7.0` 7.0.0-28.28~24.04.1 source with
> `342981fff328` as the only change, then swapped into the running `-28` kernel.
> Same machine, same ROCm 7.14 userspace, same reproducer, same file, same
> gcc 13.3.0 Canonical built `-28` with; `.config` taken verbatim from
> `/boot/config-7.0.0-28-generic`, with only the Rust options dropped because the
> box has no `rustc`, and no embedded `.BTF` because there is no locally built
> `vmlinux`. Neither touches this code path.
>
> | amdgpu | `rw-p` resident, 32 MiB |
> |---|---|
> | stock `-28` | 16 019.7 / 16 019.6 / 16 019.1 / 16 019.7 / 16 019.0 ms |
> | `-28` + `342981fff328` | **17.0 / 17.0 / 17.0 ms** |
>
> The reproducer's other three cases do not move: `r--p` resident stays at
> 3.0-3.1 ms and `rw-p` not-resident at 14.5-14.6 ms, before and after. Only the
> case that entered the futile retry changes, which is what applying the one commit
> that fixes that retry should do and nothing more.
>
> The residual is what the arithmetic predicted. 16 019.7 ms is sixteen 1000 ms
> windows plus 19.7 ms of real work, and 17.0 ms lands in the same band
> `7.0.0-14-generic` produces natively, 18.6 to 20.2 ms.
>
> **One thing this does not settle.** Building a module from Ubuntu's source is not
> the same as Canonical shipping one, and the machine is still a VFIO guest, so it
> remains untested whether a bare-metal Ubuntu `-28` behaves identically.
>
> The writable mapping is not cleared either. Across the three fast kernels a
> read-only copy of the same bytes took 3.0 to 5.6 ms against 18.6 to 28.6 ms
> writable and resident, so **4× to 8×**, not the flat 3–4× quoted earlier in this
> section — the read-only side got faster on the newer kernels while the writable
> side did not. The patched module measures 17.0 against 3.0 ms, i.e. 5.6×, right
> inside that band. That penalty is everywhere, and `342981fff328` does not address
> it; what `-28` added on top was the stack of one-second timeouts.
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
>   [ROCm#2433](https://github.com/ROCm/legacy-rocm-build/issues/2433); measured here it leaves the
>   pathological case unchanged (16 036 vs 16 020 ms) and makes the read-only fast
>   path *worse* (8 905 → 844 MiB/s).
> - **A 17× asymmetry between the two TP ranks** in a single load (~190 ms vs ~11 ms
>   per ioctl), which we could not reproduce with concurrent processes, memory
>   pressure, or torchrun + RCCL.
> - ~~**Whether an affected *Ubuntu* kernel on bare metal behaves the same.**~~
>   **Closed 2026-07-30, by someone else.** We never had such a machine: our
>   bare-metal row is a Proxmox `7.0.14` build, so it said nothing about
>   Ubuntu's `-28` outside a VM. @shineday999 reported one in
>   [ROCm#6523](https://github.com/ROCm/legacy-rocm-build/issues/6523) — RX 7900 XTX,
>   ROCm 7.2.1, `7.0.0-28`, no virtualisation — with the same pathology and the
>   same integer window counts. What remains untested is a module as *Canonical*
>   ships it: both of us ran either a self-built module or a userspace
>   workaround.
>
> **Reported upstream as [ROCm#6523](https://github.com/ROCm/legacy-rocm-build/issues/6523)**, where
> AMD confirmed the copy-on-write trigger and named `kfd_svm.c`, and to Ubuntu as
> [LP#2161985](https://bugs.launchpad.net/ubuntu/+source/linux-hwe-7.0/+bug/2161985)
> against `linux-hwe-7.0`, asking for the backport. Our first attempt at a
> root cause for this section was disproved by our own minimal reproducer, so the bar
> for the second one was that a stranger can run it and see the same thing. The
> reproducers are [`benchmarks/repro-mmap-prot.py`](../benchmarks/repro-mmap-prot.py)
> and, for machines with no PyTorch,
> [`benchmarks/repro-mmap-prot.hip.cpp`](../benchmarks/repro-mmap-prot.hip.cpp).
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

> **Unit note, added 2026-08-29.** This table's rates are MiB/s labelled MB/s.
> Against the 1.5 GB/s above, its own two rows give **19× and 52×**, not the
> 19–48× in this section's heading, which divided by a decimal-MB rate. The
> heading is left as written because two other documents cite that band; the
> derivation is here. Nothing downstream depends on which of 48 and 52 is used:
> both halves of the answer below are measured directly.

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

## 9. Why is the hybrid-SSM model's *baseline* decode only 12.1 tok/s? — **ANSWERED: the checkpoint's kernel**

> **Answered 2026-08-27.** The checkpoint is asymmetric int4, so every quantised
> linear misses vLLM's native gfx1100 W4A16 kernel and runs on Triton instead.
> The same model with a symmetric checkpoint is **3.24× faster at 1 K**, and the
> penalty is a flat ~60 ms per decode step across a 32× context range
> ([benchmarks/w4a16-symmetry](../benchmarks/w4a16-symmetry/)).
>
> **What is answered:** the 2× against llama.cpp at short context is the
> quantisation kernel the checkpoint lands on, not the architecture. **What is
> not:** the per-kernel profile of gemma-4-31B that the paragraph below asks
> for was never run, and is no longer needed to answer this question; it would
> only refine how the remaining Triton time divides.

<details>
<summary>How it was reasoned about before it was answered (kept for the mistake)</summary>

The reasoning below eliminated the right suspect on a bad comparison, and is
left in place because the mistake is the reusable part: "gemma-4-31B is also
w4a16" is true and irrelevant, because *also w4a16* is not *also the same
kernel*. gemma-4-31B is symmetric and runs the native HIP kernel; this model
is asymmetric and runs Triton. Two kernels behind one word in a config file.
The 77 % figure below was right the whole time.

*As written before 2026-08-27:* the context **slope** is settled — it is the
paged-attention fallback, see [hybrid-decode-on-rdna.md](hybrid-decode-on-rdna.md).
The baseline is not.

At 512 tokens of context, before the slope has done anything, `Qwen3.6-27B` does
12.1 tok/s under vLLM while llama.cpp on the same two cards does 24.89. Something
costs 2× before long context is involved at all.

`architecture-notes.md` offers four candidates, all about the gated-delta-net
kernels. The profile makes all four look too small to matter: at 1 K context the
two GDN kernels together are **0.56 %** of decode time. The dominant item is
`triton_w4a16_gemm_kernel` at **77 %** — but gemma-4-31B is also w4a16 and decodes
at 43.2 tok/s, so "the quantised GEMM is slow" is not an answer either.

What would settle it: profile gemma-4-31B the same way and compare the per-step
breakdown. If its w4a16 GEMM share is much lower, the question becomes what about
this model's shapes makes the same kernel so much more expensive. That run has not
been done.

</details>

---

## 10. Does the bandwidth-utilisation derivation hold on gfx1100? — **not answerable on this hardware**

Every utilisation figure this repository publishes is derived: decode tok/s
times the checkpoint's size, assuming a decode step reads every weight byte from
memory once. On 2026-09-02 that assumption was measured for the first time, on
an A100-SXM4-40GB under `ncu`
([`cuda-a100/campaign-2026-09-02`](../benchmarks/cuda-a100/campaign-2026-09-02/README.md)):
a decode step reads **81.6 %** of the checkpoint on `gemma-4-12B` and **85.6 %**
on `gemma-4-31B`, so the derivation overstates by 17–23 % there, and the two
factors differ by 4.7 % — it is a property of the model, not a constant.

**The same measurement cannot be made on the machine the figures describe, and
on 2026-09-02 the reason was narrowed but not settled.** `rocprofv3` runs in the
VFIO guest and its kernel trace is correct — names, grid sizes and durations all
land. Its counters divide cleanly by hardware block:

    SQ_WAVES            6120     SQ block, the shader engines -- works
    GL2C_EA_RDREQ_32B    0.0     GL2C block, the L2 and its memory interface
    GL2C_EA_RDREQ_64B    0.0
    GL2C_EA_RDREQ_128B   0.0
    GL2C_HIT / _MISS     0.0
    GL2C_MC_RDREQ        0.0
    GRBM_COUNT           0.0     GRBM block
    FETCH_SIZE           0.0     derived from the four GL2C_EA_RDREQ_* above

Each was accepted, profiled six dispatches of a 2048² fp32 matmul, and returned
rows — with every value zero outside the SQ block.

Two things this rules out. It is **not** an unsupported architecture:
`derived_counters.xml` defines `FETCH_SIZE` under `<gfx11 base="common_derived">`
as `(GL2C_EA_RDREQ_32B*32 + _64B*64 + _96B*96 + _128B*128)/1024`, and
`basic_counters.xml` defines every one of those for gfx11. And an earlier
reading here that `TCC_*` counters were "blocked" was **wrong**: `TCC_*` is the
CDNA name for that block, gfx11 calls it `GL2C_*`, and those names simply do not
exist on this ASIC.

**What is left is three explanations, and this campaign separated none of them:**

1. VFIO passthrough does not expose the GL2C and GRBM performance-monitor
   registers, while the SQ block's survive;
2. consumer RDNA3 does not enable those blocks' perfmon at all, in which case
   bare metal reads zero too;
3. `rocprofv3`'s gfx11 path accepts and reports those counters without wiring
   them, which would be an upstream defect.

**So "bare metal would settle it" is not established** — it settles only (1).
What separates all three is one command on any bare-metal gfx1100, which the
community has many of, or a search of ROCm's tracker for GL2C counters reading
zero on gfx11. Neither needs this machine opened.

Until then the gfx1100 figures stand as an upper bound, and the A100's factor is
not transferable to them: those run `RDNA3W4A16LinearKernel` where the A100 runs
Marlin, and a kernel that reads its weights differently would have a different
factor. The unread amount there was 1.89 GB on the 12B and 3.35 GB on the 31B,
and that campaign did not attribute it kernel by kernel.

---

## What we looked for before reporting any of this

Checked 2026-07-26, so that nobody repeats the search:

| candidate | verdict |
|---|---|
| [safetensors#183](https://github.com/huggingface/safetensors/issues/183) and [diffusers#2507](https://github.com/huggingface/diffusers/issues/2507), "loading directly to GPU is slower than to CPU then moving" | Not the same. Both are NVIDIA V100 / CUDA 11.6 from 2023, and the gap is 1.2x. Our mechanism is in `kfd_ioctl_svm`/`hmm_range_fault`, which does not exist on NVIDIA, and our gap is 4x to 4400x. The same shape of symptom, an unrelated cause. |
| [ROCm#2433](https://github.com/ROCm/legacy-rocm-build/issues/2433), an SVM change in ROCm 5.6 that slowed `hipHostRegister`, fixed by `HSA_USE_SVM=0` | Not the same, and tested: `HSA_USE_SVM=0` leaves the pathological case untouched (16 036 ms against 16 020 ms) and makes the read-only fast path *worse* (8 905 → 844 MiB/s). |
| [ROCm#5952](https://github.com/ROCm/legacy-rocm-build/issues/5952), SVM mapping failure during sequential model loads on RDNA3 | Possibly the same subsystem. Theirs crashes, ours crawls, but both are `svm_range_*` in amdgpu during weight loading on RDNA3, and their log says "VRAM loading crawls extremely slowly". Notably bare metal. |

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
