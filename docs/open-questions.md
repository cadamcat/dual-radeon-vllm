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

## 3. Is `COLLTRACE=OFF` alone sufficient?

We applied `NDEBUG` globally, which removes **both** hostcall sources at once
(device `assert()` and, indirectly, the trace path). We never tested
`-DCOLLTRACE=OFF` on its own.

If device `assert()` alone is enough to pull in `__assert_fail`, then
`COLLTRACE=OFF` by itself will **not** fix it — but this is untested. Worth
knowing, because `COLLTRACE=OFF` is a supported upstream option whereas
`add_compile_definitions(NDEBUG)` is a patch.

---

## 4. Does virtualized Instinct hit this?

The mechanism depends on the *root port*, not the GPU. A QEMU `pcie-root-port`
advertises `AtomicOpsCap: Routing-` regardless of what is behind it, so
passthrough Instinct should be affected too. We have no Instinct hardware to
confirm. If true, this materially raises the severity of the upstream issue,
since it would mean RCCL collectives are broken in virtualized datacentre
deployments and not merely on consumer desktops.

---

## 5. Can the guest be given PCIe atomics instead?

We concluded no, without exhausting it: QEMU 11.0.2's `pcie-root-port` device
model does not implement AtomicOp completion/routing, and there is no PVE-level
switch. Patching QEMU was out of scope. Additionally our host's own Zen 1 root
port reports `Routing-`, so even a fixed QEMU would not have helped *us* — but it
might help someone on a newer host platform. Untested.

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

## 8. Why is weight loading 20–50× slower than the disk?

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
