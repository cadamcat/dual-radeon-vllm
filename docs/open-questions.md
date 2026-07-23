# Open questions — what we have *not* proven

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
