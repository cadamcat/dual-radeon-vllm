# Diagnosis: is this actually your bug?

Work down this page. Each step is cheap, and each one either confirms or
eliminates. **Step 2 is decisive on its own** — if you only do one thing, do that.

---

## Step 0 — Does the shape of the failure match?

| Question | This bug |
|---|---|
| Does single-GPU work fine? | **Yes, always.** If single-GPU also fails, it is something else |
| When exactly does it die? | At the **first cross-GPU collective**, after the communicator has been created |
| Is it deterministic? | **Yes.** Not a race, not load-dependent, not intermittent |
| Do NCCL/HSA env vars change anything? | **No.** If some knob fixes it, it is not this |
| Does it depend on model / framework? | **No.** vLLM, raw `torchrun`, `rccl-tests` all fail identically |

If any answer disagrees, stop — you are probably chasing a different problem.

---

## Step 1 — Platform triage (no compiler needed)

```bash
./diagnose/check-platform.sh                 # or pass a specific library:
./diagnose/check-platform.sh /opt/rocm/lib/librccl.so.1
```

It reports three things and prints a verdict:

1. whether `amdgpu` logged `PCIE atomic ops is not supported`
2. whether the GPUs' upstream ports advertise `AtomicOpsCap: Routing+/-`
3. whether the RCCL you are using contains `hidden_hostcall_buffer`

**"No atomics" + "RCCL needs hostcall" = affected.**

Exit codes: `0` not affected · `1` affected · `2` inconclusive.

> **Caveat — the `kpack` layout.** Recent ROCm distributions installed as pip
> wheels (`_rocm_sdk_libraries`, used by current `rocm/vllm` images) do **not**
> embed device code in `librccl.so`. Its `.hip_fatbin` section is `NOBITS`, and
> the real device image sits in a separate per-architecture container at
> `<sdk>/.kpack/rccl_lib_gfx<arch>.kpack` (magic `KPAK`, compressed). That format
> is opaque to `llvm-readelf`/`llvm-objdump`, so **step 1 cannot inspect a stock
> library on such installs** and will say so rather than guess.
>
> This does not block you: **step 2 is a runtime probe and does not care about
> library layout.** Static inspection still works normally for a library *you*
> build, which is what `build/verify-nohostcall.sh` is really for.

---

## Step 2 — The decisive probe ⭐

This removes RCCL, PyTorch and vLLM from the picture completely. Two kernels,
identical launch path; only one needs a hostcall.

```bash
hipcc --offload-arch=gfx1100 -O2 diagnose/hipgate3.cpp -o hipgate3
./hipgate3
```

**Affected:**

```
--- dev0 ---
[plain    ] launch:no error   sync:no error
[hostcall ] launch:no error   sync:the operation cannot be performed in the present state
```

**Not affected:** both lines report `no error`.

That asymmetry *is* the bug: the platform cannot service a hostcall, so any
kernel that needs one is refused at dispatch. RCCL's kernels need one only
because of debug-time `assert()`.

> Replace `gfx1100` with your architecture (`rocminfo | grep gfx`).

---

## Step 3 — Confirm on the real stack (30 seconds)

Reproduces the production failure without loading a model:

```bash
torchrun --nproc_per_node=2 diagnose/ar.py
```

Broken:

```
RuntimeError: NCCL error: unhandled cuda error
... HIP failure 'the operation cannot be performed in the present state' at .../enqueue.cc:2061
```

Fixed:

```
[rank0] all_reduce OK sum=2.0
[rank1] all_reduce OK sum=2.0
```

Use this — not a vLLM start-up — for every before/after test. It exercises the
same code path in seconds instead of minutes.

---

## Step 4 — See the actual reason in the driver

The error surfaced to userspace (`unhandled cuda error`) hides the cause. Turn
the driver log up:

```bash
AMD_LOG_LEVEL=4 torchrun --nproc_per_node=2 diagnose/ar.py 2>&1 | tee amdlog.txt
grep -nE 'hostcall|AQL dispatch|IllegalState|ncclDevKernel' amdlog.txt
```

You are looking for these four consecutive lines:

```
rocvirtual.cpp:4151  ShaderName : ncclDevKernel_Generic_4(ncclDevKernelArgsStorage<4096ul>)
rocvirtual.cpp:4208  Pcie atomics not enabled, hostcall not supported
rocvirtual.cpp:4636  AQL dispatch failed!
hip_module.cpp:605   hipModuleLaunchKernel: Returned hipErrorIllegalState
```

**This is the single most valuable thing to attach to any bug report** — ours or
AMD's. Neither of the public upstream reports contains it, which is very likely
why they have gone months without a root cause.

Our captured copy: [`diagnose/logs/amdlog-crash-excerpt.txt`](../diagnose/logs/amdlog-crash-excerpt.txt).

---

## Step 5 — Things that will *not* help

Documented so you do not spend a day on them. All were tested; none changed the
signature:

- `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`, `NCCL_CUMEM_ENABLE=0/1`
- `HSA_ENABLE_SDMA=0`, `HSA_FORCE_FINE_GRAIN_PCIE=1`, `HSA_ENABLE_INTERRUPT=0`
- `GPU_MAX_HW_QUEUES=1`, `AMD_SERIALIZE_KERNEL=3`, `HIP_LAUNCH_BLOCKING=1`
- `--enforce-eager`, changing `--distributed-executor-backend`
- adding `iommu=pt` (source reads it as a string for a warning; it gates nothing)
- more guest RAM, a newer guest kernel
- **pipeline parallel instead of tensor parallel** — `ncclSend/Recv` kernels need
  hostcall too (`diagnose/ar_p2p.py` demonstrates this)

The full 12-combination sweep is in [`diagnose/sweep.sh`](../diagnose/sweep.sh).

---

## Step 6 — Fix it

[docs/deploy-vllm.md](deploy-vllm.md) — and note that the rebuild alone is not
enough; there are two further traps that look like unrelated failures.
