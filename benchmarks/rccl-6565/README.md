# ROCm#6565 does not reproduce at RCCL 2.30.4 on a contrasting dual-gfx1100 box

[`ROCm/legacy-rocm-build#6565`](https://github.com/ROCm/legacy-rocm-build/issues/6565)
(BoJl4apa, 4 August, formerly `ROCm/ROCm#6565`) reports deterministic silent
corruption of `all_gather`/`reduce_scatter` on 2x Radeon Pro W7800, root-caused
to an init-time `hipMemcpyAsync` H2D whose payload is not in device memory when
`hipStreamSynchronize` returns. RCCL's 8-byte `devRingUserRanks` array stays
zeroed, every gather offset collapses to slot 0, and `all_reduce` is unaffected
because its ring kernel indexes with `ring->index` instead.

They measured it on **RCCL 2.27.7 (ROCm 7.2.3)** and **2.28.3 (ROCm 7.13)**,
"identically". Nothing above 2.28.3 has been reported either way.

This directory is a second dual-gfx1100 machine at **RCCL 2.30.4 / ROCm 7.14**.
**135 of 135 cold communicator initialisations across eight configurations are
correct.** We do not reproduce it.

## What was run

The reporter's own ground-truth reproducer, byte for byte: `rccl_allgather_truth.py`
is extracted verbatim from the issue body (md5 `bffbc297cad9f1956c8bb2b7e8a4bb0f`),
not retyped. Rank `r` fills a tensor with `r`; after `all_gather` every rank must
hold `[0, 1]`, checked against constructed truth rather than against another
collective. One invocation covers 12 cases, `{fp32, fp16, bf16} x n in {1, 2, 64,
4096}`, and prints `ALL CORRECT` only if all twelve pass.

Their failure is deterministic on their box. Ours could have been probabilistic,
so every arm runs the whole thing under a **fresh `torchrun`**, which means a
cold `ncclCommInitRank` each time. A single pass is not evidence of absence.

| stage | arm | env | cold inits | result |
|---|---|---|---:|---|
| 1 | `default` | none | 20 | 20/20 correct |
| 1 | `p2pdisable` | `NCCL_P2P_DISABLE=1` | 20 | 20/20 correct |
| 1 | `prod` | `NCCL_P2P_DISABLE=1 HSA_ENABLE_SDMA=0` | 20 | 20/20 correct |
| 2A | `ch1` | `+ MIN/MAX_NCHANNELS=1` | 15 | 15/15 correct |
| 2A | `ch4` | `+ MIN_NCHANNELS=4` | 15 | 15/15 correct |
| 2A | `ch8` | `+ MIN_NCHANNELS=8` | 15 | 15/15 correct |
| 2A | `ch16` | `+ MIN_NCHANNELS=16` | 15 | 15/15 correct |
| 2A | `shmoff` | `NCCL_SHM_DISABLE=1` | 15 | 15/15 correct |

Totals recomputed from the logs by `../analyze/verify_doc_figures.py`.

## Why the channel sweep is the load-bearing part

A clean run only means something if the configuration could have shown the
defect. The reporter's mechanism is specific: the **first-issued** copy, on
channel 0, is the one lost, on both ranks, while channel 1's lands; and they
state that a naive single-device reproducer of the same alloc → memset →
8-byte async copy → sync → read sequence does **not** reproduce (0/200), so the
concurrent multi-device init pattern is a necessary ingredient.

On this box RCCL builds **2 channels** by default (`Channel 00/02`,
`Channel 01/02` in `logs/stage1.log`), which is exactly the shape their
first-loses/second-lands observation needs. Stage 2A then drives the number of
concurrent init copies from 1 to 16 and the transport from SHM to `NET/Socket`
(`isAllDirectP2p 1` under `NCCL_SHM_DISABLE=1`). All of it stays correct.

## The machine, against theirs

| | BoJl4apa (#6565) | here |
|---|---|---|
| GPUs | 2x Radeon Pro W7800 48G, gfx1100 | 2x RX 7900 XT, gfx1100 |
| platform | bare metal | **VFIO guest** (Proxmox passthrough) |
| kernel | 6.8.0-136 | **7.0.0-30** |
| ROCm / RCCL | 7.2.3 / 2.27.7 **and** 7.13 / 2.28.3 | **7.14 / 2.30.4-HEAD:2b22ab0** |
| PCIe atomics | `AtomicOpsCtl: ReqEn+` | `ReqEn+` (**same**) |
| IOMMU | tested with `amd_iommu=off` too | 0 groups in guest (**same effect**) |
| P2P | selected, IPC, and 7000x slow (their #6576) | none; `isAllDirectP2p 0`, SHM |
| link | CPU-attached Gen4 x8/x8, separate root ports | cross-die, `AtomicOpsCap: Routing-` |

The atomics row is worth stating because it is the axis this repository is
otherwise known for: since 2026-08-23 this guest advertises PCIe AtomicOp
completer support (see [vfio-atomics.md](../../docs/vfio-atomics.md)), so it is
**not** an atomics-starved cell and does not differ from theirs there. `dmesg`
carries zero "PCIE atomic ops is not supported" lines (`logs/environment.txt`).

## What this does and does not bound

It bounds one thing usefully: the defect is not universal to gfx1100 + RCCL, and
it does not appear at 2.30.4 on this hardware under any transport or channel
count tried.

It does not by itself separate the **version** axis from the **platform** axis.
Two things would, and neither is ours to do cheaply:

1. The reporter running 2.30.4 on their box. They build RCCL from source already
   and their repro is seconds-fast, so this is minutes for them and would settle
   it outright.
2. Us running 2.28.3 on this box. The published ROCm 7.13 image they used is
   ~73 GB and this host has 40 GB free, and reclaiming it would destroy
   container state we need for other work, so it is not free here.

## Limitation: the rccl-tests witness did not come up

The intent was a second witness that does not go through PyTorch, using the
reporter's Reproducer 1. `rccl-tests` builds cleanly against this container's
toolchain (`scripts/build_rccltests.sh`) with gfx1100 code objects confirmed in
both `all_gather_perf` and `verifiable.o`, but every collective aborts at the
data-init kernel on rank 1:

```
[1] [FATAL ERROR]: HIP failure: 'invalid device function'
Test NCCL failure .../common.cu.cpp:650
```

It fails identically with and without `-c 1`, so it is not the verification
kernels. The clone reports `Version develop_deprecated:40b1b17`; this looks like
the deprecated rccl-tests branch against RCCL 2.30.4 rather than anything about
the collectives, and the PyTorch ground-truth reproducer above is the reporter's
own primary one. Recorded rather than dropped: `logs/stage2b.log`.

## Files

- `rccl_allgather_truth.py` — the reporter's script, verbatim from the issue
- `scripts/` — the arm runners and stage drivers actually used
- `logs/stage1.log` — three transport arms, RCCL version banner, topology dump
- `logs/stage2a.log` — channel sweep and `NCCL_SHM_DISABLE=1`
- `logs/stage2b.log` — the rccl-tests attempt
- `logs/environment.txt` — machine fingerprint at capture time
- `results.json` — the per-arm tallies parsed back out of the logs
