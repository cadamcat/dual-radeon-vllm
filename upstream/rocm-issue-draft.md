# [Performance] Host→device copy is ~4400x slower from a writable private file mapping whose pages are resident

**Draft — not yet submitted.** Target: `ROCm/ROCm` (or `ROCm/clr`). Review before filing.

---

## Summary

On gfx1100 with ROCm 7.14, copying host memory to the GPU runs at **~8.6 GiB/s** (8 777 MiB/s) when the
source is a read-only file mapping (`r--p`) and at **2.0 MiB/s** — roughly **4400x
slower** — when the source is a *writable* private file mapping (`rw-p`) whose pages
have already been faulted into the process.

Same file, same bytes, same size, same process. The only difference is the `prot`
argument to `mmap` and whether the CPU has touched the pages first.

This is not a corner case: **it is the default path for loading any `safetensors`
checkpoint into PyTorch**, because `safetensors` hands PyTorch a storage created by
`torch.UntypedStorage.from_file(..., shared=False)`, and PyTorch maps that writable.
In practice it makes vLLM model loading 20–50x slower than the disk it reads from
(e.g. a 15.26 GiB checkpoint takes 206 s instead of the ~19 s the workaround achieves (the disk itself would do it in ~11 s)).

## Measurements

| source mapping | pages faulted in first | 32 MiB copy | rate |
|---|---|---:|---:|
| `MAP_PRIVATE \| PROT_READ` (`r--p`) | yes | 3.6 ms | **8 777 MiB/s** |
| `MAP_PRIVATE \| PROT_READ\|PROT_WRITE` (`rw-p`) | no | 17.2 ms | 1 865 MiB/s |
| `MAP_PRIVATE \| PROT_READ\|PROT_WRITE` (`rw-p`) | **yes** | **16 021 ms** | **2.0 MiB/s** |

Reproduced twice in a row at 32 MiB, and at 256 MiB (128 s, also 2.0 MiB/s), so the
cost tracks the number of resident pages rather than being a fixed penalty.

Anonymous memory of the same size copies at ~13 GiB/s, i.e. the hardware and the
PCIe link are fine.

## Reproducer

Dependency-free — PyTorch only, no safetensors, no vLLM. **The test file must live on a
real filesystem**; tmpfs or overlayfs does not show the effect.

```python
import os, mmap, time, torch

PATH, N, DEV = "/data/repro.bin", 32 << 20, "cuda:0"
if not os.path.exists(PATH) or os.path.getsize(PATH) < N * 8:
    with open(PATH, "wb") as f:
        for _ in range(N * 8 >> 20):
            f.write(os.urandom(1 << 20))
with open(PATH, "rb") as f:                       # warm the page cache
    while f.read(1 << 24):
        pass

def run(writable, pretouch):
    fh = open(PATH, "r+b" if writable else "rb")
    prot = mmap.PROT_READ | (mmap.PROT_WRITE if writable else 0)
    mm = mmap.mmap(fh.fileno(), N * 8, flags=mmap.MAP_PRIVATE, prot=prot)
    src = torch.frombuffer(mm, dtype=torch.uint8, count=N * 8)[:N]
    if pretouch:
        src.max()                                 # bulk read: faults the pages in
    torch.cuda.synchronize(); t0 = time.perf_counter()
    g = src.to(DEV); torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    perm = [l.split()[1] for l in open("/proc/self/maps") if "repro.bin" in l][0]
    print(f"{perm} pretouch={pretouch!s:<5} {dt*1000:9.1f} ms  {N/dt/2**20:8.1f} MiB/s")
    del g, src; torch.cuda.empty_cache(); mm.close(); fh.close()

run(False, True)     # r--p           ->  ~8800 MiB/s
run(True,  False)    # rw-p, cold     ->  ~1900 MiB/s
run(True,  True)     # rw-p, resident ->     ~2 MiB/s
```

## Where the time goes

`perf` on a vLLM worker during a real checkpoint load:

```
99.11%  ioctl
        └─ __x64_sys_ioctl → kfd_ioctl → kfd_ioctl_svm → svm_ioctl
           └─ svm_range_set_attr → svm_range_validate_and_map
              └─ 94.19% amdgpu_hmm_range_get_pages → 86.95% hmm_range_fault
```

`strace` over a 12 s window of the same load: **54 ioctls, ~189 ms each**.

Our reading — offered as a hypothesis, not a claim — is that mapping a
`MAP_PRIVATE` writable range for DMA forces copy-on-write to be broken page by page,
and that this is what `hmm_range_fault` is spending its time on. It would explain why
the cost appears only when the pages are already present as COW-shared, and why a
read-only mapping of the same file is unaffected.

## Why this hits real workloads

`safetensors`, on its PyTorch path, does not use its own (read-only) mapping. It calls:

```rust
// safetensors/bindings/python/src/lib.rs
torch.UntypedStorage.from_file(filename, shared=False, nbytes=size)
```

PyTorch maps that file **writable**, so every tensor a framework loads from a
safetensors checkpoint is backed by an `rw-p` mapping. Confirmed with
`/proc/<pid>/maps`, same file opened two ways in one process:

```
rw-p ... model-00001-of-00005.safetensors   <- safetensors framework="pt"
r--p ... model-00001-of-00005.safetensors   <- safetensors framework="np"
```

Effect on vLLM startup on this machine, with and without a workaround that copies
each tensor into anonymous memory before the device copy:

| model | as shipped | tensor cloned to anon memory first |
|---|---:|---:|
| Qwen3-8B, 15.26 GiB BF16 | 206 s | **18.7 s** |
| gemma-4-12B, 9.56 GiB w4a16 | 328 s | **10.5 s** |
| gemma-4-31B, 21.67 GiB w4a16 | 569 s | **25.1 s** |

## Possibly related, and one thing that does *not* help

**[ROCm/ROCm#5952](https://github.com/ROCm/ROCm/issues/5952)** — "SVM mapping failure
during sequential model loads (RDNA3 / RX 7900 GRE)" — reports crashes rather than
slowness, but it is the same subsystem, the same generation of hardware and the same
activity: its kernel log shows `svm_range_restore_work [amdgpu] hogged CPU for
>10000us`, and the reporter notes that "VRAM loading crawls extremely slowly". That
report is from a **bare-metal** Ryzen 7800X3D machine, which is worth noting given our
own inability to test outside a passthrough guest. These may be two faces of the same
problem.

**`HSA_USE_SVM=0` does not help.** Suggested by
[ROCm/ROCm#2433](https://github.com/ROCm/ROCm/issues/2433), where it recovers pre-5.6
`hipHostRegister` performance. Measured here: the pathological case is unchanged
(16 036 ms vs 16 020 ms), and the read-only fast path gets *worse* (8 905 → 844
MiB/s) — so SVM is what makes read-only mappings fast, while the writable path is
slow for some other reason.

## Ruled out

| hypothesis | result |
|---|---|
| disk throughput | 1.5 GB/s measured on the same file with `dd iflag=direct` |
| `HSA_USE_SVM=0` | no effect on the pathological case; degrades the fast case |
| page cache cold vs warm | 192 vs 212 MiB/s — no meaningful difference |
| file-backed vs anonymous *per se* | a read-only file mapping is full speed |
| pages not yet faulted in | that is the *faster* of the two writable cases |
| a fresh source range per copy | no effect on a read-only mapping |
| ext4 vs overlayfs | effect needs a real filesystem, but is not ext4-specific |
| page-unaligned range start, bf16 vs uint8 | no effect |
| new device allocation per copy vs one reused buffer | no effect |
| swap / memory pressure | swap usage 0 throughout |

## Environment

- 2× Radeon RX 7900 XT (gfx1100), 20 GiB each
- ROCm **7.14**, PyTorch 2.11, safetensors 0.8.0 — **also reproduces on ROCm 7.0.0
  with PyTorch 2.9**, so this is long-standing rather than a regression
- Guest: Ubuntu, kernel 6.14 HWE, running under Proxmox VE / QEMU with **VFIO
  passthrough**
- Checkpoints on ext4 on an NVMe SSD

**Caveat we cannot close ourselves:** we have no bare-metal machine to test on, so we
cannot rule out that IOMMU/VFIO is a necessary ingredient. If someone can run the
reproducer on bare metal, that single data point would tell you whether this is a
general ROCm issue or specific to passthrough guests.

## What would help

1. Confirmation (or not) on bare metal and on CDNA.
2. If the COW-breaking reading is right, is there a cheaper path for
   `MAP_PRIVATE|PROT_WRITE` ranges — or should the runtime fall back to a staging
   copy instead of trying to map such ranges for DMA?
3. Independently, `torch.UntypedStorage.from_file(shared=False)` mapping writable is
   what exposes every PyTorch user to this; that may be worth a separate conversation
   with the PyTorch maintainers, but the 4400x factor looks like it belongs here.

## Everything behind this report

https://github.com/2462381442/dual-radeon-vllm

- `benchmarks/repro-mmap-prot.py` is the reproducer quoted above, verbatim.
- `docs/open-questions.md` section 8 lists every hypothesis we tested and discarded,
  including an earlier root cause we published and then disproved with our own
  reproducer. If something here looks wrong, that section is the place to check first.
