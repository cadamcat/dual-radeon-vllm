# Who else asks for a hostcall — 2026-09-04

This repository's RCCL finding is that a kernel which merely *declares*
`hidden_hostcall_buffer` in its metadata cannot be dispatched on a platform
without PCIe AtomicOps, whatever the kernel actually does
([root-cause.md](../../docs/root-cause.md),
[open-questions §0](../../docs/open-questions.md)). Every measurement behind it
was made on one library. The obvious question — *is RCCL unusual, or is this
requirement everywhere?* — had never been asked.

It is not everywhere, and it is not only RCCL.

    gfx1100, ROCm 7.14 container (the one the pair serves from)

    librccl.so.1.0                     13 of    105 kernels
    lib/libtorch_hip.so                21 of 41 407 kernels
    vllm/_rocm_C.abi3.so              348 of  2 350 kernels
    five of torch's 8 test binaries    21 of     42 kernels
    everything else                     0 of 63 181 kernels
    ----------------------------------------------------------
                                      403 of 107 085 kernels

    three shipped libraries out of 153, plus five of eight test binaries

`librocblas`, `libhipblaslt`, `librocfft`, `librocrand`, `librocsparse`,
`libMIOpen*`, `libtorchvision`, and all 124 loose code objects on disk for
this target declare none. The requirement is real, it reaches well past the
collective library, and it is still a small minority of what ships.

A library is counted once. A kpack-backed `.so` also carries an empty
`.hip_fatbin` (below); its payload is counted under the kpack that holds it,
not a second time under the shared object.

---

## The scan

`scan_hostcall.py` walks a ROCm installation, finds every object carrying
device code for one gfx target, and for each one counts kernels — by `.symbol:`
in the AMDGPU metadata note — and the subset whose argument list declares
`hidden_hostcall_buffer`. That is the same `llvm-readelf --notes` count
[root-cause.md](../../docs/root-cause.md) uses, applied to everything instead
of to one file.

**Doing that honestly took four readers, not one**, because ROCm ships device
code in four shapes and three of them defeat the obvious inspection:

| shape | where the device code is | what reads it |
|---|---|---|
| `fatbin` | `.hip_fatbin` in the `.so`, plain bundle | `llvm-objdump --offloading` |
| `ccob` | same section, zstd-compressed, magic `CCOB` | `clang-offload-bundler --unbundle` |
| `kpack` | **not in the `.so` at all** — a separate per-architecture KPAK archive | AMD's `librocm_kpack.so` |
| `loose` | `.hsaco` / `.co` on disk, usually themselves bundles | the two bundlers above |

The `kpack` row is the one that matters, and it is the trap this campaign
walked into first:

```
$ strings -a librccl.so.1 | grep -c hidden_hostcall_buffer
0
```

That library's kernels declare thirteen. ROCm 7.14's python-wheel SDK moved
device code out of the shared object into `.kpack/rccl_lib_<arch>.kpack`, named
by a `.rocm_kpack_ref` section, and the `.so` **still declares a
`.hip_fatbin` of the full size** — 525 569 016 bytes for `librccl.so.1` — as
**`NOBITS`**, a section header with no bytes behind it. Across the 7.14
container, 26 libraries declare 2 739 667 243 bytes of device code that is not
in the files. `strings`, `grep`, and `llvm-objdump --offloading` all answer
zero on them, without error.

This repository's own harness has the shortcut:
`benchmarks/allreduce-2026-09-02/allreduce.py` records
`hidden_hostcall_buffer` with `strings -a … | grep -c`. That reading is
**correct for the library it was pointed at** — the locally built no-hostcall
RCCL 2.27.7, a classic build whose device code is in the file — and would
silently read 0 for any kpack-backed library. Anything reusing it needs the
`.rocm_kpack_ref` check.

`CCOB` is the smaller version of the same trap: `llvm-objdump --offloading`
extracts nothing from a compressed bundle and reports no error. Of this
container's 124 loose objects for gfx1100, the 67 `.co` files are `CCOB` and
the 57 `.hsaco` are not. All 67 came back as "no device code" until
`clang-offload-bundler` was added; with it the 124 are 8 603 kernels, none of
which declares a hostcall.

---

## Two ROCm versions

| | ROCm 7.14 · vLLM 0.23 | ROCm 10.0 · vLLM 0.27 |
|---|---|---|
| device-code units scanned (kpack / elf / loose) | 16 / 13 / 124 | 18 / 14 / 3 032 |
| of which libraries, and libraries declaring a hostcall | 153 · **3** | 3 064 · **4** |
| kernels | 107 085 | 113 146 |
| kernels declaring a hostcall | 403 | 447 |
| `librccl.so.1.0` | 13 of 105 | 13 of 138 |
| `lib/libtorch_hip.so` | 21 of 41 407 | 15 of 42 761 |
| `lib/libtorch_rocshmem.so` | not shipped | **50 of 68** |
| `vllm/_rocm_C.abi3.so` | 348 of 2 350 | 348 of 3 040 |
| `.hip_fatbin` NOBITS libraries | 26 | 29 |

Both containers report `RCCL version 2.30.4`. They are different binaries with
different kernel counts (105 against 138) and different md5s, recorded in
[PROVENANCE.json](PROVENANCE.json). **The RCCL version string is not an
identifier**; two libraries that call themselves 2.30.4 differ by 33 kernels.

`libtorch_rocshmem.so` is new in the 10.0 image and is the densest case found:
50 of its 68 kernels declare a hostcall. A device-side shared-memory library is
exactly where device `printf` and `assert` would be expected, so this is a
sensible number rather than a surprising one — but it means a second AMD
communication library ships the same unnegotiated requirement.

---

## What the thirteen and the 348 are

RCCL, both containers, identical lists:

```
_Z23ncclDevKernel_Generic_1 … _2 … _4                     3
ncclSymkDevKernel_ReduceScatter_RailA2A_LsaLD_{sum,avg}_
  {bf16,f16,f32,f8e4m3,f8e5m2}                           10
```

The three `Generic` kernels are the ones
[open-questions §0](../../docs/open-questions.md) bisected to RCCL's device
linker. The ten `Symk` kernels are new to 2.30.4 and had not been counted here
before; the section's static-inspection table records **3**, which is correct
for the build it describes — a local 2.30.4 built *with* `NDEBUG` — and the
shipped library without `NDEBUG` has thirteen. The difference is a datum for
[CAL experiment B1](../../../dual-radeon-vllm-workspace/CAL-OUTLINE-hostcall.md):
on 2.30.4, `NDEBUG` removes the ten and leaves the three.

torch's are ordinary device code, not diagnostics:
`c10d::checkForNaN` (6), `c10d::symmetric_memory::{barrier,put_signal,
wait_signal}_kernel` (3), and `at::native::tinygemm_m16n8k16_chunk_kernel`
(12). Five of the eight test binaries torch ships inside its kpack also declare
one; those are counted separately throughout and are not in the "three
libraries" figure.

vLLM's 348 are all one family — `paged_attention_ll4mi_QKV_mfma4_kernel`, the
custom ROCm paged attention, built for gfx1100 and present in the gfx1100
image.

**Whether any of these is ever dispatched on this box is not measured here.**
The scan counts declarations. It is worth flagging that
`vllm/platforms/rocm.py:328` `use_rocm_custom_paged_attention` does **not**
exclude RDNA — the `_ON_GFX1X` branch enables it for `head_size == 128`,
`block_size == 16`, `gqa_ratio` 3–16 — so the vLLM family is reachable in
principle on gfx1100 rather than dead weight. Testing that against the
capability toggle is a separate experiment.

---

## Cross-architecture: the requirement is in the source, not the target

`scan_crossarch.py` unpacks each `<lib>_<arch>.kpack` for its own architecture.

    RCCL, ROCm 7.14      105 kernels, 13 hostcall — identical on all 20 targets
    RCCL, ROCm 10.0      138 kernels, 13 hostcall — identical on all 21 targets

Twenty-one targets including `gfx90a`, `gfx942` and `gfx950`. **RCCL's hostcall
requirement is a property of its source and of the ABI, not of the
architecture**; what differs between an MI300X and this box is only whether the
platform can satisfy it.

Identical by name, not only by count. `scan_crossarch.py` records each target's
whole hostcall kernel list — `hostcall_names_complete` is true on every row —
and a single md5, `096a1303985cef43`, covers all 41 target-and-version
pairs. The counts here were also produced twice, in independent runs of the
same scan against the same two images, and every cell reproduced.

torch is not invariant, and the shape of the variation is informative:

    ROCm 7.14   42 on 21 RDNA/gfx908 targets · 30 on gfx90a, gfx942, gfx950
    ROCm 10.0   36 on 19 targets · 86 on gfx1100 and gfx1201 · 74 on gfx90a,
                gfx942, gfx950 · 36 on gfx1250

The twelve missing on CDNA in 7.14 are exactly the `tinygemm` int4 kernels,
which are not built there. So a per-architecture count is a claim about that
architecture's build, and must be made per architecture — [INV-013] in the
scanning direction.

---

## What this licenses, and what it does not

**Licensed.** Three of 192 shipped device libraries for gfx1100 in the
container this box serves from carry an unnegotiated platform requirement, and
they are the collective library, the framework, and the inference engine — one
from each layer of the stack, rather than one vendor library in isolation. The
requirement is architecture-invariant for RCCL across 21 targets. Nothing in
the ABI, the loader, or any of these libraries' interfaces declares the
requirement to the platform before the dispatch that fails.

**Not licensed.** That any of these kernels is dispatched in a normal run —
not measured. That a no-atomics platform fails on them — inferred from this
repository's RCCL result, not re-measured per library; the capability toggle
([CAL B2](../../../dual-radeon-vllm-workspace/CAL-OUTLINE-hostcall.md)) is
where that would be established. That the counts hold for a non-wheel ROCm
install — both containers here use the python-wheel SDK, and a `/opt/rocm`
distribution packages device code differently. [INV-013]: this is two
containers on one host.

---

## Reproducing

CPU only. No GPU, no lease, no services stopped; both cards stayed at the
27 971 584 baseline throughout.

```bash
docker run -d --name c1scan --entrypoint sleep \
  rocm/vllm:rocm7.14.0_rdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0 infinity
docker cp scan_hostcall.py c1scan:/tmp/
docker exec c1scan python3 /tmp/scan_hostcall.py --arch gfx1100 --out /tmp/r.jsonl
python3 summarize.py r.jsonl
```

~50 s for the full 7.14 scan, ~85 s for 10.0. `scan_crossarch.py` adds ~16 s
per library for every architecture it ships.

The scanner needs `llvm-readelf`, `llvm-objdump`, `clang-offload-bundler` and
`librocm_kpack.so.0`; all four are inside these images and it exits with a
named error if one is missing.
