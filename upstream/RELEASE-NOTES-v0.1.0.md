# v0.1.0 — RCCL 2.27.7 rebuilt without hostcall

Prebuilt RCCL for AMD GPUs behind a root complex that does not route PCIe AtomicOps:
consumer chipsets, and every QEMU/VFIO passthrough guest. Without this, the first
cross-GPU collective dies with:

```
HIP failure 'the operation cannot be performed in the present state' at enqueue.cc:2061
```

Root cause, evidence and a 30-line reproducer: see the repository. **Check first**
whether this is actually your problem; `./diagnose/check-platform.sh` answers that in
one command.

## Which file

| Asset | Size | Targets | Needs a companion stub? |
|---|---:|---|---|
| `librccl-nohostcall-2.27.7-gfx1100.so` | 19 MB | gfx1100 only | **No** |
| `librccl-nohostcall-2.27.7-multiarch.so` | 97 MB | gfx908, gfx1030, gfx1100, gfx1101, gfx1102, gfx1200, gfx1201 | **Yes** — see below |

**If you have an RX 7900 XT or XTX, take the first one.** It is the build every
number in the repository was measured on, and it needs nothing else.

## What is actually verified

- **gfx1100 (RX 7900 XT/XTX): verified end to end.** `all_reduce` across two cards,
  vLLM TP=2, gemma-4-31B at 43.2 tok/s, both GPUs at 265 W synchronised, and a
  292-measurement benchmark campaign.
- **The other six targets: built and statically checked, never run.** Each device
  image passes the check that matters (`hidden_hostcall_buffer` = 0), but we own
  only 7900 XTs. If you run one of them, please say so in an issue either way; that
  is the single most useful thing anyone can contribute right now.
- The multi-arch build was verified *equivalent* on gfx1100 (same three checks, same
  vLLM smoke test) before being published, so the packaging is sound even though the
  other architectures are untested on hardware.

## Installing

```bash
sha256sum -c SHA256SUMS

# inside a ROCm/vLLM container, from a checkout of the repository:
./deploy/deploy-tp2.sh /path/to/librccl-nohostcall-2.27.7-gfx1100.so
```

The script also builds and installs the `rsmi` stub and the `sitecustomize.py`
pre-init, both of which are required; see `docs/deploy-vllm.md` for what each one
does and why.

### The multi-arch build needs one extra step

PyTorch links `ncclCommDump`, an API that postdates RCCL 2.27.7. The gfx1100 build
exports it; the multi-arch build does not, because its shim was written with C linkage
while torch imports the C++ mangled name. Build the companion stub from source in the
repository:

```bash
clang++ -shared -fPIC -o librccl_dumpstub.so.1 deploy/rccl_dumpstub.cpp
patchelf --add-needed librccl_dumpstub.so.1 librccl-nohostcall-2.27.7-multiarch.so
```

The source shim has since been corrected upstream in our build tree, so a future
rebuild will not need this.

## Provenance

Built from `ROCm/rccl`, branch `release/rocm-rel-7.1.1.1` (RCCL 2.27.7), with
`add_compile_definitions(NDEBUG)`, and nothing else is patched. The build script is
`build/build-rccl-nohostcall.sh`; verify any binary yourself with
`build/verify-nohostcall.sh`, which checks that hostcall, assert and fprintf counts
are all zero, and `build/check-symbols.sh`, which checks all 38 nccl symbols PyTorch
needs are present.

## Known limitation

**This does not fix RCCL 2.30.4**, which is what ROCm 7.13 and 7.14 ship. There,
`NDEBUG` is necessary but no longer sufficient: the device-linking step declares
`hidden_hostcall_buffer` even though the linked image contains zero `__ockl_*`
symbols. Details in `docs/open-questions.md` §0. 2.27.7 remains the only route we
have verified.

## Licence

RCCL is MIT-licensed by AMD. These binaries are a rebuild of unmodified upstream
source with one compile definition added; the licence and copyright are unchanged.
See `NOTICE.md`.
