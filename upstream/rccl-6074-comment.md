<!--
INTERNAL NOTE — not part of the comment. HTML comments do not render on GitHub.
Short pointer comment for ROCm/ROCm#6074, to accompany #6520.
Written after reading @harkgill-amd's 2026-07-21 comment, so it does not re-announce
the atomics dependency as a discovery.
-->

@harkgill-amd, on the keep-or-remove question for the atomics dependency: it may be
worth knowing that on 2.30.4 the dependency does not live in the code any more.

We built 2.30.4 with `NDEBUG` for seven architectures. The linked device image has
**zero** `__ockl_*` symbols, so nothing in it can call hostcall, and it still fails at
`enqueue.cc:2118` because all three `ncclDevKernel_Generic_*` kernels still *declare*
`hidden_hostcall_buffer` in their metadata. Bisected to the device-link step:
`common.o` before the link has 0, `device.elf` after it has 3.

So removing the device-side `assert()` would not be enough on current RCCL; the
declaration is being added later, by `tools/rccl-device-compile --link`.

Written up with the static checks, a reproducer that needs no RCCL, and a 2.27.7
workaround that does work, in #6520.

For anyone landing here from a search engine and wondering whether they are affected,
this is decided by the root port rather than by the GPU or the slot width:

```bash
dmesg | grep "PCIE atomic"
# affected: amdgpu ...: PCIE atomic ops is not supported
# not affected: no output
```
