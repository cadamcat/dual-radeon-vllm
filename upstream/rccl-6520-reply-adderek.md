<!--
INTERNAL NOTE — not part of the comment. HTML comments do not render on GitHub.
Reply to @adderek's 2026-07-26 comment on ROCm/ROCm#6520.
-->

Thank you — this is a much better test case than mine, and it closes the caveat I could
not close myself.

**Bare metal with IOMMU entirely disabled settles the virtualisation question.** I only
had a VFIO guest, so the honest position in my write-up was "cannot rule out that
IOMMU/passthrough is a necessary ingredient". Your machine removes that. I have updated
the repository to say so and credited you.

**The split machine is the cleanest evidence in either thread.** One affected GPU
(chipset-attached) and one healthy one (CPU-direct), same host, same driver, same RCCL
build, differing only in which lanes the card sits on. That is a controlled experiment I
could not have run with two identical slots.

**Your `COLLTRACE=OFF` counts answer the question I asked, and the corollary looks more
important than the answer.** For anyone skimming:

| build | `__assert_fail` | `__ockl_fprintf` | `hidden_hostcall_buffer` | collectives |
|---|---|---|---|---|
| distro package, a Release build | 5 | 3 | 6 | fail |
| `-DCOLLTRACE=OFF` | 4 | 3 | 3 | fail |
| `+ -DNDEBUG` | 0 | 0 | 0 | pass |

So `CMAKE_BUILD_TYPE=Release` does not carry `NDEBUG` into RCCL's device compile. If
that holds for other distro and vendor packages, every 2.27.7-era RCCL binary is
affected out of the box on any machine without atomics — and adding `NDEBUG` to the
device compile fixes that entire class at the source, independently of whatever is
decided for 2.30.4's device linker. @harkgill-amd, that seems like the cheapest half of
the keep-or-remove decision.

**One correction to my own tooling, from your output.** You reported that
`hipDeviceSynchronize()` returns success while the dispatch is refused, with the failure
visible only through `hipGetLastError()` and the device `printf` silently never
arriving. My `hipgate3.cpp` read only the launch and sync return codes and printed an
empty string from the device, so **on a machine like yours it would have reported a
pass**. Fixed: it now reads launch, sync and `hipGetLastError()`, prints a marker from
the device so its absence is visible, and iterates per device so a split machine like
yours shows up as split. Your name is in the comment header.

I would also like to fold your bridge-chain refinement into `diagnose/check-platform.sh`
— checking each bridge on the path rather than just reporting `AtomicOpsCap`, since
`00:01.2 Routing+` with `03:00.0 Routing-` is exactly the case where someone would
otherwise waste an afternoon moving the card between chipset slots. Happy to write it,
or to take a PR if you would rather.

Two smaller things, for the record:

- Your downgrade result matches mine: swapping in the older RCCL binary does not help,
  even though the version correlation is real. Both of us verified the older library
  actually loaded, which is the part people usually skip.
- Agreed on the error message. `hipErrorIllegalState` at `enqueue.cc` cost me several
  days and sent me through two wrong root causes; "hostcall unavailable: PCIe atomics
  not enabled on device N" would have ended it in minutes.

On the `-sm tensor` path in llama.cpp: if you do try it, I would be glad to hear the
result either way. My benchmarks are all vLLM tensor-parallel, so a llama.cpp data point
on the same fix would broaden the picture usefully.
