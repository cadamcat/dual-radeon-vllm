# Notices and attribution

## What this repository does and does not contain

**Contains:** diagnostic programs, a build recipe, deployment scripts and
documentation — all original work, released under the [MIT Licence](LICENSE).

**Does not contain:** any RCCL source code. The build script fetches RCCL from
upstream and applies a one-line configuration change. Nothing from AMD's or
NVIDIA's codebase is redistributed here.

This distinction matters: because no upstream source is vendored, this
repository is **not** a derivative distribution of RCCL, and carries only its own
MIT terms.

## If you distribute a compiled library

A `librccl.so` produced by `build/build-rccl-nohostcall.sh` **is** a derivative
work of RCCL and is governed by RCCL's **BSD-3-Clause** licence. If you
redistribute such a binary — including as a GitHub Release asset or inside a
container image — you must:

1. Reproduce the RCCL copyright notice, the BSD-3 conditions and the disclaimer
   in your accompanying documentation or materials.
2. Not use the names of the copyright holders to endorse or promote your build.

RCCL upstream and its licence text:
<https://github.com/ROCm/rccl> · <https://github.com/ROCm/rccl/blob/develop/LICENSE.txt>

> RCCL is Copyright (c) 2019–2025 Advanced Micro Devices, Inc.
> Portions are Copyright (c) 2015–2020, NVIDIA CORPORATION.
> Licensed under BSD-3-Clause.

Any binaries published from this repository carry a copy of that licence
alongside them: the conditions and disclaimer are reproduced in the release notes,
and `LICENSE.rccl.txt` is attached to the release as a file.

An earlier version of this paragraph also promised "the exact base image digest
and source commit they were built from". That was not true of the v0.1.0 release,
which carried neither, and the claim is withdrawn rather than quietly dropped.
What is known: the source is RCCL's `release/rocm-rel-7.1.1.1` branch, and the
library reports itself as `2.27.7-release/rocm-rel-7.1.1.1:bf3ebf5+` at runtime,
the `+` marking the local NDEBUG change. The build script clones that branch
rather than pinning a commit, so a future release should pin one and record the
base image digest with it.

## Not affiliated with AMD

This is an independent investigation by a hobbyist. It is **not** affiliated with,
endorsed by, sponsored by or supported by Advanced Micro Devices, Inc. or NVIDIA
Corporation. "ROCm", "RCCL", "Radeon", "Instinct" and "AMD" are trademarks of
their respective owners and are used here only to describe the software this
work applies to.

## Warranty

None. This modifies a low-level communication library in your runtime. Read
[docs/root-cause.md](docs/root-cause.md) before using it, verify the result with
`build/verify-nohostcall.sh`, and satisfy yourself that the change is
appropriate for your system.
