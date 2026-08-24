# The RCCL bug in a VM is one line of VM configuration

If your GPUs are passed through to a QEMU/KVM guest, the RCCL failure this
repository is built around is very likely **not** something you need to rebuild
a library to fix. It is Proxmox passing the card's audio function alongside the
GPU, which stops QEMU from advertising PCIe AtomicOp completer support.

```
hostpci0: 0000:0b:00     →   hostpci0: 0000:0b:00.0
```

Measured on this machine 2026-08-23, same session, same two cards, same
container image, only that line changed:

| | `0000:0b:00` | `0000:0b:00.0` |
|---|---|---|
| QEMU device line | `multifunction=on` | single function |
| guest root port | `AtomicOpsCap: 32bit- 64bit-` | **`32bit+ 64bit+`** |
| GPU requester | `AtomicOpsCtl: ReqEn-` | **`ReqEn+`** |
| `amdgpu` at boot | `PCIE atomic ops is not supported`, twice | silent |
| **stock RCCL 2.30.4, 2 ranks** | **`the operation cannot be performed in the present state`** | **correct** |

The test is `all_reduce` and `all_gather_into_tensor`, checked elementwise
against ground truth rather than against each other, for `float32` / `float16` /
`bfloat16` at 1 024 and 1 048 576 elements: twelve cases, all correct on the
right and none reached on the left. Reverting the line reproduced the failure,
so this is not a side effect of reseating the cards. No rebuilt RCCL was involved in the working column: the
stock library shipped in `rocm/vllm:rocm7.14.0_rdna...` was used unmodified.

## 1. Why one character does this

QEMU already knows how to advertise AtomicOp completer support on an emulated
root port. `vfio_pci_enable_rp_atomics()` in `hw/vfio/pci.c` does it
automatically, and has since v8.1.0. It gives up if any of these is true:

```c
if (pci_bus_is_root(bus) || !parent || !parent->exp.exp_cap ||
    pcie_cap_get_type(parent) != PCI_EXP_TYPE_ROOT_PORT ||
    pcie_cap_get_version(parent) != PCI_EXP_FLAGS_VER2 ||
    pdev->devfn ||
    pdev->cap_present & QEMU_PCI_CAP_MULTIFUNCTION) {
    return;
}
```

The last condition is the one that fires. Its own comment explains the intent:

> The single function requirement avoids conflicting requirements should a slot
> be composed of multiple devices with differing capabilities.

A Radeon card presents the GPU at `.0` and an HDMI audio device at `.1`. Writing
`hostpci0: 0000:0b:00` without the function number passes both, which is what
the "All Functions" checkbox in the Proxmox web interface does, and QEMU then
puts `multifunction=on` on the device line. From there the guest never sees
AtomicOps, `amdgpu` disables them, ROCr will not dispatch any kernel whose
metadata declares a hostcall buffer, and RCCL dies.

Nothing in that chain is a defect in AMD's software. Every layer behaves as
designed; the composition is what fails.

## 2. What has to be true on the host

The emulated root port can only advertise what the physical path can deliver, so
QEMU asks the kernel and the kernel looks at the real topology. On this machine
the full chain is capable:

```
0000:0b:00.0  the GPU            AtomicOpsCap: 32bit+ 64bit+
0000:0a:00.0  Navi switch, down  AtomicOpsCap: Routing+
0000:09:00.0  Navi switch, up    AtomicOpsCap: Routing+
0000:00:03.1  X399 root port     AtomicOpsCap: Routing- 32bit+ 64bit+
```

`pci_enable_atomic_ops_to_root()` in the kernel wants completer bits at the root
port and routing on every bridge in between. Check yours before changing
anything:

```bash
lspci -vv -s <root port> | grep AtomicOpsCap
```

If the root port above your card reports `32bit- 64bit-`, this page does not
apply to you and the rebuilt RCCL is still the fix. That is the case for a card
behind a consumer chipset switch, which is why
[ROCm#6520](https://github.com/ROCm/ROCm/issues/6520)'s bare-metal reproduction
is not solved by any amount of VM configuration.

## 3. Then who is the rebuild for

Still everyone whose hardware genuinely cannot deliver AtomicOps:

- cards behind a consumer chipset switch, on bare metal
- hosts whose root ports do not advertise completer support
- QEMU older than 8.1.0
- any case where the card must keep its audio function in the guest

The mechanism this repository documents — hostcall buffers declared in device
metadata, refused at dispatch without atomics — is unchanged, and the `-DNDEBUG`
rebuild is what this repository verified for those cases. It is not the only
conceivable route: an old QEMU can be upgraded, ROCm 7.1.1's own stock RCCL
already carries no hostcall (§2 of root-cause.md), and we never tested whether
the audio function can live in the same guest on its own root port while the GPU
keeps atomics. What changed is the *ordering*: in a VM, check the configuration
first.

## 4. Why this took so long to find

The symptom appears three layers below the cause. People see RCCL fail, so they
debug RCCL, ROCm, or the kernel. Nobody looks at the PCI function layout of
their own VM.

The prevailing advice actively points the wrong way: guides for GPU passthrough
on Proxmox routinely say to tick "All Functions" along with Rom-Bar and PCIE.
Proxmox's own documentation describes the `00:02` shorthand as a convenience and
does not mention what it costs. Searching its `pve-devel` list for AtomicOps
discussion returns nothing.

We sent a two-patch series to `pve-devel` on 2026-08-24 against `pve-docs`: one
patch documents the caveat next to the paragraph that introduces the shorthand,
the other appends the function to the GPU passthrough example, which currently
pairs `pcie=on` with the all-functions form. Message-ID
`20260824170828.42821-1-Xy2462381442@gmail.com`. Whether it gets applied is
Proxmox's call.

Upstream has been circling it. Robin Voetter proposed an `x-atomic-completion`
property on `pcie-root-port` in April 2023; the automatic vfio path landed
instead. In February 2026 Manojlo Pekovic proposed extending it to multifunction
devices by intersecting the capabilities of all functions, and Alex Williamson
declined it, saying he was "not convinced it's QEMU's job" and that this belongs
to "VM builders and management tools". So the gap sits between three projects
with no owner, and the user in the middle sees only one line about atomics.

**This repository walked the long way round too.** `open-questions.md` §5
already concluded that a QEMU-side fix was "a real avenue for *us*", then said
patching QEMU was out of scope. The mistake was in the premise: no patch was
needed. The mechanism had been in our own QEMU the whole time, held off by a
configuration default.

## 5. Not established

- **One host.** X399/Threadripper 1950X, PVE 9.2.4, QEMU 11.0.2. Whether other
  boards' root ports advertise completer support is theirs to check.
- **Audio function.** We did not test whether the audio device can be passed to
  the same guest on a separate root port while the GPU keeps atomics.
- **Other hypervisors.** Only Proxmox/QEMU was tested. The QEMU condition is
  generic, but libvirt and others compose devices differently.
- **Performance.** Whether stock RCCL with real atomics is faster or slower than
  our no-hostcall build was not measured. Only correctness was.
