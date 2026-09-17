#!/usr/bin/env python3
"""Every kernel in a device image, by name, with whether its metadata declares
hidden_hostcall_buffer.

`benchmarks/hostcall-dispatch-2026-09-05/list_hostcall_kernels.py` with the
gfx1100 paths, architecture and output file turned into arguments, because this
image is not that one: the AMD Developer Cloud container is not the ROCm SDK
wheel layout the gfx1100 scan ran in. The reader is unchanged — llvm-readelf
--notes, where the hostcall line precedes the kernel's .symbol line — so a row
from this machine is produced the same way as the one it will sit beside.

    python3 list_hostcall_kernels_arch.py --so <lib.so> --arch gfx942 \
        --llvm-bin /opt/rocm/llvm/bin --out kernels-gfx942.tsv
"""
import argparse, glob, os, re, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("--so", required=True, help="shared object carrying offload bundles")
ap.add_argument("--arch", default="gfx942")
ap.add_argument("--llvm-bin", default="/opt/rocm/llvm/bin")
ap.add_argument("--out", required=True)
ap.add_argument("--family", default="paged_attention",
                help="substring whose instantiations are counted separately")
a = ap.parse_args()

objdump, readelf, cxxfilt = (os.path.join(a.llvm_bin, t) for t in
                             ("llvm-objdump", "llvm-readelf", "llvm-cxxfilt"))
for tool in (objdump, readelf, cxxfilt):
    if not os.path.exists(tool):
        sys.exit(f"required tool not found: {tool}")
if not os.path.exists(a.so):
    sys.exit(f"shared object not found: {a.so}")

# objdump writes the extracted images NEXT TO the input; a .so holds several
# offload bundles, each with its own image for the architecture.
subprocess.run([objdump, "--offloading", a.so], capture_output=True, text=True)
imgs = sorted(glob.glob(f"{a.so}.*hipv4-amdgcn-amd-amdhsa--{a.arch}"))
print(f"images for {a.arch}:", [os.path.basename(p) for p in imgs])
if not imgs:
    sys.exit(f"no {a.arch} image extracted from {a.so}: the library ships no device code "
             f"for this architecture, or it is packed (kpack) rather than embedded")

names, cur = [], False
for img in imgs:
    r = subprocess.run([readelf, "--notes", img], capture_output=True, text=True, errors="replace")
    cur = False
    for line in r.stdout.splitlines():
        if "hidden_hostcall_buffer" in line:
            cur = True
        m = re.search(r"\.symbol:\s*(\S+)", line)
        if m:
            names.append((m.group(1), cur)); cur = False
dem = subprocess.run([cxxfilt], input="\n".join(n for n, _ in names),
                     capture_output=True, text=True).stdout.splitlines()
with open(a.out, "w") as fh:
    fh.write("hostcall\tmangled\tdemangled\n")
    for (n, hc), dm in zip(names, dem):
        fh.write(f"{int(hc)}\t{n}\t{dm}\n")

hc = [(n, dm) for (n, h), dm in zip(names, dem) if h]
print(f"kernels={len(names)} hostcall={len(hc)} -> {a.out}")
fam = {}
for n, dm in hc:
    fam.setdefault(dm.split("<")[0].split("(")[0], 0)
    fam[dm.split("<")[0].split("(")[0]] += 1
print("declaring families:", fam)
sel = [(n, dm, h) for (n, h), dm in zip(names, dem) if a.family in dm]
print(f"{a.family} instantiations: {len(sel)}, declaring: {sum(1 for _, _, h in sel if h)}")
for label, want in (("DECLARING", True), ("NOT declaring", False)):
    print(f"--- {label}, first 4 ---")
    for n, dm, h in [x for x in sel if x[2] == want][:4]:
        print("  ", dm[:230])
