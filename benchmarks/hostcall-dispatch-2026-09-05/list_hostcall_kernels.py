#!/usr/bin/env python3
"""Every kernel in vllm/_rocm_C's gfx1100 device image, by name, with whether
its metadata declares hidden_hostcall_buffer. Same reader as the C1 scanner
(readelf --notes; the hostcall line precedes the kernel's .symbol line), but
every name is kept and demangled. CPU only. Writes a TSV and a summary."""
import glob, os, re, subprocess, sys, tempfile
SP = "/opt/python/lib/python3.14/site-packages"
SO = f"{SP}/vllm/_rocm_C.abi3.so"
LLVM = f"{SP}/_rocm_sdk_devel/lib/llvm/bin"
OUT = "/rb/pa/rocm_C-gfx1100-kernels.tsv"
# objdump writes the extracted images NEXT TO the input; the .so holds several
# offload bundles, each with its own gfx1100 image, so all of them are read
subprocess.run([f"{LLVM}/llvm-objdump", "--offloading", SO], capture_output=True, text=True)
imgs = sorted(glob.glob(SO + ".*hipv4-amdgcn-amd-amdhsa--gfx1100"))
print("images for gfx1100:", [os.path.basename(p) for p in imgs])
assert imgs, "no gfx1100 image extracted"
names, cur = [], False
for img in imgs:
    r = subprocess.run([f"{LLVM}/llvm-readelf", "--notes", img], capture_output=True, text=True, errors="replace")
    cur = False
    for line in r.stdout.splitlines():
        if "hidden_hostcall_buffer" in line:
            cur = True
        m = re.search(r"\.symbol:\s*(\S+)", line)
        if m:
            names.append((m.group(1), cur)); cur = False
d = subprocess.run([f"{LLVM}/llvm-cxxfilt"], input="\n".join(n for n, _ in names), capture_output=True, text=True).stdout.splitlines()
with open(OUT, "w") as fh:
    fh.write("hostcall\tmangled\tdemangled\n")
    for (n, hc), dm in zip(names, d):
        fh.write(f"{int(hc)}\t{n}\t{dm}\n")
hc = [(n, dm) for (n, h), dm in zip(names, d) if h]
print(f"kernels={len(names)} hostcall={len(hc)}")
fam = {}
for n, dm in hc:
    key = dm.split("<")[0].split("(")[0]
    fam[key] = fam.get(key, 0) + 1
print("declaring families:", fam)
pa = [(n, dm, h) for (n, h), dm in zip(names, d) if "paged_attention_ll4mi_QKV_mfma4_kernel" in dm]
print(f"paged_attention_ll4mi_QKV_mfma4_kernel instantiations: {len(pa)}, declaring: {sum(1 for _,_,h in pa if h)}")
# what distinguishes the declaring instantiations: print a few of each
for label, want in (("DECLARING", True), ("NOT declaring", False)):
    print(f"--- {label}, first 4 ---")
    for n, dm, h in [x for x in pa if x[2] == want][:4]:
        print("  ", dm[:230])
