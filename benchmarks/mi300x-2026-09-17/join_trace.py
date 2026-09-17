#!/usr/bin/env python3
"""Join a launch trace against the image's declaration list.

The shim prints the mangled symbol; the TSV carries mangled and demangled, and
the first join keyed on the wrong one. Keyed on the mangled name, a kernel the
trace names and the scan does not know is a join failure and says so, rather
than being reported as "does not declare".

    BENCH_WORK=/work BENCH_ARCH=gfx942 python3 join_trace.py <tag> <log> [<log>...]
"""
import json, os, re, sys, collections as C

W = os.environ.get("BENCH_WORK", "/work")
ARCH = os.environ.get("BENCH_ARCH", "gfx942")
tag, logs = sys.argv[1], sys.argv[2:]

decl = {}
for line in open(f"{W}/rocm_C-{ARCH}-kernels.tsv").read().splitlines()[1:]:
    hc, mangled, dem = line.split("\t", 2)
    # the metadata note names the kernel descriptor, `<symbol>.kd`; the launch
    # shim prints the function symbol, so the two join only after the suffix
    decl[mangled] = int(hc)
    decl[mangled[:-3] if mangled.endswith(".kd") else mangled] = int(hc)
    decl.setdefault(dem, int(hc))

res = {}
for path in logs:
    label = os.path.basename(path).split(".")[0]
    names = C.Counter()
    if os.path.exists(path):
        for l in open(path, errors="replace"):
            m = re.search(r"LAUNCH api=\S+ name=(.+)", l.strip())
            if m:
                names[m.group(1)] += 1
    known = {k: v for k, v in names.items() if k in decl}
    res[label] = {
        "kernels_launched": len(names), "launches": sum(names.values()),
        "kernels_in_this_image": len(known),
        "kernels_not_in_rocm_C": len(names) - len(known),
        "declaring_kernels_launched": {k: {"launches": v} for k, v in known.items() if decl[k]},
        "launched_declaring": sum(v for k, v in known.items() if decl[k]),
        "paged_attention": {k: {"launches": v, "declares": decl.get(k)}
                            for k, v in names.items() if "paged_attention" in k or "QKV_mfma" in k},
        "wvSplitK": {k: {"launches": v, "declares": decl.get(k)}
                     for k, v in names.items() if "wvSplitK" in k},
    }
out = f"{W}/trace-join-{tag}.json"
json.dump(res, open(out, "w"), indent=1)
for label, r in res.items():
    print(f"{label}: {r['kernels_launched']} distinct kernels, {r['launches']} launches; "
          f"{r['kernels_in_this_image']} are _rocm_C's ({r['kernels_not_in_rocm_C']} from elsewhere — "
          f"torch, hipBLASLt, RCCL: outside this scan); "
          f"declaring kernels launched: {len(r['declaring_kernels_launched'])} "
          f"({r['launched_declaring']} launches)")
    for k, v in list(r["declaring_kernels_launched"].items())[:5]:
        print(f"    declaring: {k[:110]} x{v['launches']}")
    for k, v in list(r["paged_attention"].items())[:4]:
        print(f"    paged-attn: declares={v['declares']} x{v['launches']} {k[:90]}")
print("->", out)
