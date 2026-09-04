#!/usr/bin/env python3
"""Roll one scan_hostcall.py jsonl up into the table the letter needs."""
import collections, json, sys

path = sys.argv[1]
rows = [json.loads(l) for l in open(path)]
meta = next(r for r in rows if r.get("kind") == "meta")
print(f"== {path}")
print(f"   candidates: {meta['n_so']} .so, {meta['n_loose']} loose, "
      f"{len(meta['kpack_dirs'])} kpack dirs, arch {meta['arch']}")

# One "shipped device library" = one file that carries device code for this
# arch, wherever the code physically lives.
lib = collections.defaultdict(lambda: {"kern": 0, "hc": 0, "img": 0,
                                       "carrier": set(), "names": []})
for r in rows:
    if r.get("carrier") is None or not r.get("device_code"):
        continue
    if r.get("image") is None:
        # Device code is declared but none was extracted for this target: a
        # kpack-backed .so (an empty NOBITS .hip_fatbin, its payload counted
        # under the kpack carrier), or a library built for other targets only.
        # Counting it here would count the same library twice.
        continue
    if r["carrier"] == "kpack":
        # A kpack stores one library's device code in numbered shards
        # (`librccl.so.1.0#0`, `#18`, ...). The unit the letter counts is the
        # library, so the shard index is stripped.
        key = r["binary"].split("#")[0]
    else:
        key = r["file"]
    e = lib[key]
    e["carrier"].add(r["carrier"])
    e["img"] += 1
    e["kern"] += r.get("kernels") or 0
    e["hc"] += r.get("hostcall_kernels") or 0
    e["names"] += r.get("examples") or []

with_dev = len(lib)
with_hc = sum(1 for v in lib.values() if v["hc"] > 0)
tot_k = sum(v["kern"] for v in lib.values())
tot_h = sum(v["hc"] for v in lib.values())
def is_test(name):
    return "/test" in name or name.split("/")[-1].startswith(("hip_", "test_"))


libs = {k: v for k, v in lib.items() if not is_test(k)}
tests = {k: v for k, v in lib.items() if is_test(k)}
# The denominator is only meaningful per carrier: a kpack ships whole
# libraries, while `loose` counts individual Tensile code objects, of which one
# library has hundreds.
for c in ("kpack", "elf", "loose"):
    sel = {k: v for k, v in libs.items() if c in v["carrier"]}
    hcn = sum(1 for v in sel.values() if v["hc"] > 0)
    print(f"   {c:6s} device-code units {len(sel):5d}   declaring hostcall {hcn:3d}"
          f"   kernels {sum(v['kern'] for v in sel.values()):7d}"
          f"   hostcall {sum(v['hc'] for v in sel.values()):5d}")
print(f"   shipped test binaries: "
      f"{sum(1 for v in tests.values() if v['hc'] > 0)} of {len(tests)} "
      f"declare a hostcall")
print(f"   kernels: {tot_k};  kernels declaring hostcall: {tot_h}")

nob = [r for r in rows if r.get("fatbin_nobits")]
print(f"   .hip_fatbin NOBITS (payload lives in a kpack): {len(nob)} libraries, "
      f"{sum(r.get('fatbin_size', 0) for r in nob)} bytes declared, 0 present")
print("   units declaring a hostcall:")
for k, v in sorted(lib.items(), key=lambda x: -x[1]["hc"]):
    if v["hc"] == 0:
        continue
    print(f"     {v['hc']:5d}/{v['kern']:<7d} {'+'.join(sorted(v['carrier'])):6s} {k[-84:]}")
