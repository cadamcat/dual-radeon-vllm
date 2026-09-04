#!/usr/bin/env python3
"""Is the hostcall requirement a property of the library or of the target?"""
import collections, json, sys

for f in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f)]
    a = [r for r in rows if r.get("kind") == "arch"]
    done = any(r.get("kind") == "done" for r in rows)
    sig = collections.Counter((r["kernels"], r["hostcall_kernels"]) for r in a)
    print(f"{f:34s} archs={len(a):3d} complete={done}")
    for (k, h), n in sig.most_common():
        archs = sorted(r["arch"] for r in a
                       if (r["kernels"], r["hostcall_kernels"]) == (k, h))
        print(f"    kernels={k:7d} hostcall={h:4d}  {n:2d} archs: {' '.join(archs)}")
