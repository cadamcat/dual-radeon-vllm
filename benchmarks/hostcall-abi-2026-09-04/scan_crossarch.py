#!/usr/bin/env python3
"""Cross-architecture pass: is hidden_hostcall_buffer a property of the library
or of the architecture it was built for?

For every `<name>_<arch>.kpack` matching --prefix, unpack that archive's own
architecture and count kernels and hostcall declarations. If the counts are the
same across every gfx target, the requirement lives in the library's source and
the ABI, and only the platform's ability to satisfy it differs.
"""
import argparse, hashlib, json, os, sys, tempfile, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scan_hostcall import Kpack, count_notes, images_from_blob, GFX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kpack-dir", default="/opt/python/lib/python3.14/site-packages/_rocm_sdk_libraries/.kpack")
    ap.add_argument("--prefix", default="rccl_lib_")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kpack-so", default="/opt/python/lib/python3.14/site-packages/_rocm_sdk_core/lib/librocm_kpack.so.0")
    ap.add_argument("--workdir", default="/data/c1scan/work")
    a = ap.parse_args()

    os.makedirs(a.workdir, exist_ok=True)
    kp = Kpack(a.kpack_so)
    t0 = time.time()
    with open(a.out, "w") as out:
        out.write(json.dumps({"kind": "meta", "kpack_dir": a.kpack_dir,
                              "prefix": a.prefix, "ts": t0}) + "\n")
        for name in sorted(os.listdir(a.kpack_dir)):
            if not (name.startswith(a.prefix) and name.endswith(".kpack")):
                continue
            arch = GFX.findall(name)
            if not arch:
                continue
            arch = arch[0]
            path = os.path.join(a.kpack_dir, name)
            h = kp.open(path)
            archs = kp.archs(h)
            tot_k = tot_hc = n_bin = 0
            hc_names = []
            for b in kp.binaries(h):
                if arch not in archs:
                    continue
                blob, rc = kp.kernel(h, b, arch)
                if blob is None:
                    continue
                n_bin += 1
                with tempfile.TemporaryDirectory(dir=a.workdir) as td:
                    for img in images_from_blob(blob, td, "k.bin", arch):
                        k, hc, names, err = count_notes(img)
                        tot_k += k or 0
                        tot_hc += hc or 0
                        hc_names.extend(names)
            kp.lib.kpack_close(h)
            # The complete name list, not a sample: "the same kernels on every
            # target" is a claim about names, and six of thirteen sampled names
            # only supports "the same count". `examples` in the full scan caps
            # at five per image for size; here the lists are short enough to
            # keep whole, and the digest makes cross-target identity one
            # comparison. Note the names are complete only where each image's
            # own cap did not bite -- assert len(hostcall_names) == hostcall.
            names = sorted(hc_names)
            rec = {"kind": "arch", "kpack": name, "arch": arch,
                   "archs_in_archive": archs, "binaries": n_bin,
                   "kernels": tot_k, "hostcall_kernels": tot_hc,
                   "hostcall_names": names,
                   "hostcall_names_complete": len(names) == tot_hc,
                   "hostcall_names_md5": hashlib.md5(
                       "\n".join(names).encode()).hexdigest(),
                   "example_names": names[:6],
                   "bytes": os.path.getsize(path)}
            out.write(json.dumps(rec, sort_keys=True) + "\n")
            out.flush()
            print(f"{arch:10s} binaries={n_bin:4d} kernels={tot_k:7d} hostcall={tot_hc:5d}", flush=True)
        out.write(json.dumps({"kind": "done", "wall_s": round(time.time()-t0, 1)}) + "\n")


if __name__ == "__main__":
    main()
