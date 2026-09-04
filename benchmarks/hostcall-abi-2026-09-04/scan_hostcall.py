#!/usr/bin/env python3
"""Count `hidden_hostcall_buffer` declarations in every shipped device code
object of a ROCm installation.

Why this is not a one-line grep. ROCm ships device code in four shapes, and
three of them defeat the obvious inspection:

  fatbin  an ELF `.so` with an uncompressed clang offload bundle in
          `.hip_fatbin`; `llvm-objdump --offloading` splits it per gfx target
  ccob    the same bundle, zstd-compressed, magic `CCOB`. `llvm-objdump
          --offloading` extracts NOTHING from it and reports no error, so a
          scan built on objdump alone silently calls these "no device code".
          `clang-offload-bundler --unbundle` is what reads them.
  kpack   ROCm 7.14's python-wheel SDK moves device code OUT of the `.so`
          entirely, into a per-architecture compressed KPAK archive named by a
          `.rocm_kpack_ref` section. `strings librccl.so.1 | grep -c
          hidden_hostcall_buffer` therefore answers 0 for a library whose
          kernels do require a hostcall. AMD's own librocm_kpack.so is used
          here to unpack it rather than a guess at the container format.
  loose   `.hsaco` / `.co` code objects on disk (Tensile, MIOpen, CK), which
          are themselves usually bundles, and increasingly CCOB ones.

For every device image: kernels are counted by `.symbol:` in the AMDGPU
metadata note, and a kernel counts as requiring a hostcall when its argument
list declares `hidden_hostcall_buffer`. One JSON record per code object.

    scan_hostcall.py --arch gfx1100 --out results.jsonl
"""
import argparse, ctypes, hashlib, json, os, re, shutil, subprocess, sys
import tempfile, time

SYMBOL = re.compile(r"\.symbol:\s*(\S+)")
GFX = re.compile(r"gfx[0-9a-z]+")
ELF_MAGIC = b"\x7fELF"
BUNDLE_MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"
CCOB_MAGIC = b"CCOB"
DEVICE_SECTIONS = (".hip_fatbin", ".hipFatBinSegment", ".rocm_kpack_ref",
                   ".nv_fatbin")


def find_tool(name):
    for base in ("/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel/lib/llvm/bin",
                 "/opt/rocm/llvm/bin", "/usr/bin", "/usr/local/bin"):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    return shutil.which(name)


READELF = find_tool("llvm-readelf")
OBJDUMP = find_tool("llvm-objdump")
BUNDLER = find_tool("clang-offload-bundler")


def count_notes(path):
    """(kernels, hostcall_kernels, example names, error) for one code object."""
    r = subprocess.run([READELF, "--notes", path], capture_output=True,
                       text=True, errors="replace")
    if r.returncode != 0:
        return None, None, [], r.stderr.strip()[:200]
    kernels = hostcall = 0
    names, cur = [], False
    for line in r.stdout.splitlines():
        if "hidden_hostcall_buffer" in line:
            cur = True
        m = SYMBOL.search(line)
        if m:
            kernels += 1
            if cur:
                hostcall += 1
                if len(names) < 5:
                    names.append(m.group(1))
            cur = False
    return kernels, hostcall, names, None


def unbundle(path, arch, workdir):
    """Device images for `arch` out of a bundle file, compressed or not."""
    r = subprocess.run([BUNDLER, "--type=o", "--list", f"--input={path}"],
                       capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        return [], r.stderr.strip()[:200]
    targets = [t for t in r.stdout.split() if arch in t]
    out = []
    for i, t in enumerate(targets):
        dst = os.path.join(workdir, f"u{i}.{t}")
        rr = subprocess.run([BUNDLER, "--type=o", "--unbundle",
                             f"--input={path}", f"--targets={t}",
                             f"--output={dst}"], capture_output=True, text=True)
        if rr.returncode == 0 and os.path.exists(dst):
            out.append(dst)
    return out, None


def images_from_blob(blob, workdir, tag, arch):
    """Write a blob, then return the device images it holds for `arch`."""
    p = os.path.join(workdir, tag)
    with open(p, "wb") as f:
        f.write(blob)
    return images_from_file(p, workdir, arch)


def images_from_file(p, workdir, arch):
    with open(p, "rb") as f:
        head = f.read(4096)
    if head[:4] == ELF_MAGIC:
        # Already a bare code object; count_notes reports 0 kernels honestly if
        # it turns out to hold none.
        return [p]
    if head[:4] == CCOB_MAGIC or BUNDLE_MAGIC in head:
        imgs, _err = unbundle(p, arch, workdir)
        return imgs
    return []


def elf_sections(path):
    """Which device-code sections an ELF carries, and — for `.hip_fatbin` — its
    declared size and whether it is NOBITS.

    NOBITS is the whole point. A kpack-backed library still declares a
    `.hip_fatbin` of the full device-code size, but the section occupies no
    bytes in the file: the payload is fetched from the KPAK archive at load
    time. That is why `strings librccl.so.1 | grep -c hidden_hostcall_buffer`
    answers 0 for a library whose kernels do require a hostcall."""
    r = subprocess.run([READELF, "-S", "--wide", path], capture_output=True,
                       text=True, errors="replace")
    if r.returncode != 0:
        return None, {}
    found, detail = set(), {}
    for line in r.stdout.splitlines():
        for sec in DEVICE_SECTIONS:
            if f" {sec} " in line:
                found.add(sec)
                if sec == ".hip_fatbin":
                    parts = line.split()
                    i = parts.index(sec)
                    detail["fatbin_type"] = parts[i + 1]
                    detail["fatbin_nobits"] = parts[i + 1] == "NOBITS"
                    try:
                        detail["fatbin_size"] = int(parts[i + 4], 16)
                    except (ValueError, IndexError):
                        pass
    return found, detail


def kpack_ref(path):
    """The archive a `.rocm_kpack_ref` section names, as a readable string."""
    r = subprocess.run([READELF, "-x", ".rocm_kpack_ref", path],
                       capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        return None
    # Decode the hex columns, not the ASCII gutter: the gutter is 16 characters
    # wide and its real spaces are indistinguishable from padding.
    raw = bytearray()
    for line in r.stdout.splitlines():
        if not line.startswith("0x"):
            continue
        for group in line.split()[1:5]:
            if re.fullmatch(r"[0-9a-fA-F]{2,8}", group):
                raw += bytes.fromhex(group)
    text = raw.decode("latin-1")
    m = re.search(r"(\.\./\.kpack/[\x20-\x7e]+?\.kpack)", text)
    return m.group(1) if m else None


# ---------------------------------------------------------------- kpack ------

class Kpack:
    """ctypes binding to AMD's librocm_kpack, the archive's own reader."""

    def __init__(self, sopath):
        L = self.lib = ctypes.CDLL(sopath)
        L.kpack_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        L.kpack_open.restype = ctypes.c_int
        L.kpack_close.argtypes = [ctypes.c_void_p]
        for fn in ("kpack_get_architecture_count", "kpack_get_binary_count"):
            f = getattr(L, fn)
            f.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
            f.restype = ctypes.c_int
        for fn in ("kpack_get_architecture", "kpack_get_binary"):
            f = getattr(L, fn)
            f.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                          ctypes.POINTER(ctypes.c_char_p)]
            f.restype = ctypes.c_int
        L.kpack_get_kernel.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                       ctypes.c_char_p,
                                       ctypes.POINTER(ctypes.c_void_p),
                                       ctypes.POINTER(ctypes.c_size_t)]
        L.kpack_get_kernel.restype = ctypes.c_int
        L.kpack_free_kernel.argtypes = [ctypes.c_void_p]

    def open(self, path):
        h = ctypes.c_void_p()
        rc = self.lib.kpack_open(path.encode(), ctypes.byref(h))
        if rc != 0:
            raise OSError(f"kpack_open rc={rc} on {path}")
        return h

    def _list(self, h, count_fn, get_fn):
        n = ctypes.c_size_t()
        if getattr(self.lib, count_fn)(h, ctypes.byref(n)) != 0:
            return []
        out = []
        for i in range(n.value):
            s = ctypes.c_char_p()
            if getattr(self.lib, get_fn)(h, i, ctypes.byref(s)) == 0 and s.value:
                out.append(s.value.decode())
        return out

    def archs(self, h):
        return self._list(h, "kpack_get_architecture_count",
                          "kpack_get_architecture")

    def binaries(self, h):
        return self._list(h, "kpack_get_binary_count", "kpack_get_binary")

    def kernel(self, h, binary, arch):
        data, size = ctypes.c_void_p(), ctypes.c_size_t()
        rc = self.lib.kpack_get_kernel(h, binary.encode(), arch.encode(),
                                       ctypes.byref(data), ctypes.byref(size))
        if rc != 0:
            return None, rc
        blob = ctypes.string_at(data, size.value)
        self.lib.kpack_free_kernel(data)
        return blob, 0


# ------------------------------------------------------------------ scan -----

def emit(out, rec):
    out.write(json.dumps(rec, sort_keys=True) + "\n")
    out.flush()


def scan_kpacks(kp, kdir, arch, out, workdir):
    for name in sorted(os.listdir(kdir)):
        if not name.endswith(".kpack"):
            continue
        path = os.path.join(kdir, name)
        try:
            h = kp.open(path)
        except OSError as e:
            emit(out, {"carrier": "kpack", "file": path, "error": str(e)})
            continue
        archs, bins = kp.archs(h), kp.binaries(h)
        for b in bins:
            if arch not in archs:
                continue
            blob, rc = kp.kernel(h, b, arch)
            if blob is None:
                emit(out, {"carrier": "kpack", "file": path, "binary": b,
                           "arch": arch, "error": f"kpack_get_kernel rc={rc}"})
                continue
            with tempfile.TemporaryDirectory(dir=workdir) as td:
                imgs = images_from_blob(blob, td, "k.bin", arch)
                if not imgs:
                    emit(out, {"carrier": "kpack", "file": path, "binary": b,
                               "arch": arch, "device_code": False,
                               "bytes": len(blob)})
                for img in imgs:
                    k, hc, names, err = count_notes(img)
                    emit(out, {"carrier": "kpack", "file": path,
                               "kpack_archs": len(archs), "binary": b,
                               "arch": arch, "device_code": True,
                               "image": os.path.basename(img),
                               "bytes": len(blob),
                               "md5": hashlib.md5(blob).hexdigest(),
                               "kernels": k, "hostcall_kernels": hc,
                               "examples": names, "error": err})
        kp.lib.kpack_close(h)


def scan_elf(paths, arch, out, workdir):
    for p in paths:
        try:
            with open(p, "rb") as f:
                if f.read(4) != ELF_MAGIC:
                    continue
        except OSError:
            continue
        secs, detail = elf_sections(p)
        if secs is None:
            emit(out, {"carrier": "elf", "file": p, "error": "readelf -S failed"})
            continue
        if not secs:
            emit(out, {"carrier": "elf", "file": p, "device_code": False,
                       "sections": []})
            continue
        base = {"carrier": "elf", "file": p, "sections": sorted(secs), **detail}
        if ".rocm_kpack_ref" in secs:
            base["kpack_ref"] = kpack_ref(p)
        with tempfile.TemporaryDirectory(dir=workdir) as td:
            local = os.path.join(td, "lib.so")
            shutil.copy2(p, local)
            subprocess.run([OBJDUMP, "--offloading", "lib.so"], cwd=td,
                           capture_output=True)
            imgs = [os.path.join(td, f) for f in sorted(os.listdir(td))
                    if f.startswith("lib.so.") and arch in f]
            if not imgs:
                imgs, _ = unbundle(local, arch, td)
            if not imgs:
                emit(out, dict(base, device_code=True, arch=arch,
                               images_for_arch=0))
                continue
            for img in imgs:
                k, hc, names, err = count_notes(img)
                emit(out, dict(base, device_code=True, arch=arch,
                               image=os.path.basename(img), kernels=k,
                               hostcall_kernels=hc, examples=names, error=err))


def scan_loose(paths, arch, out, workdir):
    """Loose .hsaco/.co objects. The architecture is a property of the file, so
    it is read from the name; a file naming a different one is skipped rather
    than mislabelled."""
    for p in paths:
        found = GFX.findall(os.path.basename(p))
        if found and arch not in found:
            continue
        with tempfile.TemporaryDirectory(dir=workdir) as td:
            imgs = images_from_file(p, td, arch)
            if not imgs:
                emit(out, {"carrier": "loose", "file": p, "device_code": False,
                           "arch_hint": found[0] if found else None})
                continue
            for img in imgs:
                k, hc, names, err = count_notes(img)
                emit(out, {"carrier": "loose", "file": p, "device_code": True,
                           "arch_hint": found[0] if found else None,
                           "arch": arch, "image": os.path.basename(img),
                           "kernels": k, "hostcall_kernels": hc,
                           "examples": names, "error": err})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gfx1100")
    ap.add_argument("--out", required=True)
    ap.add_argument("--roots", nargs="*",
                    default=["/opt/python/lib/python3.14/site-packages"])
    ap.add_argument("--kpack-so",
                    default="/opt/python/lib/python3.14/site-packages/"
                            "_rocm_sdk_core/lib/librocm_kpack.so.0")
    ap.add_argument("--workdir", default="/data/c1scan/work")
    ap.add_argument("--only", choices=["kpack", "elf", "loose"], nargs="*")
    a = ap.parse_args()

    for tool, name in ((READELF, "llvm-readelf"), (OBJDUMP, "llvm-objdump"),
                       (BUNDLER, "clang-offload-bundler")):
        if not tool:
            sys.exit(f"required tool not found: {name}")

    os.makedirs(a.workdir, exist_ok=True)
    t0 = time.time()
    so_paths, loose_paths, kpack_dirs = [], [], set()
    for root in a.roots:
        for dirpath, _dirs, files in os.walk(root):
            if os.path.basename(dirpath) == ".kpack":
                kpack_dirs.add(dirpath)
                continue
            for f in files:
                p = os.path.join(dirpath, f)
                if os.path.islink(p) or not os.path.isfile(p):
                    continue
                if ".so" in f:
                    so_paths.append(p)
                elif f.endswith((".hsaco", ".co")):
                    loose_paths.append(p)

    want = set(a.only) if a.only else {"kpack", "elf", "loose"}
    with open(a.out, "w") as out:
        emit(out, {"kind": "meta", "arch": a.arch, "roots": a.roots,
                   "readelf": READELF, "objdump": OBJDUMP, "bundler": BUNDLER,
                   "n_so": len(so_paths), "n_loose": len(loose_paths),
                   "kpack_dirs": sorted(kpack_dirs), "carriers": sorted(want),
                   "host": os.uname().nodename, "ts": t0})
        if "kpack" in want:
            kp = Kpack(a.kpack_so)
            for kdir in sorted(kpack_dirs):
                scan_kpacks(kp, kdir, a.arch, out, a.workdir)
        if "elf" in want:
            scan_elf(so_paths, a.arch, out, a.workdir)
        if "loose" in want:
            scan_loose(loose_paths, a.arch, out, a.workdir)
        emit(out, {"kind": "done", "wall_s": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
