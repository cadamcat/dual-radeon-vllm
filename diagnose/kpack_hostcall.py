#!/usr/bin/env python3
"""Count hostcall-requiring kernels in a ROCm kpack-backed shared library.

ROCm wheel installs keep the device images outside the shared object.  The
shared object has a ``.rocm_kpack_ref`` section naming the KPAK archive, and
AMD's ``librocm_kpack.so`` is the supported way to read that archive.

This file deliberately has no imports from the repository.  It is intended to
be copied into, and run directly inside, a ROCm container.
"""

import argparse
import ctypes
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile


NO_KPACK_REF = 2
ARCHIVE_NOT_FOUND = 3
KPACK_LIBRARY_NOT_FOUND = 4
ARCHITECTURE_NOT_FOUND = 5
READELF_NOT_FOUND = 6

# Set when the architecture was chosen for the caller rather than given.
ARCH_FALLBACK = []

SYMBOL = re.compile(r"\.symbol:\s*(\S+)")
ELF_MAGIC = b"\x7fELF"
BUNDLE_MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"
CCOB_MAGIC = b"CCOB"


class DiagnosticError(Exception):
    """A user-actionable failure with the command's public exit status."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def find_tool(name):
    """Find an ROCm tool without depending on the checkout's environment."""
    for base in (
        "/opt/python/lib/python3.14/site-packages/"
        "_rocm_sdk_devel/lib/llvm/bin",
        "/opt/rocm/llvm/bin",
        "/usr/bin",
        "/usr/local/bin",
    ):
        candidate = os.path.join(base, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return shutil.which(name)


def _readelf_or_error(readelf_path):
    if not readelf_path:
        raise DiagnosticError(READELF_NOT_FOUND, "llvm-readelf not found")
    if not (os.path.isfile(readelf_path) and os.access(readelf_path, os.X_OK)):
        raise DiagnosticError(READELF_NOT_FOUND, "llvm-readelf not found")
    return readelf_path


def _readelf_hex(readelf_path, path):
    """Return the bytes printed by ``llvm-readelf -x`` for one section."""
    try:
        result = subprocess.run(
            [readelf_path, "-x", ".rocm_kpack_ref", path],
            capture_output=True,
            text=True,
            errors="replace",
        )
    except OSError as exc:
        raise DiagnosticError(
            READELF_NOT_FOUND,
            f"llvm-readelf not found or could not run: {exc}",
        ) from exc
    if result.returncode != 0:
        return None

    raw = bytearray()
    for line in result.stdout.splitlines():
        if not line.lstrip().startswith("0x"):
            continue
        # The first field is the offset.  The next four are the hex columns;
        # ignoring the ASCII gutter avoids treating its spaces as data.
        for group in line.split()[1:5]:
            if (
                len(group) % 2 == 0
                and re.fullmatch(r"[0-9a-fA-F]{2,8}", group)
            ):
                raw.extend(bytes.fromhex(group))
    return bytes(raw)


def kpack_ref(path, readelf_path):
    """Return the archive reference stored in ``path``, or ``None``."""
    raw = _readelf_hex(readelf_path, path)
    if raw is None:
        return None
    text = raw.decode("latin-1")
    # The section is msgpack, so the string is preceded by a type byte and a
    # length byte -- and a length of 34 is 0x22, which is an ASCII double
    # quote. Scanning for "any printable run" therefore captures a leading `"`
    # and the path never resolves. Match the path's own alphabet instead;
    # @GFXARCH@ is part of it because the reference is a template, not a file.
    # Measured 2026-09-04 against the real librccl.so.1 in the ROCm 7.14 image.
    matches = re.findall(r"[A-Za-z0-9_.@/+-]+\.kpack", text)
    return matches[-1] if matches else None


def _ancestor_dirs(start):
    current = os.path.abspath(start)
    while True:
        yield current
        parent = os.path.dirname(current)
        if parent == current:
            return
        current = parent


def _kpack_files_below(root, max_depth=3):
    """Yield likely kpack reader libraries below one upward-search root."""
    root = os.path.abspath(root)
    try:
        walker = os.walk(root, topdown=True, followlinks=False)
        for directory, dirs, files in walker:
            relative = os.path.relpath(directory, root)
            depth = 0 if relative == "." else relative.count(os.sep) + 1
            dirs.sort()
            files.sort()
            if depth >= max_depth:
                dirs[:] = []
            for name in files:
                if name.startswith("librocm_kpack.so"):
                    candidate = os.path.join(directory, name)
                    if os.path.isfile(candidate):
                        yield candidate
    except OSError:
        return


def find_kpack_library(so_path):
    """Search upward from the shared object's directory for librocm_kpack."""
    start = os.path.dirname(os.path.abspath(so_path))
    for directory in _ancestor_dirs(start):
        # The direct check handles the normal /opt/rocm/lib layout.  The
        # bounded subtree check handles the wheel layout where the reader is
        # in a sibling package: site-packages/_rocm_sdk_core/lib/.
        matches = list(_kpack_files_below(directory))
        if matches:
            return matches[0]
        if directory == os.path.dirname(directory):
            break
    return None


class Kpack:
    """ctypes binding to AMD's archive reader."""

    def __init__(self, library_path):
        try:
            library = ctypes.CDLL(library_path)
        except OSError as exc:
            raise DiagnosticError(
                KPACK_LIBRARY_NOT_FOUND,
                f"librocm_kpack.so not found or could not be loaded: {exc}",
            ) from exc
        self.lib = library

        try:
            library.kpack_open.argtypes = [
                ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            library.kpack_open.restype = ctypes.c_int
            library.kpack_close.argtypes = [ctypes.c_void_p]
            for name in ("kpack_get_architecture_count", "kpack_get_binary_count"):
                function = getattr(library, name)
                function.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_size_t),
                ]
                function.restype = ctypes.c_int
            for name in ("kpack_get_architecture", "kpack_get_binary"):
                function = getattr(library, name)
                function.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_char_p),
                ]
                function.restype = ctypes.c_int
            library.kpack_get_kernel.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_size_t),
            ]
            library.kpack_get_kernel.restype = ctypes.c_int
            library.kpack_free_kernel.argtypes = [ctypes.c_void_p]
        except AttributeError as exc:
            raise DiagnosticError(
                KPACK_LIBRARY_NOT_FOUND,
                f"librocm_kpack.so not found or is incompatible: {exc}",
            ) from exc

    def open(self, path):
        handle = ctypes.c_void_p()
        rc = self.lib.kpack_open(path.encode(), ctypes.byref(handle))
        if rc != 0:
            raise DiagnosticError(
                ARCHIVE_NOT_FOUND,
                f"archive not found or could not be opened: {path} (rc={rc})",
            )
        return handle

    def close(self, handle):
        self.lib.kpack_close(handle)

    def _list(self, handle, count_name, get_name):
        count = ctypes.c_size_t()
        if getattr(self.lib, count_name)(handle, ctypes.byref(count)) != 0:
            raise DiagnosticError(
                ARCHIVE_NOT_FOUND,
                f"archive read failed while calling {count_name}",
            )
        values = []
        for index in range(count.value):
            value = ctypes.c_char_p()
            rc = getattr(self.lib, get_name)(
                handle, index, ctypes.byref(value)
            )
            if rc != 0:
                raise DiagnosticError(
                    ARCHIVE_NOT_FOUND,
                    f"archive read failed while calling {get_name}",
                )
            if value.value:
                values.append(value.value.decode("utf-8", errors="replace"))
        return values

    def archs(self, handle):
        return self._list(
            handle,
            "kpack_get_architecture_count",
            "kpack_get_architecture",
        )

    def binaries(self, handle):
        return self._list(handle, "kpack_get_binary_count", "kpack_get_binary")

    def kernel(self, handle, binary, arch):
        data = ctypes.c_void_p()
        size = ctypes.c_size_t()
        rc = self.lib.kpack_get_kernel(
            handle,
            binary.encode(),
            arch.encode(),
            ctypes.byref(data),
            ctypes.byref(size),
        )
        if rc != 0:
            raise DiagnosticError(
                ARCHIVE_NOT_FOUND,
                f"archive read failed while getting {binary} for {arch} (rc={rc})",
            )
        blob = ctypes.string_at(data, size.value)
        self.lib.kpack_free_kernel(data)
        return blob


def count_notes(path, readelf_path):
    """Return ``(kernels, hostcall_kernels)`` for one device code object."""
    try:
        result = subprocess.run(
            [readelf_path, "--notes", path],
            capture_output=True,
            text=True,
            errors="replace",
        )
    except OSError as exc:
        raise DiagnosticError(
            READELF_NOT_FOUND,
            f"llvm-readelf not found or could not run: {exc}",
        ) from exc
    if result.returncode != 0:
        raise DiagnosticError(
            ARCHIVE_NOT_FOUND,
            "llvm-readelf could not read a kpack device image",
        )

    kernels = 0
    hostcall_kernels = 0
    hostcall_for_next_symbol = False
    for line in result.stdout.splitlines():
        if "hidden_hostcall_buffer" in line:
            hostcall_for_next_symbol = True
        match = SYMBOL.search(line)
        if match:
            kernels += 1
            if hostcall_for_next_symbol:
                hostcall_kernels += 1
            hostcall_for_next_symbol = False
    return kernels, hostcall_kernels


def unbundle(path, arch, workdir, bundler_path):
    """Extract the requested target from a kpack blob when it is bundled."""
    if not bundler_path:
        raise DiagnosticError(
            ARCHIVE_NOT_FOUND,
            "archive read failed: clang-offload-bundler not found",
        )
    try:
        result = subprocess.run(
            [bundler_path, "--type=o", "--list", f"--input={path}"],
            capture_output=True,
            text=True,
            errors="replace",
        )
    except OSError as exc:
        raise DiagnosticError(
            ARCHIVE_NOT_FOUND,
            f"archive read failed: clang-offload-bundler could not run: {exc}",
        ) from exc
    if result.returncode != 0:
        return []
    targets = [target for target in result.stdout.split() if arch in target]
    images = []
    for index, target in enumerate(targets):
        destination = os.path.join(workdir, f"u{index}.{target}")
        try:
            extracted = subprocess.run(
                [
                    bundler_path,
                    "--type=o",
                    "--unbundle",
                    f"--input={path}",
                    f"--targets={target}",
                    f"--output={destination}",
                ],
                capture_output=True,
                text=True,
                errors="replace",
            )
        except OSError as exc:
            raise DiagnosticError(
                ARCHIVE_NOT_FOUND,
                f"archive read failed: clang-offload-bundler could not run: {exc}",
            ) from exc
        if extracted.returncode == 0 and os.path.exists(destination):
            images.append(destination)
    return images


def images_from_file(path, workdir, arch, bundler_path):
    with open(path, "rb") as stream:
        header = stream.read(4096)
    if header[:4] == ELF_MAGIC:
        return [path]
    if header[:4] == CCOB_MAGIC or BUNDLE_MAGIC in header:
        return unbundle(path, arch, workdir, bundler_path)
    return []


def images_from_blob(blob, workdir, tag, arch, bundler_path):
    path = os.path.join(workdir, tag)
    with open(path, "wb") as stream:
        stream.write(blob)
    return images_from_file(path, workdir, arch, bundler_path)


def count_archive(
    archive_path,
    kpack_library,
    arch,
    readelf_path,
    kpack_factory=Kpack,
    bundler_path=None,
):
    """Count all kpack binaries for one architecture."""
    try:
        kpack = kpack_factory(kpack_library)
    except DiagnosticError:
        raise
    except OSError as exc:
        raise DiagnosticError(
            KPACK_LIBRARY_NOT_FOUND,
            f"librocm_kpack.so not found or could not be loaded: {exc}",
        ) from exc

    handle = None
    try:
        handle = kpack.open(archive_path)
        archive_archs = kpack.archs(handle)
        if arch not in archive_archs:
            available = ", ".join(archive_archs) or "none"
            raise DiagnosticError(
                ARCHITECTURE_NOT_FOUND,
                f"architecture not in archive: {arch} (available: {available})",
            )

        total_kernels = 0
        total_hostcall = 0
        binaries = kpack.binaries(handle)
        with tempfile.TemporaryDirectory(prefix="kpack-hostcall-") as workdir:
            for binary in binaries:
                blob = kpack.kernel(handle, binary, arch)
                images = images_from_blob(
                    blob, workdir, "k.bin", arch, bundler_path
                )
                for image in images:
                    kernels, hostcall = count_notes(image, readelf_path)
                    total_kernels += kernels
                    total_hostcall += hostcall
        return total_hostcall, total_kernels
    finally:
        if handle is not None:
            kpack.close(handle)


def inspect_library(
    library_path,
    arch=None,
    readelf_path=None,
    kpack_factory=Kpack,
    kpack_locator=find_kpack_library,
    bundler_path=None,
):
    """Inspect one .so and return ``(hostcall_count, kernel_count, arch)``."""
    if readelf_path is None:
        readelf_path = find_tool("llvm-readelf")
    readelf_path = _readelf_or_error(readelf_path)

    reference = kpack_ref(library_path, readelf_path)
    if reference is None:
        raise DiagnosticError(
            NO_KPACK_REF,
            f"no .rocm_kpack_ref section in {library_path}",
        )
    # The reference names one archive per architecture through a @GFXARCH@
    # placeholder: ../.kpack/rccl_lib_@GFXARCH@.kpack. With --arch it is
    # substituted; without one, the placeholder is globbed and the architecture
    # is read back off the filename, so the default is a real target rather
    # than a guess.
    def _resolve(ref):
        if os.path.isabs(ref):
            return os.path.normpath(ref)
        return os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(library_path)), ref))

    if "@GFXARCH@" in reference:
        if arch:
            archive_path = _resolve(reference.replace("@GFXARCH@", arch))
        else:
            import glob as _glob
            hits = sorted(_glob.glob(_resolve(reference.replace("@GFXARCH@", "*"))))
            if not hits:
                raise DiagnosticError(
                    ARCHIVE_NOT_FOUND,
                    f"no archive matches {_resolve(reference)}")
            archive_path = hits[0]
            m = re.search(r"(gfx[0-9a-z]+)", os.path.basename(archive_path))
            if m:
                arch = m.group(1)
            # Without --arch this is the archive's FIRST architecture, not
            # this machine's. A diagnostic that quietly answers about gfx1010
            # on a gfx1100 box is worse than one that says which it used.
            ARCH_FALLBACK.append(arch)
    else:
        archive_path = _resolve(reference)
    if not os.path.isfile(archive_path):
        raise DiagnosticError(ARCHIVE_NOT_FOUND, f"archive not found: {archive_path}")

    kpack_library = kpack_locator(library_path)
    if not kpack_library:
        raise DiagnosticError(
            KPACK_LIBRARY_NOT_FOUND,
            "librocm_kpack.so not found (searched upward from the .so)",
        )

    # count_archive accepts a factory so --selftest can exercise the archive
    # and architecture paths without requiring a ROCm shared library on a
    # laptop.  Normal execution always uses Kpack above.
    def factory(_ignored_path):
        return kpack_factory(kpack_library)

    if bundler_path is None:
        bundler_path = find_tool("clang-offload-bundler")
    # Opening once here selects the archive's first architecture when --arch
    # was omitted.  count_archive opens it again to perform the full scan.
    try:
        kpack = factory(kpack_library)
    except DiagnosticError:
        raise
    except (AttributeError, OSError) as exc:
        raise DiagnosticError(
            KPACK_LIBRARY_NOT_FOUND,
            f"librocm_kpack.so not found or could not be loaded: {exc}",
        ) from exc
    handle = None
    try:
        handle = kpack.open(archive_path)
        archive_archs = kpack.archs(handle)
        used_arch = arch if arch is not None else (
            archive_archs[0] if archive_archs else None
        )
        if used_arch is None or used_arch not in archive_archs:
            requested = used_arch if used_arch is not None else "<none>"
            available = ", ".join(archive_archs) or "none"
            raise DiagnosticError(
                ARCHITECTURE_NOT_FOUND,
                f"architecture not in archive: {requested} (available: {available})",
            )
    finally:
        if handle is not None:
            kpack.close(handle)

    # count_archive constructs a second reader, so normal Kpack execution does
    # not retain an archive handle while temporary extracted images are read.
    hostcall, kernels = count_archive(
        archive_path,
        kpack_library,
        used_arch,
        readelf_path,
        kpack_factory=kpack_factory,
        bundler_path=bundler_path,
    )
    return hostcall, kernels, used_arch


def build_parser():
    parser = argparse.ArgumentParser(
        description="Count hidden_hostcall_buffer kernels in a kpack-backed .so"
    )
    parser.add_argument("--lib", help="path to librccl.so.1")
    parser.add_argument("--arch", help="target architecture; default: archive's first")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="exercise named failure paths without ROCm",
    )
    return parser


def _write_executable(path, contents):
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(contents)
    mode = os.stat(path).st_mode
    os.chmod(path, mode | stat.S_IXUSR)


def _hex_dump_for(text):
    data = text.encode() + b"\x00"
    padding = (-len(data)) % 4
    data += b"\x00" * padding
    lines = []
    for offset in range(0, len(data), 16):
        chunk = data[offset : offset + 16]
        groups = " ".join(
            chunk[index : index + 4].hex() for index in range(0, len(chunk), 4)
        )
        lines.append(f"  0x{offset:08x} {groups}")
    return "\n".join(lines)


class _SelftestKpack:
    """Small fake used only by --selftest to reach the arch failure."""

    def __init__(self, _library_path):
        pass

    def open(self, _archive_path):
        return object()

    def close(self, _handle):
        pass

    def archs(self, _handle):
        return ["gfx900"]


def _expect_failure(case, expected_status, function):
    try:
        function()
    except DiagnosticError as error:
        if error.status == expected_status and case in str(error):
            print(f"PASS {case} (status {error.status})")
            return True
        print(
            f"FAIL {case}: got status {error.status} and message {error}",
        )
        return False
    except Exception as error:  # pragma: no cover - diagnostic for the user
        print(f"FAIL {case}: unexpected {type(error).__name__}: {error}")
        return False
    print(f"FAIL {case}: expected status {expected_status}")
    return False


def selftest():
    """Run failure-path tests that do not need a GPU, ROCm, or a container."""
    results = []
    with tempfile.TemporaryDirectory(prefix="kpack-hostcall-selftest-") as root:
        sdk = os.path.join(root, "sdk")
        libdir = os.path.join(sdk, "lib")
        kpackdir = os.path.join(sdk, ".kpack")
        os.makedirs(libdir)
        os.makedirs(kpackdir)
        library = os.path.join(libdir, "librccl.so.1")
        with open(library, "wb") as stream:
            stream.write(b"not an ELF; the fake readelf supplies the section")

        no_ref_readelf = os.path.join(root, "llvm-readelf-no-ref")
        _write_executable(no_ref_readelf, "#!/bin/sh\nexit 1\n")
        results.append(
            _expect_failure(
                "no .rocm_kpack_ref",
                NO_KPACK_REF,
                lambda: inspect_library(
                    library, readelf_path=no_ref_readelf
                ),
            )
        )

        reference = "../.kpack/rccl_lib_gfx1100.kpack"
        reference_readelf = os.path.join(root, "llvm-readelf-ref")
        dump = _hex_dump_for(reference)
        _write_executable(
            reference_readelf,
            "#!/bin/sh\nprintf '%b\\n' " + repr(dump) + "\n",
        )
        results.append(
            _expect_failure(
                "archive not found",
                ARCHIVE_NOT_FOUND,
                lambda: inspect_library(
                    library, readelf_path=reference_readelf
                ),
            )
        )

        archive = os.path.join(kpackdir, "rccl_lib_gfx1100.kpack")
        with open(archive, "wb") as stream:
            stream.write(b"KPAK selftest")
        results.append(
            _expect_failure(
                "librocm_kpack.so not found",
                KPACK_LIBRARY_NOT_FOUND,
                lambda: inspect_library(
                    library, readelf_path=reference_readelf
                ),
            )
        )

        kpack_library = os.path.join(libdir, "librocm_kpack.so.0")
        with open(kpack_library, "wb") as stream:
            stream.write(b"fake shared library")
        results.append(
            _expect_failure(
                "architecture not in archive",
                ARCHITECTURE_NOT_FOUND,
                lambda: inspect_library(
                    library,
                    arch="gfx1100",
                    readelf_path=reference_readelf,
                    kpack_factory=_SelftestKpack,
                ),
            )
        )

        missing_readelf = os.path.join(root, "does-not-exist-llvm-readelf")
        results.append(
            _expect_failure(
                "llvm-readelf not found",
                READELF_NOT_FOUND,
                lambda: inspect_library(library, readelf_path=missing_readelf),
            )
        )

        parser = build_parser()
        defaults = parser.parse_args(["--selftest"])
        explicit = parser.parse_args(
            ["--lib", library, "--arch", "gfx999"]
        )
        argument_ok = (
            defaults.selftest
            and defaults.arch is None
            and explicit.lib == library
            and explicit.arch == "gfx999"
        )
        if argument_ok:
            print("PASS argument handling")
        else:
            print("FAIL argument handling")
        results.append(argument_ok)

    return 0 if all(results) else 1


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.lib:
        parser.error("--lib is required unless --selftest is used")
    try:
        hostcall, kernels, used_arch = inspect_library(args.lib, args.arch)
    except DiagnosticError as error:
        print(f"error: {error}", file=sys.stderr)
        return error.status
    print(
        f"hidden_hostcall_buffer={hostcall} kernels={kernels} arch={used_arch}"
        + (" (the archive's first architecture, not this machine's;"
           " pass --arch for yours)" if ARCH_FALLBACK else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
