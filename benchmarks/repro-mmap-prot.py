#!/usr/bin/env python3
"""Host->device copy from a WRITABLE private file mapping whose pages are resident
runs at ~2 MiB/s on ROCm — about 4400x slower than the same bytes copied from a
read-only mapping of the same file.

This is the default path for every safetensors checkpoint loaded into PyTorch:
safetensors' PyTorch path calls torch.UntypedStorage.from_file(shared=False), and
PyTorch maps that file writable (visible as `rw-p` in /proc/<pid>/maps).

See docs/open-questions.md section 8 for every hypothesis tested and discarded.

The test file must live on a REAL filesystem — tmpfs/overlayfs will not show it:

    REPRO_FILE=/data/repro.bin python3 repro-mmap-prot.py
"""
import os, mmap, time, torch

PATH = os.environ.get("REPRO_FILE", "/data/repro-mmap-prot.bin")
DEV = os.environ.get("REPRO_DEV", "cuda:0")
N = 32 * 1024 * 1024
SPAN = N * 8

if not os.path.exists(PATH) or os.path.getsize(PATH) < SPAN:
    print(f"creating {SPAN/2**20:.0f} MiB at {PATH}", flush=True)
    with open(PATH, "wb") as f:
        chunk = os.urandom(1 << 20)
        for _ in range(SPAN >> 20):
            f.write(chunk)
with open(PATH, "rb") as f:                    # warm the page cache: not disk I/O
    while f.read(1 << 24):
        pass

TAG = os.path.basename(PATH)

def run(writable, pretouch, rep=2):
    for r in range(rep):
        fh = open(PATH, "r+b" if writable else "rb")
        prot = mmap.PROT_READ | (mmap.PROT_WRITE if writable else 0)
        mm = mmap.mmap(fh.fileno(), SPAN, flags=mmap.MAP_PRIVATE, prot=prot)
        base = torch.frombuffer(mm, dtype=torch.uint8, count=SPAN)
        src = base[r * N * 2: r * N * 2 + N]
        if pretouch:
            src.max()                          # bulk read: faults the pages in
        torch.cuda.synchronize(); t0 = time.perf_counter()
        g = src.to(DEV); torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        perm = [l.split()[1] for l in open("/proc/self/maps") if TAG in l][0]
        print(f"  {perm}  pretouch={str(pretouch):<5} run{r}: "
              f"{dt*1000:9.1f} ms  {N/dt/2**20:8.1f} MiB/s", flush=True)
        del g, base, src; torch.cuda.empty_cache(); mm.close(); fh.close()

print(f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}, {N//2**20} MiB per copy\n")
run(False, True)      # r--p, resident   -> full speed
run(True, False)      # rw-p, cold       -> ~4-5x slower
run(True, True)       # rw-p, resident   -> ~4400x slower
