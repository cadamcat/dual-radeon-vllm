#!/usr/bin/env python3
"""allreduce.py — time one TP=2 all-reduce on this box, directly.

Why this file exists
--------------------
This repository has published three numbers for the TP=2 all-reduce floor and
withdrawn all three (docs/benchmarks.md, 2026-08-30). Every one of them was read
off a *fitted intercept* — `a` in `T(S) = a + bS + cS²` — and the intercept
turned out to be an artefact of which campaign supplied the shallow rungs: the
8B's fixed costs went 19.7 → 79.3 ms in July and 28.9 → 30.5 ms in August, so
"+76 ms of collective floor" and "1.05 ms per all-reduce" were properties of one
fit, not of this machine. The a100 article still says, correctly, "No all-reduce
was timed here."

So time it. No fit, no ladder, no engine: two ranks, one tensor, the shapes a
served step actually reduces, on the library vLLM actually calls.

What the numbers mean, and what they do not
-------------------------------------------
* **The library is the right one.** `G31-tp2`'s serve log records
  `cuda_communicator.py: Using ['PYNCCL'] all-reduce backends ... out of
  potential backends: ['NCCL_SYMM_MEM', 'QUICK_REDUCE', 'FLASHINFER', 'CUSTOM',
  'SYMM_MEM', 'PYNCCL']` and `pynccl.py: vLLM is using nccl==2.27.7`. Every
  faster path was rejected on this topology, so vLLM's collective *is*
  `librccl` 2.27.7 through the same `ncclAllReduce` this script reaches through
  `torch.distributed`. It is not a stand-in for vLLM's collective; it is it.

* **Three timings, because one number would be a choice dressed as a fact.**
  `t_stream_us` runs N collectives back to back on one stream with a single host
  sync, so each launch hides behind the previous one's execution.
  `t_sync_us` synchronises after every one, exposing the full launch and
  completion round trip — the dearest. `t_graph_us` captures N of them into a
  HIP graph and replays it, which is what a served step actually does: vLLM
  captures its decode step, and a replayed collective pays no per-call host
  dispatch at all. The engine's number is the graph one; the other two bracket
  it and show how much of the isolated figure is host overhead.

* **A per-step cost is arithmetic, not a measurement.** `2 × layers ×
  t` assumes every layer's two `RowParallelLinear` reductions cost what an
  isolated one does. It is reported in `derive.py` as a derivation, labelled as
  one.

* **The shapes are the models', read from their configs on this box**, not
  remembered: gemma-4-12B is hidden 3840 over **48** layers (the withdrawn claim's
  "36 layers" was Qwen3-8B's, which is hidden 4096).

Run
---
    # inside the vllm-tp2 container, which carries the no-hostcall RCCL
    NCCL_P2P_DISABLE=1 HSA_ENABLE_SDMA=0 \
      torchrun --nproc_per_node=2 /rb/ar0902/allreduce.py
"""
import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, "/rb")
try:
    from harness.telemetry import Sampler, describe
except Exception as _e:                                    # noqa: BLE001
    Sampler = None
    describe = None
    _TELE_ERR = str(_e)
else:
    _TELE_ERR = None

OUT = os.environ.get("AR_OUT", "/rb/ar0902/results.jsonl")
MACHINE = os.environ.get("BENCH_MACHINE", "RX 7900 XT")

# hidden sizes read from the checkpoints on this box on 2026-09-02:
#   gemma-4-26B-A4B-AWQ     2816   30 layers
#   gemma-4-12B-it-qat      3840   48 layers
#   Qwen3-8B                4096   36 layers
#   Qwen3.8-27B-INT4-sym    5120   64 layers
#   gemma-4-31B-it-qat      5376   60 layers
HIDDEN = [2816, 3840, 4096, 5120, 5376]

# batch-1 decode reduces one row. The rest of the ladder is there so the
# latency floor can be told apart from the bandwidth slope: if the 1-token and
# the 64-token reduction cost the same, the cost is not bandwidth.
NTOK = [1, 2, 4, 8, 16, 32, 64, 256, 1024, 4096, 16384]

DTYPE = torch.bfloat16      # what these models' activations are, w4a16 included
WARMUP = 30

# The buffer is zeros, not ones. `all_reduce` is in place, so a buffer of ones
# doubles on every call and reaches bf16 infinity inside 40 iterations; zeros
# stay zeros for any number of them. RCCL's reduction is a plain elementwise
# add with no data-dependent path — no compression, no zero-skipping on this
# backend — so the contents do not change what is being timed, only whether the
# numbers stay finite.


def _nccl_version():
    """What torch *reports*, which is not what is loaded.

    `torch.cuda.nccl.version()` returns 2.30.4 in this container -- a
    compile-time constant of the torch build. The library that actually served
    the collective is RCCL 2.27.7, the no-hostcall build this deployment
    installs, and 2.30.4 is the version that does *not* work here at all
    (docs/open-questions.md section 0). Reporting the constant alone would have
    published the wrong library; `_loaded_rccl()` reports the real one.
    """
    try:
        return ".".join(str(x) for x in torch.cuda.nccl.version())
    except Exception as e:                                 # noqa: BLE001
        return f"unavailable: {e}"


def _loaded_rccl():
    """The collective library actually mapped in, after a first collective.

    Read from /proc/self/maps rather than from any version API, with the
    version string and the hostcall count taken out of the file itself -- the
    two facts that decide whether this is the deployment's patched build.

    Matches `nccl` as well as `rccl` since 2026-09-03, so the same script runs
    on a rented NVIDIA pair. RCCL is NCCL's ROCm build and answers the same
    `ncclAllReduce`, so nothing else here is platform-specific; on CUDA the
    hostcall count is simply 0, which is what that build has.
    """
    import subprocess
    out = {}
    try:
        paths = sorted({ln.split()[-1] for ln in open("/proc/self/maps")
                        if "rccl" in ln.lower() or "nccl" in ln.lower()}
                       - {"(deleted)"})
        out["mapped"] = paths
        for p in paths:
            real = os.path.realpath(p)
            out["realpath"] = real
            out["md5"] = subprocess.run(["md5sum", real], capture_output=True,
                                        text=True).stdout.split()[0]
            out["version_string"] = subprocess.run(
                f"strings -a {real} | grep -Eio '(rccl|nccl) version [0-9.]*' | sort -u",
                shell=True, capture_output=True, text=True).stdout.strip()
            out["hidden_hostcall_buffer"] = int(subprocess.run(
                f"strings -a {real} | grep -c hidden_hostcall_buffer",
                shell=True, capture_output=True, text=True).stdout.strip() or -1)
            # ...and that count is a LIE whenever the device code is not
            # plainly present in the file, which is now the common case. Two
            # shapes defeat it, both measured 2026-09-04 (see
            # benchmarks/hostcall-abi-2026-09-04/):
            #   kpack -- ROCm 7.14's wheel SDK moves device code out of the .so
            #     into a per-architecture KPAK archive, leaving an empty NOBITS
            #     .hip_fatbin behind;
            #   CCOB  -- the fatbin is present but zstd-compressed, and every
            #     locally built RCCL on this box is this shape. The B1 pair's
            #     two arms declare 0 and 6 hostcall buffers and `strings` reads
            #     0 for both.
            # So read the notes when the tools are there, and say which method
            # answered. A 0 from `strings` alone means nothing.
            out["device_code_external"] = bool(subprocess.run(
                f"grep -c rocm_kpack_ref {real}",
                shell=True, capture_output=True, text=True).stdout.strip()
                not in ("", "0"))
            out["device_code_compressed"] = bool(subprocess.run(
                f"grep -c CCOB {real}", shell=True,
                capture_output=True, text=True).stdout.strip() not in ("", "0"))
            out["hidden_hostcall_buffer_method"] = "strings"
            hc, method = _hostcall_from_notes(real)
            if hc is not None:
                out["hidden_hostcall_buffer"] = hc
                out["hidden_hostcall_buffer_method"] = method
            elif out["device_code_external"] or out["device_code_compressed"]:
                out["hidden_hostcall_buffer_method"] = "unreadable: %s" % method
                out["hidden_hostcall_buffer"] = None
                out["hidden_hostcall_buffer_note"] = (
                    "the strings count cannot read this library's device code "
                    "and the note reader was unavailable; the value is unknown, "
                    "not zero.")
    except Exception as e:                                 # noqa: BLE001
        out["error"] = str(e)
    return out


def _llvm(name):
    for base in ("/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel/lib/llvm/bin",
                 "/opt/rocm/llvm/bin"):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    from shutil import which
    return which(name)


def _hostcall_from_notes(real):
    """The authoritative count: split the fatbin, read the AMDGPU metadata note.

    Returns (count, method), or (None, "<why>") when it could not answer --
    never (None, None) silently. The first version of this helper used the
    module's `subprocess`, which allreduce.py imports inside rccl_loaded()
    rather than at the top, so every call raised NameError into a broad
    `except` and reported None for all three libraries. Import what you use.
    Handles both a plain offload bundle and a CCOB-compressed one; the latter
    is what `llvm-objdump --offloading` silently extracts nothing from."""
    import glob as _glob
    import subprocess
    import tempfile
    readelf, objdump = _llvm("llvm-readelf"), _llvm("llvm-objdump")
    bundler = _llvm("clang-offload-bundler")
    if not readelf or not objdump:
        return None, "llvm-readelf or llvm-objdump not found"
    try:
        with tempfile.TemporaryDirectory() as td:
            lib = os.path.join(td, "lib.so")
            subprocess.run(["cp", real, lib], check=True, capture_output=True)
            subprocess.run([objdump, "--offloading", "lib.so"], cwd=td,
                           capture_output=True)
            imgs = sorted(_glob.glob(os.path.join(td, "lib.so.*gfx*")))
            method = "notes"
            if not imgs and bundler:
                r = subprocess.run([bundler, "--type=o", "--list",
                                    f"--input={lib}"], capture_output=True,
                                   text=True)
                for i, t in enumerate(x for x in r.stdout.split() if "gfx" in x):
                    dst = os.path.join(td, f"u{i}.{t}")
                    if subprocess.run([bundler, "--type=o", "--unbundle",
                                       f"--input={lib}", f"--targets={t}",
                                       f"--output={dst}"],
                                      capture_output=True).returncode == 0:
                        imgs.append(dst)
                method = "notes-ccob"
            if not imgs:
                return None, "no device image could be extracted"
            total = 0
            for img in imgs:
                r = subprocess.run([readelf, "--notes", img],
                                   capture_output=True, text=True,
                                   errors="replace")
                if r.returncode != 0:
                    return None, f"llvm-readelf failed on {os.path.basename(img)}"
                total += sum(1 for ln in r.stdout.splitlines()
                             if "hidden_hostcall_buffer" in ln)
            return total, method
    except Exception as e:                                  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def iters_for(nbytes):
    """More repetitions where one is short, so every cell times ~0.3-1 s."""
    if nbytes <= 1 << 16:
        return 400
    if nbytes <= 1 << 20:
        return 200
    if nbytes <= 1 << 24:
        return 50
    return 20


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    # device first, process group second: initialising first lets RCCL bind the
    # communicator to whatever device is current — device 0 for every rank — and
    # the thing measured would then not be the cross-card collective at all.
    # `device_id` pins it, as diagnose/ar.py does.
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{rank}"))
    # one collective before the library is identified: RCCL is dlopened lazily,
    # so /proc/self/maps is empty of it until something has actually reduced.
    _probe = torch.zeros(8, dtype=DTYPE, device=f"cuda:{rank}")
    dist.all_reduce(_probe)
    torch.cuda.synchronize()
    del _probe

    if rank == 0:
        meta = {"kind": "ar_meta", "ts": round(time.time(), 1), "machine": MACHINE,
                "world_size": world, "dtype": str(DTYPE),
                "torch": torch.__version__,
                "hip": getattr(torch.version, "hip", None),
                "cuda": getattr(torch.version, "cuda", None),
                "nccl_version_torch_reports": _nccl_version(),
                "rccl_loaded": _loaded_rccl(),
                "env": {k: os.environ.get(k) for k in
                        ("NCCL_P2P_DISABLE", "HSA_ENABLE_SDMA", "NCCL_ALGO",
                         "NCCL_PROTO", "NCCL_DEBUG")},
                "device_names": [torch.cuda.get_device_name(i) for i in range(world)],
                "telemetry_import_error": _TELE_ERR}
        with open(OUT, "a") as fh:
            fh.write(json.dumps(meta) + "\n")
        if describe is not None:
            with open(OUT, "a") as fh:
                fh.write(json.dumps(describe()) + "\n")

    smp = None
    if Sampler is not None and rank == 0:
        smp = Sampler()
        smp.start()

    for hidden in HIDDEN:
        for ntok in NTOK:
            x = torch.zeros(ntok, hidden, dtype=DTYPE, device=f"cuda:{rank}")
            nbytes = x.numel() * x.element_size()
            n = iters_for(nbytes)

            for _ in range(min(WARMUP, max(3, n // 5))):
                dist.all_reduce(x)
            torch.cuda.synchronize()
            dist.barrier()

            # (1) back to back on one stream: the floor, launch cost hidden
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n):
                dist.all_reduce(x)
            torch.cuda.synchronize()
            t_stream = (time.perf_counter() - t0) / n

            # (2) one at a time: the ceiling, full launch + completion exposed
            dist.barrier()
            per = []
            for _ in range(min(n, 100)):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                dist.all_reduce(x)
                torch.cuda.synchronize()
                per.append(time.perf_counter() - t0)

            # (3) captured into a HIP graph and replayed -- the mode a served
            # decode step runs in. If capture is refused (some RCCL builds
            # cannot be captured), the cell records why rather than silently
            # falling back to one of the other two numbers.
            dist.barrier()
            t_graph = None
            graph_err = None
            try:
                g = torch.cuda.CUDAGraph()
                ng = min(n, 20)
                torch.cuda.synchronize()
                with torch.cuda.graph(g):
                    for _ in range(ng):
                        dist.all_reduce(x)
                torch.cuda.synchronize()
                for _ in range(3):
                    g.replay()
                torch.cuda.synchronize()
                reps = max(3, min(50, n // ng))
                t0 = time.perf_counter()
                for _ in range(reps):
                    g.replay()
                torch.cuda.synchronize()
                t_graph = (time.perf_counter() - t0) / (reps * ng)
                del g
            except Exception as e:                         # noqa: BLE001
                graph_err = str(e)[:200]

            # (4) a local elementwise kernel of the same size, back to back:
            # what a launch of a trivial kernel costs here with no collective in
            # it, so the collective's cost is not read as launch overhead.
            dist.barrier()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n):
                x.mul_(1.0)
            torch.cuda.synchronize()
            t_local = (time.perf_counter() - t0) / n

            rec_path = OUT if rank == 0 else OUT.replace(".jsonl", f".rank{rank}.jsonl")
            if True:
                rec = {"kind": "allreduce", "rank": rank, "ts": round(time.time(), 1),
                       "machine": MACHINE, "hidden": hidden, "ntok": ntok,
                       "bytes": nbytes, "iters": n, "world_size": world,
                       "t_stream_us": round(t_stream * 1e6, 3),
                       "t_sync_us_median": round(statistics.median(per) * 1e6, 3),
                       "t_sync_us_min": round(min(per) * 1e6, 3),
                       "t_sync_us_p95": round(
                           sorted(per)[int(len(per) * 0.95) - 1] * 1e6, 3),
                       "t_sync_us_n": len(per),
                       "t_graph_us": round(t_graph * 1e6, 3) if t_graph else None,
                       "graph_error": graph_err,
                       "t_local_kernel_us": round(t_local * 1e6, 3),
                       # a 2-rank ring moves `bytes` per rank in each direction;
                       # busbw = bytes * 2(n-1)/n / t = bytes / t at n = 2
                       "bus_bw_gbs": round(nbytes / t_stream / 1e9, 3)}
                with open(rec_path, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
            if rank == 0:
                print(f"h{hidden:5d} n{ntok:6d} {nbytes/1024:9.1f} KiB  "
                      f"stream {rec['t_stream_us']:9.1f} us  "
                      f"sync {rec['t_sync_us_median']:9.1f} us  "
                      f"graph {str(rec['t_graph_us']):>9} us  "
                      f"local {rec['t_local_kernel_us']:7.1f} us  "
                      f"{rec['bus_bw_gbs']:6.2f} GB/s", flush=True)
            del x
            torch.cuda.empty_cache()

    if smp is not None:
        tele = smp.stop_and_summarise()
        with open(OUT, "a") as fh:
            fh.write(json.dumps({"kind": "ar_telemetry", "machine": MACHINE,
                                 "ts": round(time.time(), 1)} | tele) + "\n")
    if rank == 0:
        with open(OUT, "a") as fh:
            fh.write(json.dumps({"kind": "ar_complete",
                                 "ts": round(time.time(), 1)}) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
