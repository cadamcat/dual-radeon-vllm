#!/usr/bin/env python3
"""collective_correctness.py — the twelve cases, run against ground truth.

`docs/vfio-atomics.md` describes twelve correctness cases and this repository
has never carried the script that runs them; they existed as prose. CAL B3 asks
for them under both libraries on the atomics-enabled platform, so that §V can
say the NDEBUG removal changes neither results nor timing.

    2 collectives x 3 dtypes x 2 sizes
    all_reduce(SUM), all_gather_into_tensor
    float32, float16, bfloat16
    1 024 and 1 048 576 elements

Ground truth is computed on the host from the same generator that fills the
device tensors, so a collective that silently does nothing fails rather than
passing on an unchanged buffer. Run under torchrun with 2 ranks:

    AR_OUT=/rb/b1/correct-<arm>.jsonl torchrun --nproc_per_node 2 \
      collective_correctness.py
"""
import json, os, subprocess, sys, time

import torch
import torch.distributed as dist

OUT = os.environ.get("AR_OUT", "/rb/b1/correctness.jsonl")
SIZES = (1024, 1048576)
DTYPES = (("float32", torch.float32, 0.0),
          ("float16", torch.float16, 5e-3),
          ("bfloat16", torch.bfloat16, 4e-2))


def library():
    """Which RCCL is actually mapped, read from /proc/self/maps after the
    first collective. Same idea as allreduce.py, and the reason a run can be
    attributed to an arm afterwards without trusting the caller."""
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
            out["device_code_external"] = bool(subprocess.run(
                f"grep -c rocm_kpack_ref {real}", shell=True,
                capture_output=True, text=True).stdout.strip() not in ("", "0"))
    except Exception as e:                                    # noqa: BLE001
        out["error"] = str(e)
    return out


def fill(n, rank, dtype, dev):
    """A deterministic pattern that differs per rank, so a no-op collective
    cannot pass: rank r contributes r+1 scaled by position."""
    base = torch.arange(n, dtype=torch.float32, device=dev) % 97
    return ((base + 1) * (rank + 1) / 97.0).to(dtype)


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)

    rows, failures = [], 0
    for op in ("all_reduce", "all_gather_into_tensor"):
        for dname, dtype, tol in DTYPES:
            for n in SIZES:
                mine = fill(n, rank, dtype, dev)
                ref32 = [fill(n, r, dtype, dev).to(torch.float32)
                         for r in range(world)]
                t0 = time.perf_counter()
                if op == "all_reduce":
                    got = mine.clone()
                    dist.all_reduce(got, op=dist.ReduceOp.SUM)
                    want = torch.stack(ref32).sum(0)
                else:
                    got = torch.empty(n * world, dtype=dtype, device=dev)
                    dist.all_gather_into_tensor(got, mine)
                    want = torch.cat(ref32)
                torch.cuda.synchronize()
                wall = time.perf_counter() - t0

                err = (got.to(torch.float32) - want).abs()
                scale = want.abs().clamp(min=1e-6)
                rel = (err / scale).max().item()
                absmax = err.max().item()
                # a collective that never ran leaves `got` equal to its own
                # rank's contribution, so check that too rather than only tol
                unchanged = bool(op == "all_reduce" and world > 1 and
                                 torch.equal(got, mine))
                ok = (rel <= tol or absmax <= tol) and not unchanged
                failures += 0 if ok else 1
                rows.append(dict(kind="correctness", op=op, dtype=dname,
                                 elements=n, world_size=world, rank=rank,
                                 max_rel_err=rel, max_abs_err=absmax,
                                 tol=tol, unchanged=unchanged, ok=ok,
                                 wall_s=round(wall, 6)))
                if rank == 0:
                    print(f"  {'ok  ' if ok else 'FAIL'} {op:24s} {dname:8s} "
                          f"n={n:<8d} rel={rel:.3e} abs={absmax:.3e}")

    meta = dict(kind="correctness_meta", ts=time.time(), world_size=world,
                rank=rank, torch=torch.__version__,
                hip=getattr(torch.version, "hip", None),
                rccl_loaded=library(), cases=len(rows), failures=failures,
                device_names=[torch.cuda.get_device_name(i)
                              for i in range(torch.cuda.device_count())])
    path = OUT if rank == 0 else OUT.replace(".jsonl", f".rank{rank}.jsonl")
    with open(path, "a") as fh:
        fh.write(json.dumps(meta) + "\n")
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print(f"{len(rows) - failures}/{len(rows)} cases pass")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
