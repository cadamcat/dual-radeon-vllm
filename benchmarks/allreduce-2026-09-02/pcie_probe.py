#!/usr/bin/env python3
"""pcie_probe.py — the one-way ceiling of each card's link, so the all-reduce's
bus bandwidth has something to be a fraction *of*.

`allreduce.py` reports `bus_bw_gbs = bytes / t`, which at world size 2 is the
standard ring figure. Read alone it is a number without a scale: PCIe 3.0 x16 is
15.75 GB/s per direction *on paper*, and this box's practical ceiling is not the
paper number. This measures the practical one directly — pinned host memory to
device and back, per card, at the same sizes — using `hipMemcpy` through
`torch.Tensor.copy_(non_blocking=False)`.

It is deliberately not a collective: no RCCL, no second rank, nothing to
schedule. What it establishes is the width of the pipe that
`NCCL_P2P_DISABLE=1` forces every byte of the all-reduce through, since with
P2P off the path is device -> host -> device rather than device -> device.
"""
import json
import os
import time

import torch

OUT = os.environ.get("AR_OUT", "/rb/ar0902/pcie.jsonl")
MB = [1, 4, 16, 64, 256]


def bw(dst, src, n):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        dst.copy_(src)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def main():
    rows = []
    for card in range(torch.cuda.device_count()):
        torch.cuda.set_device(card)
        for mb in MB:
            nbytes = mb << 20
            host = torch.empty(nbytes, dtype=torch.uint8, device="cpu",
                               pin_memory=True)
            dev = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{card}")
            n = max(5, min(200, (1 << 28) // nbytes))
            for _ in range(5):                     # warm the mapping
                dev.copy_(host)
                host.copy_(dev)
            t_h2d = bw(dev, host, n)
            t_d2h = bw(host, dev, n)
            r = {"kind": "pcie", "ts": round(time.time(), 1),
                 "machine": os.environ.get("BENCH_MACHINE", "RX 7900 XT"),
                 "card": card, "mib": mb, "bytes": nbytes, "iters": n,
                 "h2d_gbs": round(nbytes / t_h2d / 1e9, 3),
                 "d2h_gbs": round(nbytes / t_d2h / 1e9, 3),
                 "pinned": True}
            rows.append(r)
            print(f"card{card} {mb:4d} MiB  H2D {r['h2d_gbs']:6.2f} GB/s  "
                  f"D2H {r['d2h_gbs']:6.2f} GB/s", flush=True)
            del host, dev
            torch.cuda.empty_cache()
    with open(OUT, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
