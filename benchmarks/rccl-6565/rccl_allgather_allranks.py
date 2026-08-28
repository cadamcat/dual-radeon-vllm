# Cross-rank variant of the reporter's ground-truth all_gather check.
#
# `rccl_allgather_truth.py` is theirs, verbatim from the issue body, and stays
# that way: its md5 is quoted in the README so anyone can confirm it was not
# retyped. It has one blind spot, disclosed in the README on 2026-08-27 —
# every rank computes bad_list/bad_into, but `fail += 1` and the verdict both
# sit inside `if rank == 0`, so corruption visible only on rank 1 prints
# ALL CORRECT. Under TP=2 rank 1 is the other half of the machine.
#
# This file is the tightening the README said was missing: the same twelve
# cases, the same construction, but the failure count is all-reduced across
# ranks before anything is decided, and every rank prints its own verdict so a
# one-sided failure is visible in the log rather than only in the exit code.
#
#   torchrun --nproc-per-node=2 rccl_allgather_allranks.py
#
# Exit code is 0 only when every rank saw every case correct.
import os
import sys

import torch
import torch.distributed as dist

_r = int(os.environ["RANK"])
torch.cuda.set_device(_r)
dist.init_process_group("nccl", device_id=torch.device(f"cuda:{_r}"))
rank, world = dist.get_rank(), dist.get_world_size()
dev = torch.device(f"cuda:{rank}")
fail = 0

for dt in (torch.float32, torch.float16, torch.bfloat16):
    for n in (1, 2, 64, 4096):
        x = torch.full((n,), float(rank), device=dev, dtype=dt)

        parts = [torch.full((n,), -999.0, device=dev, dtype=dt) for _ in range(world)]
        dist.all_gather(parts, x)
        torch.cuda.synchronize()
        bad_list = [r for r in range(world) if not (parts[r] == float(r)).all().item()]

        out = torch.full((n * world,), -999.0, device=dev, dtype=dt)
        dist.all_gather_into_tensor(out, x)
        torch.cuda.synchronize()
        bad_into = [r for r in range(world)
                    if not (out[r * n:(r + 1) * n] == float(r)).all().item()]

        if bad_list or bad_into:
            fail += 1
            print(f"  rank{rank} {str(dt).split('.')[-1]:9s} n={n:<5d} "
                  f"all_gather={'OK' if not bad_list else f'WRONG at {bad_list}'} "
                  f"all_gather_into_tensor="
                  f"{'OK' if not bad_into else f'WRONG at {bad_into}'}", flush=True)
            print(f"      rank{rank} got list  : {[p[0].item() for p in parts]}", flush=True)
            print(f"      rank{rank} got into  : {[out[r * n].item() for r in range(world)]}",
                  flush=True)
            print(f"      rank{rank} expected  : {[float(r) for r in range(world)]}", flush=True)

# every rank reports, and the verdict is taken on the total rather than on
# whatever rank 0 happened to see
mine = fail
tot = torch.tensor([fail], device=dev, dtype=torch.int64)
dist.all_reduce(tot, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
total = int(tot.item())
print(f"  rank{rank}: {mine} failing cases locally, {total} across all {world} ranks",
      flush=True)
if rank == 0:
    print(f"\n  ==> {'ALL CORRECT ON EVERY RANK' if total == 0 else f'{total} FAILING CASES'}",
          flush=True)
dist.destroy_process_group()
sys.exit(0 if total == 0 else 1)
