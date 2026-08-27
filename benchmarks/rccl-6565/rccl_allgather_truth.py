# MINIMAL pure-PyTorch/RCCL all_gather correctness check against GROUND TRUTH.
# No vLLM. Rank r fills its tensor with the value r; after all_gather every rank
# must hold [0,1,...,world-1]. Nothing is compared against another collective.
import os, torch, torch.distributed as dist

import os as _os
_r = int(_os.environ["RANK"])
torch.cuda.set_device(_r)
dist.init_process_group("nccl", device_id=torch.device(f"cuda:{_r}"))
rank, world = dist.get_rank(), dist.get_world_size()
dev = torch.device(f"cuda:{rank}")
fail = 0

for dt in (torch.float32, torch.float16, torch.bfloat16):
    for n in (1, 2, 64, 4096):
        x = torch.full((n,), float(rank), device=dev, dtype=dt)

        # --- list-based all_gather
        parts = [torch.full((n,), -999.0, device=dev, dtype=dt) for _ in range(world)]
        dist.all_gather(parts, x)
        torch.cuda.synchronize()
        bad_list = [r for r in range(world) if not (parts[r] == float(r)).all().item()]

        # --- single-tensor all_gather
        out = torch.full((n * world,), -999.0, device=dev, dtype=dt)
        dist.all_gather_into_tensor(out, x)
        torch.cuda.synchronize()
        bad_into = [r for r in range(world)
                    if not (out[r*n:(r+1)*n] == float(r)).all().item()]

        if rank == 0:
            s1 = "OK " if not bad_list else f"WRONG at ranks {bad_list}"
            s2 = "OK " if not bad_into else f"WRONG at ranks {bad_into}"
            print(f"  {str(dt).split('.')[-1]:9s} n={n:<5d} all_gather={s1:<22s} all_gather_into_tensor={s2}", flush=True)
            if bad_list or bad_into:
                fail += 1
                print(f"      got list  : {[p[0].item() for p in parts]}", flush=True)
                print(f"      got into  : {[out[r*n].item() for r in range(world)]}", flush=True)
                print(f"      expected  : {[float(r) for r in range(world)]}", flush=True)

if rank == 0:
    print(f"\n  ==> {'ALL CORRECT' if fail==0 else f'{fail} FAILING CASES'}", flush=True)
dist.destroy_process_group()
