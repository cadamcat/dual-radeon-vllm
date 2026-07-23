import os, torch, torch.distributed as dist
r = int(os.environ["RANK"]); dist.init_process_group("nccl")
torch.cuda.set_device(r)
x = torch.ones(1024, device=f"cuda:{r}") * (r + 1)
print(f"[rank{r}] p2p start", flush=True)
if r == 0:
    dist.send(x, dst=1)
    print(f"[rank0] send OK", flush=True)
else:
    dist.recv(x, src=0)
    torch.cuda.synchronize()
    print(f"[rank1] recv OK got={x[0].item()}", flush=True)
dist.barrier()
print(f"[rank{r}] P2P DONE", flush=True)
dist.destroy_process_group()
