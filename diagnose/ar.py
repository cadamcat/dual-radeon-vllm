import os, torch, torch.distributed as dist
r = int(os.environ["RANK"])
# select the device BEFORE the process group exists. Initialising first lets
# RCCL bind the communicator to whatever device is current -- device 0 for every
# rank -- and then the hang this script is meant to diagnose would be a
# different hang from the one the deployment hits. `device_id` pins it, the
# same way the reporter's own reproducer in benchmarks/rccl-6565 does.
torch.cuda.set_device(r)
dist.init_process_group("nccl", device_id=torch.device(f"cuda:{r}"))
print(f"[rank{r}] pg ready, dev={torch.cuda.current_device()}", flush=True)
x = torch.ones(1024, device=f"cuda:{r}")
print(f"[rank{r}] tensor on card, calling all_reduce...", flush=True)
dist.all_reduce(x)
torch.cuda.synchronize()
print(f"[rank{r}] all_reduce OK sum={x[0].item()}", flush=True)
dist.destroy_process_group()
