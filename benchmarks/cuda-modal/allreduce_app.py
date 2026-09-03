"""The TP=2 collective, on a rented pair — 2026-09-03.

    BENCH_GPU=H100:2 modal run benchmarks/cuda-modal/allreduce_app.py
    BENCH_GPU=RTX-PRO-6000:2 modal run benchmarks/cuda-modal/allreduce_app.py

`allreduce-2026-09-02` measured what a TP=2 decode step pays for its collective
on two RX 7900 XT with no P2P at all -- device to host to device, 7.5 GB/s --
and got 16.6-21.5 us. Tonight's ladders put the same five models on two H100s
**with** NVLink and on two RTX PRO 6000 **without** it (`nvidia-smi nvlink -s`
is empty on that pair, probed 2026-09-02). Those two are the interconnect
comparison; this runs the same sweep under them so the collective is measured
rather than inferred from the ladder.

Same script as the ROCm run, generalised only in which library name it looks
for in /proc/self/maps: RCCL is NCCL's ROCm build and answers the same
`ncclAllReduce`. Five hidden sizes x eleven token counts, three timing modes,
the buffer zeroed so bf16 stays finite.

The GPU is taken from BENCH_GPU at import, which is evaluated where the app is
built -- the container never reads it, and `machine` is passed as an argument
so it cannot disagree with the card that was asked for.
"""
import json
import os
import time

import modal

GPU = os.environ.get("BENCH_GPU", "H100:2")
N = int(GPU.rsplit(":", 1)[1]) if ":" in GPU else 1
HERE = os.path.dirname(os.path.abspath(__file__))
AR = os.path.join(HERE, "..", "allreduce-2026-09-02", "allreduce.py")

WORK = modal.Volume.from_name("bench-work", create_if_missing=True)
image = (modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04",
                                   add_python="3.12")
         .pip_install("vllm==0.28.0", "nvidia-ml-py")
         .add_local_file(AR, "/bench/allreduce.py")
         .add_local_file(os.path.join(HERE, "..", "harness", "telemetry.py"),
                         "/bench/harness/telemetry.py"))
app = modal.App("bench-allreduce", image=image)


@app.function(gpu=GPU, timeout=1800, volumes={"/work": WORK})
def sweep(machine: str, run: str, nproc: int):
    import subprocess
    import sys
    D = f"/work/{run}"
    os.makedirs(D, exist_ok=True)
    print(subprocess.run("nvidia-smi topo -m | head -6", shell=True,
                         capture_output=True, text=True).stdout, flush=True)
    print("nvlink: " + (subprocess.run("nvidia-smi nvlink -s 2>&1 | head -4",
                                       shell=True, capture_output=True,
                                       text=True).stdout.strip() or "(empty)"),
          flush=True)
    # The card that arrived, not the card that was asked for. `gpu="H100:4"`
    # returned four H200s on 2026-09-03 while the same string returned four
    # H100s for a ladder run an hour earlier, and the only thing that caught
    # it was `nvidia-smi nvlink -s` printing the name. Never trust the request
    # string: read the device.
    got = subprocess.run(
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
        shell=True, capture_output=True, text=True).stdout.strip().splitlines()
    print(f"  asked for {machine!r}; got {len(got)} x {got[0] if got else '?'}",
          flush=True)
    env = dict(os.environ, BENCH_MACHINE=machine, AR_OUT=f"{D}/results.jsonl",
               PYTHONPATH="/bench", PYTHONUNBUFFERED="1")
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "-m", "torch.distributed.run",
                        f"--nproc_per_node={nproc}", "/bench/allreduce.py"],
                       env=env, cwd=D)
    out = {"_rc": r.returncode, "_wall_s": round(time.perf_counter() - t0, 1),
           "_cards": got}
    for f in sorted(os.listdir(D)):
        p = os.path.join(D, f)
        if os.path.isfile(p):
            out[f] = open(p).read()[-250_000:]
    WORK.commit()
    return out


@app.local_entrypoint()
def main(machine: str = "", run: str = ""):
    machine = machine or {"H100:2": "H100-80GB-HBM3-x2",
                          "RTX-PRO-6000:2": "RTX-PRO-6000-Blackwell-x2"}.get(GPU, GPU)
    run = run or f"allreduce-{machine}-2026-09-03"
    print(f"gpu={GPU}  nproc={N}  machine={machine}  work=/work/{run}")
    res = sweep.remote(machine, run, N)
    cards = res.get("_cards") or []
    if cards:
        print(f"  cards reported by the container: {len(cards)} x {cards[0]}")
        assert all(c == cards[0] for c in cards), f"mixed cards: {cards}"
    outdir = os.path.join(HERE, "..", "allreduce-2026-09-03")
    os.makedirs(outdir, exist_ok=True)
    for name, txt in res.items():
        if name.startswith("_"):
            continue
        fn = os.path.join(outdir, f"{machine}-{name}")
        open(fn, "w").write(txt)
        print(f"  wrote {os.path.relpath(fn)}  {len(txt)} bytes")
    print(f"rc={res['_rc']}  wall {res['_wall_s']:.0f}s")
