"""Run 1 on Modal: one RTX PRO 6000, five models, TP=1.

    modal run benchmarks/cuda-pro6000/campaign-2026-09-02/app.py

What this wrapper is for, beyond starting a container:

  * **a budget that is enforced, not estimated.** Configurations run one
    subprocess each and the next one only starts if the run has time left
    inside `budget_s`. A card that turns out to be half the speed assumed
    stops after the configurations it could afford, with those configurations
    complete, rather than being killed mid-ladder by the function timeout.
  * **the checkpoints are not downloaded here.** They are on the `llm-ckpt`
    Volume, put there once by `modal-2026-09-02/fetch_models.py` on a CPU at
    $0.0473/core/h. Pulling 82 GiB inside a $3.03/h card, six runs over,
    would cost more than storing them for the month.
  * **the book is shipped, not fetched.** `get_book()` goes to gutenberg.org
    when it has nothing on disk, and on 2026-08-30 that call timed out on a
    healthy A100 whose engine had already started, taking 231 s of engine
    start with it. The copy in this directory is verified by md5 before the
    image is built, so the rented card never touches the network for it.
  * **results survive the container.** Everything is written to a Volume as it
    is produced and returned to the caller at the end. A run that dies at the
    fourth configuration still has the first three.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = "pro6000-2026-09-02"
MACHINE = "RTX-PRO-6000"
GPU = "RTX-PRO-6000"
# 5 400 s at $3.03/h is $4.55. The A100 took 38 min for four of these five
# configurations; this leaves room for the fifth, the five long rungs, and a
# card that is slower than assumed -- and stops before it is a surprise.
BUDGET_S = 5400
ORDER = ["G12", "G31", "B8", "G26A4B", "Q38"]

# One copy, where cut_prompts.py already caches it. Both campaigns ship it
# into their image so no rented card ever goes to gutenberg.org for it.
BOOK = os.path.join(HERE, "..", "..", "prompts", ".gutenberg-1228.txt")
BOOK_MD5 = "2f3418d3e506a1aa3d0a854852bb4065"
BOOK_BYTES = 970612

CKPT = modal.Volume.from_name("llm-ckpt", create_if_missing=False)
WORK = modal.Volume.from_name("bench-work", create_if_missing=True)

# vllm 0.28.0 exactly: every CUDA row in this repository -- A100 80GB, A100
# 40GB, L4, T4 -- was measured on 0.28.0 with torch 2.13.0+cu130, and a new
# machine measured on a different stack is a new machine AND a new stack.
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("vllm==0.28.0", "nvidia-ml-py")
         .env({"HF_HUB_OFFLINE": "1", "VLLM_LOGGING_LEVEL": "INFO"})
         .add_local_file(os.path.join(HERE, "run.py"), "/bench/campaign/run.py")
         .add_local_file(os.path.join(HERE, "..", "..", "harness", "telemetry.py"),
                         "/bench/harness/telemetry.py")
         .add_local_file(BOOK, "/bench/.gutenberg-1228.txt"))

app = modal.App("bench-pro6000", image=image)


# No cpu= or memory= request: Modal bills the larger of what is asked for and
# what is used, so asking for 8 cores and 32 GiB would add roughly $0.95 to a
# $4.55 run for headroom vLLM does not need. timeout is the backstop behind
# `budget_s`, which is the thing that actually stops the run.
@app.function(gpu=GPU, volumes={"/models": CKPT, "/work": WORK}, timeout=7200)
def campaign(order: list, budget_s: int):
    D = f"/work/{RUN}"
    os.makedirs(D, exist_ok=True)
    # the book, into the place get_book() looks first
    if not os.path.exists(f"{D}/origin.txt"):
        with open("/bench/.gutenberg-1228.txt", "rb") as a, open(f"{D}/origin.txt", "wb") as b:
            b.write(a.read())

    # The checkpoints, checked against the manifests fetch_models.py wrote.
    # A Volume that lost a shard reads as a model that will not load, three
    # minutes and $0.15 into a configuration, in a traceback about safetensors.
    CKPT.reload()
    for name in os.listdir("/models"):
        mf = f"/models/{name}/manifest.json"
        if os.path.exists(mf):
            m = json.load(open(mf))
            print(f"  ckpt {name:<32} {m['gib']:6.2f} GiB  {m['revision'][:12]}  "
                  f"complete={m['complete']}", flush=True)

    env = dict(os.environ, BENCH_WORK=D, BENCH_MODELS="/models",
               BENCH_MACHINE=MACHINE, PYTHONUNBUFFERED="1")
    t0 = time.perf_counter()
    ran = []
    for cid in order:
        left = budget_s - (time.perf_counter() - t0)
        if left < 300:
            print(f"\n=== budget: {left:.0f}s left, not starting {cid} ===", flush=True)
            break
        print(f"\n=== {cid} ({left / 60:.0f} min of budget left) ===", flush=True)
        t1 = time.perf_counter()
        r = subprocess.run([sys.executable, "/bench/campaign/run.py", cid],
                           env=env, cwd=D)
        ran.append({"cfg": cid, "rc": r.returncode,
                    "wall_s": round(time.perf_counter() - t1, 1)})
        print(f"=== {cid} rc={r.returncode} in {ran[-1]['wall_s']:.0f}s ===", flush=True)
        WORK.commit()

    out = {"_ran": ran, "_wall_s": round(time.perf_counter() - t0, 1)}
    for f in sorted(os.listdir(D)):
        p = os.path.join(D, f)
        if not os.path.isfile(p) or f.startswith("ladder-") or f == "origin.txt":
            continue
        b = open(p, "rb").read()
        out[f] = b[-250_000:].decode("utf-8", "ignore") if len(b) > 250_000 \
            else b.decode("utf-8", "ignore")
    WORK.commit()
    return out


@app.local_entrypoint()
def main(budget_s: int = BUDGET_S, cfgs: str = ",".join(ORDER)):
    # The book is not in git -- .gitignore has held it out since cut_prompts.py
    # first cached it -- so a clean clone fetches it once here, on the laptop,
    # and the md5 below is what makes that fetch safe to ship into the image.
    if not os.path.exists(BOOK):
        import urllib.request
        for _u in ("https://www.gutenberg.org/cache/epub/1228/pg1228.txt",
                   "https://www.gutenberg.org/files/1228/1228-0.txt"):
            try:
                _b = urllib.request.urlopen(_u, timeout=180).read()
                if len(_b) > 400000:
                    open(BOOK, "wb").write(_b)
                    break
            except Exception as _e:                          # noqa: BLE001
                print(f"  {_u}: {_e!r}")
        else:
            raise SystemExit(f"could not fetch the book to {BOOK}")
    b = open(BOOK, "rb").read()
    md5 = hashlib.md5(b).hexdigest()
    assert md5 == BOOK_MD5 and len(b) == BOOK_BYTES, \
        f"book is {len(b)} bytes md5 {md5}, expected {BOOK_BYTES} / {BOOK_MD5}"
    print(f"book ok: {len(b)} bytes, md5 {md5}")

    t0 = time.perf_counter()
    res = campaign.remote(cfgs.split(","), budget_s)
    wall = time.perf_counter() - t0
    for name, txt in res.items():
        if name.startswith("_"):
            continue
        with open(os.path.join(HERE, name), "w") as f:
            f.write(txt)
        print(f"  wrote {name}  {len(txt)} bytes")
    print(f"\nconfigs: {json.dumps(res['_ran'])}")
    print(f"in-container {res['_wall_s']:.0f}s, local wall {wall:.0f}s, "
          f"= ${wall / 3600 * 3.03:.2f} at $3.03/h plus startup")
