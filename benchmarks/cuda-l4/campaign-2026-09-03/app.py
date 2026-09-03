"""L4 on Modal, from cuda-modal/make_campaign.py.

The same L4 Colab measured twice, on the same model, to put a number on the platform difference every machine-to-machine ratio here inherits.

    modal run <this file>
    modal run <this file> --cfgs G31,G12 --run <dir> --out-suffix -retry

`budget_s` is a ceiling, not a spend: configurations start only while there is
time left for one, and only the seconds used are billed. Results are written
to a Volume as they are produced and returned at the end, so a run that dies
at the fourth configuration still has the first three.

The checkpoints are not downloaded here. They are on the `llm-ckpt` Volume,
put there once on a CPU by modal-2026-09-02/fetch_models.py; the manifests
that fetch wrote are checked before anything loads.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
# The work directory on the Volume, and therefore which results.jsonl the
# checkpointing reads. A configuration that failed is recorded as `done` --
# deliberately, so a crash is not retried in a loop -- so a rerun with a fixed
# harness needs a directory of its own rather than an edited results file.
RUN = "l4-2026-09-03"
MACHINE = "L4"
GPU = "L4"
RATE_USD_H = 0.8
# A cap, not a spend: the run stops when the work is done and only the seconds
# used are billed. 5 400 s at $3.95/h is $5.93 as a ceiling. It was 4 200 for
# the first container, and 4 200 was what made the estimator refuse the long
# rungs by three minutes -- a ceiling low enough to change what gets measured
# is a ceiling in the wrong place.
BUDGET_S = 1800
RESERVE_S = 420          # harvest, the final commit, and being wrong
ORDER = ["G12"]
LONG_CAPABLE = set(["G12"])
ALWAYS_LONG = set(["G12"])

# One copy, where cut_prompts.py already caches it. Both campaigns ship it
# into their image so no rented card ever goes to gutenberg.org for it.
BOOK = os.path.join(HERE, "..", "..", "prompts", ".gutenberg-1228.txt")
BOOK_MD5 = "2f3418d3e506a1aa3d0a854852bb4065"
BOOK_BYTES = 970612

CKPT = modal.Volume.from_name("llm-ckpt", create_if_missing=False)
WORK = modal.Volume.from_name("bench-work", create_if_missing=True)
# FlashInfer JIT-compiles its sampling module at engine start: 111 s, measured.
# The cache is keyed by flashinfer version and SM arch, so one Volume serves
# every machine in this campaign and pays the 111 s once per architecture
# instead of once per container.
JIT = modal.Volume.from_name("flashinfer-jit", create_if_missing=True)

# A CUDA 13.0 toolkit, and this is not a detail. `debian_slim` plus
# `pip install vllm` has no nvcc at all, and pointing CUDA_HOME at the nvcc
# that arrives as a pip dependency does not work either: that one is 13.3,
# and flashinfer 0.6.16's bundled cccl rejects it with "CUDA compiler and
# CUDA toolkit headers are incompatible". Both were watched failing on a real
# card before this line was written, and `gen_sampling_module().build()` was
# watched succeeding on this base image in 111 s.
#
# 13.0 is also the version every existing CUDA row in this repository was
# measured against -- Colab's own toolkit -- so this matches the machines it
# will be compared with rather than merely working.
#
# vllm 0.28.0 exactly, for the same reason: A100 80GB, A100 40GB, L4 and T4
# rows are all 0.28.0 with torch 2.13.0+cu130.
image = (modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04",
                                   add_python="3.12")
         .pip_install("vllm==0.28.0", "nvidia-ml-py")
         .env({"HF_HUB_OFFLINE": "1", "VLLM_LOGGING_LEVEL": "INFO"})
         .add_local_file(os.path.join(HERE, "run.py"), "/bench/campaign/run.py")
         .add_local_file(os.path.join(HERE, "..", "..", "harness", "telemetry.py"),
                         "/bench/harness/telemetry.py")
         .add_local_file(BOOK, "/bench/.gutenberg-1228.txt"))

app = modal.App("bench-l4", image=image)


def _ladder_walls(res_path, cid):
    """(short, long) seconds this configuration's rungs actually took.

    One request produces a prefill row and a decode row carrying the same
    `wall_s`, so summing prefill rows counts each request once.
    """
    short = long_ = 0.0
    if not os.path.exists(res_path):
        return 0.0, 0.0
    for line in open(res_path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("kind") != "prefill" or r.get("cfg") != cid:
            continue
        w = r.get("wall_s") or 0
        if r["target"] > 32000:
            long_ += w
        else:
            short += w
    return short, long_


# The FlashInfer sampling module, JIT-compiled at engine start: 111 s, measured
# on an L4 before this campaign ran. It is charged to `init_engine_s` on the
# container that pays it and to nothing afterwards, because the JIT Volume
# keeps it. An estimator that does not subtract it prices every later
# configuration as if it will pay a cost that no longer exists -- which is
# exactly what refused the long rungs on the first container.
JIT_ONCE_S = 111


def _seed(res, jit_cached):
    """What a configuration costs here, from the ones already measured here.

    A restarted container begins with an empty `ran` list but not with nothing
    known: results.jsonl on the Volume carries every completed configuration's
    engine start and every rung's wall clock. Without this the second container
    repeats the first container's guess, having just paid for the measurement
    that would have corrected it.
    """
    metas, done = {}, set()
    if os.path.exists(res):
        for line in open(res):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("kind") == "model_meta":
                metas[r["cfg"]] = float(r.get("init_engine_s") or 0)
            elif r.get("kind") == "config_complete":
                done.add(r["cfg"])
    units, longs = [], []
    for cid in sorted(done):
        short_s, long_s = _ladder_walls(res, cid)
        init = metas.get(cid, 0.0) - (JIT_ONCE_S if jit_cached else 0.0)
        units.append(max(0.0, init) + short_s)
        if long_s:
            longs.append(long_s)
        print(f"  seed {cid}: init {metas.get(cid, 0):.0f}s"
              f"{' -111 jit' if jit_cached else ''}, short {short_s:.0f}s, "
              f"long {long_s:.0f}s", flush=True)
    return (max(units) if units else 570.0), (max(longs) if longs else 600.0)


def _plan(remaining, left_s, unit_cfg_s, unit_long_s):
    """Can every remaining long-capable configuration carry the long rungs?

    All-or-nothing, and deliberately pessimistic: `unit_cfg_s` is the largest
    (engine start + eleven rungs) seen so far and `unit_long_s` the largest
    five-rung tail, both measured on the biggest model in the set. A smaller
    model costing what the 31B cost is an over-estimate in the direction that
    does not overspend.
    """
    n = len(remaining)
    k = len([c for c in remaining if c in LONG_CAPABLE])
    need = n * unit_cfg_s + k * unit_long_s
    return need + RESERVE_S <= left_s, need, n, k


@app.function(gpu=GPU, timeout=6000,
              volumes={"/models": CKPT, "/work": WORK,
                       "/root/.cache/flashinfer": JIT})
def campaign(order: list, budget_s: int, run: str):
    D = f"/work/{run}"
    os.makedirs(D, exist_ok=True)
    if not os.path.exists(f"{D}/origin.txt"):
        with open("/bench/.gutenberg-1228.txt", "rb") as a, open(f"{D}/origin.txt", "wb") as b:
            b.write(a.read())

    # The checkpoints, checked against the manifests fetch_models.py wrote. A
    # Volume that lost a shard reads as a model that will not load, three
    # minutes into a configuration, in a traceback about safetensors.
    CKPT.reload()
    for name in sorted(os.listdir("/models")):
        mf = f"/models/{name}/manifest.json"
        if os.path.exists(mf):
            m = json.load(open(mf))
            print(f"  ckpt {name:<32} {m['gib']:6.2f} GiB  {m['revision'][:12]}  "
                  f"complete={m['complete']}", flush=True)

    res = f"{D}/results.jsonl"
    base = dict(os.environ, BENCH_WORK=D, BENCH_MODELS="/models",
                BENCH_MACHINE=MACHINE, PYTHONUNBUFFERED="1")
    t0 = time.perf_counter()
    ran, decisions = [], []
    # Before anything is measured the estimate is the A100's own 38 min for
    # four, per configuration; after anything is measured it is that.
    jit_cached = bool(os.path.exists("/root/.cache/flashinfer")
                      and os.listdir("/root/.cache/flashinfer"))
    print(f"  flashinfer JIT cache: "
          f"{'populated' if jit_cached else 'empty, this container pays 111 s'}",
          flush=True)
    unit_cfg_s, unit_long_s = _seed(res, jit_cached)

    for n, cid in enumerate(order):
        left = budget_s - (time.perf_counter() - t0)
        if left < RESERVE_S:
            print(f"\n=== budget: {left:.0f}s left, not starting {cid} ===", flush=True)
            decisions.append({"cfg": cid, "skipped": "budget", "left_s": round(left)})
            break
        if cid in ALWAYS_LONG:
            go, need = True, None
        else:
            go, need, nn, kk = _plan(order[n:], left, unit_cfg_s, unit_long_s)
            print(f"  plan: {nn} left, {kk} long-capable, need {need / 60:.0f} min "
                  f"+ {RESERVE_S / 60:.0f} min reserve, have {left / 60:.0f} min "
                  f"-> long {'ON' if go else 'OFF'}", flush=True)
        long_on = cid if (go and cid in LONG_CAPABLE) else ""
        decisions.append({"cfg": cid, "long": bool(long_on), "left_s": round(left),
                          "need_s": None if need is None else round(need),
                          "unit_cfg_s": round(unit_cfg_s),
                          "unit_long_s": round(unit_long_s)})
        print(f"\n=== {cid} (long={'yes' if long_on else 'no'}, "
              f"{left / 60:.0f} min of budget left) ===", flush=True)
        t1 = time.perf_counter()
        r = subprocess.run([sys.executable, "/bench/campaign/run.py", cid],
                           env=dict(base, BENCH_LONG_CFGS=long_on), cwd=D)
        wall = time.perf_counter() - t1
        short_s, long_s = _ladder_walls(res, cid)
        ran.append({"cfg": cid, "rc": r.returncode, "wall_s": round(wall, 1),
                    "long": bool(long_on),
                    "rungs_short_s": round(short_s, 1), "rungs_long_s": round(long_s, 1)})
        # what the next decision is made from, measured
        if r.returncode == 0:
            paid_jit = 0 if jit_cached else JIT_ONCE_S
            unit_cfg_s = max(unit_cfg_s, wall - long_s - paid_jit)
            jit_cached = True          # whoever paid it, it is cached now
        if long_s > 0:
            unit_long_s = long_s
        print(f"=== {cid} rc={r.returncode} in {wall:.0f}s "
              f"(short rungs {short_s:.0f}s, long rungs {long_s:.0f}s) ===", flush=True)
        WORK.commit()
        JIT.commit()

    out = {"_ran": ran, "_decisions": decisions,
           "_wall_s": round(time.perf_counter() - t0, 1)}
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
def main(budget_s: int = BUDGET_S, cfgs: str = ",".join(ORDER),
         run: str = RUN, out_suffix: str = ""):
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
    print(f"work dir /work/{run}, configs {cfgs}")
    res = campaign.remote(cfgs.split(","), budget_s, run)
    wall = time.perf_counter() - t0
    for name, txt in res.items():
        if name.startswith("_"):
            continue
        stem, dot, ext = name.rpartition(".")
        name = f"{stem}{out_suffix}{dot}{ext}" if dot else name + out_suffix
        with open(os.path.join(HERE, name), "w") as f:
            f.write(txt)
        print(f"  wrote {name}  {len(txt)} bytes")
    print("\nconfigs:")
    for r in res["_ran"]:
        print("  " + json.dumps(r))
    print("decisions:")
    for d in res["_decisions"]:
        print("  " + json.dumps(d))
    print(f"\nin-container {res['_wall_s']:.0f}s, local wall {wall:.0f}s, "
          f"= ${wall / 3600 * RATE_USD_H:.2f} at ${RATE_USD_H}/h plus startup")
