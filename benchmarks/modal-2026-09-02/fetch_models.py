#!/usr/bin/env python3
"""Run 0a: put the five checkpoints on a Modal Volume, once, from a CPU.

Every rented card in this campaign needs the same 82.2 GiB of weights. Fetching
them inside the GPU function would charge the download to a $3.03/h card five
times over -- roughly $0.85 per run at ten minutes a pull, and there are six
runs. A Volume costs $0.09/GiB/month, so 82.2 GiB is **$0.243 a day**: the
whole campaign's storage is cheaper than one card-hour, and the download is
paid once on CPU at $0.0473/core/h.

    modal run benchmarks/modal-2026-09-02/fetch_models.py            # fetch
    modal run benchmarks/modal-2026-09-02/fetch_models.py --check    # verify only

The Volume is NOT free after the campaign ends. Delete it:

    modal volume delete llm-ckpt

What this asserts, because a silently short checkpoint is the expensive failure
mode: the file list and every file's byte count come from the Hugging Face
tree API at fetch time, and a model is only marked complete when every file
that API named is on the Volume at exactly that size. `manifest.json` records
the resolved commit sha, so a later run can say which revision it measured
rather than "main".
"""
import json
import os
import time

import modal

# The five, in the campaign's own naming -- the directory names the Radeon and
# Colab campaigns already use, so a config table moves between platforms
# unchanged.
MODELS = {
    "Qwen3-8B":                       "Qwen/Qwen3-8B",
    "gemma-4-12B-it-qat-w4a16-ct":    "google/gemma-4-12B-it-qat-w4a16-ct",
    "gemma-4-26B-A4B-AWQ":            "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
    "gemma-4-31B-it-qat-w4a16-ct":    "google/gemma-4-31B-it-qat-w4a16-ct",
    "Qwen3.8-27B-AWQ-INT4":           "cyankiwi/Qwen3.8-27B-AWQ-INT4",
    # Added 2026-09-03, after the H100 run made it worth the storage. It is the
    # missing third case: gemma-4 and Qwen3-8B read a KV that grows without
    # bound with context, Qwen3.8-27B carries a recurrent state that does not
    # grow at all, and Muse-Glimmer attends through a 2 048-token window -- a
    # term that grows and then stops. On an A100 its decode falls 9.8 % from
    # 500 to 32 000 where every other model there falls 28-44 %.
    "Muse-Glimmer-30B-INT4":          "RedHatAI/Muse-Glimmer-30B-INT4",
}

# vLLM reads safetensors and json. A repo that also ships .bin or .gguf would
# double the bill for bytes nothing opens.
SKIP_SUFFIX = (".bin", ".pth", ".gguf", ".msgpack", ".h5", ".onnx", ".onnx_data")

VOL = modal.Volume.from_name("llm-ckpt", create_if_missing=True)
MNT = "/models"

# hub 1.x dropped the hf_transfer extra; hf-xet is the accelerator now and is
# already a default dependency on x86_64. Asking for the extra makes that
# explicit rather than assumed, and HF_HUB_ENABLE_HF_TRANSFER is deliberately
# NOT set -- in 1.x it does nothing, and a no-op env var in a script reads as
# a speed claim that is not true.
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("huggingface_hub[hf-xet]==1.29.0"))

app = modal.App("ckpt-fetch", image=image)


def _wanted(repo):
    """The file list and sizes, from the API, at fetch time.

    Not from a table in this file: a table is a claim about a repository that
    can be re-uploaded. This is what the repository holds right now, and it is
    what the assertion below compares against.
    """
    from huggingface_hub import HfApi
    api = HfApi()
    sha = api.model_info(repo).sha
    # model_info(files_metadata=True) leaves `size` empty for LFS files -- that
    # was checked against all five of these repos before this ran, and it would
    # have made the completeness assertion below vacuous. The tree endpoint
    # carries real byte counts.
    files = {}
    for e in api.list_repo_tree(repo, revision=sha, recursive=True):
        p = getattr(e, "path", "")
        if not hasattr(e, "size") or p.startswith(".") or p.endswith(SKIP_SUFFIX):
            continue
        files[p] = e.size
    if not files or any(v is None for v in files.values()):
        raise RuntimeError(f"{repo}: tree gave no sizes; refusing to fetch blind")
    return sha, files


@app.function(volumes={MNT: VOL}, timeout=3600, cpu=4, memory=4096,
              max_containers=5)
def fetch(item):
    name, repo = item
    from huggingface_hub import snapshot_download
    t0 = time.perf_counter()
    dest = f"{MNT}/{name}"
    sha, want = _wanted(repo)
    print(f"[{name}] {repo} @ {sha[:12]}  {len(want)} files, "
          f"{sum(v or 0 for v in want.values()) / (1 << 30):.2f} GiB")

    # Idempotent: a container that dies halfway leaves a partial directory, and
    # the next run should finish it rather than start over. snapshot_download
    # already skips files whose size and etag match, so this is the cheap path.
    snapshot_download(repo, revision=sha, local_dir=dest,
                      ignore_patterns=[f"*{s}" for s in SKIP_SUFFIX],
                      max_workers=8)
    VOL.commit()

    missing, wrong = [], []
    for p, sz in want.items():
        fp = os.path.join(dest, p)
        if not os.path.exists(fp):
            missing.append(p)
        elif sz is not None and os.path.getsize(fp) != sz:
            wrong.append(f"{p}: {os.path.getsize(fp)} != {sz}")
    have = sum(os.path.getsize(os.path.join(dest, p))
               for p in want if os.path.exists(os.path.join(dest, p)))
    el = time.perf_counter() - t0
    ok = not missing and not wrong
    rec = {"name": name, "repo": repo, "revision": sha, "n_files": len(want),
           "bytes": have, "gib": round(have / (1 << 30), 3), "complete": ok,
           "missing": missing[:10], "wrong_size": wrong[:10],
           "fetch_s": round(el, 1),
           "mib_s": round(have / (1 << 20) / el, 1) if el > 0 else None}
    if ok:
        with open(f"{dest}/manifest.json", "w") as f:
            json.dump(rec, f, indent=1)
        VOL.commit()
    print(f"[{name}] {'OK' if ok else 'INCOMPLETE'} {rec['gib']} GiB in "
          f"{el:.0f}s ({rec['mib_s']} MiB/s)")
    return rec


@app.function(volumes={MNT: VOL}, timeout=900, cpu=2, memory=2048)
def check():
    """What is actually on the Volume, read from the Volume."""
    VOL.reload()
    out = []
    for name, repo in MODELS.items():
        d = f"{MNT}/{name}"
        mf = f"{d}/manifest.json"
        rec = {"name": name, "on_volume": os.path.isdir(d),
               "manifest": os.path.exists(mf)}
        if rec["on_volume"]:
            tot = n = 0
            for root, _, fs in os.walk(d):
                for f in fs:
                    tot += os.path.getsize(os.path.join(root, f))
                    n += 1
            rec |= {"n_files": n, "gib": round(tot / (1 << 30), 3)}
        if rec["manifest"]:
            m = json.load(open(mf))
            rec |= {"revision": m["revision"], "manifest_gib": m["gib"],
                    "complete": m["complete"]}
        out.append(rec)
    return out


@app.local_entrypoint()
def main(check_only: bool = False):
    if not check_only:
        t0 = time.perf_counter()
        recs = list(fetch.map(list(MODELS.items())))
        bad = [r for r in recs if not r["complete"]]
        print(f"\nfetched {sum(r['gib'] for r in recs):.1f} GiB in "
              f"{time.perf_counter() - t0:.0f}s wall, {len(bad)} incomplete")
        for r in recs:
            print(f"  {r['name']:<32} {r['gib']:6.2f} GiB  {r['revision'][:12]}  "
                  f"{r['fetch_s']:5.0f}s  {r['mib_s']} MiB/s"
                  + ("" if r["complete"] else f"  INCOMPLETE {r['missing']}{r['wrong_size']}"))
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "volume.json")
        json.dump(recs, open(out, "w"), indent=1)
        print(f"wrote {out}")
    print("\n--- on the Volume ---")
    for r in check.remote():
        print("  " + json.dumps(r))
