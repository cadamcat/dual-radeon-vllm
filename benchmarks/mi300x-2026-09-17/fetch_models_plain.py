#!/usr/bin/env python3
"""The six checkpoints at their pinned revisions, without Modal.

`benchmarks/modal-2026-09-02/fetch_models.py` is a Modal app: it puts these on
a Volume from a CPU function. On a rented card there is no Volume, so this is
the same list and the same revisions, downloaded straight onto the box. What it
keeps from that script is the assertion that matters: a model counts as present
only when its bytes on disk match what `volume.json` recorded, because a
silently short checkpoint is the expensive failure mode.

    BENCH_MODELS=/models python3 fetch_models_plain.py [--check]
"""
import json, os, sys

DEST = os.environ.get("BENCH_MODELS", "/models")
VOL = os.environ.get("BENCH_VOLUME", os.path.join(os.path.dirname(os.path.abspath(__file__)), "volume.json"))
CHECK = "--check" in sys.argv


def on_disk(d):
    total = 0
    for root, _, files in os.walk(d):
        for f in files:
            p = os.path.join(root, f)
            if not os.path.islink(p):
                total += os.path.getsize(p)
    return total


def main():
    models = json.load(open(VOL))
    os.makedirs(DEST, exist_ok=True)
    bad = 0
    for m in models:
        d = os.path.join(DEST, m["name"])
        have = on_disk(d) if os.path.isdir(d) else 0
        want = m["bytes"]
        if have >= want:
            print(f"ok      {m['name']:34} {have/2**30:7.2f} GiB")
            continue
        if CHECK:
            print(f"MISSING {m['name']:34} {have/2**30:7.2f} of {want/2**30:.2f} GiB")
            bad += 1
            continue
        print(f"fetch   {m['name']:34} {m['repo']}@{m['revision'][:12]}", flush=True)
        from huggingface_hub import snapshot_download
        snapshot_download(m["repo"], revision=m["revision"], local_dir=d,
                          max_workers=8)
        have = on_disk(d)
        status = "ok" if have >= want else "SHORT"
        print(f"{status:7} {m['name']:34} {have/2**30:7.2f} of {want/2**30:.2f} GiB", flush=True)
        bad += status == "SHORT"
    print(f"\n{len(models) - bad} of {len(models)} complete")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
