"""Do the downloaded checkpoint files match what the Hub advertised?

`dl_sym.py` fetched the symmetric checkpoint through hf-mirror.com rather than
from huggingface.co directly, so "we ran RedHatAI/Qwen3.8-27B-INT4" is a claim
about a mirror, not about upstream. This checks it.

`huggingface_hub` writes one `<file>.metadata` per download under
`.cache/huggingface/download/`, holding the repo commit, the server's ETag and
a timestamp. For LFS-backed files the ETag *is* the content sha256, which is
what makes this checkable after the fact: hash the local file and compare.
Small files are stored in git proper and their ETag is a 40-hex blob sha1
computed over a different preimage, so those are reported as `not_lfs` and
skipped rather than being made to look like failures.

    python3 verify_ckpt_sha.py <checkpoint_dir> [out.json]

Writes a JSON record so the provenance claim in README.md is recomputable
without re-downloading 15 GB.
"""
import hashlib
import json
import os
import sys


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/data/incoming/Qwen3.8-27B-INT4-sym"
    dest = sys.argv[2] if len(sys.argv) > 2 else "ckpt-sha256.json"
    mdir = os.path.join(root, ".cache", "huggingface", "download")
    if not os.path.isdir(mdir):
        # the asymmetric checkpoint predates this download route and carries no
        # metadata, so there is nothing upstream to compare against. Say so
        # rather than reporting a vacuous pass.
        print(f"no hub metadata under {root}; nothing to verify")
        print("SHA_UNAVAILABLE")
        return 2
    rows, commits = [], set()
    for name in sorted(os.listdir(mdir)):
        if not name.endswith(".metadata"):
            continue
        fname = name[: -len(".metadata")]
        local = os.path.join(root, fname)
        if not os.path.exists(local):
            continue
        parts = open(os.path.join(mdir, name)).read().split()
        commit, etag = parts[0], parts[1]
        commits.add(commit)
        if len(etag) != 64:
            rows.append({"file": fname, "status": "not_lfs", "etag": etag,
                         "size": os.path.getsize(local)})
            continue
        got = sha256(local)
        rows.append({"file": fname, "status": "match" if got == etag else "MISMATCH",
                     "expected_sha256": etag, "actual_sha256": got,
                     "size": os.path.getsize(local)})
        print(f"  {rows[-1]['status']:<9} {fname}", flush=True)

    lfs = [r for r in rows if r["status"] in ("match", "MISMATCH")]
    out = {
        "checkpoint": root,
        "repo_commit": sorted(commits),
        "lfs_files_checked": len(lfs),
        "lfs_files_matching": sum(1 for r in lfs if r["status"] == "match"),
        "files": rows,
    }
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"commit(s): {sorted(commits)}")
    print(f"{out['lfs_files_matching']}/{out['lfs_files_checked']} LFS files match "
          f"the ETag the Hub advertised")
    ok = lfs and all(r["status"] == "match" for r in lfs)
    print("SHA_OK" if ok else "SHA_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
