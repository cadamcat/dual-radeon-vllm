#!/usr/bin/env python3
"""Put vllm#45916's split-KV decode back into vllm-027, asserting both md5s.

Why this exists
---------------
The first attempt at this campaign ran without it and nobody noticed until the
numbers were read: `Q38-tp2-x16` decoded at **3.88 tok/s at 32 000 tokens**
against the 2026-08-29 arm's 36.47. That is not a link effect and not noise —
`../hybrid-splitkv-027/qwen38-027-depth.jsonl` records the same checkpoint at
32 768 tokens as **3.828 tok/s stock** and 35.20 with the patch, so the run had
reproduced the *stock* arm to 1.4% while claiming to reproduce the patched one.

The container had lost the patch at some point between 2026-08-29 and
2026-09-02; `chunked_prefill_paged_decode.py` read `86f68d47…`, which
`../hybrid-splitkv-027/provenance.json` records as the image's own file.
Checking #45450's two md5s before the run — which this campaign did — was not
enough, because #45916 lives in a third file nobody checked.

    python3 apply_45916.py            # -> 84c6d4f9…, the patched file
    python3 apply_45916.py --check    # report only

The patched file is `/data/50603/cppd_027_splitkv.py` on the guest, built from
`../hybrid-splitkv-027/pr45916.diff` (diff md5 `273fef16…`) and carrying the
md5 that provenance.json recorded when it was built. Both md5s are asserted
here, in both directions, so a container in a state this does not expect fails
instead of measuring the wrong arm — the rule `revert45450.py` uses for the
other patch.
"""
import subprocess
import sys

HOST = "ubuntu@192.168.31.121"
KEY = "/Users/yaoxu/.ssh/pve_key"
C = "vllm-027"
CPPD = ("/opt/python/lib/python3.14/site-packages/"
        "vllm/v1/attention/ops/chunked_prefill_paged_decode.py")
SRC = "/data/50603/cppd_027_splitkv.py"
MD5_STOCK = "86f68d47c7bdc390ced4c6d0c18025fa"
MD5_SPLITKV = "84c6d4f9b2dfe2714b3a8f43ee832b02"


def sh(cmd, check=True):
    r = subprocess.run(["ssh", "-i", KEY, "-o", "ConnectTimeout=20", HOST, cmd],
                       capture_output=True, text=True)
    if check and r.returncode != 0:
        raise SystemExit(f"failed: {cmd}\n{r.stderr[:400]}")
    return r.stdout.strip()


def in_container_md5():
    sh(f'sudo -n docker cp "{C}:{CPPD}" /tmp/_cppd.py')
    return sh("md5sum /tmp/_cppd.py").split()[0]


def main():
    if sh(f"sudo -n docker inspect -f '{{{{.State.Running}}}}' {C}") != "true":
        sh(f"sudo -n docker start {C}")
    before = in_container_md5()
    print("before:", before,
          {MD5_STOCK: "stock (no #45916)", MD5_SPLITKV: "split-KV"}.get(before, "UNKNOWN"))
    if before not in (MD5_STOCK, MD5_SPLITKV):
        raise SystemExit("refusing: the container's file is neither state on record")
    if "--check" in sys.argv:
        return
    if before == MD5_SPLITKV:
        print("already patched; nothing to do")
        return
    got = sh(f"md5sum {SRC}").split()[0]
    if got != MD5_SPLITKV:
        raise SystemExit(f"refusing: {SRC} is {got}, not the recorded {MD5_SPLITKV}")
    sh(f'sudo -n docker cp {SRC} "{C}:{CPPD}"')
    after = in_container_md5()
    print("after: ", after)
    if after != MD5_SPLITKV:
        raise SystemExit(f"refusing: file is {after} after the copy, not {MD5_SPLITKV}")
    marker = sh(f'sudo -n docker exec {C} grep -c '
                f'kernel_paged_attention_2d_splitkv {CPPD}')
    print(f"OK: {C} carries vllm#45916 ({marker} split-KV kernel references)")


if __name__ == "__main__":
    main()
