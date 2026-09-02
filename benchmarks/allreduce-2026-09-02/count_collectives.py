#!/usr/bin/env python3
"""count_collectives.py — how many all-reduces a decode step actually issues.

Every per-step figure this campaign derives multiplies one collective by
`2 x layers`, on the standard reading that each decoder layer reduces once after
attention's `o_proj` and once after the MLP's `down_proj`. That is architecture,
not measurement, and this repository's rule is that a number in the prose comes
from a file. So count them.

Method — differencing, so nothing has to be attributed
-----------------------------------------------------
RCCL logs one line per collective under `NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=COLL`. Counting the lines of a whole request would count the
prefill's collectives, the warm-up's, and the graph capture's along with the
decode steps'. Instead: send the *same prompt* twice, once with `max_tokens=A`
and once with `max_tokens=B`, and difference the counts. Everything that is not
a decode step is identical in the two and cancels.

    (lines_B - lines_A) / (B - A)  =  collectives per decode step

Two cautions this script records rather than hides:

* **A captured graph may log once at capture and not at replay.** If it does,
  the difference comes out at or near zero, which is a result about the logging
  and not about the model. `--enforce-eager` removes graphs entirely, so the
  run below is eager, and the count is of the ops the engine issues, which is
  what the arithmetic multiplies. Whether a graph replay costs what an eager
  issue costs is a separate question, and `allreduce.py` answers it with
  `t_graph_us`.
* **The count is per rank.** Both ranks issue the same collectives; rank 0's
  file is the one differenced, and rank 1's total is printed beside it.
"""
import json
import os
import re
import subprocess
import sys
import time

import requests

D = os.environ.get("CC_DIR", "/rb/ar0902/count")
URL = "http://127.0.0.1:8000/v1/chat/completions"
HEALTH = "http://127.0.0.1:8000/health"
MODEL = os.environ.get("CC_MODEL", "/models/Qwen3-8B")
OUT = os.environ.get("CC_OUT", "/rb/ar0902/collectives.jsonl")
PROMPT = "Count from one to ten, then stop."
PAIR = (8, 40)          # max_tokens for the two requests


def wait_ready(deadline_s=900):
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        try:
            if requests.get(HEALTH, timeout=5).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def ask(max_tokens):
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens, "temperature": 0.0,
            "ignore_eos": True}          # every request must run its full length
    r = requests.post(URL, json=body, timeout=600)
    r.raise_for_status()
    j = r.json()
    return j["usage"]["completion_tokens"]


def count(path):
    """AllReduce lines in one rank's RCCL debug file."""
    if not os.path.exists(path):
        return None
    n = 0
    with open(path, errors="ignore") as fh:
        for line in fh:
            if re.search(r"\bAllReduce\b", line):
                n += 1
    return n


def main():
    if not wait_ready():
        sys.exit("server never became healthy")
    logs = sorted(f for f in os.listdir(D) if f.startswith("rccl."))
    rows = []
    for mt in PAIR:
        before = {f: count(os.path.join(D, f)) for f in
                  sorted(g for g in os.listdir(D) if g.startswith("rccl."))}
        got = ask(mt)
        time.sleep(3)                    # the log is written by another process
        after = {f: count(os.path.join(D, f)) for f in
                 sorted(g for g in os.listdir(D) if g.startswith("rccl."))}
        delta = {f: (after.get(f) or 0) - (before.get(f) or 0) for f in after}
        rows.append({"max_tokens": mt, "completion_tokens": got, "delta": delta})
        print(f"max_tokens={mt} got={got} allreduce lines={delta}", flush=True)

    a, b = rows
    per_step = {}
    dtok = b["completion_tokens"] - a["completion_tokens"]
    for f in b["delta"]:
        if dtok:
            per_step[f] = (b["delta"][f] - a["delta"].get(f, 0)) / dtok
    rec = {"kind": "collective_count", "ts": round(time.time(), 1),
           "model": MODEL, "machine": os.environ.get("BENCH_MACHINE", "RX 7900 XT"),
           "requests": rows, "delta_tokens": dtok,
           "allreduce_per_decode_step": per_step,
           "logs_seen": logs}
    with open(OUT, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
