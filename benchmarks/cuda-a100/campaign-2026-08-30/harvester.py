#!/usr/bin/env python3
"""Pull a Colab session's results to local disk on a loop, with a timeout.

harvester.sh had no bound on its `colab exec`, and twice today one hung: the
loop stopped writing and the only sign was a log that had gone quiet. The VM
was reclaimed during one of those silences. A hung pull now dies after
HARD_S and the loop carries on.

Never point this at a file holding rows the VM does not have -- it overwrites
wholesale, and it will happily replace a full harvest with a fresh VM's empty
one. The guard below refuses to shrink the file by more than SHRINK_OK rows
unless the target does not exist yet.

    python3 harvester.py <harvest-dir> [session]
"""
import os, subprocess, sys, time

H = sys.argv[1]
S = sys.argv[2] if len(sys.argv) > 2 else "a100"
OUT = os.path.join(H, f"{S}-results.jsonl")   # named for the session, not the A100
PERIOD, HARD_S, SHRINK_OK = 90, 120, 5

SNIP = ("import os\n"
        "p='/content/work/results.jsonl'\n"
        "print(open(p).read() if os.path.exists(p) else '', end='')\n")


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)


while True:
    t0 = time.time()
    try:
        r = subprocess.run(["colab", "--auth=adc", "exec", "-s", S],
                           input=SNIP, capture_output=True, text=True,
                           timeout=HARD_S,
                           env={**os.environ, "COLAB_CLI_HIGH_MEM": "1"})
        out = r.stdout
    except subprocess.TimeoutExpired:
        log(f"pull timed out after {HARD_S}s, retrying next tick")
        out = ""
    except Exception as e:
        log(f"pull failed: {e!r}")
        out = ""

    n = out.count('"kind"')
    if n:
        have = 0
        if os.path.exists(OUT):
            have = sum(1 for l in open(OUT) if '"kind"' in l)
        if have and n < have - SHRINK_OK:
            log(f"REFUSING to shrink {have} -> {n} rows; not overwriting")
        else:
            tmp = OUT + ".tmp"
            with open(tmp, "w") as f:
                f.write(out)
            os.replace(tmp, OUT)
            log(f"harvested {n} rows ({time.time()-t0:.0f}s)")
    time.sleep(max(0, PERIOD - (time.time() - t0)))
