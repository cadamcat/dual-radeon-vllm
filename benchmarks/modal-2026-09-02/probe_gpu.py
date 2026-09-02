#!/usr/bin/env python3
"""What a Modal GPU actually is, in the fields this repository records.

Every machine in `prefill.jsonl` and `decode.jsonl` carries a description —
what the card is, how much memory, what its clocks and power limit are — and
until 2026-09-02 each of those came from a machine someone had to obtain and
keep. Modal rents them by the second, so a new machine's description now costs
about a cent and four seconds, and there is no reason for it to be typed by
hand from a terminal.

    python3 probe_gpu.py B300 H100 RTX-PRO-6000 ...

Writes one `kind: modal_machine` row per GPU to `machines.jsonl`. The fields are
`nvidia-smi --query-gpu` names, unchanged, so they can be compared with the
CUDA runners' own reads without a translation step.

Impossible to leave running, which matters when the budget is $30 a month:

  * `timeout=` on the Sandbox, so Modal kills it even if this process dies
  * a SIGALRM here, so a request that never gets capacity does not hang
  * `terminate()` in a `finally`, and the wall time is recorded on the row so
    the bill can be checked against `modal billing rates`

`scheduled_s` is worth reading too: it is how long Modal took to give the card
up, which is the difference between renting one and owning one.
"""
import json
import os
import signal
import sys
import time

import modal

OUT = os.environ.get("PROBE_OUT",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "machines.jsonl"))
BUDGET_S = int(os.environ.get("PROBE_BUDGET_S", "240"))   # client-side cap
SB_TIMEOUT_S = 300                                        # Modal-side cap

#: one nvidia-smi call, the field names it uses, so nothing is renamed on the
#: way in. `power.max_limit` is the ceiling the card will not exceed;
#: `power.limit` is what it is set to now, and on a rented card they can differ.
QUERY = ("name,driver_version,memory.total,power.limit,power.max_limit,"
         "clocks.max.sm,clocks.max.memory,compute_cap,"
         "pcie.link.gen.max,pcie.link.width.max")
FIELDS = QUERY.split(",")


class Deadline(Exception):
    pass


def _alarm(signum, frame):
    raise Deadline(f"no sandbox within {BUDGET_S}s")


def probe(gpu):
    t0 = time.perf_counter()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(BUDGET_S)
    sb = None
    row = {"kind": "modal_machine", "gpu_arg": gpu, "ts": round(time.time(), 1)}
    app = modal.App.lookup("gpu-probe", create_if_missing=True)
    try:
        sb = modal.Sandbox.create(app=app, image=modal.Image.debian_slim(),
                                  gpu=gpu, timeout=SB_TIMEOUT_S)
        row["scheduled_s"] = round(time.perf_counter() - t0, 2)
        row["sandbox_id"] = sb.object_id
        signal.alarm(0)
        p = sb.exec("bash", "-lc",
                    f"nvidia-smi --query-gpu={QUERY} --format=csv,noheader",
                    timeout=60)
        vals = [v.strip() for v in p.stdout.read().strip().split(",")]
        row.update(dict(zip(FIELDS, vals)))
        p = sb.exec("bash", "-lc",
                    "nvidia-smi | sed -n '3p' | grep -o 'CUDA Version: [0-9.]*'",
                    timeout=60)
        row["cuda_version"] = p.stdout.read().strip().replace("CUDA Version: ", "")
    except Deadline as e:
        row["error"] = f"deadline: {e}"
    except Exception as e:                                   # noqa: BLE001
        row["error"] = f"{type(e).__name__}: {e}"
    finally:
        signal.alarm(0)
        if sb is not None:
            sb.terminate()
        row["wall_s"] = round(time.perf_counter() - t0, 2)
    return row


def main():
    gpus = sys.argv[1:] or ["B300", "B200", "H100", "RTX-PRO-6000"]
    for g in gpus:
        r = probe(g)
        with open(OUT, "a") as fh:
            fh.write(json.dumps(r) + "\n")
        if "error" in r:
            print(f"{g:<14} FAILED  {r['error'][:80]}")
        else:
            print(f"{g:<14} {r['name']:<44} {r['memory.total']:>12}  "
                  f"{r['power.max_limit']:>10}  cc {r['compute_cap']}  "
                  f"sched {r['scheduled_s']}s  wall {r['wall_s']}s")


if __name__ == "__main__":
    main()
