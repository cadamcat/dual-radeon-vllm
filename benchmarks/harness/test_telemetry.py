#!/usr/bin/env python3
"""Gates for the telemetry sampler's phase split, run on a laptop with no GPU.

v3 exists because one summary used to serve both rows of a request. On the
H100 rows of 2026-09-03 the kept window at the 32 000 rung was up to 93 %
prefill, so a decode row's mem_busy_pct_max could be a prefill reading and
nothing on the row said so. These checks feed the sampler synthetic samples
with known times and assert that each phase sees only its own.

    python3 benchmarks/harness/test_telemetry.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from harness import telemetry as T   # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def card(mem, power=50.0):
    return {"mem_busy_pct": mem, "power_w": power, "sclk_mhz": 1500,
            "sclk_mhz_cap": 2000, "power_cap_w": 300.0, "temp_c": 60.0}


def sampler(mems, period):
    s = T.Sampler(period_s=period)
    s.cards = []
    s.rows = [[card(m)] for m in mems]
    s.times = [i * period for i in range(len(mems))]
    return s


def main():
    # 1. a prefill spike must not reach the decode row. Ten samples at 0.5 s;
    #    the first two (t = 0, 0.5) are prefill at 99 %, the rest decode at
    #    40-47 %; the first token lands at 0.9 s.
    s = sampler([99, 98, 40, 41, 42, 43, 44, 45, 46, 47], 0.5)
    ph = s.phases(0.9)
    check("decode phase excludes the prefill spike", ph["decode"]["mem_busy_pct_max"] == 46,
          f"got {ph['decode']['mem_busy_pct_max']}")
    check("prefill phase is the spike", ph["prefill"]["mem_busy_pct_max"] == 99,
          f"got {ph['prefill']['mem_busy_pct_max']}")
    check("prefill phase is untrimmed", ph["prefill"]["tele_samples"] == 2
          and ph["prefill"]["tele_samples_raw"] == 2)
    check("decode phase is trimmed like a cell", ph["decode"]["tele_samples"] == 5
          and ph["decode"]["tele_samples_raw"] == 8,
          f"{ph['decode']['tele_samples']} of {ph['decode']['tele_samples_raw']}")
    check("both phases name themselves and the split",
          ph["prefill"]["tele_phase"] == "prefill" and ph["decode"]["tele_phase"] == "decode"
          and ph["prefill"]["tele_split_s"] == ph["decode"]["tele_split_s"] == 0.9)
    # 2. the whole-request summary is what v2 wrote, and it is labelled as such
    whole = T.summarise(s.rows, period=0.5)
    check("a whole-request summary says so", whole["tele_phase"] == "request")
    check("and it carries the raw count", whole["tele_samples_raw"] == 10)
    # 3. a first token after the last sample leaves the decode phase empty, with
    #    every key still present -- a short row is a ragged schema, not a fact
    ph2 = s.phases(10.0)
    check("an empty phase keeps the schema", ph2["decode"]["tele_samples"] == 0
          and all(k in ph2["decode"] for k in T.SUMMARY_KEYS)
          and ph2["decode"]["tele_phase"] == "decode")
    check("and the prefill phase then holds everything", ph2["prefill"]["tele_samples"] == 10)
    # 4. fewer than six samples: nothing trimmed in either phase, as before
    s3 = sampler([90, 30, 31, 32], 1.5)
    ph3 = s3.phases(1.0)
    check("under six samples nothing is trimmed", ph3["decode"]["tele_samples"] == 3
          and ph3["decode"]["mem_busy_pct_max"] == 32)
    # 5. the version the module claims is the one SCHEMA.md documents
    sch = open(os.path.join(HERE, "SCHEMA.md"), encoding="utf-8").read()
    m = re.search(r"^# Campaign record schema — v(\d+)", sch, re.M)
    check("SCHEMA.md documents the module's version",
          m is not None and int(m.group(1)) == T.SCHEMA_VERSION,
          f"module {T.SCHEMA_VERSION}, doc {m.group(1) if m else 'none'}")
    for k in ("tele_phase", "tele_split_s", "tele_samples_raw"):
        check(f"SCHEMA.md names {k}", k in sch)
    # 6. both templates split at the first token
    for fn in ("runner_cuda.py", "runner_radeon.py"):
        src = open(os.path.join(HERE, fn), encoding="utf-8").read()
        check(f"{fn} uses phases()", ".phases(" in src)
    print(f"\n{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAILED: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
