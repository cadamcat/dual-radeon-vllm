#!/usr/bin/env python3
"""The CLR load-check demonstration, recomputed from logs/clr-demo.jsonl.

Two platform states x two HIP runtimes (the SDK's, and the same commit plus
clr-hostcall-load-check.patch) x two things (the 57-line probe, and the twelve
collective cases under stock RCCL 2.30.4). analyze.py keys by (state, runtime,
what) and reports the latest row; every row is kept. Exit is non-zero unless
all eight cells are present.

    python3 analyze.py [logs/clr-demo.jsonl]
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
STATES = ("atomics_present", "atomics_absent")
RUNTIMES = ("stock", "patched")
WHATS = ("probe", "collective")


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main(path=None):
    rows = load(path or os.path.join(HERE, "logs", "clr-demo.jsonl"))
    cells = {}
    for r in rows:
        cells[(r["row"], r["runtime"], r["what"])] = r
    print("=== 2 platform states x 2 runtimes x {probe, 12 collectives}: outcome / named-error lines / load-time messages ===")
    print(f"{'':20s}{'stock runtime':>36s}{'patched runtime':>40s}")
    for st in STATES:
        line = f"{st:20s}"
        for rt in RUNTIMES:
            for what in WHATS:
                r = cells.get((st, rt, what))
                if r is None:
                    line += f"{'MISSING':>19s}"; continue
                if what == "probe":
                    out = "ok" if r["rc"] == 0 and not r["error"] else (r["error"] or f"rc={r['rc']}")[:14]
                else:
                    out = f"{r['correctness_passed']}/12" if r["correctness_passed"] is not None else (r["error"] or f"rc={r['rc']}")[:14]
                line += f"{out + ' n' + str(r['named_error_lines']) + ' l' + str(r['load_messages']):>19s}"
        print(line)
    print("  n = lines naming hipErrorHostcallUnsupported, l = load-time messages naming the kernel and the device")
    for st in STATES:
        for rt in RUNTIMES:
            for what in WHATS:
                r = cells.get((st, rt, what))
                if r and (r["error"] or r["named_error_lines"] or r["load_messages"]):
                    print(f"  {st}/{rt}/{what}: rc={r['rc']} error={r['error']!r} named={r['named_error_lines']} load_messages={r['load_messages']}")
    md5s = {r.get("patched_libamdhip64_md5") for r in rows}
    print(f"patched libamdhip64 md5 across rows: {md5s}")
    caps = {st: {(r['root_ports_with_completer_support'], r['dmesg_no_atomics_lines']) for r in rows if r['row'] == st} for st in STATES}
    print(f"platform state per row (root ports with completer support, amdgpu complaints): {caps}")
    ok = len(cells) == 8
    print(f"\n{'COMPLETE' if ok else 'INCOMPLETE'}: {len(cells)}/8 distinct cells, {len(rows)} rows")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
