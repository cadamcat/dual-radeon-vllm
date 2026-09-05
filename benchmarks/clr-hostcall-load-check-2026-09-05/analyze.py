#!/usr/bin/env python3
"""The CLR load-check demonstration, recomputed from logs/clr-demo*.jsonl.

Two platform states x two HIP runtimes (the SDK's, and the same commit plus
clr-hostcall-load-check.patch) x two things (the 57-line probe, and the twelve
collective cases under stock RCCL 2.30.4), once per SDK: ROCm 7.14 / vLLM 0.23
(runtime commit 2b22ab01, logs/clr-demo.jsonl) and ROCm 10.0 / vLLM 0.27
(runtime commit 6b0e43f3, logs/clr-demo-rocm10.jsonl), and once more at 10.0 with
PR A alone — the load-time check returning the existing hipErrorNotSupported,
clr-hostcall-load-check-a.patch (logs/clr-demo-rocm10a.jsonl). analyze.py keys by
(state, runtime, what) within each file and reports the latest row; every row
is kept. Exit is non-zero unless all eight cells are present in every file.

    python3 analyze.py [logs/clr-demo.jsonl [logs/clr-demo-rocm10.jsonl ...]]
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
STATES = ("atomics_present", "atomics_absent")
RUNTIMES = ("stock", "patched")
WHATS = ("probe", "collective")
DEFAULT = ("clr-demo.jsonl", "clr-demo-rocm10.jsonl", "clr-demo-rocm10a.jsonl", "clr-demo-rocm10c.jsonl")


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def table(path):
    rows = load(path)
    cells = {}
    for r in rows:
        cells[(r["row"], r["runtime"], r["what"])] = r
    sdk = {r.get("sdk", "rocm714") for r in rows}
    commit = {r.get("runtime_commit", "2b22ab01") for r in rows}
    image = {r.get("image", "rocm/vllm:rocm7.14.0_rdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0") for r in rows}
    rccl = {r.get("rccl_version_string", "RCCL version 2.30.4 (the 7.14 image's stock library, /rb/b1/librccl-stock2304.so)") for r in rows}
    print(f"=== {os.path.basename(path)}: sdk={'/'.join(sorted(sdk))} runtime commit={'/'.join(sorted(commit))}")
    print(f"    image={'/'.join(sorted(image))}")
    print(f"    stock RCCL: {'/'.join(sorted(rccl))}")
    print("    2 platform states x 2 runtimes x {probe, 12 collectives}: outcome / named-error lines / load-time messages")
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
                line += f"{out + ' n' + str(r['named_error_lines']) + ' r' + str(r.get('refusals', '-')) + ' l' + str(r['load_messages']):>19s}"
        print(line)
    print("  n = lines naming hipErrorHostcallUnsupported, r = launch refusals (occurrences; '-' where the row did not count them), l = load-time messages naming the kernel and the device")
    for st in STATES:
        for rt in RUNTIMES:
            for what in WHATS:
                r = cells.get((st, rt, what))
                if r and (r["error"] or r["named_error_lines"] or r["load_messages"]):
                    print(f"  {st}/{rt}/{what}: rc={r['rc']} error={r['error']!r} named={r['named_error_lines']} load_messages={r['load_messages']}")
    md5s = {r.get("patched_libamdhip64_md5") for r in rows}
    print(f"  patched libamdhip64 md5 across rows: {md5s}")
    caps = {st: {(r['root_ports_with_completer_support'], r['dmesg_no_atomics_lines']) for r in rows if r['row'] == st} for st in STATES}
    print(f"  platform state per row (root ports with completer support, amdgpu complaints): {caps}")
    ok = len(cells) == 8
    print(f"  {'COMPLETE' if ok else 'INCOMPLETE'}: {len(cells)}/8 distinct cells, {len(rows)} rows\n")
    return ok


def main(*paths):
    paths = list(paths) or [os.path.join(HERE, "logs", n) for n in DEFAULT if os.path.exists(os.path.join(HERE, "logs", n))]
    oks = [table(p) for p in paths]
    print(f"{len(paths)} file(s); {'COMPLETE' if all(oks) and paths else 'INCOMPLETE'}")
    return 0 if all(oks) and paths else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
