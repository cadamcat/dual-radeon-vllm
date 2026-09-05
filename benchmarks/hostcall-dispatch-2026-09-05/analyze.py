#!/usr/bin/env python3
"""The two tables of the paged-attention dispatch experiment, recomputed from
logs/pa-cells.jsonl and logs/serve-cells.jsonl (or the files named on the
command line). Prints every field a reader would want to check and exits
non-zero if either table is incomplete, so a half-run cannot look finished."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = ("atomics_present", "atomics_absent")
ARMS = ("as_shipped", "ck_forced", "triton_forced")
BACKENDS = ("default", "triton")


def load(path, kind):
    if not os.path.exists(path):
        return []
    return [r for r in (json.loads(l) for l in open(path) if l.strip()) if r.get("kind") == kind]


def main(pa_path=None, sv_path=None, only_row=None):
    """only_row: report completeness for that row alone (the runbook's row-1 check)."""
    global ROWS
    if only_row:
        ROWS = (only_row,)
    pa = [r for r in load(pa_path or os.path.join(HERE, "logs", "pa-cells.jsonl"), "pa_cell") if r["row"] in ROWS]
    sv = [r for r in load(sv_path or os.path.join(HERE, "logs", "serve-cells.jsonl"), "serve_cell") if r["row"] in ROWS]
    cells = {(r["row"], r["arm"]): r for r in pa}
    serve = {(r["row"], r["backend_requested"]): r for r in sv}
    dup_pa = len(pa) - len(cells); dup_sv = len(sv) - len(serve)
    ok = True

    print("=== probe: 2 platform states x 3 arms (outcome / ck_op_calls / fallback / host_native_atomic) ===")
    print(f"{'':18s}" + "".join(f"{a:>28s}" for a in ARMS))
    for row in ROWS:
        line = f"{row:18s}"
        for arm in ARMS:
            r = cells.get((row, arm))
            if r is None:
                line += f"{'MISSING':>28s}"; ok = False; continue
            line += f"{r['outcome']:>16s} {str(r.get('ck_op_calls')):>3s} {'FB' if r.get('fallback_warning_seen') else '--':>3s} {str(r.get('host_native_atomic_supported')):>4s}"
        print(line)
    for row in ROWS:
        rs = [cells[(row, a)] for a in ARMS if (row, a) in cells]
        if rs:
            caps = {r["root_ports_with_completer_support"] for r in rs}; dm = {r["dmesg_no_atomics_lines"] for r in rs}
            attr = {json.dumps(r.get("host_native_atomic_supported")) for r in rs}
            print(f"  {row}: root ports with completer support {caps}, amdgpu complaints {dm}, "
                  f"hipDeviceAttributeHostNativeAtomicSupported {attr}, gate_as_shipped {[r.get('gate_as_shipped') for r in rs]}")
            for r in rs:
                if r["outcome"] in ("refused", "dispatch_error", "not_launched"):
                    print(f"    {r['arm']}: {r['outcome']} hip_last_error={r.get('hip_last_error')}:{r.get('hip_last_error_text')} exc={(r.get('exception') or '')[:100]}")
                elif r["outcome"].startswith("dispatched"):
                    print(f"    {r['arm']}: {r['outcome']} max_rel_err={r.get('max_rel_err')}")

    print("\n=== serve: Qwen3-8B TP=1, 2 platform states x 2 backends ===")
    print(f"{'':18s}{'default (engine picks)':>36s}{'--attention-backend TRITON_ATTN':>36s}")
    for row in ROWS:
        line = f"{row:18s}"
        for be in BACKENDS:
            r = serve.get((row, be))
            if r is None:
                line += f"{'MISSING':>36s}"; ok = False; continue
            state = ("answered" if r.get("answered") else
                     "died before health" if r.get("server_died_before_health") else
                     "died in request" if r.get("healthy") and not r.get("alive_after_request") else
                     "healthy, no answer" if r.get("healthy") else "never healthy")
            line += f"{r.get('backend_chosen')!s:>12s} {state:>18s} {'FB' if r.get('fallback_warning_seen') else '--':>3s}"
        print(line)
    for (row, be), r in sorted(serve.items()):
        print(f"  {row}/{be}: chosen={r.get('backend_chosen')} matches={r.get('backend_matches_request')} ok={r.get('ok')} "
              f"load_s={r.get('load_s')} tokens={r.get('completion_tokens')} error={r.get('error')} text={(r.get('text_head') or '')[:60]!r}")

    prov = {(r.get("image"), r.get("vllm"), r.get("rocm_C_md5"), r.get("guest_host")) for r in pa}
    print(f"\nprovenance sets across probe rows: {len(prov)} -> {prov}")
    if dup_pa or dup_sv:
        print(f"WARNING: duplicate cells: probe {dup_pa}, serve {dup_sv} (later rows win)")
    want_pa, want_sv = 3 * len(ROWS), 2 * len(ROWS)
    ok = ok and len(cells) == want_pa and len(serve) == want_sv
    print(f"\n{'COMPLETE' if ok else 'INCOMPLETE'}: {len(cells)}/{want_pa} distinct probe cells, {len(serve)}/{want_sv} distinct serve cells")
    return 0 if ok else 1


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--row=")]
    only = next((x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith("--row=")), None)
    sys.exit(main(*args[:2], only_row=only))
