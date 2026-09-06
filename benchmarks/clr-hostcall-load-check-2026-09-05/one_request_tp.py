#!/usr/bin/env python3
"""The end-to-end cell of the opt-in experiment: wait for a vLLM server the
caller started with --tensor-parallel-size N, or for it to die, send one
request, and record what happened -- including how many kernels the patched
runtime marked at load ("proceed with a null hostcall buffer"), how many
launches it refused, and whether the engine died of the old generic error.
Derived from hostcall-dispatch-2026-09-05/one_request.py; the backend is
recorded, not required. `ok` is: healthy, answered with tokens that mention
the subject, and alive and healthy afterwards.

    one_request_tp.py --row R --runtime {stock,patched} --tp N --serve-log L --out X.jsonl
"""
import argparse, glob, json, re, socket, time
import requests

HEALTH, URL = "http://127.0.0.1:8000/health", "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "/models/Qwen3-8B"
ERR = re.compile(r"(Memory access fault[^\n]{0,80}|Pcie atomics not enabled, hostcall not supported|AQL dispatch failed!?|"
                 r"hipErrorIllegalState|the operation cannot be performed in the present state|"
                 r"launch of \S+ refused[^\n]{0,80}|hipError[A-Za-z]+|RuntimeError: [^\n]{0,160}|EngineCore[^\n]{0,40}(?:died|failed)[^\n]{0,120})")
BACKEND = re.compile(r"(Using (\w+) backend[^\n]{0,60}|Overriding with (\w+) out of potential backends)")


def alive():
    for f in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            c = open(f, "rb").read()
            if b"vllm" in c and b"serve" in c:
                return True
        except OSError:
            continue
    return False


def healthy():
    try:
        return requests.get(HEALTH, timeout=3).status_code == 200
    except requests.RequestException:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True)
    ap.add_argument("--runtime", required=True, choices=("stock", "patched"))
    ap.add_argument("--tp", type=int, required=True)
    ap.add_argument("--serve-log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--caps", type=int, default=None)
    ap.add_argument("--dmesg", type=int, default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--image", default=None)
    ap.add_argument("--env", default="")
    ap.add_argument("--runtime-md5", default=None)
    a = ap.parse_args()
    t0 = time.time()
    row = {"kind": "serve_cell", "sdk": "rocm10c", "row": a.row, "runtime": a.runtime, "runtime_md5": a.runtime_md5,
           "env": a.env or None, "ts": t0, "model": MODEL, "tensor_parallel": a.tp,
           "root_ports_with_completer_support": a.caps, "dmesg_no_atomics_lines": a.dmesg,
           "guest_host": a.host, "image": a.image, "container_host": socket.gethostname()}
    up, died = False, False
    while time.time() - t0 < a.timeout:
        if healthy():
            up = True
            break
        if not alive():
            died = True
            break
        time.sleep(3)
    row.update(healthy=up, server_died_before_health=died, load_s=round(time.time() - t0, 1))
    answered = False
    if up:
        try:
            r = requests.post(URL, json={"model": MODEL,
                                         "messages": [{"role": "user", "content": "In one sentence, what is a PCIe root port?"}],
                                         "max_tokens": 64, "temperature": 0}, timeout=180)
            row["http_status"] = r.status_code
            try:
                j = r.json()
            except ValueError:
                j = {}
            choices = j.get("choices") or [{}]
            text = ((choices[0].get("message") or {}).get("content")) or ""
            toks = (j.get("usage") or {}).get("completion_tokens")
            row.update(completion_tokens=toks, text_head=text[:160],
                       text_mentions_pcie=("pci" in text.lower() or "root" in text.lower()),
                       request_error=None if r.status_code == 200 else json.dumps(j)[:300])
            answered = r.status_code == 200 and bool(toks) and toks > 0 and bool(text.strip())
        except Exception as e:                          # noqa: BLE001
            row["request_error"] = f"{type(e).__name__}: {e}"[:300]
    row["answered"] = answered
    row["alive_after_request"] = alive()
    row["healthy_after_request"] = healthy() if up else False
    try:
        log = open(a.serve_log, errors="replace").read()
    except OSError as e:
        log = ""
        row["serve_log_error"] = f"{type(e).__name__}: {e}"[:120]
    m = BACKEND.search(log)
    row.update(backend_line=m.group(0)[:120] if m else None, backend_chosen=((m.group(2) or m.group(3)) if m else None),
               null_buffer_lines=log.count("proceed with a null hostcall buffer"),
               refusal_lines=log.count("refused: it declares a hostcall buffer"),
               generic_error_lines=log.count("the operation cannot be performed in the present state"),
               memory_fault_lines=log.count("Memory access fault"),
               log_bytes=len(log))
    e = ERR.search(log)
    row["error"] = e.group(0)[:200] if e else None
    row["ok"] = bool(up and answered and row.get("text_mentions_pcie") and row["alive_after_request"] and row["healthy_after_request"])
    with open(a.out, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"row={a.row} runtime={a.runtime} tp={a.tp} healthy={up} died_before_health={died} answered={answered} "
          f"alive_after={row['alive_after_request']} load_s={row['load_s']} tokens={row.get('completion_tokens')} "
          f"null_buffer_lines={row['null_buffer_lines']} refusals={row['refusal_lines']} generic={row['generic_error_lines']} "
          f"faults={row['memory_fault_lines']} backend={row['backend_chosen']} error={row['error']} ok={row['ok']}", flush=True)
    print("ONE_REQUEST_DONE", flush=True)


if __name__ == "__main__":
    main()
