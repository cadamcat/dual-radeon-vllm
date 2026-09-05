#!/usr/bin/env python3
"""The end-to-end half: wait for a vLLM server the caller started, or for it
to die, then send one request and record what the engine actually chose.

    one_request.py --row R --backend {default,triton} --serve-log L --out X.jsonl

Runs INSIDE the container. An arm that asks for a backend is not an arm that
got it (gfx1100-greedy-attn-ab, 2026-09-02): the serve log is grepped for the
backend line, the row says whether it matches the request, and whether vLLM
logged a Triton fallback. Health and liveness are read again AFTER the
request, so a server that dies during generation is recorded as such. `ok`
is true only when the server answered with tokens on the backend requested.
"""
import argparse, glob, json, re, socket, time
import requests

HEALTH, URL = "http://127.0.0.1:8000/health", "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "/models/Qwen3-8B"
EXPECT = {"default": "ROCM_ATTN", "triton": "TRITON_ATTN"}
ERR = re.compile(r"(Pcie atomics not enabled, hostcall not supported|AQL dispatch failed!?|"
                 r"hipErrorIllegalState|the operation cannot be performed in the present state|"
                 r"hipError[A-Za-z]+|RuntimeError: [^\n]{0,160}|EngineCore[^\n]{0,40}(?:died|failed)[^\n]{0,120})")
BACKEND = re.compile(r"(Using (\w+) backend[^\n]{0,60}|Overriding with (\w+) out of potential backends)")
FALLBACK = "falling back to Triton"


def alive():
    """The container may not ship procps; read /proc directly."""
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
    ap.add_argument("--backend", required=True, choices=("default", "triton"))
    ap.add_argument("--serve-log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--caps", type=int, default=None)
    ap.add_argument("--dmesg", type=int, default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--image", default=None)
    a = ap.parse_args()
    t0 = time.time()
    row = {"kind": "serve_cell", "row": a.row, "backend_requested": a.backend,
           "backend_expected": EXPECT[a.backend], "ts": t0, "model": MODEL, "tensor_parallel": 1,
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
    chosen = (m.group(2) or m.group(3)) if m else None
    row.update(backend_line=m.group(0)[:120] if m else None, backend_chosen=chosen,
               backend_matches_request=(chosen == EXPECT[a.backend]),
               fallback_warning_seen=FALLBACK in log, log_bytes=len(log))
    e = ERR.search(log)
    row["error"] = e.group(0)[:200] if e else None
    row["ok"] = bool(up and answered and row.get("text_mentions_pcie") and row["backend_matches_request"]
                     and not row["fallback_warning_seen"] and row["alive_after_request"] and row["healthy_after_request"])
    with open(a.out, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"row={a.row} backend_requested={a.backend} chosen={chosen} matches={row['backend_matches_request']} "
          f"fallback={row['fallback_warning_seen']} healthy={up} died_before_health={died} answered={answered} "
          f"alive_after={row['alive_after_request']} load_s={row['load_s']} tokens={row.get('completion_tokens')} "
          f"error={row['error']} ok={row['ok']}", flush=True)
    print("ONE_REQUEST_DONE", flush=True)


if __name__ == "__main__":
    main()
