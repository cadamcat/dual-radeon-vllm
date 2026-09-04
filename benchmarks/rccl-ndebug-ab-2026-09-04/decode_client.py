#!/usr/bin/env python3
"""B1's end-to-end half: decode rate under each RCCL, on a live vLLM server.

Runs INSIDE the container, against a server the caller already started. It
does three things the caller cannot:

  1. waits for /health and refuses to measure a server that never came up;
  2. reads the SERVING process's /proc/<pid>/maps and md5s the librccl it
     actually mapped, so a row is attributed to an arm by what was loaded
     rather than by what the orchestrator believes it installed;
  3. runs the depth x repeat grid and writes one JSON object per request.

The request shape is `benchmarks/campaign-2026-09-03/runner.py`'s `chat()`,
kept identical so these rows sit beside that campaign's: streaming, a random
seed prefix so nothing is served from cache, temperature 0.8, and
decode_tps = (completion_tokens - 1) / (last_token - first_token).
"""
import argparse, glob, hashlib, json, os, random, string, subprocess, sys, time
import requests

PORT = 8000
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
HEALTH = f"http://127.0.0.1:{PORT}/health"


def rid():
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=8))


def serving_library():
    """md5 the librccl the vLLM process has actually mapped."""
    out = {"pids": []}
    for maps in glob.glob("/proc/[0-9]*/maps"):
        pid = maps.split("/")[2]
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode("utf-8", "replace")
            if "vllm" not in cmd and "VLLM" not in cmd:
                continue
            paths = {ln.split()[-1] for ln in open(maps)
                     if "rccl" in ln.lower() or "nccl" in ln.lower()}
        except (OSError, IndexError):
            continue
        for p in sorted(paths - {"(deleted)"}):
            real = os.path.realpath(p)
            if not os.path.exists(real):
                continue
            md5 = hashlib.md5(open(real, "rb").read()).hexdigest()
            out["pids"].append({"pid": pid, "mapped": real, "md5": md5})
    return out


def chat(model, prompt, max_tokens, timeout):
    body = {"model": model,
            "messages": [{"role": "user", "content": f"[seed-{rid()}] {prompt}"}],
            "max_tokens": max_tokens, "temperature": 0.8, "stream": True,
            "stream_options": {"include_usage": True}}
    t0 = time.perf_counter(); tfirst = tlast = None; usage = None
    with requests.post(URL, json=body, stream=True, timeout=(15, timeout)) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            j = json.loads(data)
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning_content"):
                    now = time.perf_counter()
                    tfirst = tfirst or now
                    tlast = now
    if usage is None or tfirst is None:
        raise RuntimeError("stream ended without usage or without a token")
    out = {"prompt_tokens": usage["prompt_tokens"],
           "completion_tokens": usage["completion_tokens"],
           "ttft": round(tfirst - t0, 4),
           "prefill_tps": round(usage["prompt_tokens"] / (tfirst - t0), 1)}
    if usage["completion_tokens"] >= 2 and tlast > tfirst:
        out["gen_time"] = round(tlast - tfirst, 4)
        out["decode_tps"] = round((usage["completion_tokens"] - 1) / (tlast - tfirst), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--expect-md5", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--depths", default="500,8000,32000")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--health-timeout", type=int, default=1800)
    a = ap.parse_args()

    t0 = time.time()
    while time.time() - t0 < a.health_timeout:
        try:
            if requests.get(HEALTH, timeout=5).status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(5)
    else:
        print(f"FATAL: /health never came up in {a.health_timeout}s", file=sys.stderr)
        sys.exit(3)
    load_s = round(time.time() - t0, 1)
    print(f"server healthy after {load_s}s")

    lib = serving_library()
    got = {p["md5"] for p in lib["pids"]}
    ok = a.expect_md5 in got
    print(f"serving process mapped: {sorted(got)}  expect {a.expect_md5}  -> "
          f"{'OK' if ok else 'MISMATCH'}")
    with open(a.out, "a") as fh:
        fh.write(json.dumps({"kind": "decode_meta", "ts": time.time(),
                             "arm": a.arm, "label": a.label, "model": a.model,
                             "expect_md5": a.expect_md5,
                             "serving_library": lib,
                             "library_matches": ok,
                             "server_load_s": load_s,
                             "max_tokens": a.max_tokens,
                             "repeats": a.repeats}) + "\n")
    if not ok:
        print("FATAL: the serving process did not map the arm's library",
              file=sys.stderr)
        sys.exit(4)

    depths = [int(d) for d in a.depths.split(",")]
    for rep in range(1, a.repeats + 1):
        for depth in depths:
            path = os.path.join(a.prompts, f"prompt_{depth}.txt")
            prompt = open(path, encoding="utf-8").read()
            rec = {"kind": "decode", "ts": time.time(), "arm": a.arm,
                   "label": a.label, "model": a.model, "depth": depth,
                   "repeat": rep, "prompt_file": path}
            try:
                rec.update(chat(a.model, prompt, a.max_tokens, 900))
            except Exception as e:                            # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {e}"
            with open(a.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"  arm={a.arm} depth={depth:>6} rep={rep} "
                  f"decode_tps={rec.get('decode_tps')} "
                  f"ptok={rec.get('prompt_tokens')} "
                  f"ctok={rec.get('completion_tokens')}"
                  f"{' ERROR ' + rec['error'] if 'error' in rec else ''}",
                  flush=True)


if __name__ == "__main__":
    main()
