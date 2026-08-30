"""Startup is not service. Fire real requests at the engine #39018 just brought up.

#38918 reports TRITON_ATTN as "server starts but crashes on first request", so a
clean startup proves nothing on its own. Two rungs, the same two the pre-flight
used: 500 to show it serves, 32000 to show the deepest rung this ladder asks for.
"""
import json, time, urllib.request

MODEL = "/content/models/gemma-4-12B-it-qat-w4a16-ct"
URL = "http://127.0.0.1:8000/v1/chat/completions"
OUT = "/content/work/headsize.jsonl"


def emit(o):
    o["ts"] = round(time.time(), 1)
    with open(OUT, "a") as f:
        f.write(json.dumps(o) + "\n")
    print("EMIT", json.dumps(o)[:500], flush=True)


def chat(prompt, max_tok, timeout):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tok, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time(); tfirst = None; usage = None; text = ""
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            d = line[6:]
            if d == "[DONE]":
                break
            j = json.loads(d)
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices", []):
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece and tfirst is None:
                    tfirst = time.time() - t0
                text += piece
    return {"ttft": round(tfirst or -1, 4), "total_s": round(time.time() - t0, 3),
            "usage": usage, "sample": text[:120]}


for target, words in ((500, 360), (32000, 23000)):
    prompt = ("Summarise the following. " + "hello world " * words)[:200000]
    try:
        r = chat(prompt, 32, 900)
        emit({"kind": "infer39018", "target": target, "ok": True, **r})
    except Exception as e:
        emit({"kind": "infer39018", "target": target, "ok": False, "err": repr(e)[:400]})
print("INFER_DONE", flush=True)
