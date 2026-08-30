"""T4 pre-flight: does compressed-tensors W4A16 load on sm75, and on what kernels?

One engine start. Turing has no bf16, so --dtype float16 is explicit; gemma-4's
config asks for bfloat16 and the engine would otherwise cast or refuse. Which
quant kernel and which attention backend it actually lands on is read out of the
serve log rather than assumed -- the campaign's rule, since a patch or a claim
about a backend is only worth anything when the kernel is really on the path.

Two rungs, not one: 500 proves it serves, 32000 proves the card holds the
deepest rung this ladder asks for. Both are the cheapest possible evidence for
the two independent ways this can fail.
"""
import json, os, re, subprocess, sys, time, urllib.request

D = "/content/work"
PORT = 8000
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
MML = int(os.environ.get("T4_MML", "33000"))
UTIL = os.environ.get("T4_UTIL", "0.90")
# The harness issues one request at a time, but vLLM sizes its activation
# buffers and captures CUDA graphs for max_num_seqs -- up to batch 512 by
# default, which on a 15 GiB card cost 4.57 GiB and left 0.65 GiB for KV.
MNS = os.environ.get("T4_MNS", "1")
# TRITON_ATTN passes the selector's validate_configuration() on sm75 and then
# fails at launch: the kernel asks for 98304 bytes of shared memory against
# Turing's 65536 per SM. So the backend has to be chosen explicitly here.
BACKEND = os.environ.get("T4_BACKEND", "")
MODEL = "/content/models/gemma-4-12B-it-qat-w4a16-ct"
LOG = f"{D}/serve-T4-G12{os.environ.get('T4_TAG', '')}.log"
OUT = f"{D}/preflight.jsonl"
GEMMA_MM = '--limit-mm-per-prompt \'{"image":0,"video":0,"audio":0}\''


def emit(o):
    o["ts"] = round(time.time(), 1)
    with open(OUT, "a") as f:
        f.write(json.dumps(o) + "\n")
    print("EMIT", json.dumps(o)[:400], flush=True)


def free_mib():
    r = subprocess.run("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits",
                       shell=True, capture_output=True, text=True).stdout.strip()
    return int(r.splitlines()[0]) if r else 0


def start():
    # Both patterns, then wait on the card itself: killing the API server does
    # not free it, because the workers run as VLLM::EngineCore.
    for pat in ("[v]llm serve", "[V]LLM::EngineCore", "vllm[.]model_executor"):
        subprocess.run(f"pkill -9 -f '{pat}' 2>/dev/null", shell=True)
    for _ in range(30):
        if free_mib() > 14000:
            break
        time.sleep(2)
    if os.path.exists(LOG):
        os.remove(LOG)                      # a stale log reads as an instant crash
    sc = f"{D}/serve-t4.sh"
    flags = (f"--dtype float16 --max-model-len {MML} --port {PORT} "
             f"--gpu-memory-utilization {UTIL} --max-num-seqs {MNS} {GEMMA_MM}"
             + (f" --attention-backend {BACKEND}" if BACKEND else ""))
    with open(sc, "w") as fh:
        fh.write(f"#!/bin/bash\nset -u\nexec vllm serve {MODEL} {flags} > {LOG} 2>&1\n")
    os.chmod(sc, 0o755)
    subprocess.Popen(["bash", sc])
    t0, hard, stall = time.time(), 2400, 600
    last = 0
    while time.time() - t0 < hard:
        txt = open(LOG).read() if os.path.exists(LOG) else ""
        if "Application startup complete" in txt:
            return "ready", txt
        # A logged traceback is not a crash: it carries its logger's
        # "<file>.py:<line>]" ahead of it. A real one heads its own line.
        real_tb = [l for l in txt.splitlines()
                   if "Traceback (most recent call last)" in l
                   and not re.search(r"\.py:\d+\]", l.split("Traceback")[0])]
        if real_tb or "EngineCore failed to start" in txt \
                or "Engine core initialization failed" in txt:
            return "crash", txt
        idle = time.time() - os.path.getmtime(LOG) if os.path.exists(LOG) else time.time() - t0
        if idle > stall:
            return "timeout", txt
        el = time.time() - t0
        if el - last > 120:
            last = el
            print(f"  ... starting ({el/60:.1f} min, free={free_mib()} MiB)", flush=True)
        time.sleep(5)
    return "timeout", open(LOG).read() if os.path.exists(LOG) else ""


def scrape(txt):
    pats = {
        "wna16_kernel":  r"Using (\w+) for CompressedTensorsWNA16",
        "attn_backend":  r"Using (\w+) backend",
        "kv_gib":        r"Available KV cache memory: ([0-9.]+) GiB",
        "kv_tokens":     r"GPU KV cache size: ([\d,]+) tokens",
        "model_load_s":  r"Model loading took [0-9.]+ GiB(?: memory)? and ([0-9.]+) seconds",
        "weights_gib":   r"Model loading took ([0-9.]+) GiB",
        "dtype":         r"dtype=torch\.(\w+)",
        "max_conc":      r"Maximum concurrency for ([\d,]+) tokens",
        "est_max_len":   r"estimated maximum model length is (\d+)",
        "kv_needed_gib": r"\(([0-9.]+) GiB KV cache is needed",
        "shmem_required": r"Required: (\d+), Hardware limit",
        "invalid_reasons": r"Selected backend \S+ is not valid[^\n]*",
    }
    m = {}
    for k, p in pats.items():
        mm = re.search(p, txt)
        if mm:
            m[k] = mm.group(1).replace(",", "")
    return m


def chat(prompt, max_tok, timeout):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tok, "temperature": 0.8, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time(); tfirst = None; tlast = None; usage = None
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
            ch = j.get("choices") or []
            if ch and ch[0].get("delta", {}).get("content"):
                if tfirst is None:
                    tfirst = time.time()
                tlast = time.time()
    out = {"prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"],
           "ttft": round(tfirst - t0, 4),
           "prefill_tps": round(usage["prompt_tokens"] / (tfirst - t0), 1)}
    if tlast and tlast > tfirst and usage["completion_tokens"] > 1:
        out["gen_time"] = round(tlast - tfirst, 4)
        out["decode_tps"] = round((usage["completion_tokens"] - 1) / (tlast - tfirst), 2)
    return out


if __name__ == "__main__":
    open(OUT, "a").close()
    print("=== GPU ===", flush=True)
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout, flush=True)
    print("=== starting engine (dtype float16) ===", flush=True)
    st, txt = start()
    meta = scrape(txt)
    meta["status"] = st
    emit({"kind": "preflight_meta", "cfg": "T4-G12", "util": UTIL, "mns": MNS, "mml": MML, "backend_flag": BACKEND or None, **meta})
    print(f"=== STATUS: {st} ===", flush=True)
    for k, v in meta.items():
        print(f"    {k:16s} {v}", flush=True)

    if st != "ready":
        print("=== TAIL OF SERVE LOG ===", flush=True)
        print(txt[-6000:], flush=True)
        print("T4_PREFLIGHT_FAILED", flush=True)
        sys.exit(0)

    for target in (500, 32000):
        p = open(f"{D}/prompts/prompt_{target}.txt", encoding="utf-8").read()
        for kind, gen in (("prefill", 1), ("decode", 512)):
            try:
                m = chat(p, gen, 900)
                rec = {"kind": kind, "cfg": "T4-G12", "target": target, "round": 1, **m}
                emit(rec)
            except Exception as e:
                emit({"kind": "error", "cfg": "T4-G12", "target": target,
                      "step": kind, "err": str(e)[:300]})
    print("=== free after ===", free_mib(), "MiB", flush=True)
    print("T4_PREFLIGHT_DONE", flush=True)
