"""The A100 half of the campaign: nine configurations, eleven rungs, two rounds.

Same ladder as 2026-07-25 -- the eleven targets cut from Darwin's Origin of
Species, Gutenberg #1228, the source benchmarks/prompts/cut_prompts.py uses.
A rung is a token count, so the ladder is cut per tokenizer.

The measurement is the campaign one: an OpenAI-compatible server, streaming,
temperature 0.8, 512 generated tokens, two rounds per rung, decode rate from the
stream's own token timings. What differs from the Radeon side is only the
machine and the stack, both of which every row records.

Checkpointed: a rung already in results.jsonl is not measured again, so a killed
session resumes instead of restarting.
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

MACHINE = os.environ.get("BENCH_MACHINE", "unknown")   # goes on every row
D = "/content/work"
RES = f"{D}/results.jsonl"
PROG = f"{D}/PROGRESS.txt"
MODELS = "/content/models"
PORT = 8000
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
HEALTH = f"http://127.0.0.1:{PORT}/health"
TARGETS = [500, 1000, 2000, 4000, 6000, 8000, 12000, 16000, 20000, 24000, 32000]
GEN = 512
MML = 33000

CFGS = [
    # 2026-08-30. The spine's model, on the same stack and the same ladder as
    # A100-G12, so the two rows differ in the card and nothing else.
    dict(id="G12", model="gemma-4-12B-it-qat-w4a16-ct"),
    # The second tier. mns is pinned here because the default capture set is
    # sized for max_num_seqs and cost 4.57 GiB on a 15 GiB T4 -- affordable on
    # an 80 GiB A100, not on a 23 GiB L4 holding 17 GiB of weights. The
    # campaign has measured mns 16 against default at under 0.7 % on this
    # family (campaign handoff §3), and the row records it either way.
    dict(id="G26A4B", model="gemma-4-26B-A4B-AWQ", mns=16),
]

# gemma-4 registers image, video and audio. vLLM only drops the mm-prefix
# backend requirement when every registered modality is zero, and without that
# FlashInfer is refused at engine init regardless of routing -- which is how the
# spec article's collapse gets measured by accident on this machine.
GEMMA_MM = '--limit-mm-per-prompt \'{"image":0,"video":0,"audio":0}\''


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} | {msg}"
    print(line, flush=True)
    with open(PROG, "a") as f:
        f.write(line + "\n")


def emit(obj):
    obj["ts"] = round(time.time(), 1)
    with open(RES, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def done_keys():
    ks = set()
    if os.path.exists(RES):
        for l in open(RES):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r.get("kind") == "decode":
                ks.add((r["cfg"], r["target"], r["round"]))
            if r.get("kind") in ("config_complete", "config_failed"):
                ks.add(("cfg", r["cfg"]))
    return ks


# --- the ladder, cut per tokenizer -----------------------------------------
BOOK = "/content/work/origin.txt"


def get_book():
    if os.path.exists(BOOK) and os.path.getsize(BOOK) > 400000:
        return open(BOOK, encoding="utf-8", errors="ignore").read()
    url = "https://www.gutenberg.org/files/1228/1228-0.txt"
    txt = urllib.request.urlopen(url, timeout=120).read().decode("utf-8", "ignore")
    open(BOOK, "w").write(txt)
    return txt


def ladder_for(model_dir):
    """one prompt per target, cut to that target in THIS model's tokens"""
    cache = f"{D}/ladder-{os.path.basename(model_dir)}.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    body = get_book()
    start = body.find("INTRODUCTION")
    body = body[start if start > 0 else 0:]
    ids = tok(body).input_ids
    out = []
    for t in TARGETS:
        take = ids[:t]
        text = tok.decode(take, skip_special_tokens=True)
        n = len(tok(text).input_ids)          # what it actually costs after decode
        out.append({"target": t, "prompt_tokens": n, "text": text})
    json.dump(out, open(cache, "w"))
    return out


def post(model, prompt, max_tokens, timeout):
    body = json.dumps({
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.8, "stream": True,
        "stream_options": {"include_usage": True}, "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft, n, usage = None, 0, {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except Exception:
                continue
            if ev.get("usage"):
                usage = ev["usage"]
            ch = (ev.get("choices") or [{}])[0]
            if (ch.get("delta") or {}).get("content"):
                if ttft is None:
                    ttft = time.time() - t0
                n += 1
    return ttft, n, time.time() - t0, usage


TOTAL_MIB = int(subprocess.run(
    "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits",
    shell=True, capture_output=True, text=True).stdout.strip().splitlines()[0])


def start_server(cfg):
    # `pkill -f 'vllm serve'` under shell=True can match its own shell, whose
    # command line contains the pattern. That is not theoretical: it left two
    # servers for one model alive at once today, and the second died in
    # init_device against a GPU the first still held -- recorded as a crash for
    # a configuration that had not been tried. The bracket stops the pattern
    # matching itself, and the wait confirms the port and the GPU are actually
    # free rather than assuming a sleep was long enough.
    # Killing the API server is not enough and waiting on a process list is not
    # enough either. vLLM's workers run as `VLLM::EngineCore`, whose command
    # line contains neither "vllm" nor "serve", so the parent dies and the
    # worker keeps the card: 72.7 GiB of 80 on this machine today, which made
    # the next configuration fail its own memory check and be recorded as a
    # crash it had nothing to do with. Kill both, then wait on the card itself.
    for pat in ("[v]llm serve", "[V]LLM::EngineCore", "vllm[.]model_executor"):
        subprocess.run(f"pkill -9 -f '{pat}' 2>/dev/null", shell=True)
    for _ in range(30):
        free = subprocess.run(
            "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits",
            shell=True, capture_output=True, text=True).stdout.strip()
        apps = subprocess.run(
            "nvidia-smi --query-compute-apps=pid --format=csv,noheader",
            shell=True, capture_output=True, text=True).stdout.strip()
        # The A100 version waited for 70 GiB free, which never comes true on a
        # 23 GiB L4. Wait for most of whatever this card is instead.
        if not apps and free and int(free.splitlines()[0]) > 0.85 * TOTAL_MIB:
            break
        time.sleep(2)
    else:
        log("WARNING: card still held at start_server; the memory check may fail")
    time.sleep(2)
    # A stale log from a previous attempt reads as an instant crash -- the same
    # trap bench_runner.py's rev2 notes fixed on the Radeon side.
    lg0 = f"{D}/serve-{cfg['id']}.log"
    if os.path.exists(lg0):
        os.remove(lg0)
    mdir = f"{MODELS}/{cfg['model']}"
    # Prefix caching OFF, and this is the whole reason this round re-measures
    # the A100 rather than reusing it. Every rung of the ladder is a strict
    # prefix of the next -- ids truncated here, sentence boundaries on the
    # Radeon -- so with the cache on, a rung's prefill is charged only for the
    # tokens the previous rung did not already leave in the KV. On the A100
    # 2026-08-29 campaign (enable_prefix_caching=True) round 2 of the 32 K rung
    # took 0.201 s against round 1's 2.932 s, and a "prefill" of 159 299 tok/s
    # was recorded. The Radeon rows are clean -- its two rounds agree to 1.00x
    # at 32 K even on the arms whose config says True -- so the fix makes the
    # CUDA side match the ROCm side rather than the other way round.
    flags = [f"--max-model-len {MML}", f"--port {PORT}",
             "--no-enable-prefix-caching",
             f"--gpu-memory-utilization {cfg.get('util', 0.90)}"]
    if cfg.get("dtype"):
        flags.append(f"--dtype {cfg['dtype']}")
    if cfg.get("mns"):
        flags.append(f"--max-num-seqs {cfg['mns']}")
    if cfg["model"].startswith("gemma-4"):
        flags.append(GEMMA_MM)
    if cfg.get("spec"):
        flags.append("--speculative-config '" + json.dumps(cfg["spec"]) + "'")
    sc = f"{D}/serve-{cfg['id']}.sh"
    with open(sc, "w") as fh:
        fh.write("#!/bin/bash\nset -u\n")
        fh.write(f"exec vllm serve {mdir} " + " ".join(flags) +
                 f" > {D}/serve-{cfg['id']}.log 2>&1\n")
    os.chmod(sc, 0o755)
    subprocess.Popen(["bash", sc])
    t0, hard, stall = time.time(), 3600, 600
    lg = f"{D}/serve-{cfg['id']}.log"
    last = 0
    while time.time() - t0 < hard:
        txt = open(lg).read() if os.path.exists(lg) else ""
        if "Application startup complete" in txt:
            return "ready", txt
        # torch logs whole formatted tracebacks at W level: triton_bundler
        # prints one per missing AOT cubin when it falls back to recompiling,
        # and injecting #45450 mid-run invalidates exactly that cache. The
        # naive test stopped a healthy server on the Radeon side today. A real
        # traceback sits at the head of its line behind only the process tag;
        # a logged one carries its logger's "<file>.py:<line>]" ahead of it.
        real_tb = [l for l in txt.splitlines()
                   if "Traceback (most recent call last)" in l
                   and not re.search(r"\.py:\d+\]", l.split("Traceback")[0])]
        if real_tb or "EngineCore failed to start" in txt \
                or "Engine core initialization failed" in txt:
            return "crash", txt[-2500:]
        idle = time.time() - os.path.getmtime(lg) if os.path.exists(lg) else time.time() - t0
        if idle > stall:
            return "timeout", f"log idle {idle:.0f}s"
        el = time.time() - t0
        if el - last > 240:
            last = el
            log(f"{cfg['id']}: still starting ({el/60:.0f} min)")
        time.sleep(5)
    return "timeout", "hard cap"


def meta_from(cfg_id, txt):
    m = {"kind": "model_meta", "cfg": cfg_id, "machine": MACHINE,
         "vram_total_mib": TOTAL_MIB}
    for k, p in {"init_engine_s": r"init engine[^\n]*took ([0-9.]+) s",
                 "model_load_s": r"Model loading took [0-9.]+ GiB(?: memory)? and ([0-9.]+) seconds",
                 "kv_gib": r"Available KV cache memory: ([0-9.]+) GiB",
                 "kv_tokens": r"GPU KV cache size: ([\d,]+) tokens",
                 # 0.28 writes this two ways from two branches of cuda.py:
                 # "Using AttentionBackendEnum.TRITON_ATTN backend." and
                 # "Using FLASH_ATTN attention backend out of potential ...".
                 # A regex for one silently misses the other, which is why the
                 # A100 campaign recorded no backend at all.
                 "backend": r"Using (?:AttentionBackendEnum\.)?([A-Z0-9_]+)(?: attention)? backend",
                 "wna16_kernel": r"Using (\w+) for CompressedTensorsWNA16",
                 "prefix_caching": r"enable_prefix_caching=(\w+)"}.items():
        mm = re.search(p, txt)
        if mm:
            m[k] = mm.group(1).replace(",", "")
    return m


def inject_45450():
    """Apply the mechanism the spec article validated, once, before it is needed.

    Not the PR's diff: that no longer applies to any tree here. This is
    benchmarks/cuda-a100/45450-validation/inject_45450.py, whose anchors are
    literal source lines and whose assertions fail loudly rather than half-patch.
    """
    import importlib
    r = subprocess.run([sys.executable, f"{D}/inject_45450.py"],
                       capture_output=True, text=True)
    log("inject_45450: " + (r.stdout + r.stderr).strip()[-300:])
    return r.returncode == 0


def run_cfg(cfg, done):
    cid = cfg["id"]
    if ("cfg", cid) in done:
        log(f"{cid}: already complete, skip")
        return
    if cfg.get("p45450") and not globals().get("_INJECTED"):
        if not inject_45450():
            emit({"kind": "config_failed", "cfg": cid, "why": "inject_45450 failed"})
            return
        globals()["_INJECTED"] = True
    st, info = start_server(cfg)
    if st != "ready":
        log(f"{cid}: {st}, FAILED")
        emit({"kind": "config_failed", "cfg": cid, "why": st, "tail": str(info)[-1200:]})
        return
    emit(meta_from(cid, info))
    # One discarded request before the ladder. Without it the very first
    # measurement of the run -- prefill, round 1, the 500 rung -- absorbs
    # everything a cold engine does once: the first CUDA graph replay, the
    # first allocation out of the KV pool, lazy JIT. On the L4 that made the
    # 500 rung 2.064 s against its own round 2's 0.287 s, a 151 % spread on a
    # rung whose every other round agrees to 0.07 %, and cost the rung its
    # chart grade. The Radeon runner has always had this, as its health gate;
    # a100_run.py never did, so every CUDA config in this repository has one
    # ungraded rung for a reason that is the harness and not the machine.
    try:
        post(f"{MODELS}/{cfg['model']}", "Say OK briefly.", 8, 180)
        log(f"{cid}: warmup ok")
    except Exception as ex:
        log(f"{cid}: warmup failed {ex!r} (continuing)")
        emit({"kind": "note", "cfg": cid, "note": f"warmup failed: {ex!r}"[:200]})
    lad = ladder_for(f"{MODELS}/{cfg['model']}")
    ok = err = 0
    for e in lad:
        if e["prompt_tokens"] + GEN + 100 > MML:
            log(f"{cid}: target {e['target']} exceeds mml, stop")
            break
        for rnd in (1, 2):
            if (cid, e["target"], rnd) in done:
                ok += 1
                continue
            try:
                ttft, n, wall, usage = post(f"{MODELS}/{cfg['model']}", e["text"], GEN, 900)
                dec = (n - 1) / (wall - ttft) if ttft and wall > ttft and n > 1 else 0.0
                emit({"kind": "prefill", "cfg": cid, "machine": MACHINE,
                      "target": e["target"], "round": rnd,
                      "prompt_tokens": usage.get("prompt_tokens", e["prompt_tokens"]),
                      "ttft": round(ttft or 0, 4),
                      "prefill_tps": round((usage.get("prompt_tokens") or e["prompt_tokens"]) / ttft, 1)
                      if ttft else 0})
                emit({"kind": "decode", "cfg": cid, "machine": MACHINE,
                      "target": e["target"], "round": rnd,
                      "prompt_tokens": usage.get("prompt_tokens", e["prompt_tokens"]),
                      "gen_tokens": n, "wall_s": round(wall, 3),
                      "decode_tps": round(dec, 4)})
                ok += 1
                log(f"{cid}: {e['target']} r{rnd} {dec:.2f} tok/s")
            except Exception as ex:
                err += 1
                log(f"{cid}: {e['target']} r{rnd} ERROR {ex!r}")
                emit({"kind": "error", "cfg": cid, "target": e["target"], "round": rnd,
                      "err": repr(ex)[:400]})
                if err >= 4:
                    emit({"kind": "config_failed", "cfg": cid, "why": "too many errors"})
                    return
    emit({"kind": "config_complete", "cfg": cid, "ok": ok, "err": err})
    log(f"{cid}: COMPLETE ({ok} ok, {err} err)")
    # A Colab VM can be reclaimed without warning; two were, at 15:58 and 16:56.
    # A copy inside the VM protects against nothing -- both copies go with it.
    # The results are printed here instead, so they reach the caller's terminal
    # and survive the machine that produced them. The poller on the other end
    # writes them down.
    try:
        print("=== HARVEST BEGIN " + cid + " ===", flush=True)
        for line in open(RES):
            print("H|" + line.rstrip(), flush=True)
        print("=== HARVEST END " + cid + " ===", flush=True)
    except Exception as ex:
        log(f"harvest failed: {ex!r}")


if __name__ == "__main__":
    os.makedirs(D, exist_ok=True)
    want = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    import vllm
    log(f"=== {MACHINE} run start, vllm {vllm.__version__} ===")
    done = done_keys()
    for cfg in CFGS:
        if want and cfg["id"] not in want:
            continue
        try:
            run_cfg(cfg, done)
        except Exception as ex:
            log(f"{cfg['id']}: unhandled {ex!r}")
            emit({"kind": "config_failed", "cfg": cfg["id"], "why": repr(ex)[:400]})
    subprocess.run("pkill -f 'vllm serve' 2>/dev/null", shell=True)
    log(f"=== {MACHINE} run end ===")
    print("A100_CAMPAIGN_DONE", flush=True)
