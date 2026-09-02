"""runner_cuda.py — the template a new CUDA campaign starts from.

Copy this beside the campaign's data, set BENCH_MACHINE, edit the config
table and run it. It is `cuda-l4/campaign-2026-08-30c/run.py` with one thing
changed, and that one thing is why this file exists: the CUDA runners
sampled no hardware at all. Not power, not clocks, not temperature. A Colab
T4 measured on 2026-09-02 ran a 300-step matmul at 1245 MHz against a
1590 MHz ceiling while pinned at its 70 W cap, and nothing in the old schema
could have told that from a slow kernel.

Telemetry now comes from `harness/telemetry.py`, the module the gfx1100
runner uses too, so both platforms emit the same field names. On CUDA it
reads NVML in-process: 9.2 ms for a full sample against 29.8 ms for one
nvidia-smi subprocess, measured on the T4 above.

Everything else is 30c's, unchanged. Its own notes follow.

--- 30c: The A100 half of the campaign: nine configurations, eleven rungs, two rounds.

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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from harness.telemetry import Sampler, describe   # noqa: E402

MACHINE = os.environ.get("BENCH_MACHINE", "unknown")   # goes on every row
# /content is Colab's. A rented container mounts its Volume somewhere else,
# and a runner that can only be run on Colab cannot be the template.
D = os.environ.get("BENCH_WORK", "/content/work")
RES = f"{D}/results.jsonl"
PROG = f"{D}/PROGRESS.txt"
MODELS = os.environ.get("BENCH_MODELS", "/content/models")
PORT = 8000
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
HEALTH = f"http://127.0.0.1:{PORT}/health"
# The default ladder and the default context. Both are per-config overridable
# -- `targets=` and `mml=` on a config row -- because a card with 96 GiB can
# carry a model past 32 000 and the eleven rungs below stop there.
TARGETS = [500, 1000, 2000, 4000, 6000, 8000, 12000, 16000, 20000, 24000, 32000]
GEN = 512
MML = 33000
# How long a configuration may take to reach "Application startup complete",
# and how long its log may go quiet. The old 3600/600 was written for a Colab
# session that costs nothing per second; on a rented card an hour of a hung
# compile is real money, and no engine in this repository has ever taken more
# than 290 s to start.
HARD_START_S = int(os.environ.get("BENCH_HARD_START_S", 1200))
STALL_S = int(os.environ.get("BENCH_STALL_S", 420))

# BENCH_CFGS picks a subset by id, as the Radeon runner does.
CFGS = [
    # 2026-08-30, third L4 attempt: the two arms the second one could not fit.
    #
    # Both failed at util 0.95 with mns=16, G31 with four "no room for KV" retries
    # down to mml 2062. That is a measurement of THAT configuration, not of the
    # card: vLLM sizes activations and CUDA graphs for max_num_seqs and charges
    # them against the same budget as the KV pool, and the T4 pre-flight put that
    # at 4.57 GiB of a 13.50 GiB budget for a 12B model at the default mns.
    #
    # So: mns=1, which is what the harness actually uses -- it issues one request
    # at a time -- and an --enforce-eager fallback if that is still not enough.
    # Raising util past 0.95 is deliberately NOT tried: the runner's own rev2 note
    # says these cards keep scratch above that, and on the Radeon it produced
    # HSA_STATUS_ERROR_OUT_OF_RESOURCES rather than a bigger KV pool.
    #
    # gemma-4-31B is 18.7 GiB of weights against the L4's 22.49, and its KV is
    # 204.6 KiB/token -- the heterogeneous 256/512 head dims. Every GiB freed is
    # about 5 100 tokens, so the difference between no ladder and six rungs is
    # roughly 2 GiB.
    dict(id="G31", model="gemma-4-31B-it-qat-w4a16-ct",
         util=0.95, mns=1, eager_fallback=True),
    dict(id="Q38", model="Qwen3.8-27B-AWQ-INT4",
         util=0.95, mns=1, eager_fallback=True),
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
BOOK = f"{D}/origin.txt"


def get_book():
    """Gutenberg #1228, from disk if setup already fetched it.

    `cut_prompts.py` runs during setup and caches the same book beside itself
    as `.gutenberg-1228.txt`, so on a machine this round built there is no
    reason to go to the network at all. On 2026-08-30 this function's single
    un-retried `urlopen` timed out on a healthy A100 whose engine had already
    started and warmed up, and took the configuration down with it -- 231 s of
    engine start thrown away for a text file that was already on the disk.
    """
    for p in (BOOK, os.path.join(D, ".gutenberg-1228.txt")):
        if os.path.exists(p) and os.path.getsize(p) > 400000:
            return open(p, encoding="utf-8", errors="ignore").read()
    urls = ("https://www.gutenberg.org/cache/epub/1228/pg1228.txt",
            "https://www.gutenberg.org/files/1228/1228-0.txt")
    last = None
    for attempt in range(3):
        for url in urls:
            try:
                txt = urllib.request.urlopen(url, timeout=180).read().decode("utf-8", "ignore")
                if len(txt) > 400000:
                    open(BOOK, "w").write(txt)
                    return txt
                last = f"{url}: only {len(txt)} bytes"
            except Exception as e:
                last = f"{url}: {e!r}"
                log(f"get_book attempt {attempt + 1}: {last}")
        time.sleep(10)
    raise RuntimeError(f"could not fetch the book: {last}")


def ladder_for(model_dir, targets):
    """one prompt per target, cut to that target in THIS model's tokens

    Keyed by model AND by target. It used to be a bare list keyed by the model
    directory alone, written by whichever configuration ran first: a second
    configuration asking for a longer ladder got the first one's list back, and
    its extra rungs simply did not exist -- no error, no missing file, no row
    saying why. One entry per target means a longer ladder reuses what is
    already cut and cuts only the rest.
    """
    cache = f"{D}/ladder-{os.path.basename(model_dir)}.json"
    have = {}
    if os.path.exists(cache):
        try:
            raw = json.load(open(cache))
        except Exception:
            raw = None
        if isinstance(raw, list):            # the old format, read once
            have = {str(e["target"]): e for e in raw}
        elif isinstance(raw, dict):
            have = raw
    need = [t for t in targets if str(t) not in have]
    if need:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        body = get_book()
        start = body.find("INTRODUCTION")
        body = body[start if start > 0 else 0:]
        ids = tok(body).input_ids
        # A ladder longer than the book does not fail: ids[:t] silently returns
        # the whole book, and the rung records whatever that came to while
        # claiming the target it asked for. At 32 000 there was no way to hit
        # this; at 128 000 there is.
        if max(need) > len(ids):
            raise RuntimeError(f"book is {len(ids)} tokens in this tokenizer, "
                               f"ladder asks for {max(need)}")
        for t in need:
            text = tok.decode(ids[:t], skip_special_tokens=True)
            n = len(tok(text).input_ids)      # what it actually costs after decode
            have[str(t)] = {"target": t, "prompt_tokens": n, "text": text}
        json.dump(have, open(cache, "w"))
    return [have[str(t)] for t in targets]


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


def classify(txt):
    """What a serve log says has gone wrong, or None while it is still trying.

    Extracted from `start_server` so it can be gated on a laptop with no GPU
    against the real messages, which is the only way to know that a retry
    branch fires on the text a card actually produced rather than on the text
    someone remembered it producing.
    """
    # Before the crash test: this condition raises a ValueError, so its own
    # traceback would otherwise be read as a crash. The message names the
    # length that would fit; 0.27 has a second phrasing with no number.
    m = re.search(r"estimated maximum model length is (\d+)", txt)
    if m:
        return "capacity", int(m.group(1))
    if "No available memory for the cache blocks" in txt:
        return "capacity", -1
    # A hybrid-SSM model reserves one Mamba cache block per decode
    # sequence, and vLLM refuses to capture CUDA graphs when it cannot
    # reserve max_num_seqs of them. Same shape as the KV retry above --
    # the message names the value that would work -- but a different
    # knob, and the KV retry cannot fix it: lowering max_model_len is
    # what frees KV, and this is the Mamba state pool. Qwen3.8-27B hit
    # this on an 80 GiB H100 at mml 132 000 having been fine at 33 000,
    # so it is the long ladder that provokes it, not the card.
    m = re.search(r"exceeds available Mamba cache blocks \((\d+)\)", txt)
    if m:
        return "mns", int(m.group(1))
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
    return None


def start_server(cfg, mml=None, mns=None):
    """`mml` is the effective max-model-len for this attempt, not the constant.

    The Radeon runner has had a capacity retry since rev2; this one did not, and
    on 2026-08-30 that cost four L4 configurations. vLLM raises a ValueError when
    the KV pool cannot hold one request at `--max-model-len`, and the message
    carries the length that would fit. Without the retry the traceback that
    ValueError produces is caught by the crash test below and the configuration
    is recorded as a crash, which is what happened to B8, Q38S, G31 and Q38 --
    B8 by 0.13 GiB, needing 4.53 against 4.40 available.
    """
    mml = (cfg.get("mml") or MML) if mml is None else mml
    mns = cfg.get("mns") if mns is None else mns
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
    flags = [f"--max-model-len {mml}", f"--port {PORT}",
             "--no-enable-prefix-caching",
             f"--gpu-memory-utilization {cfg.get('util', 0.90)}"]
    if cfg.get("dtype"):
        flags.append(f"--dtype {cfg['dtype']}")
    if mns:
        flags.append(f"--max-num-seqs {mns}")
    if cfg.get("tp"):
        flags.append(f"--tensor-parallel-size {cfg['tp']}")
    # CUDA graph capture is sized for max_num_seqs and is charged against the
    # same budget as the KV pool. On a 15 GiB T4 the default capture set cost
    # 4.57 GiB; --enforce-eager skips capture entirely, which is the largest
    # lever available when a model loads but leaves no room for one KV block.
    if cfg.get("eager"):
        flags.append("--enforce-eager")
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
    t0, hard, stall = time.time(), HARD_START_S, STALL_S
    lg = f"{D}/serve-{cfg['id']}.log"
    last = 0
    while time.time() - t0 < hard:
        txt = open(lg).read() if os.path.exists(lg) else ""
        if "Application startup complete" in txt:
            return "ready", txt
        st = classify(txt)
        if st:
            return st
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
    # the one-time machine description goes out beside the first model_meta, so
    # a reader of results.jsonl alone can see how many cards there were, what
    # they are, and -- in `absent` -- what this platform cannot measure
    emit(describe())
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
    mml = cfg.get("mml") or MML
    mns = cfg.get("mns")
    info = None
    for _ in range(4):
        st, info = start_server(cfg, mml, mns)
        if st == "ready":
            break
        if st == "mns":
            new = max(1, int(info))
            log(f"{cid}: Mamba cache holds {info} blocks -> retry mns {new}")
            emit({"kind": "note", "cfg": cid,
                  "note": f"mamba_blocks={info}, mns->{new}"})
            mns = new
            continue
        if st == "capacity":
            if info == -1:
                mml = max(1200, mml // 2)
                log(f"{cid}: no room for KV -> retry mml {mml}")
                emit({"kind": "note", "cfg": cid, "note": f"no-kv-room, mml->{mml}"})
                continue
            if info < 2000:
                log(f"{cid}: KV holds only {info} tok -> not measurable, FAILED")
                emit({"kind": "config_failed", "cfg": cid,
                      "why": f"kv_max_len={info} too small at util={cfg.get('util', 0.90)}"})
                return
            newmml = max(1200, int(info * 0.99))
            log(f"{cid}: KV holds only {info} tok -> retry mml {newmml}")
            emit({"kind": "note", "cfg": cid, "note": f"kv_max_len={info}, mml->{newmml}"})
            mml = newmml
            continue
        log(f"{cid}: {st}, FAILED")
        emit({"kind": "config_failed", "cfg": cid, "why": st, "tail": str(info)[-1200:]})
        return
    else:
        # Running out of capacity retries is not the same as "will not fit":
        # every attempt so far still captured CUDA graphs. Try once more with
        # capture off, under its own id so the row says which it is.
        if cfg.get("eager_fallback") and not cfg.get("eager"):
            log(f"{cid}: capacity retries exhausted -> retrying with --enforce-eager")
            emit({"kind": "note", "cfg": cid, "note": "capacity exhausted, retrying eager"})
            return run_cfg(dict(cfg, eager=True, eager_fallback=False,
                                id=cid + "-eager"), done)
        emit({"kind": "config_failed", "cfg": cid, "why": "startup retries exhausted"})
        return
    emit(meta_from(cid, info) | {"mml": mml, "util": cfg.get("util", 0.90),
                                "mns": mns})
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
    lad = ladder_for(f"{MODELS}/{cfg['model']}", cfg.get("targets") or TARGETS)
    ok = err = 0
    for e in lad:
        if e["prompt_tokens"] + GEN + 100 > mml:
            log(f"{cid}: target {e['target']} exceeds mml, stop")
            break
        for rnd in (1, 2):
            if (cid, e["target"], rnd) in done:
                ok += 1
                continue
            try:
                smp = Sampler()
                with smp:
                    ttft, n, wall, usage = post(f"{MODELS}/{cfg['model']}",
                                                e["text"], GEN, 900)
                dec = (n - 1) / (wall - ttft) if ttft and wall > ttft and n > 1 else 0.0
                # one request produces both rows, so one sampler covers both.
                # wall_s is the request's on both rows rather than the sampler's,
                # or the field would mean two things depending on the row.
                tele = dict(smp.result, wall_s=round(wall, 3))
                emit({"kind": "prefill", "cfg": cid, "machine": MACHINE,
                      "target": e["target"], "round": rnd,
                      "prompt_tokens": usage.get("prompt_tokens", e["prompt_tokens"]),
                      "ttft": round(ttft or 0, 4), "gen_tokens": 0,
                      "prefill_tps": round((usage.get("prompt_tokens") or e["prompt_tokens"]) / ttft, 1)
                      if ttft else 0} | tele)
                emit({"kind": "decode", "cfg": cid, "machine": MACHINE,
                      "target": e["target"], "round": rnd,
                      "prompt_tokens": usage.get("prompt_tokens", e["prompt_tokens"]),
                      "gen_tokens": n, "decode_tps": round(dec, 4)} | tele)
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
    want = (sys.argv[1].split(",") if len(sys.argv) > 1
            else (os.environ["BENCH_CFGS"].split(",")
                  if os.environ.get("BENCH_CFGS") else None))
    import vllm
    # Record the stack. Nothing in the 2026-08-29 A100 logs names a torch or a
    # CUDA version, and the L4's were lost with its VM, so both had to be left
    # null in the projection. A run that does not write down what it ran on
    # cannot be compared to one that did.
    def _v(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None
    import torch
    smi = subprocess.run(
        "nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version "
        "--format=csv,noheader", shell=True, capture_output=True, text=True).stdout.strip()
    emit({"kind": "run_meta", "machine": MACHINE, "vllm": vllm.__version__,
          "torch": _v("torch"), "transformers": _v("transformers"),
          "cuda": torch.version.cuda, "gpu": smi})
    log(f"=== {MACHINE} run start, vllm {vllm.__version__}, torch {_v('torch')}, "
        f"cuda {torch.version.cuda} ===")
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
