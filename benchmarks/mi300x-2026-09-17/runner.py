"""runner.py — the template a rented single-card ROCm campaign starts from.

`runner_cuda.py` with the platform layer swapped and nothing else touched, so a
diff against it is the list of things that are actually different between a
CUDA card and a CDNA one. Everything that is vLLM-level — the ladder cut per
tokenizer, the capacity/max-position/Mamba retries, the crash classifier, the
checkpointing, the harvest — is that file's, and the reasons those branches
exist are in its comments.

Written for one MI300X (gfx942) on a rented box: 192 GiB, one card, TP=1. It is
NOT a second copy of the Radeon runner: that one drives a local docker
container and stops two GPU services first, neither of which exists here.

What is different from runner_cuda.py, and why:

  * VRAM total, VRAM free and "who is holding the card" come from sysfs and
    /dev/kfd instead of nvidia-smi. amdgpu has no --query-compute-apps.
  * `classify()` gains the two ROCm phrasings of out-of-memory. The other
    branches are vLLM's own messages and are platform-independent.
  * `inject_45450()` is gone. That patch is a CUDA-side experiment.
  * The stack is asserted, not just recorded. BENCH_VLLM_EXPECT and
    BENCH_ROCM_EXPECT, when set, must match the installed versions or the run
    refuses to start. Every cross-machine comparison this repository has had to
    weaken was weakened by a version that differed and was noticed afterwards;
    on a rented card there is no afterwards.
  * A telemetry pre-flight. `harness/telemetry.py` reads sysfs nodes that exist
    on Navi 31 and may not exist on gfx942 (`mem_busy_percent`, the hwmon
    power nodes). It returns None rather than lying, which means a whole
    campaign can come back with no power column and nothing saying so. The
    pre-flight samples once and writes down which fields are actually readable,
    before any model is loaded.
  * Model pre-flight against `benchmarks/modal-2026-09-02/volume.json`: the six
    checkpoints and their pinned revisions. A missing one prints the exact
    download line rather than failing four minutes into a serve.

Environment: BENCH_MACHINE, BENCH_WORK, BENCH_MODELS, BENCH_CFGS as in
runner_cuda.py; BENCH_VLLM_EXPECT / BENCH_ROCM_EXPECT for the assertions.

runner_cuda.py's own notes follow, because everything below still applies.

runner_cuda.py — the template a new CUDA campaign starts from.

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
# The sixteen-rung ladder of 2026-09-03, so a row from this machine sits beside
# the pair's and the rented CUDA cards' rows without an interpolation anywhere.
TARGETS = [500, 1000, 2000, 4000, 6000, 8000, 12000, 16000, 20000, 24000, 32000,
           48000, 64000, 80000, 96000, 128000]
GEN = 512
MML = 132000
# How long a configuration may take to reach "Application startup complete",
# and how long its log may go quiet. The old 3600/600 was written for a Colab
# session that costs nothing per second; on a rented card an hour of a hung
# compile is real money, and no engine in this repository has ever taken more
# than 290 s to start.
HARD_START_S = int(os.environ.get("BENCH_HARD_START_S", 1200))
STALL_S = int(os.environ.get("BENCH_STALL_S", 420))

# BENCH_CFGS picks a subset by id, as the Radeon runner does.
CFGS = [
    # The six checkpoints of 2026-09-03, one card, TP=1. Two carry their own
    # position cap and vLLM refuses `mml` above it at configuration time: the
    # Radeon runner learned this the expensive way on 2026-09-03 (two arms lost
    # to it, one runner revision to fix), and the value is in each config.json,
    # recorded in benchmarks/campaign-2026-09-03/position_caps.json.
    dict(id="G12", model="gemma-4-12B-it-qat-w4a16-ct", tp=1),
    dict(id="G26A4B", model="gemma-4-26B-A4B-AWQ", tp=1),
    dict(id="G31", model="gemma-4-31B-it-qat-w4a16-ct", tp=1),
    dict(id="B8", model="Qwen3-8B", tp=1, mml=40960),
    dict(id="MG30", model="Muse-Glimmer-30B-INT4", tp=1, mml=131072),
    # The hybrid: its Mamba state pool holds fewer blocks than the default
    # max_num_seqs at a long mml, on every machine tried so far (969 on an
    # H100, 198 then 161 on the pair). The retry in run_cfg reads the number
    # out of the refusal; this is only the starting point.
    dict(id="Q38", model="Qwen3.8-27B-AWQ-INT4", tp=1),
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


# The sysfs root, so this file can be gated on a laptop with no GPU against a
# fake tree. runner_cuda.py's own test exists because the failures it catches
# are silent; the same applies here, and a runner whose platform layer can
# only be exercised on the rented card is a runner tested by spending money.
SYSFS = os.environ.get("BENCH_SYSFS_ROOT", "/sys/class/drm")


def _amd_cards():
    """The render nodes amdgpu exposes, in the order telemetry.py uses."""
    import glob as _g
    return sorted(d for d in _g.glob(os.path.join(SYSFS, "card*", "device"))
                  if os.path.exists(os.path.join(d, "mem_info_vram_total")))


def _sysfs_int(path, default=0):
    try:
        return int(open(path).read().strip())
    except Exception:
        return default


_CARDS = _amd_cards()
assert _CARDS, f"no amdgpu render node under {SYSFS}; is this a ROCm box?"
TOTAL_MIB = _sysfs_int(os.path.join(_CARDS[0], "mem_info_vram_total")) // (1024 * 1024)


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
    # ROCm says it two ways and neither is CUDA's. HSA_STATUS_ERROR_OUT_OF_RESOURCES
    # is what the Radeon side got at gpu-memory-utilization above 0.95; the HIP
    # phrasing is torch's allocator. Both are capacity, not crashes, and without
    # this branch they take the whole configuration down as one.
    if "HSA_STATUS_ERROR_OUT_OF_RESOURCES" in txt or "hipErrorOutOfMemory" in txt \
            or "HIP out of memory" in txt:
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
    # A campaign picks one max_model_len for a ladder and applies it to every
    # model in the set. The models do not agree on what they will accept:
    # gemma-4 says 262 144, Muse-Glimmer 131 072, Qwen3-8B 40 960. vLLM
    # refuses before it loads anything and names the number it derived, so
    # this costs 31 s and is recoverable -- Muse-Glimmer was asked for 132 000
    # on 2026-09-03 and this branch is why that is a retry and not a lost
    # configuration.
    m = re.search(r"derived max_model_len \(max_position_embeddings=([\d.]+)", txt)
    if m:
        return "maxlen", int(float(m.group(1)))
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
    # amdgpu has no --query-compute-apps. What it has is the used-VRAM node and
    # /dev/kfd, which every ROCm process holds open for as long as it has a
    # context: `fuser` on it is the direct answer to "is anything still on this
    # card", and it is what the Radeon side has used since a zombie worker held
    # 72 GiB through the next configuration's memory check.
    for _ in range(30):
        used_mib = _sysfs_int(os.path.join(_CARDS[0], "mem_info_vram_used")) // (1024 * 1024)
        holders = subprocess.run("fuser /dev/kfd 2>/dev/null", shell=True,
                                 capture_output=True, text=True).stdout.strip()
        if not holders and used_mib < 0.15 * TOTAL_MIB:
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
    # An arm that differs only by a serve flag, as campaign-2026-09-07's runner
    # expressed the attention A/B: `extra="--attention-backend TRITON_ATTN"`.
    # It goes last so a configuration can override anything above it.
    if cfg.get("extra"):
        flags.append(cfg["extra"])
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


def run_cfg(cfg, done):
    cid = cfg["id"]
    if ("cfg", cid) in done:
        log(f"{cid}: already complete, skip")
        return
    mml = cfg.get("mml") or MML
    mns = cfg.get("mns")
    info = None
    for _ in range(4):
        st, info = start_server(cfg, mml, mns)
        if st == "ready":
            break
        if st == "maxlen":
            log(f"{cid}: model accepts at most {info} -> retry mml {info}")
            emit({"kind": "note", "cfg": cid, "note": f"max_position_embeddings={info}, mml->{info}"})
            mml = int(info)
            continue
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


VOLUME = os.environ.get("BENCH_VOLUME", f"{D}/volume.json")


def preflight_stack():
    """Assert the stack, then record it. An unasserted version is a footnote
    in a footnote six weeks later saying the arms are not comparable."""
    # The cheap check first: a wrong vLLM should stop the run before anything
    # imports torch, both because it is the likelier mismatch and because this
    # ordering is what lets the whole function be gated on a laptop.
    import vllm
    want_v = os.environ.get("BENCH_VLLM_EXPECT")
    if want_v and not vllm.__version__.startswith(want_v):
        raise SystemExit(f"vllm is {vllm.__version__}, expected {want_v}; "
                         "set BENCH_VLLM_EXPECT to what this arm is supposed to be")
    import torch
    hip = getattr(torch.version, "hip", None)
    want_r = os.environ.get("BENCH_ROCM_EXPECT")
    if want_r and not (hip or "").startswith(want_r):
        raise SystemExit(f"torch reports hip {hip}, expected {want_r}")
    return vllm.__version__, hip


def preflight_telemetry():
    """Sample once and write down which fields this platform actually gives.

    telemetry.py reads sysfs nodes that exist on Navi 31. gfx942 may not have
    `mem_busy_percent` or the same hwmon layout, and the module returns None
    rather than guessing -- which is right, and also means a campaign can come
    back with an empty power column and nothing in it saying so. This runs
    before the first model and emits the answer as a row.
    """
    from harness.telemetry import cards
    cs = cards()
    got, missing = {}, []
    for i, c in enumerate(cs):
        smp = c.sample()
        for k, v in smp.items():
            (got.setdefault(k, []).append(v) if v is not None else missing.append(f"card{i}.{k}"))
    row = {"kind": "telemetry_preflight", "machine": MACHINE, "cards": len(cs),
           "readable": sorted(got), "absent": sorted(set(missing)),
           "static": [c.static() for c in cs]}
    emit(row)
    log(f"telemetry: {len(cs)} card(s), readable {sorted(got)}, absent {sorted(set(missing))}")
    for k in ("power_w", "sclk_mhz", "vram_used_b"):
        if k not in got:
            log(f"WARNING: {k} is not readable on this platform; rows will carry nulls for it")
    return row


def preflight_models():
    """Every configured checkpoint is on disk, at the revision the sweep pinned.

    volume.json is benchmarks/modal-2026-09-02/volume.json: name, repo and the
    commit revision each was fetched at. A model that is merely present is not
    the same model as the one the other machines measured.
    """
    try:
        vol = {v["name"]: v for v in json.load(open(VOLUME))}
    except Exception as ex:
        log(f"volume.json not readable ({ex!r}); model revisions are NOT pinned for this run")
        vol = {}
    missing = []
    for cfg in CFGS:
        d = f"{MODELS}/{cfg['model']}"
        if not os.path.isdir(d):
            v = vol.get(cfg["model"])
            missing.append((cfg["model"], v))
    for name, v in missing:
        if v:
            log(f"MISSING {name}: hf download {v['repo']} --revision {v['revision']} "
                f"--local-dir {MODELS}/{name}   # {v.get('gib', '?')} GiB")
        else:
            log(f"MISSING {name}: and it is not in volume.json either")
    if missing:
        raise SystemExit(f"{len(missing)} checkpoint(s) missing; the lines above fetch them")
    emit({"kind": "models_meta", "machine": MACHINE,
          "models": [{"name": c["model"], "revision": (vol.get(c["model"]) or {}).get("revision")}
                     for c in CFGS]})


if __name__ == "__main__":
    os.makedirs(D, exist_ok=True)
    want = (sys.argv[1].split(",") if len(sys.argv) > 1
            else (os.environ["BENCH_CFGS"].split(",")
                  if os.environ.get("BENCH_CFGS") else None))
    def _v(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None
    vllm_v, hip_v = preflight_stack()
    import torch
    # rocm-smi is not on every image and its output format has changed twice.
    # The two things that identify the card are in sysfs and in torch, and both
    # are there on any box that can run vLLM at all.
    props = torch.cuda.get_device_properties(0)
    gpu = (f"{props.name}, {props.total_memory // (1024**2)} MiB, "
           f"gfx{getattr(props, 'gcnArchName', '?')}")
    emit({"kind": "run_meta", "machine": MACHINE, "vllm": vllm_v,
          "torch": _v("torch"), "transformers": _v("transformers"),
          "cuda": None, "rocm": hip_v, "gpu": gpu,
          "gcn_arch": getattr(props, "gcnArchName", None),
          "vram_total_mib": TOTAL_MIB})
    log(f"=== {MACHINE} run start, vllm {vllm_v}, torch {_v('torch')}, "
        f"hip {hip_v}, {gpu} ===")
    preflight_telemetry()
    preflight_models()
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
    print("ROCM_CAMPAIGN_DONE", flush=True)
