"""runner.py -- two RX 7900 XT, TP=2, context to 128 000.

Every ladder this repository has run stopped at 32 000, and until 2026-09-03
that looked like the machine's limit. It was the campaign's: every arm set
`mml` 33 000 and the KV pools it reports are 2.5x to 11.9x larger than the
ladder asked for. gemma-4-12B's TP=2 pool held 355 078 tokens for a ladder
whose longest rung is 32 512.

So this is the same six models the rented sweep of the same day measured, on
the machine this repository is actually about, with the ladder carried as far
as each configuration will hold it. `points_for` recomputes which rungs fit
after every capacity retry, so a model that cannot hold 128 000 settles on the
longest ladder it can and says so in a `note` row rather than failing.

**The prompts are a new cut and that is deliberate.** `/data/rccl-build/v2/` holds sixteen
rungs per tokenizer, cut in one pass on 2026-09-03. The committed eleven were
not extended, because they cannot be reproduced: the same 2 158 characters that
`manifest-gemma.json` records as 481 tokens are 511 under the tokenizer in this
container today -- same book, both Gutenberg URLs give the identical passage
after the anchor, so it is the tokenizer that moved and not the text. Rewriting
the old files would have silently changed what every earlier campaign is
recorded as having measured. Instead the whole ladder here is cut once with one
tokenizer, which makes it internally consistent, and its overlap with the old
rungs at 500-32 000 is a measurable difference rather than a hidden one.

Patch state, read out of the container before the run (the rule
campaign-2026-09-02c earned): vllm 0.23.1.dev1+g9ddef7117.d20260715,
chunked_prefill_paged_decode.py 63f0505d (vllm#45916 split-KV),
triton_attn.py 7e275cdc (vllm#45450). The same container and the same two
patches as campaign-2026-09-02d -- which is `vllm-tp2`, and `CONTAINER` below
says so.

That last sentence was false for the first three attempts of this campaign.
This file is copied from `harness/runner_radeon.py`, whose `CONTAINER`
defaults to `vllm-027`; the md5s above were read out of `vllm-tp2` and the
runs went to `vllm-027`, a different ROCm and a different vLLM. The
verification was real and described a container that was not the one being
measured, which is the same failure `campaign-2026-09-02c` is about, arrived
at from the other direction. Both stacks have committed rows -- 172 on
0.23/ROCm 7.14 against 94 on 0.27/ROCm 10.0 -- so this is a choice, not a
repair, and it is made for 0.23 because the eleven rungs these sixteen extend
were measured there.
"""
import json, os, re, subprocess, sys, time, random, string, threading
import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from harness.telemetry import Sampler, describe   # noqa: E402

# Its own directory. The template carries the 2026-08-30 campaign's, and
# the first three attempts of this one appended their rows and serve logs
# to that campaign's files -- a work directory is not a detail when the
# runner appends.
D = "/data/rccl-build/bench0903"
D_IN_CONTAINER = "/rb/bench0903"
# gemma-4 cannot be served on the 0.27 ROCm image at all -- its Quark plugin
# reads head_dim off a heterogeneous config and dies before loading. Its rows
# come from the 0.23 container, which is the stack the 08-24 campaign used,
# so the MTP arm has that campaign's own ladder as its control.
CONTAINER = os.environ.get("BENCH_CONTAINER", "vllm-tp2")
OTHER_CONTAINERS = ("vllm-027", "vllm-tp2")
# the 2026-09-03 cut: sixteen rungs per tokenizer in one pass. See the
# docstring for why the committed eleven were not extended in place.
MUSE_P = "/data/rccl-build/v2/prompts-muse"
RES = f"{D}/results.jsonl"
PROG = f"{D}/PROGRESS.txt"
MACHINE = os.environ.get("BENCH_MACHINE", "RX 7900 XT")
PORT = 8000
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
HEALTH = f"http://127.0.0.1:{PORT}/health"
GEMMA_P = "/data/rccl-build/v2/prompts"
QWEN_P = "/data/rccl-build/v2/prompts-qwen"
P26 = "/data/rccl-build/v2/prompts-26b"
MML = 132000         # 128 000 tok + 512 output + template; the capacity
                     # retry lowers it per configuration to what fits
DEFAULT_UTIL = 0.85  # 0.90 leaves no scratch headroom on 20 GiB cards (see rev2 note)

CFGS = [
    # Ordered cheapest-first: the 12B is the model every machine in this
    # repository has, so if the long ladder is going to break it breaks here.
    dict(id="A-12B-tp2-long", model="/models/gemma-4-12B-it-qat-w4a16-ct",
         tp=2, prompts=GEMMA_P),
    dict(id="B-8B-tp2-long", model="/models/Qwen3-8B", tp=2, prompts=QWEN_P),
    dict(id="E-26B-tp2-long", model="/models/gemma-4-26B-A4B-AWQ",
         tp=2, prompts=P26),
    dict(id="G-30B-tp2-long", model="/models/Muse-Glimmer-30B-INT4",
         tp=2, prompts=MUSE_P),
    dict(id="D8-27B-tp2-long", model="/models/Qwen3.8-27B-AWQ-INT4",
         tp=2, prompts=QWEN_P),
    # Last, and expected to settle short: its pool held 85 766 tokens at
    # mml 33 000, so 128 000 is not reachable and the retry will find where it
    # stops. That number is the measurement, not a failure.
    dict(id="C-31B-tp2-long", model="/models/gemma-4-31B-it-qat-w4a16-ct",
         tp=2, prompts=GEMMA_P),
]

def sh(cmd, timeout=180):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 1, "", "timeout")

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
        for ln in open(RES):
            try:
                j = json.loads(ln)
            except Exception:
                continue
            if j.get("kind") == "config_complete":
                ks.add(("cfg", j["cfg"]))
            elif j.get("kind") in ("prefill", "decode"):
                ks.add((j["cfg"], j["target"], j["kind"], j["round"]))
    return ks

def rid():
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=8))

def chat(model, prompt, max_tokens, timeout):
    body = {"model": model, "messages": [{"role": "user", "content": f"[seed-{rid()}] {prompt}"}],
            "max_tokens": max_tokens, "temperature": 0.8, "stream": True,
            "stream_options": {"include_usage": True}}
    t0 = time.perf_counter(); tfirst = None; tlast = None; usage = None
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
                    if tfirst is None:
                        tfirst = now
                    tlast = now
    if usage is None or tfirst is None:
        raise RuntimeError("stream ended without usage/token")
    out = {"prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"],
           "ttft": round(tfirst - t0, 4), "prefill_tps": round(usage["prompt_tokens"] / (tfirst - t0), 1)}
    if usage["completion_tokens"] >= 2 and tlast > tfirst:
        out["gen_time"] = round(tlast - tfirst, 4)
        out["decode_tps"] = round((usage["completion_tokens"] - 1) / (tlast - tfirst), 2)
    return out

def hostlog(cfg_id):
    return f"{D}/serve-logs/{cfg_id}.log"

def server_alive():
    try:
        return requests.get(HEALTH, timeout=5).status_code == 200
    except Exception:
        return False

def start_server(cfg, mml, util):
    """returns (state, info): ready|capacity|crash|timeout"""
    # a restart alone can leave the previous vllm holding :8000, and the next
    # config then dies with "Address already in use" -- which reads as a real
    # failure in the log and is not one
    # ...and the holder can be the *other* container rather than this one.
    # vllm-tp2 is bridge-networked and publishes 8000, so its docker-proxy owns
    # the host port for the container's whole life even with nothing serving
    # inside it; vllm-027 is host-networked and binds the host port directly.
    # An in-container pkill cannot see that holder, which is why the first fix
    # did not take. Both p45450 attempts died this way, and both of the arms
    # that succeeded ran before vllm-tp2 was started.
    for _o in OTHER_CONTAINERS:
        if _o != CONTAINER:
            sh(f"sudo docker stop {_o} || true", timeout=180)
    sh(f"sudo docker exec {CONTAINER} pkill -f 'vllm serve' || true", timeout=60)
    sh(f"sudo docker restart {CONTAINER}", timeout=180); time.sleep(10)
    sh(f"sudo rm -f {hostlog(cfg['id'])}")   # stale log = false 'ready' (rev2)
    flags = (f"--tensor-parallel-size {cfg['tp']} --gpu-memory-utilization {util} "
             f"--max-model-len {mml} --port {PORT}")
    if cfg.get("eager"):
        flags += " --enforce-eager"
    if cfg.get("mns"):
        flags += f" --max-num-seqs {cfg['mns']}"
    # speculation, which the July runner had no way to express. Two shapes:
    # a separate drafter checkpoint, and a head inside the target's own weights.
    if cfg.get("spec"):
        flags += " --speculative-config '" + json.dumps(cfg["spec"]) + "'"
    if cfg.get("extra"):
        flags += " " + cfg["extra"]
    # gemma-4's head dimensions are heterogeneous (head_dim 256 local,
    # global_head_dim 512). vLLM 0.27's converter reads them with
    #     head_dim = getattr(self.hf_text_config, "head_dim", 0)
    # expecting AttributeError, but transformers 5.x raises
    # AmbiguousGlobalPerLayerAttributeError, which a getattr default does not
    # swallow, and the engine dies before loading anything. The exception names
    # its own switch; this sets it.
    if cfg.get("hf_overrides"):
        flags += " --hf-overrides '" + json.dumps(cfg["hf_overrides"]) + "'"
    env = "VLLM_CLONE_MMAP=1 NCCL_P2P_DISABLE=1 HSA_ENABLE_SDMA=0"
    # The serve command goes into a file rather than through docker exec's
    # quoting. --speculative-config takes JSON, and JSON's double quotes do not
    # survive host shell -> docker exec -> bash -c: the first attempt reached
    # vLLM as {model: /models/..., method: draft_model} with every quote gone.
    # The file is also the record of exactly what was run for this cell.
    sc = f"{D}/serve-{cfg['id']}.sh"
    with open(sc, "w") as fh:
        fh.write("#!/bin/bash\nset -u\nexport " + env.replace(" ", "\nexport ") + "\n")
        fh.write(f"exec vllm serve {cfg['model']} {flags} "
                 f"> {D_IN_CONTAINER}/serve-logs/{cfg['id']}.log 2>&1\n")
    os.chmod(sc, 0o755)
    sh(f"sudo docker exec -d {CONTAINER} bash {D_IN_CONTAINER}/serve-{cfg['id']}.sh")
    # Activity-aware deadline (rev3): a cold torch.compile of one graph takes 23+ min on
    # this Zen1 box (A-12B-tp1 was killed 6 s before its 1401.95 s compile finished by the
    # old flat 1500 s cap). Keep waiting while the server is demonstrably still working —
    # i.e. its log was written to within STALL_S — up to a hard cap.
    # NOTE: Inductor writes NOTHING to the log for the whole compile (A-12B-tp1 was silent
    # from 03:31:59 to 03:55:03), so a plain idle test would kill a healthy long compile.
    # While a compile is outstanding (started but no "takes N s" yet) the stall test is
    # relaxed to COMPILE_STALL; the hard cap is the only real bound.
    HARD_CAP, STALL_S, COMPILE_STALL = 5400, 420, 5400
    hl = hostlog(cfg["id"]); t0 = time.time(); last_note = 0
    while time.time() - t0 < HARD_CAP:
        txt = open(hl).read() if os.path.exists(hl) else ""
        if "Application startup complete" in txt:
            return "ready", txt
        m = re.search(r"estimated maximum model length is (\d+)", txt)
        if m:
            return "capacity", int(m.group(1))
        # 0.27 says it a second way, with no number attached; halve and retry
        if "No available memory for the cache blocks" in txt:
            return "capacity", -1
        # "Traceback" also appears *inside* torch's own logged warnings:
        # triton_bundler.py:242 prints a whole formatted traceback at W level
        # when an AOT cubin is missing from the cache and it falls back to
        # recompiling. That is not a crash, and the naive test stopped a server
        # that was starting normally. A real traceback sits at the head of its
        # line behind only the process tag; a logged one carries its logger's
        # "<file>.py:<line>]" ahead of it. Checked against all six serve logs:
        # the three genuine startup failures still match, the two completed runs
        # and the warning storm do not.
        real_tb = [l for l in txt.splitlines()
                   if "Traceback (most recent call last)" in l
                   and not re.search(r"\.py:\d+\]", l.split("Traceback")[0])]
        if real_tb or "EngineCore failed to start" in txt:
            return "crash", txt[-2000:]
        starts = txt.count("Cache the graph of compile range")
        dones = len(re.findall(r"Compiling a graph for compile range[^\n]*takes", txt))
        compiling = starts > dones
        idle = time.time() - os.path.getmtime(hl) if os.path.exists(hl) else time.time() - t0
        if idle > (COMPILE_STALL if compiling else STALL_S):
            return "timeout", f"log idle {idle:.0f}s (compiling={compiling}) after {time.time()-t0:.0f}s"
        el = time.time() - t0
        if el - last_note > 300:   # heartbeat so a long compile does not look hung
            last_note = el
            log(f"{cfg['id']}: still starting ({el/60:.0f} min, idle {idle:.0f}s)"
                + (f" [compiling {starts-dones} graph(s)]" if compiling else ""))
        time.sleep(5)
    return "timeout", f"hard cap {HARD_CAP}s"

def load_meta(cfg_id, txt):
    meta = {"kind": "model_meta", "cfg": cfg_id}
    for k, p in {"weights_s": r"Loading weights took ([0-9.]+) seconds",
                 "model_load_s": r"Model loading took [0-9.]+ GiB(?: memory)? and ([0-9.]+) seconds",
                 "init_engine_s": r"init engine[^\n]*took ([0-9.]+) s",
                 "kv_gib": r"Available KV cache memory: ([0-9.]+) GiB",
                 "kv_tokens": r"GPU KV cache size: ([\d,]+) tokens",
                 "concurrency": r"Maximum concurrency[^\n]*?([\d.]+)x"}.items():
        m = re.search(p, txt)
        if m:
            meta[k] = m.group(1).replace(",", "")
    return meta

def points_for(cfg, mml):
    man = json.load(open(os.path.join(cfg["prompts"], "manifest.json")))
    tsel = os.environ.get("BENCH_TARGETS")
    allow = {int(x) for x in tsel.split(",")} if tsel else None
    return [(e["target"], e["est_prompt_tokens"]) for e in man
            if (not allow or e["target"] in allow) and e["est_prompt_tokens"] + 600 <= mml]

class ConfigAborted(Exception):
    pass

def run_cfg(cfg, done, util=None, attempt=1):
    cid = cfg["id"]
    if ("cfg", cid) in done:
        log(f"{cid}: already complete, skip"); return
    if util is None:
        util = cfg.get("util", DEFAULT_UTIL)
    mml = MML; txt = None
    for _ in range(3):
        st, info = start_server(cfg, mml, util)
        if st == "ready":
            txt = info; break
        if st == "capacity":
            if info == -1:            # no number given: halve the ladder and retry
                mml = max(1200, mml // 2)
                log(f"{cid}: no room for KV at mml -> retry mml {mml}")
                emit({"kind": "note", "cfg": cid, "note": f"no-kv-room, mml->{mml}"})
                continue
            if info < 2000:   # a ladder needs at least the 1K point to be meaningful
                log(f"{cid}: KV holds only {info} tok at util {util} — not measurable, FAILED")
                emit({"kind": "config_failed", "cfg": cid, "why": f"kv_max_len={info} too small at util={util}"})
                return
            newmml = max(1200, int(info * 0.99))
            log(f"{cid}: KV holds only {info} tok -> retry mml {newmml}")
            emit({"kind": "note", "cfg": cid, "note": f"kv_max_len={info}, mml->{newmml}"})
            mml = newmml; continue
        if st in ("crash", "timeout") and cfg.get("eager_fallback") and not cfg.get("eager"):
            log(f"{cid}: startup {st} -> eager fallback")
            emit({"kind": "note", "cfg": cid, "note": f"graph startup {st} -> eager", "tail": str(info)[-400:]})
            cfg = dict(cfg, eager=True, id=cid + "-eagerfb"); cid = cfg["id"]; continue
        log(f"{cid}: startup {st}, config FAILED")
        emit({"kind": "config_failed", "cfg": cid, "why": st, "tail": str(info)[-500:]}); return
    else:
        emit({"kind": "config_failed", "cfg": cid, "why": "startup retries exhausted"}); return

    emit(describe())
    emit(load_meta(cid, txt) | {"mml": mml, "util": util})
    model = cfg["model"]

    # ---- health gate (rev2): warmup must succeed, else the config is not measurable
    ok = False
    for i in range(2):
        try:
            chat(model, "Say OK briefly.", 32, 120); ok = True; break
        except Exception as e:
            log(f"{cid}: warmup attempt {i+1} failed: {str(e)[:120]}"); time.sleep(10)
    if not ok:
        emit({"kind": "config_failed", "cfg": cid, "why": "warmup failed (server unhealthy)"})
        log(f"{cid}: FAILED at health gate")
        if attempt == 1 and util > 0.78:
            log(f"{cid}: retrying whole config at util {round(util-0.03,2)}")
            return run_cfg(cfg, done, util=round(util - 0.03, 2), attempt=2)
        return

    pts = points_for(cfg, mml)
    # two rounds is what every campaign before 2026-09-02 ran, and it is enough
    # to see a rung disagree with itself but not enough to see *why*: the L4 and
    # the T4 needed five to show that the shallow rung's spread follows the
    # clock rather than the request number. `rounds` lets an arm ask for more
    # without a second runner.
    nrounds = cfg.get("rounds", 2)
    log(f"{cid}: ready (mml={mml}, util={util}), {len(pts)} points x {nrounds} rounds")
    nerr = consec = nok = 0

    def do(kind, target, rnd, base, to):
        nonlocal nerr, consec, nok
        if (cid, target, kind, rnd) in done:
            return
        # A 500-token prefill on this box takes 0.12 s. At the sampler's default
        # 1.5 s that is one sample, and on 2026-09-02b four of five rounds took
        # theirs in the idle gap before the request and recorded sclk 0. Short
        # cells get a short period; the period is recorded on the row either way.
        smp = Sampler(period_s=0.02 if kind == "prefill" else None)
        try:
            # sampled for both kinds now; a prefill too short to catch a sample
            # emits the same keys with tele_samples 0 rather than a shorter row
            with smp:
                m = chat(model, base, 512 if kind == "decode" else 1, to)
            rec = {"kind": kind, "cfg": cid, "machine": MACHINE, "target": target,
                   "round": rnd, "prompt_tokens": m["prompt_tokens"],
                   "ttft": m["ttft"]} | smp.result
            if kind == "prefill":
                rec["prefill_tps"] = m["prefill_tps"]
                rec["gen_tokens"] = 0
            else:
                rec |= {"completion_tokens": m["completion_tokens"],
                        "gen_tokens": m["completion_tokens"],
                        "decode_tps": m.get("decode_tps")}
            emit(rec); nok += 1; consec = 0
        except Exception as e:
            smp.stop_ev.set()
            nerr += 1; consec += 1
            emit({"kind": "error", "cfg": cid, "target": target, "step": f"{kind} r{rnd}", "err": str(e)[:300]})
            log(f"{cid}: {kind} r{rnd} @{target} ERROR {str(e)[:90]}")
            if consec >= 3 or (nerr >= 4 and nerr > nok):
                raise ConfigAborted(f"{nerr} errors ({consec} consecutive), server_alive={server_alive()}")
        time.sleep(1)

    try:
        for target, est in pts:
            base = open(os.path.join(cfg["prompts"], f"prompt_{target}.txt"), encoding="utf-8").read()
            to = max(300, int(est / 60) + 300)
            for rnd in range(1, nrounds + 1):
                do("prefill", target, rnd, base, to)
            for rnd in range(1, nrounds + 1):
                do("decode", target, rnd, base, to + 400)
            log(f"{cid}: point {target} done")
    except ConfigAborted as e:
        emit({"kind": "config_failed", "cfg": cid, "why": f"aborted mid-run: {e}", "ok": nok, "err": nerr})
        log(f"{cid}: ABORTED mid-run ({e})")
        if attempt == 1 and util > 0.78:
            log(f"{cid}: retrying whole config at util {round(util-0.03,2)}")
            return run_cfg(cfg, done, util=round(util - 0.03, 2), attempt=2)
        return

    if nok == 0:
        emit({"kind": "config_failed", "cfg": cid, "why": "no successful measurement"})
        log(f"{cid}: FAILED (no data)"); return
    emit({"kind": "config_complete", "cfg": cid, "mml": mml, "util": util, "ok": nok, "err": nerr})
    log(f"{cid}: COMPLETE ({nok} ok, {nerr} err)")

def main():
    global CFGS
    only = os.environ.get("BENCH_CFGS")
    if only:
        CFGS = [c for c in CFGS if c["id"] in only.split(",")]
    os.makedirs(f"{D}/serve-logs", exist_ok=True)
    log(f"=== campaign start rev2 ({[c['id'] for c in CFGS]}) ===")
    sh("sudo systemctl stop ollama llamacpp-hub"); time.sleep(2)
    r = sh("sudo fuser -v /dev/kfd 2>&1 | tail -2")
    if re.search(r"\b\d{2,}\b", r.stdout + r.stderr):
        log(f"kfd still busy -> ABORT: {(r.stdout + r.stderr)[:200]}"); sys.exit(2)
    try:
        done = done_keys()
        for cfg in CFGS:
            run_cfg(cfg, done)
    finally:
        sh(f"sudo docker stop {CONTAINER}", timeout=180)
        sh("sudo systemctl start ollama llamacpp-hub")
        log("=== campaign end: container stopped, services restored ===")

if __name__ == "__main__":
    main()
