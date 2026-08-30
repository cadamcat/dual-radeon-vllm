#!/usr/bin/env python3
"""bench_runner.py — five-model dual-GPU vLLM context-scan campaign (plan 2026-07-25, rev2).

rev2 fixes (after the 2026-07-25 02:xx false-complete incident):
  * util 0.85, not 0.90 — at 0.90 the KV pool leaves ~54 MB free and the Triton
    _fwd_kernel cannot allocate scratch at long context -> HSA_STATUS_ERROR_OUT_OF_RESOURCES.
  * capacity retry parses vLLM's real message ("estimated maximum model length is N").
  * health gate: the warmup request MUST succeed before any point is measured.
  * guardrails: consecutive/total error thresholds abort a config as FAILED;
    config_complete is only emitted when real decode data exists.
  * mid-run engine death -> one retry of the whole config at util-0.03.
"""
import json, os, re, subprocess, sys, time, random, string, threading
import requests

D = "/data/rccl-build/bench0830c"
D_IN_CONTAINER = "/rb/bench0830c"
# gemma-4 cannot be served on the 0.27 ROCm image at all -- its Quark plugin
# reads head_dim off a heterogeneous config and dies before loading. Its rows
# come from the 0.23 container, which is the stack the 08-24 campaign used,
# so the MTP arm has that campaign's own ladder as its control.
CONTAINER = os.environ.get("BENCH_CONTAINER", "vllm-027")
OTHER_CONTAINERS = ("vllm-027", "vllm-tp2")
MUSE_P = "/data/rccl-build/prompts-muse"
RES = f"{D}/results.jsonl"
PROG = f"{D}/PROGRESS.txt"
PORT = 8000
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
HEALTH = f"http://127.0.0.1:{PORT}/health"
HW = {
    "p1": "/sys/class/drm/card1/device/hwmon/hwmon0/power1_average",
    "p2": "/sys/class/drm/card2/device/hwmon/hwmon1/power1_average",
    "v1": "/sys/class/drm/card1/device/mem_info_vram_used",
    "v2": "/sys/class/drm/card2/device/mem_info_vram_used",
}
GEMMA_P = "/data/rccl-build/prompts"
QWEN_P = "/data/rccl-build/prompts-qwen"
P26 = "/data/rccl-build/prompts-26b"
MML = 33000          # max prompt is ~32,010 tok + 512 output + template
DEFAULT_UTIL = 0.85  # 0.90 leaves no scratch headroom on 20 GiB cards (see rev2 note)

# 2026-08-30. One arm, run twice with the container's triton_unified_attention.py
# in two states. The serve command is IDENTICAL between them; the only difference
# is which file is in site-packages, asserted by md5 by setstate.sh before each
# run. Both states keep vllm#45916 in chunked_prefill_paged_decode.py -- that is
# the ROCM_ATTN path and is not what this experiment varies.
#
#   stock    triton_unified_attention.py 49fab3b643bf5a88eb65303ce377996b
#   patched  vllm#52684 applied           f1d7a7e3c6656303fa63b6a4c1b8aef5
#
# On this checkpoint num_queries_per_kv = 24/4 = 6, so #52684's gate
# (max_seqlen_q >= 512 and num_queries_per_kv <= 16 and _is_gfx1100()) fires from
# the 1000 rung up: BLOCK_M 16 -> 64, BLOCK_Q 2 -> 8, num_warps pinned to 4.
_ARMS = {
    "Q38-triton-stock-tp2": dict(
        id="Q38-triton-stock-tp2", model="/models/Qwen3.8-27B-AWQ-INT4", tp=2,
        prompts=QWEN_P, mns=16, util=0.92,
        extra="--attention-backend TRITON_ATTN"),
    "Q38-triton-52684-tp2": dict(
        id="Q38-triton-52684-tp2", model="/models/Qwen3.8-27B-AWQ-INT4", tp=2,
        prompts=QWEN_P, mns=16, util=0.92,
        extra="--attention-backend TRITON_ATTN"),
    # No --attention-backend: the selector's own choice, which on gfx1100 is
    # ROCM_ATTN out of ['ROCM_ATTN', 'TRITON_ATTN']. Two cppd states:
    #   -45916  the image's own file, i.e. what upstream ships today
    #   +45916  the unmerged split-KV decode PR, i.e. the 08-29 campaign's state
    "Q38-rocm-45916-tp2": dict(
        id="Q38-rocm-45916-tp2", model="/models/Qwen3.8-27B-AWQ-INT4", tp=2,
        prompts=QWEN_P, mns=16, util=0.92),
    "Q38-rocm-nopatch-tp2": dict(
        id="Q38-rocm-nopatch-tp2", model="/models/Qwen3.8-27B-AWQ-INT4", tp=2,
        prompts=QWEN_P, mns=16, util=0.92),
}
CFGS = [_ARMS[os.environ["ARM_ID"]]]

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

class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_ev = threading.Event()
        self.rows = []
    def run(self):
        while not self.stop_ev.is_set():
            row = {}
            for k, p in HW.items():
                try:
                    row[k] = int(open(p).read().strip())
                except Exception:
                    row[k] = 0
            self.rows.append(row)
            self.stop_ev.wait(1.5)
    def stats(self):
        rs = self.rows
        if len(rs) >= 6:
            rs = rs[len(rs)//3: -max(1, len(rs)//6)]
        if not rs:
            return {}
        pw = [(r["p1"] + r["p2"]) / 1e6 for r in rs]
        return {"pw_sum_min": round(min(pw)), "pw_sum_max": round(max(pw)),
                "p1_max": round(max(r["p1"] for r in rs) / 1e6),
                "p2_max": round(max(r["p2"] for r in rs) / 1e6),
                "v1_g": round(max(r["v1"] for r in rs) / 2**30, 2),
                "v2_g": round(max(r["v2"] for r in rs) / 2**30, 2)}

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
    log(f"{cid}: ready (mml={mml}, util={util}), {len(pts)} points")
    nerr = consec = nok = 0

    def do(kind, target, rnd, base, to):
        nonlocal nerr, consec, nok
        if (cid, target, kind, rnd) in done:
            return
        smp = Sampler()
        if kind == "decode":
            smp.start()
        try:
            m = chat(model, base, 512 if kind == "decode" else 1, to)
            rec = {"kind": kind, "cfg": cid, "target": target, "round": rnd,
                   "prompt_tokens": m["prompt_tokens"], "ttft": m["ttft"]}
            if kind == "prefill":
                rec["prefill_tps"] = m["prefill_tps"]
            else:
                smp.stop_ev.set(); smp.join(3)
                rec |= {"completion_tokens": m["completion_tokens"], "decode_tps": m.get("decode_tps")} | smp.stats()
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
            for rnd in (1, 2):
                do("prefill", target, rnd, base, to)
            for rnd in (1, 2):
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
        # This experiment runs two arms back to back with a file swap between
        # them, so the container has to survive the first invocation and ollama
        # must not be allowed back onto the cards in between. BENCH_KEEP=1 keeps
        # both; the caller restores by hand and checks the VRAM baseline.
        if os.environ.get("BENCH_KEEP") == "1":
            log("=== arm end: container left up, services left stopped (BENCH_KEEP=1) ===")
        else:
            sh(f"sudo docker stop {CONTAINER}", timeout=180)
            sh("sudo systemctl start ollama llamacpp-hub")
            log("=== campaign end: container stopped, services restored ===")

if __name__ == "__main__":
    main()
