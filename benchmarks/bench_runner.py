#!/usr/bin/env python3
"""bench_runner.py — dual-GPU vLLM context-scan campaign (plan 2026-07-25, rev2).

2026-08-24: three configurations added for the patched-state re-sweep. The six
2026-07-25 configurations below are unchanged and are re-run as controls: none of
the patches that container carries touches their code path, so reproducing their
July numbers is what makes the two campaigns comparable. Select with BENCH_CFGS.

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

# --- site configuration -------------------------------------------------------
# Defaults are the paths on the machine this was measured on. Override with env
# vars; the container must see BENCH_DIR at CONTAINER_BENCH_DIR (we bind-mount
# /data/rccl-build -> /rb). Card indices and hwmon numbers are host-specific:
# check `ls /sys/class/drm/card*/device/hwmon/`.
D = os.environ.get("BENCH_DIR", "/data/rccl-build/bench0725")
D_IN_CONTAINER = os.environ.get("CONTAINER_BENCH_DIR", "/rb/bench0725")
PROMPT_ROOT = os.environ.get("PROMPT_ROOT", "/data/rccl-build")
CONTAINER = os.environ.get("BENCH_CONTAINER", "vllm-tp2")

RES = f"{D}/results.jsonl"
PROG = f"{D}/PROGRESS.txt"
PORT = int(os.environ.get("BENCH_PORT", 8000))
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
HEALTH = f"http://127.0.0.1:{PORT}/health"
HW = {
    "p1": "/sys/class/drm/card1/device/hwmon/hwmon0/power1_average",
    "p2": "/sys/class/drm/card2/device/hwmon/hwmon1/power1_average",
    "v1": "/sys/class/drm/card1/device/mem_info_vram_used",
    "v2": "/sys/class/drm/card2/device/mem_info_vram_used",
}
GEMMA_P = f"{PROMPT_ROOT}/prompts"
QWEN_P = f"{PROMPT_ROOT}/prompts-qwen"
P26 = f"{PROMPT_ROOT}/prompts-26b"
MUSE_P = f"{PROMPT_ROOT}/prompts-muse"
# gemma-3 tokenises every rung of the gemma ladder identically to gemma-4
# (verified 2026-08-24, prompts/gemma3-shares-gemma-ladder.json), so it reuses it.
MML = 33000          # max prompt is ~32,010 tok + 512 output + template
DEFAULT_UTIL = 0.85  # 0.90 leaves no scratch headroom on 20 GiB cards (see rev2 note)

CFGS = [
    dict(id="B-8B-tp2",  model="/models/Qwen3-8B", tp=2, prompts=QWEN_P),
    # 8B BF16 weights are 15.26 GiB on ONE 20 GiB card: at util 0.85 only ~0.4 GiB is
    # left for KV (1,168 tok — useless). 0.90 gives 2.19 GiB / ~15,952 tok, enough for
    # the ladder up to 12K; its max context is short so Triton scratch stays small.
    dict(id="B-8B-tp1",  model="/models/Qwen3-8B", tp=1, prompts=QWEN_P, util=0.90),
    dict(id="A-12B-tp1", model="/models/gemma-4-12B-it-qat-w4a16-ct", tp=1, prompts=GEMMA_P),
    dict(id="A-12B-tp2", model="/models/gemma-4-12B-it-qat-w4a16-ct", tp=2, prompts=GEMMA_P),
    dict(id="C-31B-tp2", model="/models/gemma-4-31B-it-qat-w4a16-ct", tp=2, prompts=GEMMA_P),
    dict(id="D-27B-tp2", model="/models/Qwen3.6-27B-AWQ-INT4", tp=2, prompts=QWEN_P, mns=128, eager_fallback=True),
    # Try compiled first (the 128-expert fused-MoE graph never finished in the 20 min it was
    # given on 2026-07-22, so its real decode speed is still unknown); eager only as fallback.
    dict(id="E-26B-tp2", model="/models/gemma-4-26B-A4B-AWQ", tp=2, prompts=P26, eager_fallback=True),

    # --- added 2026-08-24, measured against the patched container -------------
    # Same architecture as D-27B (64 layers, full_attention_interval 4, head_dim
    # 256) with newer weights, so it inherits D's flags.
    dict(id="D8-27B-tp2", model="/models/Qwen3.8-27B-AWQ-INT4", tp=2, prompts=QWEN_P,
         mns=128, eager_fallback=True),
    dict(id="F-27B-tp2", model="/models/gemma-3-27b-it-w4a16", tp=2, prompts=GEMMA_P),
    # Runs through the downstream adaptation in patches/adapt-muse-glimmer.py;
    # upstream support merged 2026-08-14, after this container was built.
    dict(id="G-30B-tp2", model="/models/Muse-Glimmer-30B-INT4", tp=2, prompts=MUSE_P),
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
                # An older record has no target list, which means the whole ladder.
                # A BENCH_TARGETS subset writes config_complete too, so treating it
                # as "this configuration is finished" made a later full run skip
                # everything it had not measured.
                ks.add(("cfg", j["cfg"], tuple(j.get("targets") or ())))
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
    sh(f"sudo docker restart {CONTAINER}", timeout=120); time.sleep(4)
    sh(f"sudo rm -f {hostlog(cfg['id'])}")   # stale log = false 'ready' (rev2)
    flags = (f"--tensor-parallel-size {cfg['tp']} --gpu-memory-utilization {util} "
             f"--max-model-len {mml} --port {PORT}")
    if cfg.get("eager"):
        flags += " --enforce-eager"
    if cfg.get("mns"):
        flags += f" --max-num-seqs {cfg['mns']}"
    env = "VLLM_CLONE_MMAP=1 NCCL_P2P_DISABLE=1 HSA_ENABLE_SDMA=0"
    sh(f"sudo docker exec -d {CONTAINER} bash -c \"{env} vllm serve {cfg['model']} {flags} "
       f"> {D_IN_CONTAINER}/serve-logs/{cfg['id']}.log 2>&1\"")
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
        # The container redirects into CONTAINER_BENCH_DIR while we watch BENCH_DIR.
        # If those two do not name the same directory the log never appears here and a
        # healthy server looks like a stalled one, so say so instead of waiting it out.
        if not os.path.exists(hl) and time.time() - t0 > 90:
            return "timeout", (f"no log at {hl} after 90 s — the container writes to "
                               f"{D_IN_CONTAINER}/serve-logs/, check that CONTAINER_BENCH_DIR "
                               f"and BENCH_DIR name the same directory")
        txt = open(hl).read() if os.path.exists(hl) else ""
        if "Application startup complete" in txt:
            return "ready", txt
        m = re.search(r"estimated maximum model length is (\d+)", txt)
        if m:
            return "capacity", int(m.group(1))
        if "Traceback" in txt or "EngineCore failed to start" in txt:
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
    want = {t for t, _ in points_for(cfg, MML)}
    for key in done:
        if key[0] == "cfg" and key[1] == cid:
            have = set(key[2]) if key[2] else want
            if want <= have:
                log(f"{cid}: already complete, skip"); return
            log(f"{cid}: previous run covered {sorted(have)}, "
                f"missing {sorted(want - have)} — re-running the configuration")
            break
    if util is None:
        util = cfg.get("util", DEFAULT_UTIL)
    mml = MML; txt = None
    for _ in range(3):
        st, info = start_server(cfg, mml, util)
        if st == "ready":
            txt = info; break
        if st == "capacity":
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
    emit({"kind": "config_complete", "cfg": cid, "mml": mml, "util": util, "ok": nok, "err": nerr,
          "targets": sorted(t for t, _ in pts)})
    log(f"{cid}: COMPLETE ({nok} ok, {nerr} err)")

def main():
    global CFGS
    only = os.environ.get("BENCH_CFGS")
    if only:
        # honour the order given, so the cheapest control and the headline configs
        # can be put first; an unknown name is a typo, not something to skip
        order = [x.strip() for x in only.split(",") if x.strip()]
        by_id = {c["id"]: c for c in CFGS}
        unknown = [x for x in order if x not in by_id]
        if unknown:
            sys.exit(f"BENCH_CFGS names configs that do not exist: {unknown}")
        CFGS = [by_id[x] for x in order]
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
        sh(f"sudo docker stop {CONTAINER}", timeout=120)
        sh("sudo systemctl start ollama llamacpp-hub")
        log("=== campaign end: container stopped, services restored ===")

if __name__ == "__main__":
    main()
