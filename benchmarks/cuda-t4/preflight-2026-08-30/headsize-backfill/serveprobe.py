"""Start the engine exactly as the 2026-08-30 pre-flight's TRITON_ATTN cell did,
and let the recorder installed by headprobe.py capture what the kernel is called
with before it fails. Auto backend selection, same flags, same model.
"""
import json, os, re, subprocess, sys, time

D = "/content/work"
MODEL = "/content/models/gemma-4-12B-it-qat-w4a16-ct"
LOG = f"{D}/serve-T4-G12-headprobe.log"
REC = f"{D}/kernel-args.txt"
OUT = f"{D}/headsize.jsonl"
MML, UTIL, MNS = "33000", "0.95", "1"
MM = '--limit-mm-per-prompt \'{"image":0,"video":0,"audio":0}\''


def free_mib():
    r = subprocess.run("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits",
                       shell=True, capture_output=True, text=True).stdout.strip()
    return int(r.splitlines()[0]) if r else 0


def emit(o):
    o["ts"] = round(time.time(), 1)
    with open(OUT, "a") as f:
        f.write(json.dumps(o) + "\n")
    print("EMIT", json.dumps(o)[:600], flush=True)


for pat in ("[v]llm serve", "[V]LLM::EngineCore", "vllm[.]model_executor"):
    subprocess.run(f"pkill -9 -f '{pat}' 2>/dev/null", shell=True)
for _ in range(30):
    if free_mib() > 14000:
        break
    time.sleep(2)
for p in (LOG, REC, REC + ".err"):
    if os.path.exists(p):
        os.remove(p)

sc = f"{D}/serve-head.sh"
flags = (f"--dtype float16 --max-model-len {MML} --port 8000 "
         f"--gpu-memory-utilization {UTIL} --max-num-seqs {MNS} {MM}")
open(sc, "w").write(f"#!/bin/bash\nset -u\nexec vllm serve {MODEL} {flags} > {LOG} 2>&1\n")
os.chmod(sc, 0o755)
subprocess.Popen(["bash", sc])
print("serve launched (auto backend, matching the pre-flight's triton cell)", flush=True)

t0, hard, stall, last = time.time(), 2400, 600, 0
state = "timeout"
while time.time() - t0 < hard:
    txt = open(LOG).read() if os.path.exists(LOG) else ""
    if "Application startup complete" in txt:
        state = "ready"; break
    real_tb = [l for l in txt.splitlines()
               if "Traceback (most recent call last)" in l
               and not re.search(r"\.py:\d+\]", l.split("Traceback")[0])]
    if real_tb or "EngineCore failed to start" in txt or "Engine core initialization failed" in txt:
        state = "crash"; break
    if os.path.exists(REC):          # the recorder fired; that is all this needs
        time.sleep(5)
        state = "recorded"; break
    idle = time.time() - os.path.getmtime(LOG) if os.path.exists(LOG) else time.time() - t0
    if idle > stall:
        break
    el = time.time() - t0
    if el - last > 120:
        last = el
        print(f"  ... {el/60:.1f} min, free={free_mib()} MiB, rec={os.path.exists(REC)}", flush=True)
    time.sleep(5)

txt = open(LOG).read() if os.path.exists(LOG) else ""
rec = open(REC).read().strip().splitlines() if os.path.exists(REC) else []
err = open(REC + ".err").read()[:400] if os.path.exists(REC + ".err") else ""
emit({"kind": "serve", "state": state, "recorder_lines": rec, "recorder_err": err,
      "backend": (re.search(r"Using (\S+) backend", txt) or [None, None])[1],
      "shmem_required": (re.search(r"Required: (\d+), Hardware limit: (\d+)", txt) or [None, None, None])[1],
      "shmem_limit": (re.search(r"Required: (\d+), Hardware limit: (\d+)", txt) or [None, None, None])[2],
      "wna16": (re.search(r"Using (\w+) for CompressedTensorsWNA16", txt) or [None, None])[1]})
print("RECORDER:", flush=True)
for l in rec:
    print("   ", l, flush=True)
print("SERVEPROBE_DONE", flush=True)
