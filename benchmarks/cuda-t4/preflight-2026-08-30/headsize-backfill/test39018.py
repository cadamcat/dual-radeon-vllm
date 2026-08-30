"""Does vllm#39018 make gemma-4-12B serve on a T4?

#39018 drops the Triton tile size and the flex block sizes when the padded head
size is >= 512 and the card has < 96 KiB of shared memory. Whether it covers
this case depends on what head_size the kernel actually sees, which headprobe.py
measures. This applies the PR to the installed 0.28.0 and starts the engine the
same way; the recorder from headprobe.py stays in place, so the launch config it
writes is the PR's, not the default's.
"""
import glob, hashlib, json, os, pathlib, re, subprocess, sys, time

D = "/content/work"
MODEL = "/content/models/gemma-4-12B-it-qat-w4a16-ct"
LOG = f"{D}/serve-T4-G12-39018.log"
REC = f"{D}/kernel-args-39018.txt"
OUT = f"{D}/headsize.jsonl"
MM = '--limit-mm-per-prompt \'{"image":0,"video":0,"audio":0}\''


def emit(o):
    o["ts"] = round(time.time(), 1)
    with open(OUT, "a") as f:
        f.write(json.dumps(o) + "\n")
    print("EMIT", json.dumps(o)[:700], flush=True)


def free_mib():
    r = subprocess.run("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits",
                       shell=True, capture_output=True, text=True).stdout.strip()
    return int(r.splitlines()[0]) if r else 0


SP = [p for p in glob.glob("/usr/local/lib/python3*/dist-packages")
      if os.path.exists(os.path.join(p, "vllm"))][0]
tua = pathlib.Path(SP, "vllm/v1/attention/ops/triton_unified_attention.py")
flex = pathlib.Path(SP, "vllm/v1/attention/backends/flex_attention.py")

# point the recorder at a fresh file so the two runs cannot be confused
body = tua.read_text().replace("/content/work/kernel-args.txt", REC)
tua.write_text(body)

r = subprocess.run(f"cd {SP} && patch -p1 --forward < {D}/pr39018.diff",
                   shell=True, capture_output=True, text=True)
print(r.stdout, r.stderr, flush=True)
applied = ("_select" in tua.read_text()) or ("head_size_padded" in tua.read_text())
emit({"kind": "patch39018", "rc": r.returncode, "stdout": r.stdout[-500:],
      "tua_md5": hashlib.md5(tua.read_bytes()).hexdigest(),
      "flex_md5": hashlib.md5(flex.read_bytes()).hexdigest(),
      "has_head_size_padded": "head_size_padded" in tua.read_text(),
      "recorder_still_installed": "_seen_args" in tua.read_text()})
for pyc in glob.glob(SP + "/vllm/**/__pycache__/*.pyc", recursive=True):
    os.remove(pyc)

for pat in ("[v]llm serve", "[V]LLM::EngineCore", "vllm[.]model_executor"):
    subprocess.run(f"pkill -9 -f '{pat}' 2>/dev/null", shell=True)
for _ in range(30):
    if free_mib() > 14000:
        break
    time.sleep(2)
for p in (LOG, REC, REC + ".err"):
    if os.path.exists(p):
        os.remove(p)

sc = f"{D}/serve-39018.sh"
open(sc, "w").write("#!/bin/bash\nset -u\nexec vllm serve " + MODEL +
                    f" --dtype float16 --max-model-len 33000 --port 8000 "
                    f"--gpu-memory-utilization 0.95 --max-num-seqs 1 {MM} > {LOG} 2>&1\n")
os.chmod(sc, 0o755)
subprocess.Popen(["bash", sc])
print("serve launched with vllm#39018 applied", flush=True)

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
    idle = time.time() - os.path.getmtime(LOG) if os.path.exists(LOG) else time.time() - t0
    if idle > stall:
        break
    el = time.time() - t0
    if el - last > 120:
        last = el
        print(f"  ... {el/60:.1f} min, free={free_mib()} MiB", flush=True)
    time.sleep(5)

txt = open(LOG).read() if os.path.exists(LOG) else ""
oor = re.search(r"Required: (\d+),? Hardware limit:? ?(\d+)", txt)
emit({"kind": "serve39018", "state": state,
      "recorder_lines": open(REC).read().strip().splitlines() if os.path.exists(REC) else [],
      "backend": (re.search(r"Using (\S+) backend", txt) or [None, None])[1],
      "kv_gib": (re.search(r"Available KV cache memory: ([0-9.]+) GiB", txt) or [None, None])[1],
      "kv_tokens": (re.search(r"GPU KV cache size: ([\d,]+) tokens", txt) or [None, None])[1],
      "shmem_required": oor.group(1) if oor else None,
      "shmem_limit": oor.group(2) if oor else None,
      "tail": txt[-1200:] if state != "ready" else ""})
print("TEST39018_DONE", flush=True)
