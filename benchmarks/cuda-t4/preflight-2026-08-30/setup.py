"""T4 pre-flight setup: vLLM plus the one checkpoint the spine needs.

Only gemma-4-12B-it-qat-w4a16-ct (9.6 G). The question this VM exists to
answer is whether compressed-tensors W4A16 has an sm75 kernel at all, so
nothing else is worth downloading until it is answered.
"""
import os, subprocess, sys, time

LOG = "/content/setup.log"
D = "/content/work"


def sh(cmd, timeout=7200):
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    with open(LOG, "a") as f:
        f.write(f"\n$ {cmd}\n[{time.time()-t0:.0f}s rc={r.returncode}]\n"
                f"{r.stdout[-2000:]}\n{r.stderr[-2000:]}\n")
    return r


MODELS = [("google/gemma-4-12B-it-qat-w4a16-ct", "gemma-4-12B-it-qat-w4a16-ct")]

if __name__ == "__main__":
    open(LOG, "w").close()
    os.makedirs(D, exist_ok=True)
    os.makedirs("/content/models", exist_ok=True)

    print("=== installing vllm ===", flush=True)
    t0 = time.time()
    sh("pip install -q -U vllm 2>&1 | tail -5", timeout=3600)
    # Colab preinstalls torchaudio for CUDA 12.8; vLLM's torch is 13.0 and
    # transformers' is_torchaudio_available() raises on the version check
    # rather than skipping. Nothing here needs audio.
    sh("pip uninstall -y -q torchaudio 2>&1 | tail -2", timeout=600)
    for mod in ("vllm", "torch", "transformers"):
        v = subprocess.run([sys.executable, "-c", f"import {mod}; print({mod}.__version__)"],
                           capture_output=True, text=True)
        print(f"{mod}: {v.stdout.strip() or v.stderr.strip()[-200:]}", flush=True)
    print(f"install took {time.time()-t0:.0f}s", flush=True)

    print("=== downloading checkpoint ===", flush=True)
    for repo, local in MODELS:
        d = f"/content/models/{local}"
        if os.path.exists(d + "/config.json"):
            print(f"  have  {local}", flush=True); continue
        t0 = time.time()
        sh(f"hf download {repo} --local-dir {d} --max-workers 8")
        ok = os.path.exists(d + "/config.json")
        sz = subprocess.run(["du", "-sh", d], capture_output=True, text=True).stdout.split()[0] if ok else "-"
        print(f"  {'ok  ' if ok else 'FAIL'} {local:34s} {sz:>7s}  {time.time()-t0:5.0f}s", flush=True)

    print(subprocess.run(["df", "-h", "/"], capture_output=True, text=True).stdout, flush=True)
    print("T4_SETUP_DONE", flush=True)
