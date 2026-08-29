"""A100 session setup: vLLM, the five checkpoints, the two drafters, the ladders.

Runs on the Colab VM. Everything here is wall-clock that does not need the
Radeon host, so it starts first and runs while that side is being built.

The prompt ladders are cut from Darwin's Origin of Species, Gutenberg #1228,
the same source benchmarks/prompts/cut_prompts.py uses, to the same eleven
targets, with each model's own tokenizer -- a rung is a token count, not a
character count, so the ladder is per tokenizer.
"""
import json
import os
import subprocess
import sys
import time

LOG = "/content/setup.log"


def sh(cmd, timeout=7200):
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    with open(LOG, "a") as f:
        f.write(f"\n$ {cmd}\n[{time.time()-t0:.0f}s rc={r.returncode}]\n"
                f"{r.stdout[-2000:]}\n{r.stderr[-2000:]}\n")
    return r


# Trimmed for the fourth VM. Only the three remaining arms' checkpoints:
# G31 + its assistant, G26A4B + its assistant, and Q38, whose mtp head is in
# its own weights. That is ~61 GB instead of ~153, because four VMs have now
# been reclaimed mid-campaign and setup is the part that does not survive.
# gemma-4-12B and both Muse-Glimmer checkpoints belong to configurations that
# are already complete and saved.
MODELS = [
    ("google/gemma-4-31B-it-qat-w4a16-ct", "gemma-4-31B-it-qat-w4a16-ct"),
    ("google/gemma-4-31B-it-assistant", "gemma-4-31B-it-assistant"),
    ("cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit", "gemma-4-26B-A4B-AWQ"),
    ("google/gemma-4-26B-A4B-it-assistant", "gemma-4-26B-A4B-it-assistant"),
    ("cyankiwi/Qwen3.8-27B-AWQ-INT4", "Qwen3.8-27B-AWQ-INT4"),
]

if __name__ == "__main__":
    open(LOG, "w").close()
    print("=== disk before ===", flush=True)
    print(subprocess.run(["df", "-h", "/"], capture_output=True, text=True).stdout, flush=True)

    print("=== installing vllm ===", flush=True)
    r = sh("pip install -q -U vllm 2>&1 | tail -5", timeout=3600)
    # Colab preinstalls a torchaudio built for CUDA 12.8; vLLM's torch is 13.0,
    # and transformers' `if is_torchaudio_available(): import torchaudio` then
    # raises on the version check rather than skipping. Nothing here needs audio.
    sh("pip uninstall -y -q torchaudio 2>&1 | tail -2", timeout=600)
    v = subprocess.run([sys.executable, "-c", "import vllm; print(vllm.__version__)"],
                       capture_output=True, text=True)
    print("vllm:", v.stdout.strip() or v.stderr.strip()[-300:], flush=True)
    tr = subprocess.run([sys.executable, "-c", "import transformers; print(transformers.__version__)"],
                        capture_output=True, text=True)
    print("transformers:", tr.stdout.strip() or tr.stderr.strip()[-200:], flush=True)

    print("=== downloading checkpoints ===", flush=True)
    os.makedirs("/content/models", exist_ok=True)
    for repo, local in MODELS:
        d = f"/content/models/{local}"
        if os.path.exists(d + "/config.json"):
            print(f"  have  {local}", flush=True)
            continue
        t0 = time.time()
        r = sh(f"hf download {repo} --local-dir {d} --max-workers 8")
        ok = os.path.exists(d + "/config.json")
        sz = subprocess.run(["du", "-sh", d], capture_output=True, text=True).stdout.split()[0] \
            if ok else "-"
        print(f"  {'ok  ' if ok else 'FAIL'} {local:36s} {sz:>7s}  {time.time()-t0:5.0f}s",
              flush=True)

    print("=== disk after ===", flush=True)
    print(subprocess.run(["df", "-h", "/"], capture_output=True, text=True).stdout, flush=True)
    print("A100_SETUP_DONE", flush=True)
