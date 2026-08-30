"""Setup for a CUDA VM in the 2026-08-30 round: vLLM, two checkpoints, two ladders.

Only what this round measures. gemma-4-12B is the spine's model, gemma-4-26B-A4B
is the second tier -- about 27 GB against the 2026-08-29 campaign's 153, because
setup is the part that does not survive a reclaim and four VMs were reclaimed
that day.

The ladders are cut here, by benchmarks/prompts/cut_prompts.py itself rather
than a re-implementation, from Gutenberg #1228 with each model's own tokenizer.
"""
import json, os, subprocess, sys, time

LOG = "/content/setup.log"
D = "/content/work"

MODELS = [
    ("google/gemma-4-12B-it-qat-w4a16-ct", "gemma-4-12B-it-qat-w4a16-ct"),
    ("cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit", "gemma-4-26B-A4B-AWQ"),
]


def complete(d):
    """Every shard the safetensors index names is present, or the single file is."""
    if not os.path.exists(d + "/config.json"):
        return False
    idx = f"{d}/model.safetensors.index.json"
    if os.path.exists(idx):
        try:
            names = set(json.load(open(idx))["weight_map"].values())
        except Exception:
            return False
        return all(os.path.exists(f"{d}/{n}") for n in names)
    return os.path.exists(f"{d}/model.safetensors")


def sh(cmd, timeout=7200):
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    with open(LOG, "a") as f:
        f.write(f"\n$ {cmd}\n[{time.time()-t0:.0f}s rc={r.returncode}]\n"
                f"{r.stdout[-2000:]}\n{r.stderr[-2000:]}\n")
    return r


if __name__ == "__main__":
    open(LOG, "w").close()
    os.makedirs(D, exist_ok=True)
    os.makedirs("/content/models", exist_ok=True)
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout, flush=True)

    print("=== installing vllm ===", flush=True)
    t0 = time.time()
    sh("pip install -q -U vllm 2>&1 | tail -5", timeout=3600)
    # Colab preinstalls a torchaudio for CUDA 12.8; vLLM's torch is 13.0 and
    # transformers' is_torchaudio_available() raises on the version check
    # rather than skipping. Nothing here needs audio.
    sh("pip uninstall -y -q torchaudio 2>&1 | tail -2", timeout=600)
    for mod in ("vllm", "torch", "transformers"):
        v = subprocess.run([sys.executable, "-c", f"import {mod}; print({mod}.__version__)"],
                           capture_output=True, text=True)
        print(f"{mod}: {v.stdout.strip() or v.stderr.strip()[-200:]}", flush=True)
    print(f"install {time.time()-t0:.0f}s", flush=True)

    print("=== checkpoints ===", flush=True)
    for repo, local in MODELS:
        d = f"/content/models/{local}"
        # `config.json` is written early by `hf download`, so its presence says
        # the download started, not that it finished. On 2026-08-30 a killed
        # download left config.json and one of four shards; setup skipped it as
        # "have", and the engine died on
        #   FileNotFoundError: Weight files referenced in index but missing
        # after the run had already spent a full engine start on it. Check what
        # the index actually references instead, and re-fetch what is missing.
        if complete(d):
            print(f"  have  {local}", flush=True); continue
        t0 = time.time()
        sh(f"hf download {repo} --local-dir {d} --max-workers 8")
        ok = complete(d)
        sz = subprocess.run(["du", "-sh", d], capture_output=True, text=True).stdout.split()[0] if ok else "-"
        print(f"  {'ok  ' if ok else 'FAIL'} {local:34s} {sz:>7s}  {time.time()-t0:5.0f}s", flush=True)

    print("=== ladders ===", flush=True)
    r = subprocess.run([sys.executable, f"{D}/cut_prompts.py", "--models-dir", "/content/models",
                        "--only", "gemma,gemma26b", "--out", D],
                       capture_output=True, text=True, cwd=D)
    print(r.stdout[-2500:], flush=True)
    if r.returncode != 0:
        print("LADDER STDERR:", r.stderr[-1500:], flush=True)
    print(subprocess.run(["df", "-h", "/"], capture_output=True, text=True).stdout, flush=True)
    print("CUDA_SETUP_DONE", flush=True)
