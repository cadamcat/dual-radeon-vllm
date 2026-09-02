
"""t4d: the 32 000 rung of gemma-4-12B on a T4, both rounds, one engine.

t4c measured round 1 (ttft 331.4023 s, decode 8.8631 tok/s) and died before
round 2. Three sessions have now died before this rung. The nineteen rungs
below it are seeded into results.jsonl, so the runner's checkpoint skips them
and the ladder starts at 32 000 -- but the engine still starts once and warms
up once, so both rounds are measured inside a single engine instance, which is
what "two rounds" means everywhere else in this repository.
"""
import hashlib, json, os, subprocess, sys, time

D = "/content/work"
LOG = f"{D}/setup.log"
MODEL_REPO = "google/gemma-4-12B-it-qat-w4a16-ct"
MODEL_DIR = "/content/models/gemma-4-12B-it-qat-w4a16-ct"
PATCHED = ["vllm/v1/attention/backends/flex_attention.py",
           "vllm/v1/attention/ops/triton_unified_attention.py"]


def sh(cmd, timeout=7200):
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    with open(LOG, "a") as f:
        f.write(f"\n$ {cmd}\n[{time.time()-t0:.0f}s rc={r.returncode}]\n"
                f"{r.stdout[-3000:]}\n{r.stderr[-3000:]}\n")
    print(f"[chain] {cmd[:70]} rc={r.returncode} {time.time()-t0:.0f}s", flush=True)
    return r


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def complete(d):
    """Every shard the index names is present. `config.json` alone means the
    download STARTED -- on 2026-08-30 that cost a full engine start."""
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


prov = {"session": "t4d", "purpose": "gemma-4-12B @32000, rounds 1 and 2, T4"}

# --- 1. the stack, PINNED ---------------------------------------------------
# t4c ran vllm 0.28.0 / torch 2.13.0+cu130 / cuda 13.0. `pip install -U vllm`
# would silently take whatever is newest today, and a row measured on a
# different vLLM is not the second round of a row measured on 0.28.0.
print("=== installing vllm==0.28.0 ===", flush=True)
sh("pip install -q vllm==0.28.0 2>&1 | tail -5", timeout=3600)
sh("pip uninstall -y -q torchaudio 2>&1 | tail -2", timeout=600)
r = subprocess.run([sys.executable, "-c",
                    "import vllm, os; print(vllm.__version__); "
                    "print(os.path.dirname(os.path.dirname(vllm.__file__)))"],
                   capture_output=True, text=True)
if r.returncode != 0:
    print("[chain] FATAL vllm import failed:", r.stderr[-1500:], flush=True)
    sys.exit(1)
vllm_ver, site = r.stdout.strip().splitlines()
prov["vllm"] = vllm_ver
print(f"[chain] vllm {vllm_ver} at {site}", flush=True)
if vllm_ver != "0.28.0":
    print(f"[chain] FATAL wanted 0.28.0, got {vllm_ver}", flush=True)
    sys.exit(1)

# --- 2. vllm#39018, asserted both ways --------------------------------------
# Without it the engine dies at kernel load: 98304 bytes of shared memory asked
# against Turing's 65536. Every row this session writes carries the patch, so
# the patch has to be proved present, not assumed.
before = {p: md5(f"{site}/{p}") for p in PATCHED}
r = sh(f"cd {site} && patch -p1 --forward --batch < {D}/pr39018.diff")
after = {p: md5(f"{site}/{p}") for p in PATCHED}
prov["md5_before"], prov["md5_after"] = before, after
changed = [p for p in PATCHED if before[p] != after[p]]
src = open(f"{site}/vllm/v1/attention/ops/triton_unified_attention.py").read()
marker = "head_size_padded >= 512" in src
prov["patch_rc"] = r.returncode
prov["files_changed"] = changed
prov["marker_present"] = marker
print(f"[chain] patch rc={r.returncode} changed={changed} marker={marker}", flush=True)
if len(changed) != 2 or not marker:
    print("[chain] FATAL vllm#39018 did not apply", flush=True)
    json.dump(prov, open(f"{D}/PROVENANCE.json", "w"), indent=1)
    sys.exit(1)

# --- 3. the checkpoint ------------------------------------------------------
print("=== checkpoint ===", flush=True)
if complete(MODEL_DIR):
    print("[chain] have the checkpoint", flush=True)
else:
    t0 = time.time()
    sh(f"hf download {MODEL_REPO} --local-dir {MODEL_DIR} --max-workers 8")
    if not complete(MODEL_DIR):
        print("[chain] FATAL checkpoint incomplete", flush=True)
        json.dump(prov, open(f"{D}/PROVENANCE.json", "w"), indent=1)
        sys.exit(1)
    print(f"[chain] checkpoint ok in {time.time()-t0:.0f}s", flush=True)
prov["checkpoint_ok"] = True
json.dump(prov, open(f"{D}/PROVENANCE.json", "w"), indent=1)

# --- 4. the run -------------------------------------------------------------
print("=== run ===", flush=True)
env = {**os.environ, "BENCH_MACHINE": "T4", "BENCH_CFGS": "G12"}
print("T4_PROVISION_DONE", json.dumps(prov), flush=True)
