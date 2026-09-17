#!/bin/bash
# Discovery, provenance and the model download. Run first, inside the container.
# Nothing here loads a model, so it overlaps the 103 GiB download by design.
set -euo pipefail
: "${BENCH_WORK:=/work}" "${BENCH_MODELS:=/models}"
mkdir -p "$BENCH_WORK" "$BENCH_MODELS"
cd "$BENCH_WORK"
exec > >(tee -a "$BENCH_WORK/preflight.log") 2>&1
echo "== 00-preflight $(date -Is)"

# The probes and the scanners come from the repository, so the tools this row
# used are identified by a commit rather than by whatever was uploaded.
[ -d repo ] || git clone --quiet --depth 1 https://github.com/cadamcat/dual-radeon-vllm repo
REPO_HEAD=$(git -C repo rev-parse HEAD)

# Discovery, not assumption: this image is not the one the gfx1100 scans ran in.
PY_SP=$(python3 -c 'import site;print(site.getsitepackages()[0])')
ROCM_C=$(ls "$PY_SP"/vllm/_rocm_C*.so 2>/dev/null | head -1 || true)
LLVM_BIN=$(dirname "$(command -v llvm-readelf || echo /opt/rocm/llvm/bin/llvm-readelf)")
VLLM_V=$(python3 -c 'import vllm;print(vllm.__version__)')
VLLM_COMMIT=$(python3 - <<'PY'
try:
    import vllm, vllm.version as v
    print(getattr(v, "__commit__", "") or getattr(vllm, "__commit__", "") or "unknown")
except Exception as e:
    print("unknown")
PY
)
TORCH_V=$(python3 -c 'import torch;print(torch.__version__)')
ARCH=$(python3 -c 'import torch;print(torch.cuda.get_device_properties(0).gcnArchName)' 2>/dev/null || rocminfo | awk '/gfx/{print $2; exit}')
python3 - "$REPO_HEAD" "$PY_SP" "$ROCM_C" "$LLVM_BIN" "$VLLM_V" "$VLLM_COMMIT" "$TORCH_V" "$ARCH" <<'PY'
import json, os, subprocess, sys, time
head, sp, rocm_c, llvm, vllm_v, vllm_commit, torch_v, arch = sys.argv[1:9]
def sh(cmd):
    try: return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception as ex: return f"<{ex!r}>"
prov = {
  "purpose": "one MI300X, TP=1: declaration versus launch on CDNA, the hostcall declaration count in this image, the sixteen-rung ladder and the batch dimension",
  "date": time.strftime("%Y-%m-%d"), "host": "AMD Developer Cloud GPU droplet, 1x MI300X 192 GB",
  "repo_commit": head, "site_packages": sp, "rocm_C": rocm_c, "llvm_bin": llvm,
  "vllm": vllm_v, "vllm_commit": vllm_commit, "torch": torch_v, "gcnArchName": arch,
  "rocm_version": sh("cat /opt/rocm/.info/version 2>/dev/null || hipconfig --version"),
  "driver": sh("cat /sys/module/amdgpu/version 2>/dev/null"), "kernel": sh("uname -a"),
  "rocm_smi": sh("rocm-smi --showproductname --showdriverversion --csv 2>/dev/null | head -20"),
  "disk": sh("df -h /models /work | tail -3"),
  "container_image": os.environ.get("BENCH_IMAGE", "<set BENCH_IMAGE to the image reference and digest>"),
}
json.dump(prov, open(os.environ.get("BENCH_WORK", "/work") + "/PROVENANCE.json", "w"), indent=1)
print(json.dumps(prov, indent=1))
PY

# hipgate3: the 57-line probe, compiled for this architecture. On a cloud host
# AtomicOps are present, so both kernels are expected to pass; the row that
# matters is the attribute and the fact that the probe builds and runs here.
hipcc -O1 repo/diagnose/hipgate3.cpp -o hipgate3
./hipgate3 | tee "$BENCH_WORK/hipgate3.out" || echo "hipgate3 exit $?"
python3 - <<'PY' | tee -a "$BENCH_WORK/hipgate3.out"
import torch
p = torch.cuda.get_device_properties(0)
print("gcnArchName", p.gcnArchName, "| total GiB", round(p.total_memory/2**30, 1))
PY

# Weights, with the pinned revisions. Runs in the background: the scan and the
# probe do not need them, and the GPU is idle either way.
# fetch_models.py in the repository is a Modal app; fetch_models_plain.py is
# the same list, the same pinned revisions and the same size assertion, run here.
[ -f volume.json ] || cp repo/benchmarks/modal-2026-09-02/volume.json volume.json
nohup python3 fetch_models_plain.py > fetch.log 2>&1 &
echo "fetch_models.py started (pid $!), log $BENCH_WORK/fetch.log"
echo "== preflight done; next: 05-scan.sh while the download runs"
