#!/bin/bash
# Stage 1 for ROCm/legacy-rocm-build#6565 on the dual RX 7900 XT VFIO guest.
# Runs BoJl4apa's exact rccl_allgather_truth.py (md5 bffbc297...) N times with a
# COLD communicator init each time: the defect they report is init-timing
# dependent, so a single pass is not evidence of absence.
# argv: N [arm]
set -u
N="${1:-1}"
ARM="${2:-p2pdisable}"
cd /work

case "$ARM" in
  default)     ENVV=() ;;
  p2pdisable)  ENVV=(NCCL_P2P_DISABLE=1) ;;
  prod)        ENVV=(NCCL_P2P_DISABLE=1 HSA_ENABLE_SDMA=0) ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac

echo "=== arm=$ARM  env: ${ENVV[*]:-<none>}  N=$N ==="
echo "--- stack ---"
env "${ENVV[@]}" NCCL_DEBUG=VERSION python -c "
import torch, os
print('torch', torch.__version__)
print('hip', torch.version.hip)
print('devices', torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
" 2>&1 | sed 's/^/    /'
echo "--- librccl in the load path (proves stock, not our NDEBUG rebuild) ---"
python - <<'PY' 2>&1 | sed 's/^/    /'
import glob, hashlib, os
for p in glob.glob('/opt/python/lib/python3.*/site-packages/_rocm_sdk_*/lib/librccl.so.1*'):
    try:
        h = hashlib.md5(open(p,'rb').read()).hexdigest() if os.path.isfile(p) and not os.path.islink(p) else 'symlink->'+os.readlink(p)
    except Exception as e:
        h = str(e)
    print(f"{p}  {h}")
PY

pass_n=0; fail_n=0
for i in $(seq 1 "$N"); do
  out=$(env "${ENVV[@]}" torchrun --nproc-per-node=2 /work/rccl_allgather_truth.py 2>&1)
  verdict=$(echo "$out" | grep -o 'ALL CORRECT\|[0-9]* FAILING CASES' | tail -1)
  if [ "$verdict" = "ALL CORRECT" ]; then
    pass_n=$((pass_n+1)); echo "run $i/$N: ALL CORRECT"
  else
    fail_n=$((fail_n+1)); echo "run $i/$N: >>> $verdict <<<"
    echo "$out" | sed 's/^/      /'
  fi
done
echo "=== arm=$ARM RESULT pass=$pass_n fail=$fail_n of $N cold inits ==="
