#!/bin/bash
# Stage 3: the same eight arms and the same cold-init counts as stages 1 and 2A,
# 135 in total, re-run through the cross-rank variant. This is what replaces the
# rank-0 caveat on the 135/135 with a result.
set -u
cd /work
fail=0
echo "########## stack, and which librccl is loaded ##########"
NCCL_P2P_DISABLE=1 NCCL_DEBUG=VERSION torchrun --nproc-per-node=2 /work/rccl_allgather_allranks.py 2>&1 \
  | grep -i "NCCL version\|RCCL version\|rccl" | head -5
python - <<'PY'
import glob, hashlib, os
for p in glob.glob('/opt/python/lib/python3.*/site-packages/_rocm_sdk_*/lib/librccl.so.1*'):
    h = hashlib.md5(open(p,'rb').read()).hexdigest() if os.path.isfile(p) and not os.path.islink(p) else 'symlink->'+os.readlink(p)
    print(f"{p}  {h}")
PY
echo
for spec in \
  "20 default" \
  "20 p2pdisable NCCL_P2P_DISABLE=1" \
  "20 prod       NCCL_P2P_DISABLE=1 HSA_ENABLE_SDMA=0" \
  "15 ch1        NCCL_P2P_DISABLE=1 NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1" \
  "15 ch4        NCCL_P2P_DISABLE=1 NCCL_MIN_NCHANNELS=4" \
  "15 ch8        NCCL_P2P_DISABLE=1 NCCL_MIN_NCHANNELS=8" \
  "15 ch16       NCCL_P2P_DISABLE=1 NCCL_MIN_NCHANNELS=16" \
  "15 shmoff     NCCL_SHM_DISABLE=1" \
  ; do
  /work/run6565_allranks.sh $spec || fail=$((fail+1))
  echo
done
# the DONE marker a poller greps for must not appear when an arm failed
[ "$fail" -eq 0 ] || { echo "STAGE3_ALLRANKS_FAILED arms=$fail"; exit 1; }
echo STAGE3_ALLRANKS_DONE
