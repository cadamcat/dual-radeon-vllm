#!/bin/bash
# Stage 1 driver: RCCL version banner + three transport arms x 20 cold inits.
set -u
cd /work
echo "########## RCCL version banner (from a real comm init) ##########"
NCCL_P2P_DISABLE=1 NCCL_DEBUG=VERSION torchrun --nproc-per-node=2 /work/rccl_allgather_truth.py 2>&1 \
  | grep -i "NCCL version\|RCCL version\|rccl" | head -5
echo
echo "########## topology as RCCL sees it ##########"
NCCL_P2P_DISABLE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH torchrun --nproc-per-node=2 /work/rccl_allgather_truth.py 2>&1 \
  | grep -iE "via |Channel |isAllDirectP2p|P2P|SHM|Trees|Rings|nChannels" | head -25
echo
for arm in default p2pdisable prod; do
  echo "##########################################################"
  /work/run6565.sh 20 "$arm"
  echo
done
echo "STAGE1_DONE"
