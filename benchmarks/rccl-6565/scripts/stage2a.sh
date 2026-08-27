#!/bin/bash
# Stage 2A: make the negative result load-bearing.
# The reporter's mechanism is specifically that the FIRST-ISSUED init copy
# (channel 0) is lost while a later channel's lands, and they state the
# concurrent multi-device init pattern is a necessary ingredient. So sweep the
# number of channels RCCL builds, which is the number of concurrent init
# copies, and force the other transport.
set -u
cd /work
for spec in \
  "15 ch1   NCCL_P2P_DISABLE=1 NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1" \
  "15 ch4   NCCL_P2P_DISABLE=1 NCCL_MIN_NCHANNELS=4" \
  "15 ch8   NCCL_P2P_DISABLE=1 NCCL_MIN_NCHANNELS=8" \
  "15 ch16  NCCL_P2P_DISABLE=1 NCCL_MIN_NCHANNELS=16" \
  "15 shmoff NCCL_SHM_DISABLE=1" \
  ; do
  /work/run6565b.sh $spec
done
echo "STAGE2A_DONE"
