#!/bin/bash
# Both runners, same model and same speculative config, one fresh container per
# cell so an edited site-packages cannot leak between arms. Routing only -- no
# timing is taken and none should be read out of this.
#
# The DONE marker requires each arm to have actually constructed a runner. A
# first version checked only the container exit code, and the probe exits 0 on
# purpose so that a failed MRv2 start is recorded rather than crashing -- so both
# arms failed on an unmounted model path and the driver still printed DONE.
set -u
IMG=rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0
CTX="${1:-2048}"
SPEC="${2:-spec}"
MODEL="${3:-/data/incoming/Qwen3-8B}"
fail=0
for arm in v1 v2; do
  echo "########## arm=$arm spec=$SPEC ctx=$CTX  $(date -u +%H:%M:%S) ##########"
  sudo docker run --rm --name "mrv2-$arm-$SPEC" --device /dev/kfd --device /dev/dri \
    --group-add video --ipc host --shm-size 8g \
    -v /data/50603:/work -v /data:/data -w /work \
    --entrypoint bash "$IMG" -c "python -u /work/probe_mrv2_spec.py $arm $CTX $SPEC "$MODEL"" \
    > "/data/50603/mrv2-$arm-$SPEC.log" 2>&1
  rc=$?
  log="/data/50603/mrv2-$arm-$SPEC.log"
  echo "exit=$rc  $(date -u +%H:%M:%S)"
  grep -E "RUNNER_CONSTRUCTED|BUILDER_INIT|GUARD |ENGINE_FAILED|RECORDER_ERRORS" "$log" | head -14
  if [ "$rc" -ne 0 ]; then
    echo "  arm=$arm: container exited $rc"; fail=$((fail+1)); continue
  fi
  if grep -q "RUNNER_CONSTRUCTED: \[\]" "$log"; then
    echo "  arm=$arm: no model runner was constructed -- the cell measured nothing"
    fail=$((fail+1))
  fi
done
[ "$fail" -eq 0 ] || { echo "MRV2_SPEC_FAILED cells=$fail"; exit 1; }
echo MRV2_SPEC_DONE
