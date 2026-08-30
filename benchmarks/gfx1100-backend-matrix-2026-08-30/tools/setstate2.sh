#!/bin/bash
# setstate2.sh <tua:stock|patched> <cppd:stock|p45916>
# Puts the container's three attention files into a named state and asserts it.
#   tua  = vllm/v1/attention/ops/triton_unified_attention.py   (TRITON_ATTN path)
#   cppd = vllm/v1/attention/ops/chunked_prefill_paged_decode.py (ROCM_ATTN path)
# triton_attn.py always goes back to the image's own bytes: the 08-29 campaign
# left vllm#45450 in it and this experiment must not carry that.
set -euo pipefail
TUA="$1"; CPPD="$2"
D=/data/rccl-build/bench0830c
SP=/opt/python/lib/python3.14/site-packages/vllm
C=vllm-027

M_TUA_STOCK=49fab3b643bf5a88eb65303ce377996b     # image, ROCm/vllm@f46a9dfe2
M_TUA_52684=f1d7a7e3c6656303fa63b6a4c1b8aef5     # + vllm#52684
M_TATTN_STOCK=f0a1379d724c870fa2703330524100f9
M_CPPD_STOCK=86f68d47c7bdc390ced4c6d0c18025fa    # image, 493 lines
M_CPPD_45916=84c6d4f9b2dfe2714b3a8f43ee832b02    # + vllm#45916, 1083 lines

case "$TUA" in
  stock)   SRC_TUA=$D/stock/triton_unified_attention.py;   W_TUA=$M_TUA_STOCK ;;
  patched) SRC_TUA=$D/patched/triton_unified_attention.py; W_TUA=$M_TUA_52684 ;;
  *) echo "tua must be stock|patched"; exit 2 ;;
esac
case "$CPPD" in
  stock)  SRC_CPPD=$D/stock/chunked_prefill_paged_decode.py; W_CPPD=$M_CPPD_STOCK ;;
  p45916) SRC_CPPD=$D/p45916/chunked_prefill_paged_decode.py; W_CPPD=$M_CPPD_45916 ;;
  *) echo "cppd must be stock|p45916"; exit 2 ;;
esac

sudo docker cp "$SRC_TUA"  "$C:$SP/v1/attention/ops/triton_unified_attention.py"
sudo docker cp "$SRC_CPPD" "$C:$SP/v1/attention/ops/chunked_prefill_paged_decode.py"
sudo docker cp "$D/stock/triton_attn.py" "$C:$SP/v1/attention/backends/triton_attn.py"
sudo docker exec "$C" bash -c "find $SP -name '__pycache__' -type d -prune -exec rm -rf {} + || true"

echo "--- asserting tua=$TUA cppd=$CPPD ---"
sudo docker exec "$C" bash -c "
set -e
a=\$(md5sum $SP/v1/attention/ops/triton_unified_attention.py | cut -d' ' -f1)
b=\$(md5sum $SP/v1/attention/backends/triton_attn.py | cut -d' ' -f1)
c=\$(md5sum $SP/v1/attention/ops/chunked_prefill_paged_decode.py | cut -d' ' -f1)
echo \"  tua   \$a\"; echo \"  tattn \$b\"; echo \"  cppd  \$c\"
[ \"\$a\" = \"$W_TUA\" ]          || { echo 'FAIL tua md5'; exit 1; }
[ \"\$b\" = \"$M_TATTN_STOCK\" ]  || { echo 'FAIL triton_attn md5'; exit 1; }
[ \"\$c\" = \"$W_CPPD\" ]         || { echo 'FAIL cppd md5'; exit 1; }
n=\$(grep -c _select_query_block $SP/v1/attention/ops/triton_unified_attention.py || true)
k=\$(grep -c kernel_paged_attention_2d_splitkv $SP/v1/attention/ops/chunked_prefill_paged_decode.py || true)
echo \"  _select_query_block=\$n  splitkv_kernel=\$k\"
echo '  OK'
"
