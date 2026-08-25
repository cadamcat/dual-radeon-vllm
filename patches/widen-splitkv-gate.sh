#!/usr/bin/env bash
# widen-splitkv-gate.sh — make vllm#45916's split-KV decode kernel reachable on
# RDNA3.
#
# The PR ships kernel_paged_attention_2d_splitkv but gates it to on_gfx12x(), so
# on gfx1100 it is inert and decode falls back to the kernel that walks the whole
# sequence. Widening the gate to on_gfx1x() is the whole change; nothing else in
# the PR is touched.
#
# This is what the D8-27B-tp2 configuration of the 2026-08-24 campaign was
# measured on. It is a two-token edit rather than a diff because the container's
# pre-change backup predates it, so no recorded diff of it exists — the sed below
# is the change as it was made and as it can be checked.
#
#     ./widen-splitkv-gate.sh /opt/python/lib/python3.14/site-packages/vllm
#
set -euo pipefail
VLLM="${1:-}"
[ -n "$VLLM" ] && [ -d "$VLLM" ] || { echo "usage: $0 /path/to/site-packages/vllm" >&2; exit 2; }
F="$VLLM/v1/attention/ops/chunked_prefill_paged_decode.py"
[ -f "$F" ] || { echo "not found: $F" >&2; exit 2; }

# Two different states both show zero on_gfx12x: already widened, and a vLLM that
# never carried #45916 at all. Reporting success for the second would tell you the
# split-KV kernel is reachable when it is not present.
if ! grep -q 'use_splitkv_decode' "$F"; then
  echo "ERROR: this vLLM does not carry vllm#45916 — there is no split-KV decode" >&2
  echo "       path to widen. Apply the PR first; it is not vendored here." >&2
  exit 3
fi
before=$(grep -c 'on_gfx12x' "$F" || true)
echo "on_gfx12x occurrences before: ${before:-0}"
[ "${before:-0}" -eq 0 ] && { echo "already widened"; exit 0; }

cp -n "$F" "$F.pre-splitkv-gate"
sed -i 's/\bon_gfx12x\b/on_gfx1x/g' "$F"

after=$(grep -c 'on_gfx12x' "$F" || true)
echo "on_gfx12x occurrences after:  ${after:-0}"
[ "${after:-0}" -eq 0 ] || { echo "ERROR: some remain" >&2; exit 1; }
echo "backup at $F.pre-splitkv-gate"
echo
echo "verify at runtime: the profile of a long-context decode should contain"
echo "kernel_paged_attention_2d_splitkv and not kernel_paged_attention_2d."
