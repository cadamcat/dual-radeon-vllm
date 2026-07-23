#!/usr/bin/env bash
# check-symbols.sh — does this librccl export everything PyTorch needs?
#
# Usage: ./check-symbols.sh /path/to/librccl.so.1.0 [required-symbols.txt]
#
# Run this BEFORE deploying a rebuilt library. A missing symbol surfaces at
# `import torch` as an opaque loader error, long after you have spent an hour
# building — this turns that into a two-second check.

set -uo pipefail
LIB="${1:-}"
LIST="${2:-$(dirname "$0")/required-symbols-torch.txt}"

[ -z "$LIB" ] && { echo "usage: $0 /path/to/librccl.so.1.0 [symbol-list.txt]"; exit 2; }
[ -f "$LIB" ]  || { echo "not a file: $LIB"; exit 2; }
[ -f "$LIST" ] || { echo "symbol list not found: $LIST"; exit 2; }

OD=$(command -v llvm-objdump || command -v objdump || true)
[ -z "$OD" ] && { echo "need llvm-objdump or objdump on PATH"; exit 2; }

echo "library: $LIB"
echo "list:    $LIST"

exported=$("$OD" -T "$LIB" 2>/dev/null | grep -oE 'nccl[A-Za-z0-9_]*' | sort -u)
if [ -z "$exported" ]; then
  echo "no nccl* symbols found at all — wrong file, or not a dynamic library?"
  exit 2
fi

# Symbols that appear C++-mangled (name followed by an encoded signature), so an
# exact match will not find them. Kept as an explicit allowlist: a blanket prefix
# match would wrongly accept e.g. ncclReduce because ncclReduceScatter exists.
MANGLED_OK="ncclCommDump"

missing=0; ok=0
while read -r sym; do
  case "$sym" in ''|\#*) continue ;; esac
  if printf '%s\n' "$exported" | grep -qx "$sym"; then
    ok=$((ok+1))
  elif printf ' %s ' "$MANGLED_OK" | grep -q " $sym " && \
       printf '%s\n' "$exported" | grep -q "^${sym}[A-Za-z0-9_]"; then
    ok=$((ok+1))
    echo "  (mangled ok: $sym)"
  else
    echo "  MISSING: $sym"
    missing=$((missing+1))
  fi
done < "$LIST"

echo
echo "exported nccl* symbols in library: $(printf '%s\n' "$exported" | wc -l | tr -d ' ')"
echo "required present: $ok   missing: $missing"
if [ "$missing" -eq 0 ]; then
  echo "RESULT: PASS — PyTorch should be able to load this library."
  exit 0
else
  echo "RESULT: FAIL — torch will refuse to load this build."
  echo "        The source you built is older than the runtime expects."
  exit 1
fi
