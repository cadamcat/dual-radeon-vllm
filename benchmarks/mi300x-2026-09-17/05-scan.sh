#!/bin/bash
# The declaration side, item 2, and the static list item 1 joins against.
# CPU only; run it while the weights download.
set -euo pipefail
: "${BENCH_WORK:=/work}"
cd "$BENCH_WORK"
exec > >(tee -a "$BENCH_WORK/scan.log") 2>&1
echo "== 05-scan $(date -Is)"
PY_SP=$(python3 -c 'import site;print(site.getsitepackages()[0])')
LLVM_BIN=$(dirname "$(command -v llvm-readelf || echo /opt/rocm/llvm/bin/llvm-readelf)")
ARCH=${BENCH_ARCH:-gfx942}

# 1. Every shared object with device code for this architecture, and how many of
#    its kernels declare the buffer. --arch is the scanner's own option; the
#    roots are this image's, discovered above rather than assumed.
python3 repo/benchmarks/hostcall-abi-2026-09-04/scan_hostcall.py \
  --arch "$ARCH" --out "$BENCH_WORK/scan-$ARCH.jsonl" \
  --roots "$PY_SP" /opt/rocm/lib /usr/lib/x86_64-linux-gnu \
  --workdir "$BENCH_WORK/scanwork" || echo "scan_hostcall exit $? (read the message above; do not treat a missing tool as zero declarations)"

# 2. The engine's own device image, kernel by kernel: the list item 1 joins the
#    launch trace against. On gfx1100 this is where the 348 declaring kernels
#    and the 256 CDNA template instantiations came from.
ROCM_C=$(ls "$PY_SP"/vllm/_rocm_C*.so | head -1)
python3 list_hostcall_kernels_arch.py --so "$ROCM_C" --arch "$ARCH" \
  --llvm-bin "$LLVM_BIN" --out "$BENCH_WORK/rocm_C-$ARCH-kernels.tsv" \
  --family paged_attention | tee "$BENCH_WORK/rocm_C-$ARCH-summary.txt"
echo "== scan done"
