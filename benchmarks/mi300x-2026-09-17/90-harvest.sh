#!/bin/bash
# Everything leaves the box before it is destroyed. Results are not a file on a
# machine that is about to stop existing.
set -euo pipefail
: "${BENCH_WORK:=/work}"
cd "$BENCH_WORK"
T="mi300x-harvest-$(date -u +%Y%m%dT%H%M%SZ).tgz"
tar czf "/tmp/$T" \
  PROVENANCE.json results.jsonl PROGRESS.txt volume.json \
  preflight.log scan.log trace.log fetch.log \
  hipgate3.out trace-join.json trace-probe.txt pa-probe.jsonl \
  rocm_C-*-kernels.tsv rocm_C-*-summary.txt scan-*.jsonl \
  serve-*.log trace-serve.log 2>/dev/null || true
ls -l "/tmp/$T"
echo "rows: $(grep -c . results.jsonl 2>/dev/null || echo 0)   configs complete: $(grep -c config_complete results.jsonl 2>/dev/null || echo 0)"
echo
echo "Copy it off, then DESTROY the droplet — stopping it keeps billing:"
echo "  scp root@<droplet-ip>:/tmp/$T ."
