#!/bin/bash
# One cell per invocation. The caller runs a fresh container per cell so the
# source patch cannot leak from a widened cell into the next stock cell.
set -u
python -u /work/probe_stage3.py "$1" "$2" /work/stage3-027.jsonl
