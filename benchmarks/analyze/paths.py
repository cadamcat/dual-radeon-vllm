"""Where the campaign data lives.

One line, previously copied into four files and twice into one of them, which
is how `analyze.py` ended up rebinding `RESULTS` and re-importing `os` halfway
down itself. Overridable so a script can be pointed at another campaign:

    BENCH_RESULTS=../results-2026-08-24.jsonl python3 summarize.py
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("BENCH_RESULTS") or os.path.join(_HERE, "..", "results.jsonl")
