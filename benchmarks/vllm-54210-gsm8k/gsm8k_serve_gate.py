#!/usr/bin/env python3
"""Serve one arm of vllm#54210's gate, with the dispatch's verdict recorded.

    python3 gsm8k_serve_gate.py stock|widened /work/route-<arm>.txt

The patch-and-record block is `benchmarks/vllm-50603/probe_stage3.py`'s, kept
identical on purpose: that is the mechanism the PR's own end-to-end numbers were
taken with, so an lm-eval score taken through this one is comparable to them.
What differs is the last step -- Stage 3 built an offline `LLM()` and timed it,
this execs `vllm serve` so lm_eval can drive it over HTTP from outside the
container. lm_eval is deliberately NOT installed in the container: this box's
comparison rests on the container's file md5s, and pip resolving lm_eval's
dependencies could move torch or vllm underneath them.

Both assertions matter and neither is decorative. The first says the anchor is
where it was; the second says the edit took, or in the stock arm that the tree
is pristine -- a stock arm measured on an accidentally-patched tree would read
as "no difference" and be reported as such.
"""
import os
import pathlib
import sys

ARM = sys.argv[1]
ROUTE = sys.argv[2]
assert ARM in ("stock", "widened"), ARM

import vllm.platforms.rocm as rp                                    # noqa: E402

src_path = pathlib.Path(rp.__file__)
src = src_path.read_text()
# "gqa_ratio >= 3" is the gfx11 branch and occurs once; "gqa_ratio >= 1" already
# occurs once, in the CDNA branch, which is the point of the change.
OLD, NEW = "gqa_ratio >= 3", "gqa_ratio >= 1"
assert src.count(OLD) == 1, f"gfx11 anchor x{src.count(OLD)}, expected 1"
assert src.count(NEW) == 1, f"CDNA baseline x{src.count(NEW)}, expected 1"
if ARM == "widened":
    src_path.write_text(src.replace(OLD, NEW))

after = src_path.read_text()
if ARM == "widened":
    assert after.count(OLD) == 0 and after.count(NEW) == 2, "patch did not take"
else:
    assert after.count(OLD) == 1 and after.count(NEW) == 1, "tree is not pristine"

# Under TP=2 attention runs in spawned workers, so a counter in this process
# sees nothing; rewriting the .py does reach them.
import vllm.v1.attention.ops.chunked_prefill_paged_decode as CPPD   # noqa: E402

dpath = pathlib.Path(CPPD.__file__)
dsrc = dpath.read_text()
ANCHOR = "    if use_custom:\n"
assert dsrc.count(ANCHOR) == 1
probe_src = (
    "    try:\n"
    "        import os as _os\n"
    "        _k = (num_queries_per_kv, head_size, block_size, sliding_window,\n"
    "              bool(use_custom))\n"
    "        _s = getattr(chunked_prefill_paged_decode, '_seen', None)\n"
    "        if _s is None:\n"
    "            _s = set(); chunked_prefill_paged_decode._seen = _s\n"
    "        if _k not in _s:\n"
    "            _s.add(_k)\n"
    "            with open('" + ROUTE + "', 'a') as _fh:\n"
    "                _fh.write('pid=%d %s\\n' % (_os.getpid(), _k))\n"
    "    except Exception:\n"
    "        pass\n"
)
if "_seen" not in dsrc:
    dpath.write_text(dsrc.replace(ANCHOR, probe_src + ANCHOR))

print(f"ARM={ARM} patched={ARM == 'widened'} route={ROUTE}", flush=True)
os.execvp("vllm", ["vllm", "serve", sys.argv[3],
                   "--tensor-parallel-size", "2",
                   "--gpu-memory-utilization", "0.92",
                   "--max-model-len", "4096",
                   "--max-num-seqs", "16",
                   "--port", "8000"])
