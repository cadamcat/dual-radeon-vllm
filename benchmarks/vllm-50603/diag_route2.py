"""Instrument the dispatch IN THE WORKER, by patching its source on disk.

TP=2 runs attention in spawned workers, so monkeypatching module attributes in
the parent is blind: that is why the first attempt reported ck_calls=0 for both
arms. Rewriting the .py before any worker starts does reach them, because each
worker imports the file fresh.

Writes one line per distinct (gqa, head, block, window, verdict) to
/work/route-<arm>.txt, from inside whichever process actually runs attention.
"""
import pathlib, sys

ARM = sys.argv[1]
import vllm.platforms.rocm as rp
import vllm.v1.attention.ops.chunked_prefill_paged_decode as CPPD

from vllm import LLM, SamplingParams


def main():
    # 1. the candidate gate change
    rpath = pathlib.Path(rp.__file__); rsrc = rpath.read_text()
    assert rsrc.count("gqa_ratio >= 3") == 1
    if ARM == "widened":
        rpath.write_text(rsrc.replace("gqa_ratio >= 3", "gqa_ratio >= 1"))

    # 2. record what the dispatch decides, from inside the worker
    dpath = pathlib.Path(CPPD.__file__); dsrc = dpath.read_text()
    ANCHOR = "    if use_custom:\n"
    assert dsrc.count(ANCHOR) == 1, dsrc.count(ANCHOR)
    probe = (
        "    try:\n"
        "        import os as _os\n"
        "        _k = (num_queries_per_kv, head_size, block_size, sliding_window,\n"
        "              bool(use_custom), bool(has_native_layout), bool(is_pow2))\n"
        "        _f = '/work/route-" + ARM + ".txt'\n"
        "        _seen = getattr(chunked_prefill_paged_decode, '_seen', None)\n"
        "        if _seen is None:\n"
        "            _seen = set(); chunked_prefill_paged_decode._seen = _seen\n"
        "        if _k not in _seen:\n"
        "            _seen.add(_k)\n"
        "            with open(_f, 'a') as _fh:\n"
        "                _fh.write('pid=%d %s\\n' % (_os.getpid(), _k))\n"
        "    except Exception as _e:\n"
        "        pass\n"
    )
    dpath.write_text(dsrc.replace(ANCHOR, probe + ANCHOR))
    import ast; ast.parse(dpath.read_text())
    print(f"instrumented for arm={ARM}", flush=True)

    llm = LLM(model="/data/incoming/gemma-3-27b-it-w4a16", tensor_parallel_size=2,
              max_model_len=2048, gpu_memory_utilization=0.92)
    llm.generate(["hello " * 400], SamplingParams(temperature=0, max_tokens=8))
    print("DIAG2_DONE", flush=True)


if __name__ == "__main__":
    main()
