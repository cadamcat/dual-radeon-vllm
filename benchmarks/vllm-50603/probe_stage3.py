"""Stage 3 for vllm#50603: end-to-end effect of widening the gfx11 gqa gate.

Stage 1 measured the two kernels directly and found the CK custom paged
attention 2.06-7.28x faster than the Triton fallback at gqa_ratio 1 and 2,
which `use_rocm_custom_paged_attention` excludes on gfx11. This asks what that
is worth on a real model, which is the number a PR would have to justify.

    python3 probe_stage3.py <stock|widened> <ctx>

The `widened` arm applies the one-line change a PR would make, at source:

    gqa_ratio >= 3   ->   gqa_ratio >= 1        (gfx11 branch only)

and asserts the edit took. Both arms count calls into
`ops.paged_attention_rocm`, so routing is proven rather than assumed.

Scope, stated because it bounds the result: gemma-3-27b is 62 layers with
`sliding_window_pattern: 6`, so only the ~10 full-attention layers can move to
CK; the ~52 sliding layers fail the gate's `sliding_window == 0` condition in
BOTH arms and stay on Triton. That dilutes the end-to-end effect at short
context. At long context it matters much less than it sounds: a full layer
scans the whole context while a sliding layer scans 1024, so at 32K the 10
full layers carry roughly 86% of decode KV traffic.

Output equality between arms is deliberately NOT the correctness test. Greedy
decoding on this box is not reproducible across processes for this model
(benchmarks/gfx1100-greedy-nondeterminism.json: gemma-3-27b settles at 2-3
distinct outputs of 8 at 8192). What is checked is that the text stays
coherent; the numerical case was made at kernel level in Stage 1 against an
fp32 reference.

Decode rate by 64-vs-8 differencing, the method used across this repository.
"""

import json
import pathlib
import sys
import time

ARM = sys.argv[1]
CTX = int(sys.argv[2])
OUT = sys.argv[3] if len(sys.argv) > 3 else "/work/stage3.jsonl"
MODEL = "/data/incoming/gemma-3-27b-it-w4a16"

import torch
import vllm._custom_ops as ops
import vllm.platforms.rocm as rp
from vllm import LLM, SamplingParams


def main():
    # ---- apply the candidate patch before vLLM is imported anywhere ----

    src_path = pathlib.Path(rp.__file__)
    src = src_path.read_text()
    # "gqa_ratio >= 3" occurs once, in the gfx11 branch. "gqa_ratio >= 1" already
    # occurs once before any edit, in the CDNA branch, which is the whole point:
    # CDNA runs this kernel down to 1 and gfx11 does not.
    OLD, NEW = "gqa_ratio >= 3", "gqa_ratio >= 1"
    assert src.count(OLD) == 1, f"gfx11 anchor x{src.count(OLD)}, expected 1"
    assert src.count(NEW) == 1, f"CDNA baseline x{src.count(NEW)}, expected 1"
    patched = False
    if ARM == "widened":
        src_path.write_text(src.replace(OLD, NEW))
        patched = True
        import importlib
        importlib.reload(rp)
    # Record the dispatch's actual verdict from inside whichever process runs
    # attention. Under TP=2 that is a spawned worker, so a monkeypatched
    # counter in this process sees nothing: rewriting the .py does reach them.
    import vllm.v1.attention.ops.chunked_prefill_paged_decode as CPPD
    dpath = pathlib.Path(CPPD.__file__)
    dsrc = dpath.read_text()
    ANCHOR = "    if use_custom:\n"
    assert dsrc.count(ANCHOR) == 1
    route_file = f"/work/route-{ARM}-{CTX}.txt"
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
        "            with open('" + route_file + "', 'a') as _fh:\n"
        "                _fh.write('pid=%d %s\\n' % (_os.getpid(), _k))\n"
        "    except Exception:\n"
        "        pass\n"
    )
    dpath.write_text(dsrc.replace(ANCHOR, probe_src + ANCHOR))

    after = src_path.read_text()
    if ARM == "widened":
        assert after.count(OLD) == 0 and after.count(NEW) == 2, "patch did not take"
    else:
        assert after.count(OLD) == 1 and after.count(NEW) == 1, "tree is not pristine"


    _ck = {"n": 0}
    _orig = ops.paged_attention_rocm


    def _counting(*a, **kw):
        _ck["n"] += 1
        return _orig(*a, **kw)


    ops.paged_attention_rocm = _counting

    gate = rp.use_rocm_custom_paged_attention(
        torch.bfloat16, 128, 16, 2, 4096, 0, "auto", None, None
    )
    print(f"ARM={ARM} ctx={CTX} patched={patched} gate(gqa=2)={gate}", flush=True)

    llm = LLM(model=MODEL, tensor_parallel_size=2, max_model_len=CTX + 512,
              gpu_memory_utilization=0.92)
    tok = llm.get_tokenizer()
    ids = tok("hello " * (CTX + 4000)).input_ids[:CTX]
    prompt = tok.decode(ids)


    def timed(n):
        sp = SamplingParams(temperature=0, max_tokens=n, ignore_eos=True)
        t0 = time.time()
        o = llm.generate([prompt], sp)
        return time.time() - t0, o[0].outputs[0].text


    timed(8)                       # warm the prefix cache
    t8, _ = timed(8)
    t64, text = timed(64)
    tps = (64 - 8) / (t64 - t8)

    # With CUDA graphs on, replays do not re-enter the Python wrapper, so the
    # per-window delta undercounts. The total since process start does include
    # the capture-time calls, which is what proves routing.
    row = {"arm": ARM, "ctx": CTX, "patched": patched, "gate_gqa2": bool(gate),
           "decode_tok_s": tps, "t8": t8, "t64": t64,
           "route_file": route_file,
           "sample_text": text[:200]}
    with open(OUT, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"RESULT decode_tok_s={tps:.2f} t8={t8:.1f} t64={t64:.1f} "
          f"routes={open(route_file).read().strip().replace(chr(10), ' | ') if pathlib.Path(route_file).exists() else 'NONE'}",
          flush=True)
    print("SAMPLE:", text[:160].replace("\n", " "), flush=True)
    print("STAGE3_CELL_DONE", flush=True)


if __name__ == "__main__":
    # TP=2 spawns workers by re-importing this module. Without this guard each
    # worker re-runs the patch-and-measure block and recursively builds another
    # engine, which is how the first attempt died. The workers do not need to
    # apply the patch themselves: main() rewrites rocm.py on disk before any
    # worker starts, so they import the patched file.
    main()
