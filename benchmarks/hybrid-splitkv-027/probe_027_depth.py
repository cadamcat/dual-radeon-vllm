"""Qwen3.8-27B decode against context depth on 0.27, with and without vllm#45916.

The gap this closes: the 37.32 tok/s figure for this checkpoint is 0.27 at
ctx 1024, and the 10.68 tok/s at 32K is the 0.23.1-era stack with #45916's
split-KV decode applied. Nothing measured both on one stack, so neither number
could be read as "where this model is today".

    python3 probe_027_depth.py <stock|splitkv> <ctx> <out.jsonl>

The splitkv arm is not a hand-port. `gh pr diff 45916` applies to 0.27's
chunked_prefill_paged_decode.py with no rejected hunks; the result is shipped
in as a whole file and its md5 asserted here, so the arm is the PR's code
rather than an approximation of it.

Both the file swap and the routing recorder are installed BEFORE any vllm
import, and __pycache__ is cleared. Stage 3 on 0.27 lost its routing records
to exactly that: the workers ran unmodified bytecode although the file on disk
had changed.
"""

import glob
import hashlib
import json
import os
import pathlib
import sys
import time

ARM, CTX = sys.argv[1], int(sys.argv[2])
OUT = sys.argv[3] if len(sys.argv) > 3 else "/work/qwen38-027-depth.jsonl"
MODEL = "/data/incoming/Qwen3.8-27B-AWQ-INT4"
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "16"))
assert ARM in ("stock", "splitkv"), ARM


def main():
    assert "vllm" not in sys.modules, "vllm imported before the edits"

    MD5_STOCK = "86f68d47c7bdc390ced4c6d0c18025fa"
    MD5_SPLITKV = "84c6d4f9b2dfe2714b3a8f43ee832b02"
    md5 = lambda p: hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()

    SP = glob.glob("/opt/python/lib/python*/site-packages")
    assert len(SP) == 1, SP
    SP = SP[0]
    cppd = pathlib.Path(SP, "vllm/v1/attention/ops/chunked_prefill_paged_decode.py")
    assert md5(cppd) == MD5_STOCK, f"0.27 file is not what was patched against: {md5(cppd)}"

    if ARM == "splitkv":
        src = pathlib.Path("/work/cppd_027_splitkv.py")
        assert md5(src) == MD5_SPLITKV, f"shipped file is not the patched one: {md5(src)}"
        cppd.write_bytes(src.read_bytes())

    body = cppd.read_text()
    marker = "kernel_paged_attention_2d_splitkv"
    if ARM == "splitkv":
        assert marker in body, "split-KV arm has no split-KV kernel"
    else:
        assert marker not in body, "stock arm already has split-KV"

    # Record the dispatch's verdict from inside whichever process runs attention.
    ROUTE = f"/work/route38-{ARM}-{CTX}.txt"
    if ARM == "splitkv":
        ANCHOR = "        if use_splitkv_decode:\n"
        assert body.count(ANCHOR) == 1, body.count(ANCHOR)
        rec = (
            "        try:\n"
            "            import os as _os\n"
            "            _k = (head_size, sliding_window, bool(use_splitkv_decode))\n"
            "            _s = getattr(chunked_prefill_paged_decode, '_seen', None)\n"
            "            if _s is None:\n"
            "                _s = set(); chunked_prefill_paged_decode._seen = _s\n"
            "            if _k not in _s:\n"
            "                _s.add(_k)\n"
            "                with open('" + ROUTE + "', 'a') as _fh:\n"
            "                    _fh.write('pid=%d %s\\n' % (_os.getpid(), _k))\n"
            "        except Exception as _e:\n"
            "            with open('" + ROUTE + ".err', 'a') as _fh:\n"
            "                _fh.write('%r\\n' % (_e,))\n"
        )
        cppd.write_text(body.replace(ANCHOR, rec + ANCHOR, 1))
    for pyc in glob.glob(SP + "/vllm/**/__pycache__/*.pyc", recursive=True):
        os.remove(pyc)

    import vllm                                                     # noqa: E402
    from vllm import LLM, SamplingParams                            # noqa: E402

    print(f"ARM={ARM} ctx={CTX} vllm={vllm.__version__} max_num_seqs={MAX_NUM_SEQS}",
          flush=True)

    llm = LLM(model=MODEL, tensor_parallel_size=2, max_model_len=CTX + 512,
              gpu_memory_utilization=0.92, max_num_seqs=MAX_NUM_SEQS)
    tok = llm.get_tokenizer()
    ids = tok("hello " * (CTX + 4000)).input_ids[:CTX]
    prompt = tok.decode(ids)


    def timed(n):
        s = SamplingParams(temperature=0, max_tokens=n, ignore_eos=True)
        t0 = time.time()
        o = llm.generate([prompt], s)
        return time.time() - t0, o[0].outputs[0].text


    timed(8)                     # warm the prefix cache
    t8, _ = timed(8)
    t64, text = timed(64)
    tps = (64 - 8) / (t64 - t8)

    routes = sorted(set(pathlib.Path(ROUTE).read_text().strip().splitlines())) \
        if pathlib.Path(ROUTE).exists() else []
    row = {"arm": ARM, "ctx": CTX, "model": MODEL, "vllm": vllm.__version__,
           "max_num_seqs": MAX_NUM_SEQS, "decode_tok_s": tps, "t8": t8, "t64": t64,
           "routes": routes, "sample_text": text[:120]}
    with open(OUT, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"RESULT arm={ARM} ctx={CTX} decode_tok_s={tps:.2f} t8={t8:.1f} t64={t64:.1f} "
          f"routes={routes}", flush=True)
    err = pathlib.Path(ROUTE + ".err")
    if err.exists():
        print("RECORDER_ERRORS:", err.read_text()[:300], flush=True)
    print("DEPTH_CELL_DONE", flush=True)


if __name__ == "__main__":
    # TP=2 spawns its workers, which re-import this module. Without the
    # guard each worker re-runs the file swap -- and would then fail its
    # own md5 assert, because the parent has already swapped the file.
    main()
