"""Sliding-window block skip: correctness and cost, before and after.

The decode kernel in chunked_prefill_paged_decode.py iterates every block of the
sequence and then masks everything outside the window to -10000. Blocks entirely
below seq_len - SLIDING_WINDOW contribute exp(-10000 - m), which is zero, so
skipping them should be exactly equivalent.

"Should be" is the part that needs testing. This script compares token ids
before and after, and that comparison turned out not to be the correctness
argument: greedy decoding is not reproducible on this machine even with the
patch absent, so identical output is not evidence and differing output is not
either. See docs/sliding-window-block-skip.md §7 and
benchmarks/gfx1100-greedy-nondeterminism.json.

What the correctness argument rests on instead is upstream's own kernel suite,
which passes with no case changing outcome, and fifteen boundary cases that are
bit-identical under torch.equal. What this script is still good for is the cost
half: the timing here is the same differencing method used everywhere else.
"""
import json
import os
import sys
import time

MODEL = "/models/Muse-Glimmer-30B-INT4"
DEPTHS = [8192, 32768]
NGEN = 64
os.environ.setdefault("VLLM_ROCM_CLONE_MMAP_WEIGHTS", "1")
os.environ.pop("VLLM_CLONE_MMAP", None)


def main():
    tag = sys.argv[1]  # before | after
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(model=MODEL, tensor_parallel_size=2, max_model_len=max(DEPTHS) + 512,
              max_num_seqs=128, gpu_memory_utilization=0.92, disable_log_stats=True)

    cfg = llm.llm_engine.vllm_config
    print(f"[gate] block_size={cfg.cache_config.block_size} dtype={cfg.model_config.dtype}",
          flush=True)

    rows = []
    for d in DEPTHS:
        p = TokensPrompt(prompt_token_ids=[1000 + (i % 20000) for i in range(d)])
        sp_a = SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True)
        sp_b = SamplingParams(max_tokens=NGEN, temperature=0.0, ignore_eos=True)
        llm.generate([p], sp_a)
        t0 = time.perf_counter(); llm.generate([p], sp_a); ta = time.perf_counter() - t0
        t0 = time.perf_counter(); out = llm.generate([p], sp_b); tb = time.perf_counter() - t0
        ms = (tb - ta) / (NGEN - 8) * 1000
        ids = list(out[0].outputs[0].token_ids)
        rows.append({"depth": d, "ms_per_token": round(ms, 2),
                     "tok_per_s": round(1000 / ms, 2), "token_ids": ids})
        print(f"[rate] {tag} ctx={d:>6} {ms:7.2f} ms/tok {1000 / ms:6.2f} tok/s "
              f"first8={ids[:8]}", flush=True)

    with open(f"/rb/wintest-{tag}.json", "w") as f:
        json.dump({"tag": tag, "model": MODEL, "ngen": NGEN, "rows": rows}, f, indent=2)
    print("=== WINTEST DONE ===", flush=True)


if __name__ == "__main__":
    main()
