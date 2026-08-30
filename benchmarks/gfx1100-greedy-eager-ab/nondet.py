"""Is the non-determinism within a process or across processes?

vllm#50603 says a warm-up call fixes first-call non-determinism. Every earlier
measurement here already had a warm-up and still varied across processes, so the
two possibilities are: it varies within a process too (the warm-up claim does not
hold here), or it is stable inside a process and only differs between them (a
different phenomenon, worth describing separately).

Same process, same prompt, warm-up first, then eight identical greedy
generations.
"""
import json
import os
import sys

MODELS = {
    "muse": "/models/Muse-Glimmer-30B-INT4",
    "gemma3": "/models/gemma-3-27b-it-w4a16",
}
DEPTHS = [512, 8192]
NGEN = 64
REPEATS = 8
os.environ.setdefault("VLLM_ROCM_CLONE_MMAP_WEIGHTS", "1")
os.environ.pop("VLLM_CLONE_MMAP", None)


def main():
    which, proc = sys.argv[1], sys.argv[2]
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(model=MODELS[which], tensor_parallel_size=2,
              max_model_len=max(DEPTHS) + 512, max_num_seqs=128,
              gpu_memory_utilization=0.92, disable_log_stats=True)
    sp = SamplingParams(max_tokens=NGEN, temperature=0.0, ignore_eos=True)

    rows = []
    for d in DEPTHS:
        p = TokensPrompt(prompt_token_ids=[1000 + (i % 20000) for i in range(d)])
        llm.generate([p], SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True))
        seqs = []
        for i in range(REPEATS):
            out = llm.generate([p], sp)
            seqs.append(list(out[0].outputs[0].token_ids))
        distinct = {tuple(s) for s in seqs}
        first = [next((j for j, (x, y) in enumerate(zip(seqs[0], s)) if x != y), None)
                 for s in seqs[1:]]
        rows.append({"depth": d, "repeats": REPEATS, "distinct": len(distinct),
                     "first_divergence_vs_run0": first, "seqs": seqs})
        print(f"[within] {which} proc={proc} ctx={d:>6} 同进程 {REPEATS} 次 -> "
              f"{len(distinct)} 种输出  首次分歧位置={first}", flush=True)

    with open(f"/rb/nondet-{which}-p{proc}.json", "w") as f:
        json.dump({"which": which, "proc": proc, "rows": rows}, f)
    print("=== NONDET DONE ===", flush=True)


if __name__ == "__main__":
    main()
