"""Does --enforce-eager remove the greedy non-determinism? (vllm#50603)

@AIwork4me could not reproduce either symptom on a rebuilt 0.25.1 with
enforce_eager on. The original measurements here ran with it OFF -- nondet.py
never passes it, and nondet-muse-p1.log records `enforce_eager=False` with 58
`Capturing CUDA graph` lines -- so graphs are an axis that differs between the
two results and has never been tested here.

Identical to nondet.py in every other respect, deliberately, so the eager=0 arm
reproduces the published cells rather than merely resembling them.

    nondet_eager.py <muse|gemma3> <proc-tag> <0|1 eager>
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
    which, proc, eager = sys.argv[1], sys.argv[2], bool(int(sys.argv[3]))
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(model=MODELS[which], tensor_parallel_size=2,
              max_model_len=max(DEPTHS) + 512, max_num_seqs=128,
              gpu_memory_utilization=0.92, disable_log_stats=True,
              enforce_eager=eager)
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
        print(f"[within] {which} eager={int(eager)} proc={proc} ctx={d:>6} -> "
              f"{len(distinct)} distinct of {REPEATS}  first_div={first}", flush=True)

    with open(f"/rb/nondet-eager-{which}-e{int(eager)}-p{proc}.json", "w") as f:
        json.dump({"which": which, "proc": proc, "enforce_eager": bool(eager),
                   "rows": rows}, f)
    print("=== NONDET DONE ===", flush=True)


if __name__ == "__main__":
    main()
