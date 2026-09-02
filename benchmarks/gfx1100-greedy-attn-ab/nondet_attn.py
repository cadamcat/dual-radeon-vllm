"""Is the greedy non-determinism the attention backend or the W4A16 kernel?

The 36-cell set in ../gfx1100-greedy-nondeterminism.json has a confound nobody
had named. Reading the quantisation kernel out of every serve log for the four
models in it gives:

    model              W4A16 kernel                 attention    unstable cells
    gemma-3-27b-w4a16  RDNA3W4A16LinearKernel       ROCM_ATTN     3 of 14
    Muse-Glimmer-30B   RDNA3W4A16LinearKernel       ROCM_ATTN     7 of 14
    gemma-4-31B        RDNA3W4A16LinearKernel       TRITON_ATTN   0 of 4
    Qwen3.8-27B        RDNAHybridW4A16LinearKernel  ROCM_ATTN     0 of 4

Every unstable cell is `RDNA3W4A16LinearKernel` **and** `ROCM_ATTN`, and each of
the two stable models differs from the unstable ones on a *different* one of
those two axes. So the set is consistent with either being the cause and cannot
choose. The published reading picks the attention backend ("the affected models
are on ROCM_ATTN and its Triton kernel_paged_attention_2d"); the open prediction
in the handoff picks the kernel (qwen38 is stable "because it routes to
RDNAHybrid"). Both are guesses at a confound.

This separates them. Same two unstable models, same harness, same depths, same
container: the only variable is the attention backend, which the W4A16 kernel
does not follow -- both arms keep `RDNA3W4A16LinearKernel`, and the log of each
run records that it did.

    stable under TRITON_ATTN  ->  the attention backend is the axis, and
                                  vllm#54706, which patches the RDNA3 W4A16
                                  split-K epilogue, is off this path
    still unstable            ->  the attention backend is not the axis, and
                                  #54706 is worth the ROCm rebuild it needs
                                  (it changes csrc/rocm/q_gemm_rdna3*.cu, so
                                  there is no file to swap -- the extension has
                                  to be compiled)

`nondet_attn.py` is `nondet_eager.py` with the third argument changed from an
eager flag to a backend name and `VLLM_ATTENTION_BACKEND` set before vLLM is
imported. Every measurement constant is identical, so the ROCM_ATTN arm
reproduces the published cells rather than merely resembling them.

**The env var does not work here.** The first attempt set
`VLLM_ATTENTION_BACKEND=TRITON_ATTN` before importing vLLM, on the strength of
"0.27 removed it in favour of `--attention-backend`" from the 2026-08-29
campaign -- and both arms came out on ROCM_ATTN, because `rocm.py`'s selector
reaches `Overriding with ROCM_ATTN` without ever consulting it. The driver's log
grep caught it; the arms differed in nothing. 0.23 *does* have the flag
(`arg_utils.py:905`, `attention_backend: AttentionBackendEnum | None`), so it is
passed to `LLM(...)` as the enum, which is the path `--attention-backend` takes.
`run_attn_ab.sh` greps every log for the backend actually chosen, and an arm
whose log does not name the one it asked for is not a measurement of it.

    nondet_attn.py <muse|gemma3> <proc-tag> <ROCM_ATTN|TRITON_ATTN>
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
    which, proc, backend = sys.argv[1], sys.argv[2], sys.argv[3]
    assert backend in ("ROCM_ATTN", "TRITON_ATTN"), backend
    from vllm import LLM, SamplingParams
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.inputs import TokensPrompt

    llm = LLM(model=MODELS[which], tensor_parallel_size=2,
              max_model_len=max(DEPTHS) + 512, max_num_seqs=128,
              gpu_memory_utilization=0.92, disable_log_stats=True,
              attention_backend=AttentionBackendEnum[backend])
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
        print(f"[within] {which} backend={backend} proc={proc} ctx={d:>6} -> "
              f"{len(distinct)} distinct of {REPEATS}  first_div={first}", flush=True)

    with open(f"/rb/nondet-attn-{which}-{backend}-p{proc}.json", "w") as f:
        json.dump({"which": which, "proc": proc, "attn_backend": backend,
                   "rows": rows}, f)
    print("=== NONDET DONE ===", flush=True)


if __name__ == "__main__":
    main()
