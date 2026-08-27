"""Force the native RDNA3 W4A16 kernel onto an ASYMMETRIC checkpoint.

The A/B in `probe_w4a16_ab.py` compared two checkpoints from two publishers, so
"symmetry is what matters" still rests on the two arms being otherwise alike.
This removes the publisher as a variable entirely: one checkpoint, two kernels.

`RDNA3W4A16LinearKernel` declines `uint4` on a single line, and its docstring
calls itself a drop-in replacement for exllama. It does not avoid zero points —
`apply_weights` asserts one is present, and the *symmetric* path fabricates a
constant `qzeros` of `weight_type.bias - 1`, packs it with
`pack_quantized_values_into_int32`, and hands it to `ops.gptq_gemm_rdna3`. So
the HIP kernel is already a general asymmetric GPTQ dequant. An asymmetric
checkpoint arrives with real per-group zero points already registered under
`weight_zero_point`, and `process_weights_after_loading` transforms `w_q` and
`w_s` but *never touches* `w_zp`.

Two conventions therefore have to agree for this to work, and neither is
checked anywhere:

  1. the GPTQv1 "+1 quirk" — the kernel adds 1 to the stored zero, which is why
     the symmetric path encodes `bias - 1` rather than `bias`;
  2. the packing and layout of the zero-point tensor, which the symmetric path
     produces itself and the asymmetric path inherits from the checkpoint.

    python3 probe_w4a16_forced.py <stock|patched> <out.jsonl>

`patched` admits `uint4` into `SUPPORTED_QUANT_TYPES` by editing the module on
disk, before the engine starts, because TP=2 builds layers in spawned workers
that re-import the file. Nothing else is changed.

Outcomes and what each means:

  engine or kernel raises        a shape/dtype assumption differs; the error
                                 names which one
  runs, output is gibberish      the kernel ran on zero points it read
                                 differently than they were written — the +1
                                 or the packing
  runs, output is coherent       the conventions already agree, and the type
                                 gate is the only thing in the way

Coherence is scored, not eyeballed: same greedy prompt on both arms, and the
mean logprob of the generated tokens. Gibberish scores far lower than text.
Speed is measured separately at ctx=1024 so it is comparable to the A/B.
"""

import ast
import json
import pathlib
import sys
import time

ARM = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "/work/w4a16-forced.jsonl"
MODEL = "/data/incoming/Qwen3.8-27B-AWQ-INT4"
assert ARM in ("stock", "patched"), ARM

import torch

import kernel_record
import vllm.model_executor.kernels.linear.mixed_precision.rdna3_w4a16 as RK
from vllm import LLM, SamplingParams

QUESTION = (
    "Answer in two or three sentences. What is the capital city of France, "
    "and what river runs through it?"
)




def admit_uint4():
    """the one-line change under test, applied on disk so workers see it."""
    p = pathlib.Path(RK.__file__)
    src = p.read_text()
    old = "    SUPPORTED_QUANT_TYPES = [scalar_types.uint4b8]\n"
    assert src.count(old) == 1, src.count(old)
    new = "    SUPPORTED_QUANT_TYPES = [scalar_types.uint4b8, scalar_types.uint4]\n"
    p.write_text(src.replace(old, new))
    ast.parse(p.read_text())
    assert "scalar_types.uint4]" in p.read_text()
    print("PATCH applied: uint4 admitted to SUPPORTED_QUANT_TYPES", flush=True)


def main():
    rec = kernel_record.install(f"forced-{ARM}")
    if ARM == "patched":
        admit_uint4()
    print(f"ARM={ARM} model={MODEL}", flush=True)

    llm = LLM(model=MODEL, tensor_parallel_size=2, max_model_len=1536,
              gpu_memory_utilization=0.92, max_num_seqs=128)
    tok = llm.get_tokenizer()

    # 1) coherence: greedy, scored by mean logprob rather than by eye
    sp = SamplingParams(temperature=0, max_tokens=64, logprobs=1)
    out = llm.generate([QUESTION], sp)[0].outputs[0]
    lps = [next(iter(d.values())).logprob for d in (out.logprobs or []) if d]
    mean_lp = sum(lps) / len(lps) if lps else None
    answer = out.text

    # 2) speed at ctx=1024, same shape as the A/B so the numbers are comparable
    ids = tok("hello " * 5000).input_ids[:1024]
    prompt = tok.decode(ids)

    def timed(n):
        s = SamplingParams(temperature=0, max_tokens=n, ignore_eos=True)
        t0 = time.time()
        o = llm.generate([prompt], s)
        return time.time() - t0, o[0].outputs[0].text

    timed(8)
    t8, _ = timed(8)
    t64, _ = timed(64)
    tps = (64 - 8) / (t64 - t8)

    chosen = kernel_record.read(rec)

    row = {"arm": ARM, "model": MODEL, "decode_tok_s": tps, "t8": t8, "t64": t64,
           "mean_logprob": mean_lp, "n_logprobs": len(lps),
           "answer": answer, "kernels": chosen}
    with open(OUT, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"RESULT decode_tok_s={tps:.2f} mean_logprob="
          f"{'NA' if mean_lp is None else format(mean_lp, '.4f')}", flush=True)
    print(f"ANSWER {answer!r}", flush=True)
    print(f"KERNELS {chosen}", flush=True)
    print("FORCED_CELL_DONE", flush=True)


if __name__ == "__main__":
    main()
