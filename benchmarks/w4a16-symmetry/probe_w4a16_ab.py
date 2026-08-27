"""Symmetric vs asymmetric int4 on gfx1100: does the native W4A16 kernel show up?

vllm#50264's profile puts ~80 ms of Qwen3.8-27B's 85.85 ms decode step in
`triton_w4a16_gemm_kernel`, against 5.7 ms in attention. vLLM has had a native
gfx1100 W4A16 kernel since #41394, but `RDNA3W4A16LinearKernel.can_implement`
rejects `scalar_types.uint4`, and compressed-tensors maps `symmetric: false` to
exactly that. Our checkpoint is asymmetric; a symmetric one of the same model
should therefore land on the native kernel instead.

    python3 probe_w4a16_ab.py <asym|sym> <ctx>

Two checkpoints of the same model, same architecture, same TP, differing in the
quantization scheme:

  asym  cyankiwi/Qwen3.8-27B-AWQ-INT4   compressed-tensors, symmetric=false, g32
  sym   RedHatAI/Qwen3.8-27B-INT4       compressed-tensors, symmetric=true,  g128

Which linear kernel each layer actually gets is recorded from inside the TP
workers by patching `choose_mp_linear_kernel` on disk. A monkeypatch in this
process would see nothing, because the model runs in spawned workers.

Speculative decoding is left off on both, even though both checkpoints ship an
MTP head, so the comparison is plain decode on both sides.
"""

import json
import pathlib
import sys
import time

ARM = sys.argv[1]
CTX = int(sys.argv[2])
OUT = sys.argv[3] if len(sys.argv) > 3 else "/work/w4a16-ab.jsonl"
MODELS = {
    "asym": "/data/incoming/Qwen3.8-27B-AWQ-INT4",
    "sym": "/data/incoming/Qwen3.8-27B-INT4-sym",
}

import torch

import kernel_record
from vllm import LLM, SamplingParams


def main():
    # instrumentation goes on before the engine starts: TP=2 builds the layers
    # in spawned workers, which re-import the patched module from disk
    rec = kernel_record.install(f"{ARM}-{CTX}")
    print(f"ARM={ARM} ctx={CTX} model={MODELS[ARM]}", flush=True)

    # max_num_seqs is pinned for BOTH arms. The asymmetric checkpoint carries
    # zero-points and a 32-element group, so it leaves fewer Mamba cache blocks
    # than the symmetric one at the same utilization, and the default 256 aborts
    # graph capture there ("exceeds available Mamba cache blocks (214)"). Left
    # at the default this would have measured only the arm that happened to fit.
    llm = LLM(model=MODELS[ARM], tensor_parallel_size=2, max_model_len=CTX + 512,
              gpu_memory_utilization=0.92, max_num_seqs=128)
    tok = llm.get_tokenizer()
    ids = tok("hello " * (CTX + 4000)).input_ids[:CTX]
    prompt = tok.decode(ids)

    def timed(n):
        sp = SamplingParams(temperature=0, max_tokens=n, ignore_eos=True)
        t0 = time.time()
        o = llm.generate([prompt], sp)
        return time.time() - t0, o[0].outputs[0].text

    timed(8)                      # warm the prefix cache
    t8, _ = timed(8)
    t64, text = timed(64)
    tps = (64 - 8) / (t64 - t8)

    chosen = kernel_record.read(rec)
    row = {"arm": ARM, "model": MODELS[ARM], "ctx": CTX, "decode_tok_s": tps,
           "t8": t8, "t64": t64, "kernels": chosen, "sample_text": text[:160]}
    with open(OUT, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"RESULT decode_tok_s={tps:.2f} t8={t8:.1f} t64={t64:.1f}", flush=True)
    print(f"KERNELS {chosen}", flush=True)
    print("AB_CELL_DONE", flush=True)


if __name__ == "__main__":
    main()
