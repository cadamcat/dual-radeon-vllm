"""One vLLM decode step, bracketed in an NVTX range so ncu profiles only that.

Loads a checkpoint at TP=1, warms up, then generates GEN tokens from a
500-token prompt inside nvtx range "decode". Eager mode: ncu replays each
kernel to read counters, and a captured CUDA graph would hide the kernels
from it. Bytes moved per token do not depend on that; launch overhead does,
which is why tok/s is taken from a separate graph-mode run and not from here.

    python3 decode_step.py <model-dir> [gen_tokens]
"""
import json, os, sys, time
import torch
MODEL = sys.argv[1]
GEN = int(sys.argv[2]) if len(sys.argv) > 2 else 8
from vllm import LLM, SamplingParams
cache = "/content/work/ladder-" + os.path.basename(MODEL) + ".json"
if os.path.exists(cache):                        # the runner cut it on demand
    prompt = next(e["text"] for e in json.load(open(cache)) if e["target"] == 500)
else:                                            # cut it here the same way, from the book on disk
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    # get_book() in the runner tries BOOK first and then the cutter's own cache;
    # the same two, in the same order, so this cannot disagree with the ladder
    for bp in ("/content/work/origin.txt", "/content/work/.gutenberg-1228.txt"):
        if os.path.exists(bp):
            body = open(bp, encoding="utf-8", errors="ignore").read(); break
    else:
        raise SystemExit("no book on disk: run setup first")
    i = body.find("INTRODUCTION"); body = body[i if i > 0 else 0:]
    prompt = tok.decode(tok(body).input_ids[:500], skip_special_tokens=True)
llm = LLM(model=MODEL, tensor_parallel_size=1, enforce_eager=True, max_model_len=2048,
          gpu_memory_utilization=0.90, max_num_seqs=1,
          limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0})
sp = SamplingParams(temperature=0.0, max_tokens=GEN, ignore_eos=True)
llm.generate([prompt], sp)                        # warm-up, outside the range
torch.cuda.synchronize()
torch.cuda.nvtx.range_push("decode")
t0 = time.perf_counter()
out = llm.generate([prompt], sp)
torch.cuda.synchronize()
dt = time.perf_counter() - t0
torch.cuda.nvtx.range_pop()
n = len(out[0].outputs[0].token_ids)
print(f"DECODE_STEP tokens={n} wall_s={dt:.4f} (prefill of 500 included)", flush=True)
