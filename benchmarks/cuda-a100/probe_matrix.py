#!/usr/bin/env python3
"""probe_matrix.py — one cell of the gemma-4 backend/speculation matrix.

    python3 probe_matrix.py <backend> <spec> <ctx>

backend: FLASHINFER | TRITON_ATTN | AUTO   (AUTO = leave selection to vLLM,
         which on Ampere with vllm#47547 applied yields FA2 for the sliding
         groups and FlashInfer for the 512-head full-attention group)
spec:    mtp | nospec
ctx:     prompt length in tokens (30000 and 50000 in the committed data)

Decode rate is measured by 64-vs-8 differencing: two generations from the
same prompt, decode tok/s = (64-8)/(t64-t8). The warm-up generation fills
the prefix cache, so t8/t64 are decode-dominated and the differencing
removes what little prefill remains. All three modalities are zeroed:
gemma-4 registers image, video AND audio, and every one of them must be 0
before vLLM 0.28.0 drops the mm-prefix requirement that otherwise excludes
FlashInfer (see ../cuda-a100/README.md, finding 1).

Single-run probe. Clear ~/.cache/vllm/torch_compile_cache between
configurations — the AOT cache key ignores limit_mm_per_prompt (vllm#50891)
and a stale artifact crashes engine init.
"""
import sys, time
from vllm import LLM, SamplingParams

backend = sys.argv[1]
spec = sys.argv[2] == "mtp"
ctx = int(sys.argv[3])

MODEL = "google/gemma-4-31B-it-qat-w4a16-ct"
ASSIST = "google/gemma-4-31B-it-assistant"

kwargs = dict(model=MODEL,
              max_model_len=33000 if ctx == 30000 else ctx + 1000,
              gpu_memory_utilization=0.9,
              limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0})
if backend != "AUTO":
    kwargs["attention_backend"] = backend
if spec:
    kwargs["speculative_config"] = {"method": "mtp", "model": ASSIST,
                                    "num_speculative_tokens": 1}

llm = LLM(**kwargs)
tok = llm.get_tokenizer()
ids = tok("hello " * (ctx + 20000)).input_ids[:ctx]
prompt = tok.decode(ids)

def timed(n):
    sp = SamplingParams(temperature=0, max_tokens=n, ignore_eos=True)
    t0 = time.time(); llm.generate([prompt], sp); return time.time() - t0

timed(8)
t8, t64 = timed(8), timed(64)
tps = (64 - 8) / (t64 - t8)
print(f"RESULT decode_tok_s={tps:.2f}  t8={t8:.1f}s t64={t64:.1f}s", flush=True)
