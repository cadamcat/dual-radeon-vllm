"""Fixed-input greedy token-id probe: two generations in one engine.

argv: spec|nospec depth
Prints IDS0/IDS1 (token-id lists), SELF_DET, PROBE_DONE. With the
inject_45450 patch active and spec on, the engine also prints
PROBE_3D_SPEC_ACTIVE once, proving the 3D path served the verify steps.
"""
import sys, json
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

spec_on = sys.argv[1] == "spec"
depth = int(sys.argv[2])

MODEL = "google/gemma-4-31B-it-qat-w4a16-ct"
ASSIST = "google/gemma-4-31B-it-assistant"

kwargs = dict(model=MODEL, max_model_len=depth + 256, gpu_memory_utilization=0.9)
if spec_on:
    kwargs["speculative_config"] = {"method": "mtp", "model": ASSIST,
                                    "num_speculative_tokens": 1}
llm = LLM(**kwargs)
ids = [1000 + (i % 20000) for i in range(depth)]
sp = SamplingParams(temperature=0, max_tokens=64, ignore_eos=True)
outs = []
for rep in range(2):
    o = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp)
    outs.append(list(o[0].outputs[0].token_ids))
print("IDS0", json.dumps(outs[0]), flush=True)
print("IDS1", json.dumps(outs[1]), flush=True)
print("SELF_DET", outs[0] == outs[1], flush=True)
print("PROBE_DONE", flush=True)
