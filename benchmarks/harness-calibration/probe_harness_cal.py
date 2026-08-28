"""How much of the gap between this repo's two decode harnesses is the harness?

The campaign (`bench_runner.py`) drives an OpenAI server, streams, samples at
temperature 0.8, and reports (completion_tokens - 1) / (last_token - first_token)
over a 512-token generation. The probes drive `LLM.generate` offline, greedy,
and difference a 64-token run against an 8-token one. Pooling the two into one
chart needs that difference to be a stated number rather than an assumption.

Qwen3-8B is the calibration model because it is immune to every other variable
in play: head_dim 128 so vllm#45916's split-KV (head_size 256) cannot apply,
no sliding window so the block-skip patch cannot apply, bf16 so no W4A16 kernel
selection, and gqa_ratio 4 so the gfx11 gate admits it either way. The stock
0.23.1 image is therefore comparable to the campaign's patched one FOR THIS
MODEL, which is the whole reason for choosing it.

Engine knobs are the campaign's: max_model_len 33000, util 0.85, TP=2, and the
same three env vars, so only the measurement method differs.

Two probe figures per depth, not one:
  tps_64   the probe as this repo actually uses it, 8 -> 64
  tps_512  the same generation span as the campaign, 8 -> 512
If tps_512 lands on the campaign and tps_64 does not, the difference is the
growth of context during the measured window, not the harness.

    python3 probe_harness_cal.py <out.jsonl>
"""

import json
import sys
import time

OUT = sys.argv[1] if len(sys.argv) > 1 else "/work/harness-cal.jsonl"
MODEL = "/data/incoming/Qwen3-8B"
# The campaign's actual prompt_tokens at these targets, so the depths match
# rather than merely being close: 500 -> 511, 8000 -> 8009, 32000 -> 32012.
DEPTHS = [(500, 511), (8000, 8009), (32000, 32012)]


def main():
    import torch  # noqa: F401
    import vllm
    from vllm import LLM, SamplingParams

    print(f"vllm={vllm.__version__}", flush=True)
    llm = LLM(model=MODEL, tensor_parallel_size=2, max_model_len=33000,
              gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()

    fh = open(OUT, "w")
    for target, want in DEPTHS:
        # A distinct filler per depth, so a shorter prompt is not a prefix of a
        # longer one and prefix caching cannot carry across depths.
        filler = {500: "alpha ", 8000: "bravo ", 32000: "delta "}[target]
        ids = tok(filler * (want + 4000)).input_ids[:want]
        prompt = tok.decode(ids)
        got = len(tok(prompt).input_ids)

        def timed(n):
            sp = SamplingParams(temperature=0, max_tokens=n, ignore_eos=True)
            t0 = time.time()
            o = llm.generate([prompt], sp)
            return time.time() - t0, len(o[0].outputs[0].token_ids)

        timed(8)                       # warm this depth's prefix
        t8, n8 = timed(8)
        t64, n64 = timed(64)
        t512, n512 = timed(512)
        tps64 = (n64 - n8) / (t64 - t8)
        tps512 = (n512 - n8) / (t512 - t8)
        row = {"model": MODEL, "vllm": vllm.__version__, "campaign_target": target,
               "prompt_tokens_wanted": want, "prompt_tokens_got": got,
               "t8": t8, "t64": t64, "t512": t512,
               "n8": n8, "n64": n64, "n512": n512,
               "tps_64": tps64, "tps_512": tps512}
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print(f"RESULT target={target} prompt={got} tps_64={tps64:.2f} "
              f"tps_512={tps512:.2f} t8={t8:.1f} t64={t64:.1f} t512={t512:.1f}",
              flush=True)
    fh.close()
    print("HARNESS_CAL_DONE", flush=True)


if __name__ == "__main__":
    # TP=2 spawns its workers, which re-import this module.
    main()
