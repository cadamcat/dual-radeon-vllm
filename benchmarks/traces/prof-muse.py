"""Which attention kernel does Muse-Glimmer-30B actually run on gfx1100?

vllm/platforms/rocm.py:352 lets gfx1x reach the custom HIP paged-attention
kernel only when sliding_window is 0. Muse-Glimmer is 39 sliding-window layers
(window 2048) plus 13 full-attention ones, and it satisfies every other clause:
head_size 128, block_size 16, gqa_ratio 16, kv_cache_dtype auto, bf16, no alibi,
no sinks. The engine logs the fallback with warning_once, so the log cannot say
whether the 13 global layers took the custom path. Call counts can.

Read the result as counts per decode step: 52 layers total. If
paged_attention_rocm appears at roughly 13 per step and the Triton kernel at 39,
both paths are live. If the Triton kernel alone appears at 52, none are.

Method copied from prof-mtp-31b.py so the numbers sit next to the existing
traces: same profiler settings, same depth, same differencing-free single pass.
"""
import glob
import os
import shutil
import time

MODEL = "/models/Muse-Glimmer-30B-INT4"
DEPTH = 32768
PROF = "/rb/prof-muse"
os.environ.setdefault("VLLM_ROCM_CLONE_MMAP_WEIGHTS", "1")
os.environ.pop("VLLM_CLONE_MMAP", None)


def main():
    from vllm import LLM, SamplingParams
    from vllm.config import ProfilerConfig
    from vllm.inputs import TokensPrompt

    os.makedirs(PROF, exist_ok=True)
    for f in glob.glob(os.path.join(PROF, "*")):
        os.remove(f) if os.path.isfile(f) else shutil.rmtree(f, ignore_errors=True)

    llm = LLM(model=MODEL, tensor_parallel_size=2, max_model_len=DEPTH + 512,
              max_num_seqs=128, gpu_memory_utilization=0.92, disable_log_stats=False,
              profiler_config=ProfilerConfig(
                  profiler="torch", torch_profiler_dir=PROF,
                  torch_profiler_with_stack=False, torch_profiler_use_gzip=True,
                  delay_iterations=2, active_iterations=8))

    cfg = llm.llm_engine.vllm_config
    cc = cfg.cache_config
    tc = getattr(cfg.model_config.hf_config, "text_config", cfg.model_config.hf_config)
    print(f"[gate] block_size={cc.block_size} cache_dtype={cc.cache_dtype} "
          f"dtype={cfg.model_config.dtype} sliding_window={getattr(tc, 'sliding_window', None)} "
          f"layers={getattr(tc, 'num_hidden_layers', None)}", flush=True)
    lt = getattr(tc, "layer_types", None)
    if lt:
        from collections import Counter
        print(f"[gate] layer_types={dict(Counter(lt))}", flush=True)

    p = TokensPrompt(prompt_token_ids=[1000 + (i % 20000) for i in range(DEPTH)])
    sp = SamplingParams(max_tokens=64, temperature=0.0, ignore_eos=True)
    t0 = time.perf_counter()
    llm.generate([p], sp)
    print(f"[warm] first pass {time.perf_counter() - t0:.1f}s", flush=True)

    llm.start_profile()
    llm.generate([p], sp)
    llm.stop_profile()
    time.sleep(30)
    print(f"[prof] {sorted(os.path.basename(x) for x in glob.glob(os.path.join(PROF, '*')))}", flush=True)
    print("=== PROF DONE ===", flush=True)


if __name__ == "__main__":
    main()
