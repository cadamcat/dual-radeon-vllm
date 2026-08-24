# Patches

Downstream changes applied to the installed vLLM in the test container, kept here
so the measurements in `docs/` can be reproduced. None of these is a
recommendation to run in production; each says what it is for.

| File | What it is |
|---|---|
| `sliding-window-block-skip.patch` | Start the Triton paged-decode loop at the first block the sliding-window mask does not zero. An identity, not an approximation — see the header for why. Worth 3.15x at 32K on `Muse-Glimmer-30B` and 2.75x on `gemma-3-27b`; `gemma-4` is on a different backend and unaffected. Written up in [sliding-window-block-skip.md](../docs/sliding-window-block-skip.md) |
| `wintest.py` | The before/after timing harness for that patch: a 64-token generation differenced against an 8-token one at the same depth. It also records generated token ids, which was the original correctness check until that turned out not to discriminate on this machine — see `benchmarks/gfx1100-greedy-nondeterminism.json` |
| `adapt-muse-glimmer.py` | Back-adapts upstream's `muse_glimmer.py` to this container's vLLM, which predates the model. Rewrites `load_weights` to this version's `stacked_params_mapping` convention, since `WeightsMapper` here has no `orig_to_new_stacked`, and inlines the `is_vit_use_data_parallel` divisibility fallback. Idempotent, backs up once |

The three new files the adaptation copies in (`muse_glimmer.py` and its config and
processor) are upstream's, taken at the merge commit of vllm-project/vllm#51655,
and are not vendored here.
