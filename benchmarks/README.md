# Benchmarks — raw data and how to reproduce them

Sections 1 to 5 of [`docs/benchmarks.md`](../docs/benchmarks.md) are derived from
[`results.jsonl`](results.jsonl) in this directory; §6 is derived from
[`results-2026-08-24.jsonl`](results-2026-08-24.jsonl) and from the per-finding
files listed below. Nothing is extrapolated, nothing is hand-edited, and
[`analyze/verify_doc_figures.py`](analyze/verify_doc_figures.py) recomputes the
published figures from those files and exits non-zero if any disagrees.

## What is here

| Path | What it is |
|---|---|
| `results.jsonl` | **The raw data.** 309 lines, one JSON object per event: every request's prompt tokens, TTFT, decode rate, per-card power and VRAM, plus per-config engine metadata |
| `bench_runner.py` | The campaign runner that produced it — serial, checkpointed, VRAM-safe |
| `analyze/` | The scripts that turn the raw data into the tables and charts in the docs |
| `prompts/` | Prompt-ladder manifests (the token counts as measured) + the cutter that rebuilds the ladders from the public-domain source |
| `speculative-decoding/` | Results behind [speculative-decoding-on-rdna.md](../docs/speculative-decoding-on-rdna.md). `splitkv-31b-{stock,patched}.json` is the PR#45916 A/B on the 31B (identical, it runs a different attention backend); `mtp-31b-mtp.json` is the MTP depth curve; `kbench{,2}-0.json` are two constructions of the kernel-level `query_len` sweep; `mtp32k-{tuned,spec3d}.json` are the two 32K single points; `c2-{on,off}.json` carry `token_ids` for the correctness comparison; `trace-unified-attention.json` is the per-call profiler summary, the one file here derived rather than measured directly — the traces it came from are ~2 MB each and stay on the test machine |
| `hmm-kernel-three-states.json` | The mmap reproducer across three kernel states: stock `7.0.0-28`, `-28` with `342981fff328` applied by hand, and Canonical's shipped `7.0.0-30`. The third was measured 2026-08-23 and is the one that matters — until then the repository told people to upgrade on the strength of a changelog entry alone |
| `loader-flag-kernel-30.json` | **What the writable-mapping penalty is actually worth**, measured 2026-08-23 on the kernel Canonical ships. 89 cells: four load paths (default, `eager`, vllm#49991's clone flag, safetensors' `pread` backend) across four checkpoints. 73 of them have page cache controlled per cell, warm or cold; the other 16 are `asis` ordering controls that deliberately do not. `RssAnon`/`RssFile` are sampled in 33 of the 89 — the resident-set question only needed the load paths it distinguishes. The flag is worth 1.5-2.0x while the checkpoint fits in RAM and 7.5x when it does not; the 3.9-5.6x this repository and the PR published on 2026-07-28 came from an uncontrolled page cache and does not reproduce. Includes the counterexample where the flag does not help |
| `sliding-window-block-skip.json` | **The Triton paged-decode kernel reads the whole sequence and masks the sliding window away afterwards.** Two models reach that path and both gain, three runs per cell: `gemma-3-27b` 124.29 → 45.26 ms/tok at 32 K (2.75x, 8.05 → 22.09 tok/s, medians both sides) and `Muse-Glimmer-30B` 83.99 → 26.63 (3.15x), against 1.00x below each model's own window — the shape is the mechanism check. Correctness is upstream's own kernel suite with no case changing outcome, plus 15 boundary cases bit-identical under `torch.equal`. Includes the controls that draw the boundary: `gemma-4` is forced onto a different backend and is unaffected, `Qwen3.8` is the no-window path, and eight other current models were checked and do not qualify |
| `gfx1100-greedy-nondeterminism.json` | **Greedy decoding on this machine is not reproducible across processes.** 10 of 36 cells produced more than one output across three runs with no code change between them, at any depth including 512 tokens, symmetric between the two kernel states. Surfaced while verifying the row above, and it is why that row's correctness argument is kernel-level rather than end-to-end. Resembles [vllm#50603](https://github.com/vllm-project/vllm/issues/50603) except that a warm-up call, which it says fixes the problem, was already present in every measurement here |
| `mtp-qwen-draft-head.json` | Qwen3.6-27B against Qwen3.8-27B with each checkpoint's **built-in MTP draft head**, on and off, four depths. The two are the same architecture quantised the same way in 63 of 64 layers; what differs is that Qwen3.8 leaves the draft head in bf16 and Qwen3.6 quantises it. **bf16 does not pay** — the int4 head matches or beats it everywhere and the gap is widest at 32K, where int4 breaks even and bf16 is a net loss. Also this repository's first vLLM MTP numbers for a Qwen model; the existing curve is gemma-4-31B with a separate assistant checkpoint |
| `llamacpp-layer-vs-tensor.json` | `-sm layer` against `-sm tensor` on the 31B, **ROCm backend**, three independent processes per cell. Not comparable to the Vulkan figures in the top-level README: same model and same split, different backend, which is most of the 25.7 against 27.0. `tensor` is 6.4% ahead at depth 0; at 8192 the 1.7% gap sits inside an 11% process-to-process spread, so it cannot be called. Past 8192 there is nothing to compare — `-sm layer` aborts with an illegal memory access in the KV state restore, which is a known ROCm runtime bug (`layer_abort` in the file has the isolation and the upstream links) |
| `llamacpp-depth-sweep-{rocm,vulkan}.json` | `llama-bench` decode rate at six context depths, both backends, same model and machine as the vLLM campaign. The control that showed the long-context collapse is specific to vLLM's paged-attention path — see [hybrid-decode-on-rdna.md](../docs/hybrid-decode-on-rdna.md) |

## Reproducing the analysis (no GPU needed)

```bash
cd benchmarks/analyze
python3 summarize.py          # per-config tables, exactly as measured
python3 decode_slope.py       # cost of one context token, per model
python3 fit_prefill.py B-8B-tp2   # fit T(S) = a + b*S + c*S^2, report S* = sqrt(a/c)
python3 analyze.py            # TP2/TP1 speed-up, MBU, cross-model view
python3 gen_charts.py         # regenerate the SVGs in docs/assets/
python3 gen_window_chart.py    # the sliding-window block-skip chart, from its own JSON
```

All scripts default to `../results.jsonl`; override with `BENCH_RESULTS=/path/to/results.jsonl`.
The weight-traffic section of `analyze.py` and all of `arch_cmp.py` read safetensors
headers, so they need the models — point `MODELS_DIR` at them.

## Rebuilding the prompt ladders

The prompts themselves are not committed. They are cut from Darwin's *On the Origin
of Species* (Project Gutenberg #1228, public domain), which the cutter downloads:

```bash
cd benchmarks/prompts
python3 cut_prompts.py --models-dir /path/to/models              # rebuild
python3 cut_prompts.py --models-dir /path/to/models --check-only # verify only
```

It cuts one ladder per tokenizer (gemma, qwen, gemma-26B), trims to sentence
boundaries, and **prints the drift against the committed manifests** so you can see
whether your rebuild matches the ladder that produced `results.jsonl`.

On our machine the short rungs (500–8 000) come back **exactly**, and the long ones
land within **0.24 %**. That residue is not a tokenizer difference: the original
cutter stopped as soon as it was inside its tolerance, so where it landed depended on
the path it took, while the cutter here binary-searches the sentence boundaries and
is deterministic. **No published number depends on these nominal lengths** — every
analysis uses the `prompt_tokens` the server actually reported per request, which is
recorded in `results.jsonl`.

## Re-running the measurements (needs the hardware)

`bench_runner.py` runs on the VM host, driving a container that already has the
[no-hostcall RCCL](../docs/deploy-vllm.md) installed. It stops the other GPU
services, verifies `/dev/kfd` is free, then for each configuration: restarts the
container, starts `vllm serve`, waits for readiness, measures the ladder, and
finally restores everything.

```bash
python3 bench_runner.py                       # full campaign, ~3.5 h
BENCH_CFGS=C-31B-tp2 python3 bench_runner.py  # one configuration
BENCH_TARGETS=500,8000 python3 bench_runner.py   # a subset of the ladder
```

It is **checkpointed**: results already in `results.jsonl` are skipped, so an
interrupted run resumes where it stopped.

### Four things it does that a naive script would get wrong

1. **`--gpu-memory-utilization 0.85`, not 0.90.** At 0.90 the KV pool leaves
   ~54 MB free on a 20 GiB card and the Triton attention kernel cannot allocate
   its scratch at long context — `HSA_STATUS_ERROR_OUT_OF_RESOURCES`. A short-prompt
   smoke test passes; the real run dies. 0.85 leaves ~0.9 GiB; measured peak at
   32 K context is 19.36 GiB.
2. **Startup timeout is activity-aware.** A cold `torch.compile` takes 23–26
   minutes on this host **and writes nothing to the log while it runs**. A plain
   idle timeout kills a healthy compile. The runner counts started-vs-finished
   compile graphs and relaxes the stall threshold while one is outstanding.
3. **A warm-up request must succeed before any point is recorded**, and a run is
   only marked complete if it produced real measurements. An earlier version
   reported `COMPLETE` for a configuration in which all 44 requests had failed.
4. **The server log is deleted before each start.** Otherwise the previous run's
   `Application startup complete` makes the next one look ready in 6 seconds.

## Method

- **Task**: incremental truncations of Darwin's *On the Origin of Species*
  (Gutenberg #1228, public domain) with a fixed translation instruction. One
  source text means identical content and difficulty at every length.
- **Lengths**: cut with **each model's own tokenizer** (three ladders: gemma,
  qwen, gemma-26B, Muse), trimmed to sentence boundaries. Nominal targets are
  approached, not hit: the cutter stops at whole sentences, so the short rungs
  land up to 5.2 % off — 500 becomes 481 on the gemma ladder, 1 000 becomes 948
  on gemma-26B. **Nothing downstream uses the nominal length.** Every analysis
  uses the `prompt_tokens` the server reported per request, which is in
  `results.jsonl`.
- **Prefill**: `max_tokens=1`; throughput = prompt tokens / TTFT.
- **Decode**: `max_tokens=512`; rate measured **from first to last token**, so
  TTFT is excluded.
- **Prefix caching defeated** by a unique random `[seed-xxxxxxxx]` on every request.
- **Two rounds per point** (four for the two re-measured points, below).
  **Decode is reported as the mean of all rounds, prefill as the best of all rounds.**
- Power sampled from both cards' hwmon every 1.5 s during decode.

## Two anomalies, and what was done about them

The first pass produced two rounds that disagreed with their partner by more than
1 %: gemma-4-12B TP2 @2000 decode (50.3 against 58.3) and @4000 prefill (2096
against 2652). Rather than drop them, **both points were re-measured with two
further rounds** on a fresh server start:

| point | round 1 | round 2 | round 3 | round 4 | reported |
|---|---:|---:|---:|---:|---:|
| @2000 decode | 58.31 | **50.29** | 59.00 | 58.50 | **56.5** (mean of 4) |
| @4000 prefill | **2095.5** | 2651.7 | 2629.1 | 2663.1 | **2663** (best of 4) |

Neither anomaly reproduced, so both are one-off glitches. **All four rounds remain
in `results.jsonl`** — the added ones numbered 3 and 4 and tagged
`"note": "re-measured …"`. Nothing was deleted, which means the low reading still
drags the @2000 decode mean down from ~58.6 to 56.5. We would rather publish the
conservative number than curate the dataset.

Across the 142 (config, context, kind) cells, taking (max − min) / mean,
**30 exceed 1 %**: 14 decode and 16 prefill, or 28 excluding the two named above.
The widest is the 26B MoE's prefill at 4 000, 2 431.3 against 3 198.6, a 27.3 %
range. It is wider than either anomaly that was re-measured, and it was not
re-measured: the two above were singled out because they were noticed, not
because they were the worst.

> **Corrected 2026-08-25.** This paragraph used to say every other point was
> repeatable within 1 % across its two rounds. Thirty cells are not. Decode is the tighter of the two: the prefill spread is what dividing
a prompt length by a short TTFT does, and it is why prefill is reported as best
of rounds while decode is reported as the mean. Anyone comparing single cells
across campaigns should recompute the spread from `results.jsonl` rather than
assume 1 %.
