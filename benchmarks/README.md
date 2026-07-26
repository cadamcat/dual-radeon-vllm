# Benchmarks — raw data and how to reproduce them

Everything in [`docs/benchmarks.md`](../docs/benchmarks.md) is derived from
[`results.jsonl`](results.jsonl) in this directory. Nothing is extrapolated,
nothing is hand-edited.

## What is here

| Path | What it is |
|---|---|
| `results.jsonl` | **The raw data.** 309 lines, one JSON object per event: every request's prompt tokens, TTFT, decode rate, per-card power and VRAM, plus per-config engine metadata |
| `bench_runner.py` | The campaign runner that produced it — serial, checkpointed, VRAM-safe |
| `analyze/` | The scripts that turn the raw data into the tables and charts in the docs |
| `prompts/` | Prompt-ladder manifests (the token counts as measured) + the cutter that rebuilds the ladders from the public-domain source |

## Reproducing the analysis (no GPU needed)

```bash
cd benchmarks/analyze
python3 summarize.py          # per-config tables, exactly as measured
python3 decode_slope.py       # cost of one context token, per model
python3 fit_prefill.py B-8B-tp2   # fit T(S) = a + b*S + c*S^2, report S* = sqrt(a/c)
python3 analyze.py            # TP2/TP1 speed-up, MBU, cross-model view
python3 gen_charts.py         # regenerate the SVGs in docs/assets/
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
  qwen, gemma-26B), trimmed to sentence boundaries, <1 % error against target.
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

Every other point in the campaign is repeatable within 1 % across its two rounds.
