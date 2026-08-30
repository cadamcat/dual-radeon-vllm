# Benchmarks — raw data and how to reproduce them

Sections 1 to 5 of [`docs/benchmarks.md`](../docs/benchmarks.md) are derived from
[`results.jsonl`](results.jsonl) in this directory; §6 is derived from
[`results-2026-08-24.jsonl`](results-2026-08-24.jsonl) and from the per-finding
files listed below. Nothing is extrapolated, nothing is hand-edited, and
[`analyze/verify_doc_figures.py`](analyze/verify_doc_figures.py) recomputes the
headline figures from those files and exits non-zero if any disagrees.

## What is here

| Path | What it is |
|---|---|
| `results.jsonl` | **The raw data.** 309 lines, one JSON object per event: every request's prompt tokens, TTFT, decode rate, per-card power and VRAM, plus per-config engine metadata |
| `bench_runner.py` | The campaign runner that produced it — serial, checkpointed, VRAM-safe |
| `ledger.jsonl` | **Every decode point in one row format**, built by `analyze/build_ledger.py` from the three campaign files and the 0.27 probe sweeps. One row per (model, context), carrying the stack it was measured on — date, vLLM, ROCm, kernel, patches, harness, and since 2026-08-29 `spec` and `attn_backend` — rather than inheriting one from the chart it lands in, plus `runs`, `range_pct` and `chart_grade`. It is a projection, not a source: `build_ledger.py --check` fails if it no longer matches the files it was built from |
| `prefill.jsonl` | **Every prefill point in one row format, across machines**, built by `analyze/build_prefill.py`. The ledger is the Radeon box's and has no machine column; this one does, because prefill is asked across five machines. Same row discipline plus `machine`, `cuda`, `driver` and `prefix_caching`; `--fits` reports `T(S) = a + b·S + c·S²` per (machine, cfg, date, patches). Two rules it encodes are not obvious and each moved a number: rungs group by `target` rather than the measured `prompt_tokens`, because ROCm's two rounds land one to three tokens apart and grouping on the measurement gave nineteen mostly-unpaired points where CUDA gave eleven paired ones; and **only chart-grade rungs are fitted**, which is what excludes `cuda-a100/campaign-2026-08-29/`.

**What it says.** `T(S) = a + b·S + c·S²`, where `b` is compute and `c` is how badly attention
scales. `a` is not reported: it does not reproduce, and going after it is what
[docs/benchmarks.md §4](../docs/benchmarks.md#4-prefill-peaks-and-where-the-peak-sits) had to
withdraw. One card each:

| machine | gemma-4-12B b / c | gemma-4-26B-A4B b / c |
|---|---|---|
| A100 80G | **145.7** / **3.62** | **62.6** / **2.30** |
| one RX 7900 XT | 479.0 / 24.16 | 360.0 / 13.13 |
| one L4 24G | 534.7 / 8.03 | 204.4 / 5.53 |
| one Tesla T4 16G † | 3033.2 / 218.89 | — |

The A100 leads one 7900 XT 3.3× on b and 6.7× on c: the gap in attention is twice the gap in
compute. The L4 is *slower* on b (0.90×) and 3.0× better on c, so its curve starts below the
Radeon's and ends 1.58× above it at 32 K. The second card buys 1.23–1.48× on b and 1.91–2.22× on
c across the three models measured on both topologies — attention parallelises better than the
GEMMs, which is the claim §4's withdrawn 76 ms intercept was reaching for.

† **The T4's row does not compare with the three above it, for two reasons.** It is the only line
in either projection measured with a patch that changes an attention kernel: without
[vllm#39018](https://github.com/vllm-project/vllm/pull/39018) the engine does not start on sm75 at
all, and the patch halves `TILE_PREFILL` on the head_size 512 layers — which is the quadratic
term. And its `b` is not determined by its own ladder: the 32 000 rung was measured on a second VM,
the two agree to 4.61 % there, and swapping which one supplies it moves `b` by **29.9 %** and `c`
by 12.8 %. This curve is quadratic-dominated in a way no other here is — 224 s of c·S² against
97 s of b·S at 32 K — so the linear term absorbs the uncertainty, exactly as `a` does everywhere
else. Read `c`, with that ±13 %, and read it as this card *with a different kernel*.

**Qwen3-8B, the second model with more than one single card**, and it splits the same way:

| machine | Qwen3-8B b / c | backend |
|---|---|---|
| one RX 7900 XT, vLLM 0.27, stock | **206.7** / 8.87 | `ROCM_ATTN` |
| one L4 24G | 288.3 / **5.38** | `FLASH_ATTN` |

The Radeon wins the linear term 1.39× and loses the quadratic 1.65× — the same direction
gemma-4-12B shows, at very different magnitudes. **This pair is a kernel difference as well as a
card difference**, which the gemma-4-12B pair is not free of either: the Radeon's 2026-07-25 and
2026-08-24 campaigns kept no serve log, so their backend is unrecorded rather than known |
| `decode.jsonl` | **Every decode point, across machines**, built by `analyze/build_decode.py`. `ledger.jsonl` stays Radeon-only and unchanged; this is the cross-machine projection beside it. It imports its campaign table from `build_prefill.py` so the two cannot drift apart, and `--check` **recomputes the overlap against `ledger.jsonl` and fails if any cell disagrees** — two files projecting the same rows is how a repository ends up with two answers to one question.

**What it answers, and a correction.** Single-card decode, stock arms only, chart-grade rungs:

| model | machine | cfg | 500 | deepest | retained |
|---|---|---|--:|--:|--:|
| gemma-4-12B-it | A100 80G | `A100-G12` | 115.0 | 71.3 @32K | 61.9 % |
| | one RX 7900 XT | `A-12B-tp1` | 50.6 | 36.7 @32K | 72.6 % |
| | one L4 24G | `G12` | 28.2 | 25.1 @32K | 88.8 % |
| | one Tesla T4 16G | `G12` | 20.3 | **9.0 @32K** | **44.3 %** |
| gemma-4-26B-A4B | A100 80G | `A100-G26A4B` | 161.0 | 105.0 @32K | 65.2 % |
| | one RX 7900 XT | `E26-tp1-u95` | 96.9 | 79.1 @12K | 81.6 % |
| | one L4 24G | `G26A4B` | 52.4 | 44.1 @32K | 84.1 % |
| Qwen3-8B | one RX 7900 XT | `B8-tp1-u95` | 46.6 | 44.1 @6K | 94.7 % |
| | one L4 24G | `B8` | 16.6 | 13.5 @24K | 81.3 % |
| Qwen3.8-27B INT4-sym | one L4 24G | `Q38S` | 15.9 | 15.4 @8K | 96.6 % |
| gemma-4-31B-it | A100 80G | `G31` | 58.5 | 42.4 @32K | 72.5 % |
| | one L4 24G | `G31-eager` | 11.1 | 11.1 @1K | 99.7 % |

**Decode is the column the T4's patch does not touch** — vllm#39018 changes `TILE_PREFILL` and
nothing else — so its row compares directly, and it is the only card here whose decode more than
halves across the ladder: 0.72× of the L4 at 500 and **0.36× at 32 K**.

Four of these lines stop short of 32 K and each stop is arithmetic rather than an abandoned run.
`E26-tp1-u95`: 16.96 GiB of weights leave 0.93 GiB of KV, 13 149 tokens. `B8-tp1-u95`: 1.13 GiB,
8 236 tokens, and raising utilisation to 0.95 did not move it — the weights are 15.27 GiB on 0.27
rather than the 14.02 GiB 0.23 reported for the same checkpoint. `B8` on the L4: the capacity
retry stepped `max_model_len` to 31 680. `G31-eager`: gemma-4-31B does not start on a 23 GiB L4 at
all with CUDA graphs on (`Available KV cache memory: -0.8 GiB`), and `--enforce-eager` buys
**2.51 GiB**, a 2 020-token pool and two rungs — where the same flag buys Qwen3.8-27B on the same
card **0.05 GiB**, which is not enough, and that model does not fit at 23 GiB at all.

`Q38S` is **RedHatAI/Qwen3.8-27B**, symmetric compressed-tensors at group 128 — a different
checkpoint from the AWQ `Q38` elsewhere in this table's file, not another arm of it. On gfx1100 the
two land on different kernels and differ by 1.27–3.24× on decode
(`w4a16-symmetry/w4a16-ab.jsonl`), so they do not belong in one row, and this one has no
counterpart on any other machine.

The A100 leads on both models — 2.3x on the dense 12B at 500 and 1.66x on the MoE — and loses
the most with depth. The commit message of `73fa06e` says the opposite of that second figure,
that one 7900 XT is *ahead* of the A100 on the MoE. **It is wrong.** The query behind it keyed
on `(machine, model, date)`, under which the A100 has three configurations of that model —
stock, `-mtp` and `-mtp-p45450` — so it kept whichever was written last and reported a
speculative arm's 93.3 as the machine's stock figure. The projection was right; the ad-hoc query
over it was not. Group by `cfg` and filter `spec is None`; this table is generated by the same
rule the index's Figure 1 uses, and `verify_doc_figures.py` recomputes it |
| `analyze/` | The scripts that turn the raw data into the tables and charts in the docs |
| `prompts/` | Prompt-ladder manifests (the token counts as measured) + the cutter that rebuilds the ladders from the public-domain source |
| `speculative-decoding/` | Results behind [speculative-decoding-on-rdna.md](../docs/speculative-decoding-on-rdna.md). `splitkv-31b-{stock,patched}.json` is the PR#45916 A/B on the 31B (identical, it runs a different attention backend); `mtp-31b-mtp.json` is the MTP depth curve; `kbench{,2}-0.json` are two constructions of the kernel-level `query_len` sweep; `mtp32k-{tuned,spec3d}.json` are the two 32K single points; `c2-{on,off}.json` carry `token_ids` for the correctness comparison; `trace-unified-attention.json` is the per-call profiler summary, the one file here derived rather than measured directly — the traces it came from are ~2 MB each and stay on the test machine |
| `campaign-2026-08-29/` | **Speculation, both arms, on two attention backends.** Eight configurations on the Radeons: `Qwen3.8-27B` and `gemma-4-31B-it` with and without speculation, plus three more that pin Qwen3.8's kernel with `--attention-backend` instead of letting ROCm choose. Every row carries `spec` and `attn_backend`, and this is the campaign that shows they are not decoration: `vllm#45450` patches the Triton unified-attention files, ROCm routes Qwen3.8 to `ROCM_ATTN`, whose backend file imports `chunked_prefill_paged_decode` and neither of them, and the two arms differing only in whether the patch is installed agree to a mean of −1.93 % — inside their own repeat spread — while the patch's probe never prints. Pinned to `TRITON_ATTN` the probe prints twice, once per tensor-parallel worker, and the patch is worth **+187 %** at 32 K. Separately, the same stock ladder pinned to Triton is **15.0 %** faster at 32 K than the backend ROCm picks for itself — but **it is a trade and not a free flag**, which this entry said until 2026-08-30 and the prefill rows contradict: the same two arms, the same day, differing in that flag and nothing else, give **969 against 690 prefill tok/s at 32 K the other way** (`ROCM_ATTN` 1.40× ahead), and their fitted quadratic terms are 3.43 against 18.44 ns/tok². Pinning Triton buys decode at depth and sells prefill at depth. Both halves are drawn in the index's Figure 2 `provenance.json` is derived from `logs/` by `make_provenance.py` so the backend column can be recomputed rather than believed; `acceptance.txt` is the acceptance length vLLM logs every ten seconds, aligned to each rung by `acceptance.py` |
| `cuda-a100/campaign-2026-08-29/` | The other half of the same campaign, twelve configurations on an A100-SXM4-80GB under vLLM 0.28.0. `#45450` nearly doubles decode at 32 K on the two models routed onto the Triton kernel (**+94.6 %** on `gemma-4-31B`, **+98.9 %** on `gemma-4-26B-A4B`) and does **nothing** on `Qwen3.8`, which CUDA routes to `FLASH_ATTN` — 20.51 against 20.52, both arms repeating to 0.03 %. Also the DFlash arm, which needed method `dflash` rather than `draft_model` and is a net loss deepening from −28.3 % to −47.5 %. Its README has the four VM reclaims and what they cost. **Its prefill rows are not prefill measurements** and are superseded by `cuda-a100/campaign-2026-08-30/`: it ran with `enable_prefix_caching=True` and every rung of the ladder is a strict prefix of the next, so 130 of its 132 prefill rungs fail the repeatability cut. Decode is unaffected and stands |
| `campaign-2026-08-30/` | **The MoE on one card.** `gemma-4-26B-A4B` at TP=1 on a single 7900 XT: 16.96 GiB resident on a card that reports 19.98, which leaves 0.33 GiB of KV at util 0.92 and 0.93 GiB at 0.95. Both attempts are in `results.jsonl`, the first as a `config_failed`. Seven rungs of the eleven; 32 K would need util ≈ 0.97, past where `runner.py`'s own note says these cards keep any scratch, so the ceiling is the finding |
| `cuda-t4/preflight-2026-08-30/` | **Why the T4 needs a patch to run gemma-4 at all**, and until 2026-08-30 why there was no T4 row anywhere. One engine start on a `Tesla T4`, sm75. compressed-tensors W4A16 loads and takes `MarlinLinearKernel`, and memory is solvable — `--gpu-memory-utilization 0.95 --max-num-seqs 1` turns 0.65 GiB of KV into 3.5 GiB, 55 809 tokens. What cannot be solved is gemma-4's head dimensions against Turing's 65 536 bytes of shared memory per SM — heterogeneous, `head_dim` 256 on the 40 sliding layers and `global_head_dim` 512 on the 8 full-attention ones, and vLLM sizes the kernel for **512** (`model_arch_config.head_size`, measured 2026-08-30, not the 256 this row first carried): `FLASH_ATTN` is **rejected** by the selector for compute capability, while `TRITON_ATTN` and `FLEX_ATTENTION` are **accepted** and then fail at kernel load asking for 98 304 and 163 840 bytes. The only backend honest about sm75 is the one that is excluded — the selector models compute capability for one backend and shared memory for none. Three serve logs, one per backend |
| `cuda-l4/campaign-2026-08-30/` | **The spine's fourth machine.** `gemma-4-12B` and `gemma-4-26B-A4B` on one NVIDIA L4, sm89, vLLM 0.28.0, TRITON_ATTN, and the first CUDA rows here measured with **prefix caching off**. Eleven rungs each, 22 measurements, 0 errors. Against one 7900 XT on the same model the Radeon wins the linear term and loses the quadratic three times over, which is why the L4 is 1.58× faster at 32 K and would lose at a short enough prompt |
| `campaign-2026-08-30b/` | **Qwen3-8B on one 7900 XT at util 0.95, on the 0.27 image, fully stock.** Run to lift July's 6 000 ceiling by raising utilisation; it did not — 1.13 GiB and 8 236 tokens against July's 8 442. The prediction was wrong twice and the errors compound: weights are 15.27 GiB on 0.27 rather than the 14.02 that 0.23 reported for the same checkpoint, and activation overhead is **per-model** (2.58 GiB for a bf16 dense, against the 1.09 measured on an int4 MoE). What it produced instead: decode agrees to **0.21 %** across three vLLM stacks and two utilisations at all five rungs, while prefill's `b` improves 1.24× and its `c` improves **1.82×** — the gain is in the quadratic term, which a TTFT ratio cannot say. `ROCM_ATTN`, and this is the case vllm#54438 deliberately leaves alone: `head_dim` 128 with `gqa_ratio` 4 gets the actual HIP kernel |
| `cuda-l4/campaign-2026-08-30b/` | **The L4's second pass**, after `cuda_run.py` gained the capacity retry the Radeon runner has had since rev2 — its absence cost four configurations, `B8` by 0.13 GiB. Adds `Qwen3-8B` (10 rungs, 20/20 chart-grade) and `Qwen3.8-27B-INT4-sym` (6 rungs), both `FLASH_ATTN`. **`Q38S` is a different checkpoint from `Q38`, not another arm of it**: RedHatAI symmetric compressed-tensors against cyankiwi AWQ, 1.27–3.24× apart on decode on gfx1100, and with no counterpart on any other machine here. On Qwen3-8B the Radeon wins `b` 1.39× and loses `c` 1.65× — the same split gemma-4-12B shows, at very different magnitudes, and **this pair is a kernel difference as well as a card one** (`ROCM_ATTN` against `FLASH_ATTN`) |
| `cuda-a100/campaign-2026-08-30/` | **The A100 measured again with prefix caching off**, for the two models this round needs, because the 2026-08-29 prefill is not usable. Same cell at 32 K: 2.9320 s and 0.2010 s there, 8.3826 s and 8.3796 s here. Round 1 of the old data was wrong by 2.9× as well. No serve logs — the VM was reclaimed after the run finished — so `harvester.py` is committed beside the data as the reason the data exists |
| `cuda-l4/campaign-2026-08-30c/` | **What fits on a 23 GB L4 and what does not**, at `max_num_seqs 1` with an `--enforce-eager` fallback. `gemma-4-31B` needs the fallback and then reaches **two rungs** on a 2 020-token pool; `Qwen3.8-27B-AWQ-INT4` does not start at all. The contrast is the finding: eager buys **2.51 GiB** on the 31B (KV budget −0.8 → +1.71 GiB) and **0.05 GiB** on Qwen3.8 (−0.39 → −0.34), so one was CUDA graphs and the other is the weights — 19.24 GiB resident on a 22.49 GiB card leaves 2.13 GiB for everything that is not KV. Also: **halving `max_model_len` cannot fix a NEGATIVE budget**, which both configurations proved over four engine starts each; `Available KV cache memory` is the log line that separates that case from the too-small-pool case the length *is* the lever for. `G31-eager` is its own configuration id — eager is a different engine — and is reported, not fitted |
| `cuda-t4/campaign-2026-08-30/` | **The fifth machine, with [vllm#39018](https://github.com/vllm-project/vllm/pull/39018) applied.** `gemma-4-12B` on a `Tesla T4`, sm75, eleven rungs, 22 measurements, 0 errors — the wall the pre-flight documented, removed by halving `TILE_PREFILL` on the `head_size_padded >= 512` layers. The patch is md5-asserted before and after, and `vllm==0.28.0` pinned. Decode compares directly (the patch touches prefill only): **20.28 tok/s at 500 and 8.99 at 32 K**, 0.72× and 0.36× of the L4 — the only card here whose decode more than halves across the ladder. **Prefill does not compare**, and `patches=["vllm#39018"]` travels on every row so a query can exclude them. Assembled from three sessions, two of which died; `assemble.py` says which rows came from where and why `t4c`'s 32 000 round 1 is left out (50.8 minutes from `t4d`'s, against a 3 600 s session gap, so it would have aggregated into one cell across two VMs). **And the finding: which VM supplied that one rung moves fitted `b` by 29.9 % and `c` by 12.8 %** — a 4.6 % difference in one TTFT reading, because this curve is quadratic-dominated (224 s against 97 s at 32 K) and the linear term absorbs it. `b` is not determined by this ladder, the way `a` is not determined by any of them |
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
python3 build_prefill.py --fits   # T(S) = a + b*S + c*S^2 per machine and config
python3 fit_prefill.py B-8B-tp2   # superseded: groups by prompt_tokens, one campaign at a time
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

It cuts one ladder per tokenizer (gemma, qwen, gemma-26B, Muse), trims to sentence
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
- **Lengths**: cut with **each model's own tokenizer** (four ladders: gemma,
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
