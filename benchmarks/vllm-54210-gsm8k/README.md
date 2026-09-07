# Widening the gfx11 paged-attention gate leaves gsm8k where it was

[vllm#54210](https://github.com/vllm-project/vllm/pull/54210) lowers
`use_rocm_custom_paged_attention`'s gfx11 bound from `gqa_ratio >= 3` to
`>= 1`, which is what the CDNA branch of the same function already uses.
`benchmarks/vllm-50603/` measured what that buys — the CK kernel is 1.84x to
7.28x faster than the Triton fallback in exactly the excluded range, and
numerically equivalent to it. `tjtanaa` (MEMBER), reviewing the PR on
2026-09-06, asked for the other half: kernel correctness results, and an
end-to-end gsm8k lm-eval score.

This directory is the gsm8k half. **Stock and widened score the same.** Over
the full 1 319 questions the two arms differ by two questions on one filter and
two the other way on the other, against a standard error of about ten:

| arm | routing, both ranks | strict-match | flexible-extract |
|---|---|---|---|
| stock | `(2, 128, 16, 0, False)` | 1209/1319 = 0.9166 ± 0.0076 | 1211/1319 = 0.9181 ± 0.0076 |
| widened | `(2, 128, 16, 0, True)` | 1207/1319 = 0.9151 ± 0.0077 | 1213/1319 = 0.9196 ± 0.0075 |

Both arms answer the same documents, so the paired view is the one that carries
information. Six documents change on strict-match and eight on flexible-extract,
and the two filters move in opposite directions:

| filter | stock | widened | right -> wrong | wrong -> right |
|---|---|---|---|---|
| strict-match | 1209 | 1207 | 4 | 2 |
| flexible-extract | 1211 | 1213 | 3 | 5 |

## What makes this a score about the gate

A score is worth nothing here unless the kernel under test was reached, so the
dispatch records its own verdict rather than the harness inferring it from the
patch. `gsm8k_serve_gate.py` inserts a recorder into
`chunked_prefill_paged_decode` that writes
`(num_queries_per_kv, head_size, block_size, sliding_window, use_custom)` once
per distinct key per process. Under TP=2 attention runs in spawned workers, so a
counter in the serving process would see nothing and both ranks have to report:

    route-stock.txt      pid=148 (2, 128, 16, 0, False)   pid=149 (2, 128, 16, 0, False)
    route-widened.txt    pid=148 (2, 128, 16, 0, True)    pid=149 (2, 128, 16, 0, True)

with the sliding-window layers `(2, 128, 16, 1023, False)` in both arms. That is
the routing table the PR body gives, reproduced under load: the widened arm
takes the custom kernel on both ranks, the stock arm takes it nowhere, and the
sliding layers are untouched either way.

The other precondition is that the two arms were asked the same thing.
`prompt_hash`, `doc_hash` and `target_hash` agree on all **2 638** rows of each
arm, which is what licenses the paired table above.

## Why this checkpoint

`checkpoints.jsonl` is `enum_ckpts.py` over every `config.json` on the guest.
Of the **11** checkpoints there, exactly **1** is reached by the change:

- **7** are `head_dim` 256 (every gemma-4 and every Qwen3.x), which the gfx11
  branch refuses before `gqa_ratio` is read;
- **3** are `head_dim` 128 at `gqa_ratio` 4 or 16, which the gate admitted
  before this PR;
- **`gemma-3-27b-it-w4a16`** is `head_dim` 128 at `gqa_ratio` 2 — inside the
  band the change opens, and the only checkpoint here that is.

So the model was not chosen. It is the one the box can test the change with.

## Method

Two arms, one sitting each, launched detached at 14:24 and 17:13 UTC on
2026-09-07 (`logs/both-arms.log`). Stock ran first, because its two assertions —
that the tree is pristine, and that the routing shows `use_custom=False` — had
never been exercised before that day.

- **Server**, inside the container: `vllm serve` at TP=2,
  `--gpu-memory-utilization 0.92 --max-model-len 4096 --max-num-seqs 16`,
  vLLM `0.27.1.dev5+gf46a9dfe2.d20260827`. The arm patches
  `vllm/platforms/rocm.py` in place and asserts the edit took (widened) or that
  the tree is untouched (stock) — a stock arm run on an accidentally patched
  tree would read as "no difference" and be reported as one.
- **Harness**, outside the container: lm-eval 0.4.13 over HTTP. It is
  deliberately not installed inside, because this box's comparisons rest on the
  container's file md5s and pip resolving lm_eval's dependencies could move
  torch or vllm underneath them. `gsm8k_local.yaml` pulls in the stock 0.4.13
  `gsm8k` task with `include:` and overrides only the dataset source, so prompt,
  five shots, filters and `do_sample: false` are that version's.
- **Dataset**: openai/gsm8k at revision `740312add88f`, copied in by hand
  because the guest cannot reach Hugging Face; md5s in
  `dataset-provenance.json`, checked on both sides of the copy.
- **Stack asserted before each arm**: `chunked_prefill_paged_decode.py` at
  `84c6d4f9b2dfe2714b3a8f43ee832b02` (vllm#45916, which three campaigns rest
  on), and `config.json` identical inside the container and at the host path the
  tokeniser is loaded from. Either mismatch refuses the run.
- **Restored on every exit path**: the two edited files from a backup taken
  before the run, the container stopped, ollama and llamacpp-hub restarted, both
  cards confirmed back at `27971584` bytes.

`validate-20q/` is the same two arms at `--limit 20`, run first. Both scored
0.75 and both produced their own routing verdict; that is what the full run was
launched on.

**A defect the validation caught, kept here as evidence.**
`validate-20q/route-widened.txt` carries **8** lines where every other route
file carries 4: `pid=245` and `pid=246` are an earlier attempt's, `pid=148` and
`pid=149` are the validation's own. The recorder opens the route file with
`'a'`, and the arm deleted the copy's destination rather than the file being
appended to, so a route survived the run that earned it. It reads as harmless
here — both runs were widened and both reached the kernel — but the stock arm's
verdict is *the absence* of `use_custom=True`, and a stale `True` from a
previous widened run would have inverted exactly that. Both sides of the copy
are cleared now, which is why `route-stock.txt` in the same directory has 4
lines and the two full arms have 4 each.

The full set was run rather than a `--limit`, and that answers the reviewer's
open question without his answer: `--limit N` takes the first N documents in
order, so the first N rows of `samples-*.jsonl` are the `--limit N` score for
any N.

## Rows

    samples-stock.jsonl     2 638 rows: 1 319 documents x 2 filters, per document
    samples-widened.jsonl   the same, widened arm
    results-*.json          lm_eval's own aggregate output, verbatim
    route-*.txt             the dispatch's own verdict, per worker process
    checkpoints.jsonl       11 rows, one per checkpoint on the guest
    logs/                   both arms' progress and full lm_eval output
    validate-20q/           the 20-question validation of both arms

`samples-*.jsonl` are lm_eval's per-document records with two fields dropped by
`strip_samples.py`: `arguments`, the rendered five-shot prompt, which is 8.25 MB
of 12 MB per arm and identical in front of every question, and `doc`, which is
the pinned dataset. `prompt_hash`, `doc_hash` and `target_hash` stay, so what
was dropped is still checkable from what was kept, and `verify_doc_figures.py`
checks it.

## Not established

- **277 of the 1 319 continuations differ between the arms** while six documents
  change score. Nothing here separates that from this box's own greedy-decode
  nondeterminism, which `benchmarks/gfx1100-greedy-attn-ab/` measured at 4
  distinct continuations in 8 repeats of one prompt at 8 192 tokens and 1 at
  512. gsm8k's five-shot contexts sit between those two lengths. The control
  that would separate them is a second stock arm compared against the first; it
  was not run.
- **One checkpoint, one architecture, one box.** gfx1100 at `gqa_ratio` 2. The
  change also opens `gqa_ratio` 1, which no checkpoint here has.
- **Accuracy only.** This directory measures no latency. The speed case is
  `benchmarks/vllm-50603/`.
