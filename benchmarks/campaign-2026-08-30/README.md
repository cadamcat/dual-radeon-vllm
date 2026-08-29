# gemma-4-26B-A4B on one RX 7900 XT — 2026-08-30

The second tier of the single-card round: a 26B mixture-of-experts, 4B active,
served from **one** 7900 XT rather than the pair every other Radeon row in this
repository uses. It is here because it is the arm that nearly did not fit, and
the ladder it could reach is the result.

`results.jsonl` holds both attempts. `runner.py` is the campaign runner with
this round's single configuration; `logs/` holds a serve log for each attempt.

## What fits on one 19.98 GiB card

The card reports **21 458 059 264 bytes = 19.98 GiB** (`mem_info_vram_used`'s
sibling in sysfs, not a datasheet figure). The checkpoint is 16.01 GiB of
safetensors and **16.96 GiB resident** — vLLM's own `Model loading took` line.

| attempt | util | budget | resident | KV | reaches | outcome |
|---|--:|--:|--:|--:|--:|---|
| `E26-tp1` | 0.92 | 18.38 GiB | 16.96 | **0.33 GiB** | 1 536 tok | `config_failed` |
| `E26-tp1-u95` | 0.95 | 18.98 GiB | 16.96 | **0.93 GiB** | 13 149 tok | seven rungs |

vLLM puts the 33 000-token requirement at **1.3 GiB**, which is
0.0394 GiB per 1 000 tokens. Reaching 32 K therefore needs a budget of about
19.35 GiB, or **util ≈ 0.97** — past the point `runner.py`'s own rev2 note says
these cards keep any scratch. So the ladder stops at 12 000 and the ceiling is
the finding, in the same way Qwen3-8B's 6 000-token ceiling at TP=1 is.

**The reachable ladder is not deterministic across engine starts.** At
`--max-model-len 33000` the engine reported an estimated maximum of 32 064
tokens; restarted at 31 743 it reported 13 248; the run that served reported
13 149. Same card, same utilisation, same container, three different answers
within seven minutes. The run took the third.

`util` is on the row and in the configuration id, because no other row in
either projection was measured at 0.95.

## Why 0.23 and not 0.27

gemma-4 cannot be served on the 0.27 ROCm image at all — its Quark plugin reads
`head_dim` off a heterogeneous config and dies before loading, see the
2026-08-29 campaign handoff. So this ran in `vllm-tp2`, which is the **0.23**
image (`rocm7.14.0_rdna_..._vllm_0.23.0`) despite its name; the name is the
image it was built from, not the topology, and this arm is `tp=1`.

    vLLM 0.23.1.dev1+g9ddef7117.d20260715 · ROCm 7.14 · kernel 7.0.0-30
    TRITON_ATTN · compressed-tensors · bfloat16 · max_num_seqs 16

## Prefix caching is on, and does nothing

`enable_prefix_caching=True` in this container. It produced no hits: the two
rounds of the 12 000 rung are **6.209 s and 6.207 s**, and every rung repeats to
better than 0.7 %. That matters because the same flag on the 2026-08-29 A100
campaign *did* hit — round 2 of its 32 K rung took 0.201 s against round 1's
2.932 s — which is why that campaign's prefill cannot be used and this one's
can. The flag is recorded on the row; what gates the fit is `chart_grade`.

## The rows

Seven rungs, two rounds each, prefill and decode in one pass: **28 measurements,
0 errors**, every rung chart-grade (ranges 0.011 % to 0.683 %).

| rung | prefill TTFT | decode |
|---|--:|--:|
| 500 | 0.204 / 0.207 s | 96.85 tok/s |
| 2000 | 0.784 / 0.783 s | 93.16 |
| 8000 | 3.719 / 3.720 s | 83.21 |
| 12000 | 6.209 / 6.207 s | 79.06 |

Decode goes into `ledger.jsonl` and prefill into `prefill.jsonl`, both as
`E26-tp1-u95`.
