# The other two x8 prefill lines, re-measured — 2026-09-02c

`Q38-tp2` and `Q38-triton-tp2` are the Radeon-pair lines the front page's
Figure 2 draws for `Qwen3.8-27B`, both from the 2026-08-29 sitting, on which one
card had trained at **PCIe 3.0 x8**. `Q38-triton-tp2` also backs the "969
against 690 prefill tok/s" comparison in [`../README.md`](../README.md) — a
ratio that was always fair, because both arms sat on the same link, next to two
absolute numbers that were not.

Re-measured on the link the 2026-09-02 22:44 reboot restored:

| arm | backend | `b` on x8 | `b` on x16 | x8 was | `c` on x8 | `c` on x16 |
|---|---|--:|--:|--:|--:|--:|
| `Q38-tp2` | ROCM_ATTN | 913.2 | **761.3** | +20.0% | 3.43 | 4.46 |
| `Q38-triton-tp2` | TRITON_ATTN | 846.2 | **758.5** | +11.6% | 18.44 | 17.27 |

**The two backends agree on `b` to 0.4% once the link is equal**, where on the
narrowed one they differed by 7.9%. They still differ by nearly four times on
`c` — 4.46 against 17.27 ns/tok². That is the split the decomposition claims:
`b` carries the linear term the link throttles, `c` is attention, and attention
is what the flag actually changes. Nothing in the 2026-08-29 data showed this,
because the link was putting a different-sized thumb on each arm's `b`.

## The published comparison, corrected

| at 32 000 tokens | x8, published | x16, measured |
|---|--:|--:|
| `ROCM_ATTN` prefill | 969.0 tok/s | **1 098.8** |
| `TRITON_ATTN` prefill | 689.9 | **759.7** |
| ratio | 1.405× | **1.446×** |
| `TRITON_ATTN` decode gain | 1.150× | 1.154× |

The trade is intact and the ratio barely moves — which is what "both arms on the
same link" predicted. The absolute numbers move 10 to 13%.

## Decode, which was predicted not to move

Every rung, both arms: **−0.2% to +1.0%**. At 32 000 tokens 36.47 → 36.62
(ROCm) and 41.95 → 42.28 (Triton). The same result the 31B gave in
[`../campaign-2026-09-02/`](../campaign-2026-09-02/README.md), and the reason it
is small is measured rather than asserted in
[`../allreduce-2026-09-02/`](../allreduce-2026-09-02/README.md).

## What this changed in a published claim elsewhere

The hybrid-SSM article's section 6 said `Qwen3.8-27B`'s prefill *falls* on every
arm this repository has measured, and quoted **7.5%** for the 0.27 arm. That
7.5% was this arm on the narrowed link. On a full-width one the same arm is
**flat, +0.3%** from the 1 000 rung to 32 000. The paragraph's argument — that a
rising prefill curve is a property of the configuration and not of the
architecture — is unaffected: flat is not rising, the MoE beside it rises 21.9
to 23.8%, and pinned to Triton the same checkpoint falls 32.6%. The sentence and
its numbers were corrected in both languages; the count of stock hybrid-SSM
ladders in `prefill.jsonl` goes from six to eight.

## Two container states, so the campaign ran twice

The two arms did not run in the same container state on 2026-08-29 and could not
here either:

    Q38-tp2         vllm-027, #45450 stock    49fab3b6 / f0a1379d
    Q38-triton-tp2  vllm-027, #45450 patched  9416a868 / 8bd13173

`BENCH_CFGS` selected one arm each time, with `campaign-0829/revert45450.py`
flipping the container between — it asserts both md5s in both directions.
Reproducing the patch state matters for the Triton arm specifically, because
#45450 patches `triton_unified_attention.py` and `triton_attn.py` and that arm
is the one forced onto TRITON_ATTN. On `Q38-tp2` the patch is off the path
entirely: 0.27 routes Qwen3.8 to ROCM_ATTN, whose backend imports
`chunked_prefill_paged_decode` and neither patched file.

## The run that had to be thrown away, and why it is kept

`results-nosplitkv.jsonl` is a complete, error-free 44-measurement run of
`Q38-tp2-x16` that is **not** what it claims to be, and finding out cost the
campaign an hour.

It decoded at **3.88 tok/s at 32 000 tokens** against the 2026-08-29 arm's
36.47 — a tenfold collapse with depth where the published arm falls 1.36×.
[`../hybrid-splitkv-027/qwen38-027-depth.jsonl`](../hybrid-splitkv-027/) records
the same checkpoint at 32 768 tokens as **3.828 tok/s stock** and 35.20 with
`vllm#45916` applied. The run had reproduced the *stock* arm to 1.4% while its
`patches` field would have said `vllm#45916 split-KV`.

The container had lost the patch some time between 2026-08-29 and 2026-09-02:
`chunked_prefill_paged_decode.py` read `86f68d47…`, which
`../hybrid-splitkv-027/provenance.json` records as the image's own file. This
campaign *did* check #45450's two md5s before running. That was not enough,
because **#45916 lives in a third file nobody was checking.**

`apply_45916.py` here asserts both states of that file the way
`revert45450.py` does for the other two, and the run is kept because it is also
an independent reproduction of the stock arm — 3.88 against the probe's 3.828 at
32 K, 12.34 against 12.57 at 8 K — on a different harness, a different link and
a different day.

> **The rule this earns:** before reproducing an arm, assert every patch md5
> that arm records, not the one you happen to remember. Three files are in play
> on `vllm-027`, and a container's state is not what a campaign's `patches`
> field says it was.

## Files

    runner.py                     copied from harness/runner_radeon.py
    apply_45916.py                asserts and restores the split-KV file
    results.jsonl                 both arms, 88 measurements, 0 errors
    results-nosplitkv.jsonl       the discarded run, kept as evidence
    logs/ , logs-nosplitkv/       one serve log per arm per attempt
    serve-Q38-*.sh                the exact commands, as the runner wrote them
    host_link.json                preflight, both root ports x16, before the run
    PROGRESS.txt , PROGRESS-nosplitkv.txt
