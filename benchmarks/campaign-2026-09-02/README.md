# `G31-tp2` on a restored link — 2026-09-02

One of the two 7900 XTs trained at **PCIe 3.0 x8** from the boot of 2026-08-29
21:48 CST onward. Nothing in the guest could see it — the guest's sysfs reports
the on-card bridge link, 16 GT/s x16 always, and the trained width is visible
only at the host's root ports, three levels above the GPU. The reboot of
2026-09-02 22:44 restored it: both root ports read x16 again, under load and at
boot, with no reseat.

`G31-tp2` is the arm that shows what the narrowed link cost. It has two x16
sittings a month apart whose fitted `b` agrees to 1% — 743.9 and 736.0 µs/tok —
and one x8 sitting at 868.7, and that x8 line is what the front page's Figure 2
and the a100 article's Figure 4 draw. Until now the 868.7 was captioned as
overstating by an amount nobody had measured.

**Re-measured on the restored link: `b` = 722.6 µs/tok.**

| sitting | link | rungs used | `a` ms | `b` µs/tok | `c` ns/tok² | r² |
|---|---|--:|--:|--:|--:|--:|
| 2026-07-25 | x16/x16 | 10/11 | 117.6 | 743.9 | 28.11 | 0.999997 |
| 2026-08-24 | x16/x16 | 10/11 | 127.9 | 736.0 | 28.43 | 0.999998 |
| 2026-08-29 | **x8**/x16 | 10/11 | 115.2 | **868.7** | 29.06 | 0.999997 |
| **2026-09-02** | x16/x16 | **11/11** | 161.6 | **722.6** | 28.31 | 0.999996 |

The three x16 sittings span **3.0%** on `b`; the x8 sitting sits **18.3%** above
their mean. `c` — the attention term, which needs no communication — is 28.11,
28.43, 28.31 on the three x16 sittings and 29.06 on the x8 one: the link moved
the linear term and left the quadratic one where it was, which is what the
arithmetic said it should do and what nothing had checked.

This sitting is also the cleanest of the four: **11 of 11 rungs chart-grade**,
where each of the other three lost one. The worst rung disagrees with itself by
1.56% and nine of the eleven by 0.31% or less.

## Decode, which was predicted not to move

| rung | x16 mean of three | x8 sitting | difference |
|---|--:|--:|--:|
| 500 | 42.937 tok/s | 42.555 | **−0.89%** |
| 8 000 | 36.772 | 36.565 | −0.56% |
| 32 000 | 29.445 | 29.135 | −1.05% |

The repository has said since 2026-09-02 that decode is "not measurably affected
at either width, <1% of a step". With three x16 sittings to average, it is
0.6–1.1% — just outside "<1%" at the deepest rung. The reason it is small is now
measured rather than asserted: `../allreduce-2026-09-02` times the collective a
decode step actually pays at 16.6–21.5 µs, so a 31B step's 120 collectives are
2.30 ms of a 23.3 ms step, and a link that halves the collective's bandwidth
cannot move a step that is 90% something else.

## Reproduced exactly, so the link is the only difference

| | 2026-08-29 | 2026-09-02 |
|---|---|---|
| container | `vllm-tp2` | `vllm-tp2` |
| vLLM | 0.23.1.dev1+g9ddef7117.d20260715 | same, read from the log |
| #45450 state | patched, `4a14f86d` / `7e275cdc` | **unchanged**, both md5s re-read in the container before the run against `campaign-2026-08-29/provenance.json` |
| kernel | 7.0.0-30 | 7.0.0-30 |
| serve args | tp 2, util 0.92, mns 16, mml 33 000, `hf_overrides` | identical — the command is committed as `serve-G31-tp2-x16.sh` |
| backend | TRITON_ATTN, forced by vLLM for gemma-4's heterogeneous head dims | same line in the log, not a flag |
| collective | PYNCCL / RCCL 2.27.7, `disable_custom_all_reduce=True` | same |
| prefix caching | True | True |

The id is `G31-tp2-x16`, not `G31-tp2`: the link is part of the configuration,
so this is a different one and not a second round of the same one. `host_link()`
in `analyze/build_prefill.py` now reads this campaign's `host_link.json` rather
than inferring from the date — the reboot landed in the middle of 2026-09-02 and
campaigns are dated by day, so a date can no longer answer the question.

**One difference that is not the link, recorded because it is a difference.**
The KV pool came out 6.78 GiB / 85 766 tokens against 08-29's 6.49 / 82 106 —
4.5% more at the same utilisation, after a reboot with nothing else resident. It
changes the reported concurrency (2.60× against 2.49×) and no rung of a
33 000-token ladder.

## Files

    runner.py                 copied from harness/runner_radeon.py; CFGS, paths
    results.jsonl             44 measurements, 0 errors
    logs/G31-tp2-x16.log      the serve log
    serve-G31-tp2-x16.sh      the exact command, as the runner wrote it
    host_link.json            preflight, both root ports x16, before the run
    PROGRESS.txt
