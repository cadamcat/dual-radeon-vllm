# gemma-4-12B on a Tesla T4 — 2026-08-30

**The fifth machine, which this repository had recorded as a wall.** The
pre-flight beside this directory is why: on sm75 the selector rejects
`FLASH_ATTN` for compute capability, accepts `TRITON_ATTN` and `FLEX_ATTENTION`,
and both then die at kernel load asking 98 304 and 163 840 bytes of shared
memory against Turing's 65 536.

[vllm#39018](https://github.com/vllm-project/vllm/pull/39018) removes the wall.
It halves `TILE_PREFILL` on the `head_size_padded >= 512` layers only, and with
it applied the same card serves the whole ladder.

Eleven rungs, two rounds each: **22 measurements, 0 errors**, 10 of 11 prefill
rungs and 11 of 11 decode rungs chart-grade.

    Tesla T4 · 15360 MiB · sm75 · driver 580.82.07
    vLLM 0.28.0 + vllm#39018 · torch 2.13.0+cu130 · CUDA 13.0
    TRITON_ATTN · MarlinLinearKernel · float16 · util 0.95 · max_num_seqs 1
    enable_prefix_caching=False
    weights 8.28 GiB · KV 3.5 GiB / 55 809 tokens · max_model_len 33 000

`float16` because Turing has no bf16. `max_num_seqs 1` because vLLM sizes
activations and CUDA graphs for it and the default set costs 4.57 GiB of a
15 GiB card — the harness issues one request at a time anyway.

## The patch is asserted, not assumed

`PROVENANCE-t4d.json` carries the md5 of both files before and after
`patch -p1`, both changed, plus a grep for the patch's own marker
(`head_size_padded >= 512`) in the installed source, and `vllm==0.28.0` is
pinned rather than resolved. `logs/setup.log` is the install's own record.

One thing fell out of that assertion. The **pre-patch** md5 of
`vllm/v1/attention/ops/triton_unified_attention.py` is
`49fab3b643bf5a88eb65303ce377996b`, which is the hash
`campaign-0830c` recorded for the same file **inside the ROCm 0.27 image**. The
TRITON_ATTN prefill kernel is byte-identical between the ROCm 0.27 image and the
CUDA 0.28.0 wheel, so where a TRITON_ATTN row here is put beside a Radeon one,
the kernel source is not the variable.

## Three sessions, and which rows came from which

| session | reached | fate |
|---|---|---|
| `t4b` | 19 of 22 rungs | lost with its VM. `results-t4b-19rungs.jsonl` |
| `t4c` | 21 of 22 | lost before round 2 of 32 000. `results-t4c-21rungs.jsonl` |
| `t4d` | the 32 000 pair | seeded with `t4c`'s lower rungs so the checkpoint skipped them; both rounds inside one engine |

`results.jsonl` is `t4c`'s ten rungs below 32 000 plus `t4d`'s 32 000 pair, and
`assemble.py` is what built it, with the reasoning in its docstring. **`t4c`'s
32 000 round 1 is deliberately not in it**: it is 50.8 minutes before `t4d`'s,
and `build_ledger.SESSION_GAP_S` is 3 600 s, so `latest_session` would not have
seen a session boundary — all three values would have aggregated into one cell
with `runs=3`, averaging across two VMs.

## What agrees across VMs, and what does not

`t4b` and `t4c` share ten rungs. Decode agrees to **2.53 %** at worst and prefill
to **2.95 %**, the latter on the 500 rung, where the whole measurement is
0.87 s.

At the spliced rung the two VMs are further apart than anywhere else:

| 32 000 | `t4c` | `t4d` | |
|---|--:|--:|--:|
| decode tok/s | 8.8631 | 8.9851 (r1 8.9909, r2 8.9793) | **1.44 %** |
| prefill TTFT s | 331.4023 | 316.8056 (r2 317.5508) | **4.61 %** |

`t4d` is faster on both. It measured this rung six minutes after warm-up on a
fresh VM; `t4c` measured it after 43 minutes of continuous ladder. A 70 W
passively-cooled card throttling is the obvious reading and this data does not
demonstrate it — nothing here logs clocks or temperature.

## What that 4.61 % does to the fit, which is the finding

Fitting `T(S) = a + b·S + c·S²` over the chart-grade rungs:

| | a ms | b µs/tok | c ns/tok² |
|---|--:|--:|--:|
| nine rungs, no 32 000 at all | −681.5 | 2355.5 | 250.03 |
| **+ `t4d`'s 32 000 (316.8056 s)** — what is published | **−2717.8** | **3033.2** | **218.89** |
| + `t4c`'s 32 000 (331.4023 s) | −617.8 | 2334.3 | 251.00 |

**Which VM supplied one rung moves `b` by 29.9 % and `c` by 12.8 %** — as much
as adding the rung at all moves them (28.8 % and 12.5 %). A 4.6 % difference in
one TTFT reading propagates into a 30 % change in a fitted coefficient.

The reason is that this curve is quadratic-dominated in a way no other machine's
here is. At 32 013 tokens `c·S²` is 224 s against `b·S`'s 97 s — the quadratic
term is 2.3× the linear one, so the linear term is what absorbs the uncertainty.
On the L4 the same model gives 17.1 s linear against 8.2 s quadratic and the fit
is conditioned the other way round.

**So `b` for this machine is not determined by this ladder**, in the same way
`a` is not determined on any of them. `c` is the coefficient to read here, and
it should be read with the ±13 % this table shows.

## Against the other four machines

Decode is unaffected by #39018 — it only touches `TILE_PREFILL` — so these
compare directly:

| gemma-4-12B, TP=1 | @500 | @32 K | backend |
|---|--:|--:|---|
| A100-SXM4-80GB | 115.05 | 71.25 | `TRITON_ATTN` |
| RX 7900 XT | 50.32–50.56 | 36.72–36.77 | not recorded |
| NVIDIA L4 | 28.24 | 25.07 | `TRITON_ATTN` |
| **Tesla T4** | **20.28** | **8.99** | `TRITON_ATTN` |

The T4 holds 0.72× of the L4 at 500 and **0.36× at 32 K**. It is the only card
here whose decode more than halves across the ladder.

**Prefill does not compare.** These are the only rows in either projection
measured with a patch that changes an attention kernel's tile size, and halving
`TILE_PREFILL` lands squarely on the quadratic term. `patches=["vllm#39018"]`
travels on every row so a query can exclude them.

## The one ungraded rung

The 500 rung's two prefill rounds are 0.7767 s and 0.8909 s, 13.70 %. Every
other rung of this ladder is under 2 %. The runner's discarded warm-up request
is in place, so this is not the cold engine; it is a 0.87 s measurement in which
114 ms is 13.7 %. Decode at the same rung is 1.83 % and grades.
