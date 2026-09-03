# The collective, on seven rented configurations — 2026-09-03

`allreduce-2026-09-02` timed the all-reduce a TP=2 decode step pays on two
RX 7900 XT with no P2P at all — device to host to device — and got 16.6–21.5 µs.
This runs the same sweep on seven NVIDIA configurations, so what tensor
parallelism costs on the wire is measured on every machine whose ladders were
measured the same night.

Same script, generalised in one place: `/proc/self/maps` is now matched for
`nccl` as well as `rccl`. RCCL is NCCL's ROCm build and answers the same
`ncclAllReduce`. Five hidden sizes × eleven token counts × three timing modes,
the buffer zeroed so bf16 stays finite. **NCCL 2.29.7 on every NVIDIA run**, so
the library is not a variable among them.

## Hidden 4096 (Qwen3-8B), `t_graph_us`

| | n=1, one decode step | n=16 384 | ratio |
|---|--:|--:|--:|
| H100 ×2, NVLink | **12.06** | 474 | 39× |
| H100 ×4, NVLink | 14.76 | 617 | 42× |
| H200 ×4, NVLink | 14.97 | 616 | 41× |
| B300 ×2 | 14.24 | **290** | 20× |
| A100 ×2, NVLink | 17.74 | 759 | 43× |
| RTX PRO 6000 ×2, **no NVLink** | 14.44 | 3 775 | 261× |
| RTX PRO 6000 ×4, **no NVLink** | **39.10** | 11 294 | 289× |
| RX 7900 XT ×2, no P2P (2026-09-02) | 16.65 | **18 050** | 1 084× |

**The bandwidth end spans 62× and the latency end 3.2×** across these; among
the pairs alone the latency end spans 1.5×. Batch-1 decode reduces one row.

## Three things this settles

**Inference never enters the interconnect's range.** A B300 pair moves a
16 384-token reduction 62× faster than the Radeon pair and a one-token
reduction 1.2× faster. The second card's value tracking `mem_busy` rather than
the wire — `campaign-2026-09-02d`'s finding, reproduced on NVLink in
`cuda-h100/campaign-2026-09-03-tp2` — has a mechanism here rather than an
inference.

**NVLink is for the fourth card, not the second.**

| adding cards three and four | n=1 | n=16 384 |
|---|--:|--:|
| H100, NVLink | ×1.22 | ×1.30 |
| RTX PRO 6000, no NVLink | **×2.71** | **×2.99** |

Two cards without NVLink cost 20 % over two with it. Four cost 171 %. PCIe is a
shared bus and the ranks contend; NVLink is point-to-point and the extra hop is
nearly free.

**More memory bandwidth does not reach the collective.** H200 ×4 and H100 ×4
agree to 1 % at both ends (14.97 against 14.76 µs; 616 against 617 µs), so none
of the H200's ladder advantage in `cuda-h200/campaign-2026-09-03` comes from
the wire.

## Two labelling faults, both corrected in the files

`gpu="H100:4"` returned four **H200s** on the first four-way sweep, while the
same string returned four H100s for a ladder an hour earlier. Caught by
`nvidia-smi nvlink -s` printing the card name and confirmed by `vram_total_b`
(150 754 820 096 against 85 520 809 984 per card). The rows carry
`mislabelled_as` and `relabel_reason`; the data is unchanged.

Two runs then shared a work directory on the Volume, because both were launched
with the same `--machine`, and one file held 118 rows from two machines. Split
at the `ar_meta` boundary and identified by `vram_total_b`.

`cuda-modal/allreduce_app.py` now reads `nvidia-smi --query-gpu=name` inside
the container, prints what was asked for beside what arrived, and asserts the
cards are identical. **The request string is not evidence of what ran.**

## Files

    <machine>-results.jsonl        rank 0, one `allreduce` row per cell
    <machine>-results.rankN.jsonl  the other ranks, where kept
