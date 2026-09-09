# The 27B's depth curve was the stack's, not the model's — 2026-09-07

`campaign-2026-09-03` carried six checkpoints to 128 000 tokens on this pair and
the front page reads one of its rows as architecture: *the hybrid-SSM 27B loses
28 % to 96 000*. That row was measured on **vllm 0.23.1**.dev1 / ROCm 7.14,
chosen there deliberately, for continuity with the eleven rungs it extends.

This is the same checkpoint, the same two cards, the same TP=2, the same
vllm#45916 split-KV patch and the same runner, on **0.27.1**.dev5 / ROCm 10.0.
Eight rungs, two rounds each, 0 errors, every cell inside 0.17 %.

**The percentage was a property of the stack. The slope was not.**

| over 8 000 → 96 000, decode | intercept | slope |
|---|--:|--:|
| 0.23.1 + #45916 (the published row) | 82.9 ms/tok | **0.304 µs per context token** |
| 0.27.1 + #45916 (here) | 19.8 ms/tok | **0.236 µs per context token** |

Read as a percentage the new arm looks *worse* — it gives up 48.9 % across that
span where the old one gives up 23.9 %. Read in µs per context token, the unit
[`hybrid-decode-on-rdna.md` §6.5](../../docs/hybrid-decode-on-rdna.md) already
uses for exactly this quantity, it is **flatter, by a factor of 0.78**. A
percentage divides the depth cost by the baseline, and the baseline moved 4.19×;
the quantity that describes what depth does is the one that does not.

**And it lands inside the band.** §6.5 measured this checkpoint's decode slope at
**0.430 µs/tok** with #45916 on 0.23 and said it was still about 27 % above the
0.118–0.339 band this machine's dense models occupy. On 0.27 it is **0.236** —
inside it. The collapse that article is about is not merely reduced here; on this
stack the hybrid's decode no longer looks anomalous at all.

## What was measured

| rung | prefill 0.23 | prefill 0.27 | × | decode 0.23 | decode 0.27 | × |
|---:|---:|---:|--:|---:|---:|--:|
| 8 000 | 1 067.2 | 1 184.2 | 1.11 | 11.74 | **46.11** | 3.93 |
| 16 000 | 1 010.3 | 1 197.1 | 1.18 | 11.39 | 42.42 | 3.72 |
| 32 000 | 896.1 | 1 104.2 | 1.23 | 10.79 | 36.63 | 3.39 |
| 48 000 | 800.1 | 1 023.3 | 1.28 | 10.25 | 32.15 | 3.14 |
| 64 000 | 720.1 | 952.3 | 1.32 | 9.76 | 28.69 | 2.94 |
| 80 000 | 653.5 | 894.7 | 1.37 | 9.34 | 25.88 | 2.77 |
| 96 000 | 599.0 | 844.4 | 1.41 | 8.93 | 23.57 | 2.64 |
| **128 000** | — | **754.0** | — | — | **20.01** | — |

**128 000 is new.** The 0.23 arm's pool settled at `mml` 122 633 and its ladder
stopped at 96 221; this one holds 140 000 tokens of KV and ran the rung.

The gap is decode's, not prefill's: prefill is 11–41 % better and decode is 3–4×
better. That is where the two stacks differ in kernel, too —

| | 0.23 arm | 0.27 arm |
|---|---|---|
| W4A16 kernel | `TritonW4A16LinearKernel` | **`RDNAHybridW4A16LinearKernel`** |
| decoder attention | `ROCM_ATTN` | `ROCM_ATTN` |

The attention backend is **not** one of the differences: both logs settle it in
the same words — *Overriding with ROCM_ATTN out of potential backends:
['ROCM_ATTN', 'TRITON_ATTN']*. (An earlier draft of this file had the 0.27 arm
on `TORCH_SDPA`; that line is `MMEncoderAttention`'s, the multimodal encoder,
and the route column caught it.) What is left differing in the decode path is
the W4A16 kernel, and batch-1 decode is dominated by the weight-dequantising
GEMM, which is what that kernel is. **This is still a hypothesis the campaign
does not test** — the vLLM and ROCm versions moved too. Naming it is not
attributing to it: see "What this does not establish".

## The patch was asserted, and then confirmed against a known arm

`run.sh` refuses to measure unless `chunked_prefill_paged_decode.py` reads
`84c6d4f9b2dfe2714b3a8f43ee832b02` — vllm#45916 applied. That check exists
because [`campaign-2026-09-02c`](../campaign-2026-09-02c/) was written after the
container silently lost the patch and a run reproduced the *stock* arm while
reporting itself as the patched one.

An md5 says the file is patched, not that the path is taken, so the run is also
tied to a known number: **36.63 tok/s at 32 000 here against 36.12 at 32 768** in
[`hybrid-splitkv-027`](../hybrid-splitkv-027/)'s controlled A/B, whose stock arm
at that depth is **3.82**. 1.4 % apart from the patched arm and an order of
magnitude from the unpatched one.

## Three attempts, and what the two failures measured

The first two attempts produced no measurements and are kept —
`results-failed-util085.jsonl` and `results-failed-util092-nomns.jsonl`.

Both died on `No available memory for the cache blocks`, at every `mml` from
132 000 down to 8 250. Halving the ladder was the runner's only lever and it was
the wrong one: the shortfall is not the ladder's length. From the serve log:

    Model loading took            10.11 GiB     (per card, TP=2)
    Estimated CUDA graph memory:   6.82 GiB
    Available KV cache memory:    -1.51 GiB

Weights and the graph estimate together are 16.9 GiB of a 19.98 GiB card — a size
the log itself gives away, since its `--gpu-memory-utilization=0.9200 is
equivalent to 0.5786` line is that 6.82 GiB expressed as a fraction. It is an
**absolute reservation, not a percentage**: the same 6.82 GiB is 8.5 % of an
H100 and nobody notices it there.

The cause is not the stack. `campaign-2026-09-02c` served this checkpoint in this
container, and its log and ours agree to the digit on model loading and on the
784-token attention block the mamba page forces. They differ in one line:

| | `--max-num-seqs` | estimated CUDA graph memory | KV available |
|---|--:|--:|--:|
| `campaign-2026-09-02c` | **16** | 0.95 GiB | +4.4 GiB |
| attempts 1 and 2 here | unset | **6.82 GiB** | −1.51 GiB |

5.87 GiB, which is exactly the distance between those two KV figures. The
capture list scales with `max_num_seqs`: unset it runs 51 sizes to 512, at 16 it
is about five. The 0.23 runner this one is copied from never needed the flag,
because vLLM began charging CUDA graph memory to the budget in 0.21 — the cost
existed before and was simply taken after the KV cache was sized.

## What this does not establish

- **It is not an attribution.** Between the two arms the vLLM version, the ROCm
  version, the W4A16 kernel, `--gpu-memory-utilization` (0.85 → 0.92) and
  `--max-num-seqs` (161 → 16) all differ. The decoder's attention backend does
  not, which narrows the list but does not end it. The last two are not
  choices: 0.85 places no KV block at all on 0.27, and `mns` had to be capped for
  the reason above. Neither moves a single-stream rate — they decide how many
  sequences fit, not how fast one runs — but they are differences.
- **The 500 rung was not run**, so the published table's "500 → deepest" column
  cannot be rewritten from this file alone. The slope above does not need it:
  both fits are over the same 8 000–96 000 span.
- **Nothing here says the other five arms would move the same way.** They cannot
  be re-run on 0.27 together in any case — gemma-4 does not start on this
  image: vLLM 0.27's Gemma4 converter reads a global `head_dim` off a
  per-layer transformers config and the accessor raises before loading
  ([the traceback](../campaign-2026-08-29/logs/G31-tp2-on-027.log); fixed
  upstream in vllm#49797, which 0.27.1 lacks).

## Provenance

    container    vllm-027, rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0
    vllm         0.27.1.dev5+gf46a9dfe2.d20260827        ROCm 10.0, kernel 7.0.0-30
    patch        chunked_prefill_paged_decode.py 84c6d4f9b2dfe2714b3a8f43ee832b02 (vllm#45916)
    checkpoint   /data/incoming/Qwen3.8-27B-AWQ-INT4
    serve        TP=2, mml 130000, util 0.92, max-num-seqs 16, port 8000
    engine       weights 22.89 s, load 26.5 s, init 316.5 s, KV 4.37 GiB = 140 000 tokens, 1.08x
    host link    both cards x16 at the host root ports, read before the run (host_link.json)
    prompts      /data/rccl-build/v2/prompts-qwen, the 2026-09-03 cut, the same rungs as
                 campaign-2026-09-03 so the two sittings share their ladder
    runner       runner.py, campaign-2026-09-03/runner.py with its work directory, container
                 and arm list changed and nothing else
