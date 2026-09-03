# Muse-Glimmer, and the three controls 2026-09-03 named — 2026-09-03b

`cuda-h100/campaign-2026-09-03` published a README that named three things it
had not bought. This bought them, on the same card, and added the model that
turned out to matter most.

## Muse-Glimmer-30B: the prediction was wrong and the mechanism got clearer

Written into `run.py` before the run: gemma-4 reads a KV that grows without
bound, Qwen3.8-27B carries a recurrent state that does not grow at all, and
Muse-Glimmer attends through a 2 048-token window — a term that grows and then
stops. **It was predicted to land between the two. It landed below both.**

| 500 → 128 000 decode | H100 | A100 → H100 at 32 000 |
|---|--:|--:|
| **Muse-Glimmer-30B**, window 2 048 | **−4.8 %** | **1.37×** |
| gemma-4-12B | −11.9 % | 2.16× |
| gemma-4-26B-A4B | −17.9 % | 2.08× |
| **Qwen3.8-27B**, hybrid SSM | −21.8 % | 1.45× |
| gemma-4-31B | −22.0 % | 1.90× |

The recurrent-state model is not the flat one — it falls as far as the dense
31B. What flattens a curve is a bounded attention window, not a constant-size
state; the SSM still carries attention layers whose cost grows. Two independent
quantities, the decay and the machine ratio, give the same ordering.

One confound, and it runs against the conclusion: `MG30` is the only model that
changes W4A16 kernel between machines — **Machete** on the Hoppers, Marlin
elsewhere. Machete is the faster of the two on Hopper, so its low ratio is
despite a better kernel, not because of a worse one.

## What `max_num_seqs` is worth: 0.33 %

Two arms, same model, same `mml`, same everything else, 512 against 16:

| | decode, 11 rungs | prefill | KV pool | engine start |
|---|--:|--:|--:|--:|
| `mns` 512 → 16 | **+0.33 % mean, 0.42 % max** | −0.06 % mean | 720 978 → 739 630 tok | 68.4 → 55.9 s |

The A100 campaign pinned `mns` to 16 for this model and `cuda-h100/campaign-2026-09-03`
had it forced to 969 by the Mamba pool, so the two could not be compared
without this. The A100's own note put the effect under 0.7 % on this family;
that was someone else's control and is now reproduced here, single-variable,
same card, same sitting.

At batch 1 the knob touches nothing that runs: `cudagraph_capture_sizes` is
`[1, 2, 4, 8, 16, 24, 32]` at `mns` 16 and `[1, …, 512]` at 969, and the
shorter list is a prefix of the longer. Both replay the size-1 graph.

## What `mml` 132 000 against 33 000 is worth: 0.10 %

`G31-mml33` repeats the eleven-rung ladder at the context budget every A100 and
Radeon row uses, against the same arm at 132 000 on the same card the same day:
**−0.10 % mean, 0.22 % max**, converging with depth (−0.19 % at 500,
−0.06 % at 32 000).

Which retires the caveat `campaign-2026-09-03` opened. Together the two
controls bound the configuration difference between that run's long-ladder arms
and every row they are compared with at **under 0.4 %**, against effects of
100 % and more.

## The one that got away

`MG30`'s first attempt died in 31 s: `User-specified max_model_len (132000) is
greater than the derived max_model_len (max_position_embeddings=131072.0)`.
132 000 was chosen for gemma-4, whose ceiling is 262 144; Muse-Glimmer's is
131 072. Kept as `results-mg30-attempt1.jsonl` and
`logs/serve-MG30-maxlen-crash.log`. The runner now retries on it, beside the KV
retry and the Mamba retry — same shape each time, a different knob.

Two configurations planned here were dropped rather than run: `G12-mml33` and
`G26A4B-mml33`. The `mml` question was answered by `G31-mml33` and by the Q38
pair, and buying the same answer three more times was worth less than the card
time.
