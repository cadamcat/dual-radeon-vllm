# The A100 half of the 2026-08-29 ladder campaign

Twelve configurations, eleven rungs each, two rounds a rung, on one
**A100-SXM4-80GB** (Colab high-memory) running **vLLM 0.28.0**. The same ladder
as the Radeon side — the eleven targets cut from Darwin's *Origin of Species*,
Gutenberg #1228, per tokenizer — so a rung is a token count and the two halves
are comparable point for point.

`results.jsonl` is the raw data, 554 rows. `run.py` produced it, `setup.py`
built the machine, and `harvester.py` is what kept the results when the machine
did not — see **Four reclaims** below.

## What vllm#45450 does, and where it does nothing

Three models, each measured with speculation on, patched against unpatched.
Identical serve commands within a pair; the difference is
`45450-validation/inject_45450.py` applied to the running container.

| model | attention backend | `PROBE_3D_SPEC_ACTIVE` | mean | at 32 K |
|---|---|--:|--:|---|
| `gemma-4-31B-it` | TRITON_ATTN | 1 | +45.6 % | 17.48 → 34.02 (**+94.6 %**) |
| `gemma-4-26B-A4B` | TRITON_ATTN | 1 | +48.4 % | 32.52 → 64.68 (**+98.9 %**) |
| `Qwen3.8-27B` | **FLASH_ATTN** | **0** | **−0.08 %** | 20.51 → 20.52 (+0.04 %) |

The patch nearly doubles decode at 32 K on the two models vLLM routes onto the
Triton unified-attention kernel, and does nothing at all on the one it does
not. Not approximately nothing: 20.51 against 20.52, with the two arms
repeating to 0.00 % and 0.03 %.

**The probe predicts it.** `inject_45450.py` prints `PROBE_3D_SPEC_ACTIVE` once
per worker process when the 3D path serves a step wider than one token. It
needs three things at once — the patch installed, the Triton kernel on the
path, and speculation on. Wherever it printed, the patch moved the numbers;
wherever it stayed silent, the two ladders are indistinguishable. That held on
this machine and on the Radeons, across five patched arms, without exception.
The count is the worker count: 1 here at TP=1, 2 on the Radeons at TP=2, which
is what `45450-validation/README.md`'s "exactly once" means.

Qwen3.8 is routed away from the Triton kernel on both vendors — `FLASH_ATTN`
here, `ROCM_ATTN` on the 0.27 ROCm image — so `#45450` is inert for it on
either machine unless the backend is chosen explicitly. The Radeon half does
choose it explicitly, and there the same patch is worth **+187 %** at 32 K.

## DFlash

`A100-MG30-dflash` is `Muse-Glimmer-30B-INT4` with its block-diffusion
assistant. The method is **`dflash`**, not `draft_model`: the latter fails
validation with `Draft model vocab_size=0`, the drafter's config having no
`vocab_size` at all. On the ROCm images the arm cannot run at any setting —
0.27 and 0.23 register DFlash for `qwen3_dflash` and `laguna_dflash` only, and
`MuseGlimmerAssistantModel` is in neither their registry nor
`transformers_utils/configs`. 0.28 registers it, mapping both
`MuseGlimmerAssistantModel` and `DFlashMuseGlimmerAssistantModel` onto
`qwen3_dflash`'s `DFlashQwen3ForCausalLM`, so this is the machine where the arm
exists.

It is a **net loss at every depth, deepening with context**: −28.3 % at 500 to
−47.5 % at 32 K, k=8, every rung chart-grade (ranges 0.00–0.60 %). Its own
control barely decays at all — 66.94 to 60.41 across 64× of context — so the
slope belongs to the drafter. Acceptance length falls 5.30 → 2.36.

## Four reclaims

The VM was reclaimed without warning four times: 15:58, 16:56, ~18:21 and
~19:55 UTC. Nothing measured was lost, because `harvester.py` pulls
`results.jsonl` to the operator's machine on a loop and each restored file is
uploaded back before the next run so `run.py`'s checkpoint skips what is
already done. What did not survive is the **serve logs of the nine
configurations measured on earlier VMs**: `logs/` holds only those from the
fourth, plus `serve-A100-MG30-dflash-attempt1.log`, kept because it is the
evidence that `dflash` is accepted at all. The backend and probe columns above
are read from the logs that exist; for the nine, the results are the record.

Two things the losses taught, both now in the code here:

- Killing `vllm serve` does not free the card. The workers run as
  `VLLM::EngineCore`, a command line containing neither "vllm" nor "serve".
  One held 72.7 GiB of 80 after its parent died, and the next configuration
  failed its own memory check and was recorded as a crash it had nothing to do
  with. `run.py` now kills both and waits on `nvidia-smi`, not on a process
  list.
- `pkill -f 'vllm serve'` under `shell=True` can match its own shell, whose
  command line contains the pattern. `'[v]llm serve'` does not.
