# Six rented machines, described — 2026-09-02

Every machine in this repository's projections carries a description, and until
today each one came from a machine somebody had to obtain and keep: two Radeons
in a box under a desk, and three Colab shapes whose availability decided what
could be measured. The A100-80GB cell in the a100 article has been open since
2026-09-02 for exactly that reason — Colab Pro lapsed and `shape=hm` stopped
being offered.

Modal rents them by the second. `probe_gpu.py` asks for a GPU, reads
`nvidia-smi --query-gpu`, and terminates. **Six machines, 39 seconds,
$0.054.**

| `gpu=` | what arrives | memory | power cap | SM / mem clock | cc | PCIe | scheduled |
|---|---|--:|--:|--:|--:|--:|--:|
| `B300` | NVIDIA B300 SXM6 AC | 275 040 MiB | **1 100 W** | 2 032 / 3 996 MHz | **10.3** | **gen 6** ×16 | 1.21 s |
| `B200` | NVIDIA B200 | 183 359 MiB | 1 000 W | 1 965 / 3 996 MHz | 10.0 | gen 5 ×16 | 0.62 s |
| `H100` | NVIDIA H100 80GB HBM3 | 81 559 MiB | 700 W | 1 980 / 2 619 MHz | 9.0 | gen 5 ×16 | 0.63 s |
| `RTX-PRO-6000` | NVIDIA RTX PRO 6000 Blackwell Server Edition | 97 887 MiB | 600 W | 2 430 / **12 481 MHz** | **12.0** | gen 5 ×16 | 0.63 s |
| `L40S` | NVIDIA L40S | 49 140 MiB | 350 W | 2 520 / 9 001 MHz | 8.9 | gen 4 ×16 | 0.64 s |
| `A100-80GB` | NVIDIA A100-SXM4-80GB | 81 920 MiB | 500 W | 1 410 / 1 593 MHz | 8.0 | gen 4 ×16 | 0.62 s |

All six on driver 580.95.05, CUDA 13.0. Nothing queued: the longest wait for a
card was **1.21 s**, on the largest one.

## Three things the table says that a price list does not

**`A100-80GB` returns `NVIDIA A100-SXM4-80GB`** — the same string this
repository's existing A100 rows carry. So this is not only six new machines; it
is the machine three published comparisons already use, available without a
subscription. The open 80 GB cell costs an hour of it.

**`RTX-PRO-6000` is compute capability 12.0 and its memory clock is
12 481 MHz.** That is `sm_120` and GDDR7 — the workstation Blackwell die, not
the datacentre one that `B200` and `B300` report as 10.0 and 10.3. Kernels are
compiled per architecture, so it is a different software path from the other two
Blackwells, and it is the closest thing in a datacentre catalogue to the
consumer part this repository is about.

**`B300` reports PCIe gen 6** where the rest report 5 or 4. This repository spent
2026-09-02 establishing that one card at PCIe 3.0 x8 instead of x16 moved a
published prefill coefficient by 18.3 %, so a machine two generations further up
that axis is not a detail.

## What is deliberately not here

`power.limit` equals `power.max_limit` on every one of the six, so none of these
cards is being rented to us de-rated. That is worth knowing because it is not
true of everything: the L4 measured on Colab on 2026-09-02 ran at a 72 W cap
throughout, and the Radeon pair sits at 100.0–100.4 % of its 265 W cap during
prefill at depth. **A power cap is a measurement condition, and this file
records it before any measurement is taken.**

No performance number is in this directory. Nothing here has run a model.

## Files

    probe_gpu.py     ask for a GPU, read nvidia-smi, terminate
    machines.jsonl   one `kind: modal_machine` row per GPU, nvidia-smi's own
                     field names, plus `scheduled_s` and `wall_s` so the bill
                     is checkable against `modal billing rates`
