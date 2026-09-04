# What removing the hostcall requirement costs — 2026-09-04

`docs/open-questions.md` §6 has recorded the cost of the `NDEBUG` fix as
**assumed, not measured**, since the fix was found. The reason was platform, not
inclination: until 2026-08-23 this box had no PCIe AtomicOps, so the library
that *carries* a hostcall declaration could not be dispatched here at all and
there was nothing to compare against. It has atomics now. Both libraries run.

    RCCL 2.27.7, one source tree, two builds, differing by one line

    arm=ndebug     add_compile_definitions(NDEBUG)     hostcall 0   126 kernels
    arm=nondebug   that line removed                   hostcall 6   126 kernels

    three interleaved sweeps per arm, 55 cells each, RX 7900 XT pair, TP=2

    t_graph_us, 16 KB - 512 KB     nondebug 2.7-4.6 % slower, 29 of 30 cells
    t_graph_us, 2 MB and above     no difference (median 1.0001, 11/20, p=0.82)
    t_graph_us, 8 KB (ntok=1)      nondebug 1.3 % FASTER, 5 of 5 cells
    twelve correctness cases       12/12 under both libraries

    end to end, 2 models x 3 depths x 5 repeats, 60 runs, 0 errors
    decode rate                    no difference: <= 0.6 % in every cell,
                                   inside a noise floor of up to 4.9 %

    the capability matrix, 2 platform states x 3 libraries, one sitting
    atomics present                all three dispatch, 12/12 each
    atomics absent                 only the 0-hostcall build dispatches

**The fix is free where it matters and cheap where it does not.** At the batch-1
decode shape a served step actually reduces, the arm without the fix is not
slower; in a band from 16 KB to 512 KB it is a few percent slower; past 2 MB the
collective is bandwidth-bound and the difference vanishes.

---

## What is being compared, exactly

Not "the cost of a metadata declaration". `NDEBUG` removes the device `assert()`
calls as well as the implicit argument they cause, so the two arms differ in
both metadata *and* code:

| | arm=ndebug | arm=nondebug |
|---|---|---|
| `hidden_hostcall_buffer` | 0 | **6** |
| `__assert_fail` | 0 | 4 |
| `__ockl_fprintf` | 0 | 3 |
| kernels | 126 | 126 |
| version string | `RCCL version 2.27.7` | `RCCL version 2.27.7` |
| deployable md5 | `76f1916f…` | `666c8aae…` |

So this measures **the fix**, which is what a reader deciding whether to apply
it needs. It does not isolate the declaration from the assert code, and no
claim here should be read that way.

The kernel count is identical, which is the structural half of "the fix changes
nothing": `NDEBUG` does not add or remove kernels, only what they declare and
the assert paths inside them.

## Same source, one line apart

There is no commit to quote — the tree is a `--depth 1` clone of
`ROCm/rccl` at `release/rocm-rel-7.1.1.1` and has no `.git` — so it is hashed
instead. Both arms were generated from the same `CMakeLists.txt.bak` in one
scripted run, back to back, with identical configure options:

    tree md5 excluding CMakeLists.txt   318534aeaba1aac01516f70af34dbb08
    CMakeLists.txt.bak md5             51e72f46a456af4deaf577160a6594cc
    diff .bak -> arm=ndebug            17a18
                                       > add_compile_definitions(NDEBUG)

`arm=nondebug`'s `CMakeLists.txt` is byte-identical to the `.bak`, and its
`build.ninja` contains **zero** occurrences of `NDEBUG`, so the flag did not
leak into the control from a Release preset. `arm=ndebug`'s contains 251.

This is the comparison [CAL-OUTLINE §B1] asks for and warns about: a
2.27.7-no-hostcall library measured against a stock 2.30.4 would move version
and hostcall together and answer a different question.

**Both arms carry `COLLTRACE:BOOL=ON`** (250 `ENABLE_COLLTRACE` occurrences in
each `build.ninja`). The global `NDEBUG` alone takes `__assert_fail`,
`__ockl_fprintf` and `hidden_hostcall_buffer` to 0/0/0 with the trace path still
compiled in. @adderek's counts in [ROCm#6520], quoted in
[open-questions §3](../../docs/open-questions.md), reached 0/0/0 only with
`-DCOLLTRACE=OFF` *and* `NDEBUG`; the shipped Release build there scored 5/3/6
against this build's 4/3/6. Different branch, different box, same two ends.

## Design: interleaved, and read against its own noise

Six sweeps ran **A B A B A B**, not A A A B B B. Blocking would confound the
half hour's drift with the arm, and at n=3 that is the likeliest way to
manufacture an effect. The library is reinstalled and its md5 re-asserted
before every sweep; each run records the library it actually mapped, read from
`/proc/self/maps`.

A ratio is meaningless without the spread it sits in:

| metric | noise, (max−min)/mean within an arm | pooled ratio | slower | sign p |
|---|---|---|---|---|
| `t_graph_us` | median 2.01 %, p90 4.28 % | 1.0191 | 40/55 | 0.0010 |
| `t_stream_us` | median 2.76 %, p90 8.10 % | 1.0031 | 34/55 | 0.1048 |
| `t_sync_us_median` | median 1.61 %, p90 3.63 % | 1.0061 | 37/55 | 0.0145 |

The effect and the noise are the same size. What separates them is direction:
each cell is a paired comparison, and the count of cells pointing one way is a
sign test that does not care that a single cell's 2 % is unreadable.

**Only `t_graph_us` reproduces in all three sweeps** — slower in 42, 37 and 39
of 55, with sign p 0.0001, 0.0145, 0.0027. `t_stream_us` and `t_sync_us_median`
both reverse in sweep 3 (24/55 and 22/55), so neither is established here.
`t_graph_us` is also the one a served step pays: vLLM captures its decode step
into a HIP graph and replays it.

## Where the cost is, and where it is not

| ntok | bytes at hidden 4096 | median ratio | slower | ndebug µs | nondebug µs |
|---|---|---|---|---|---|
| 1 | 8 KB | **0.9873** | **0/5** | 19.29 | 19.10 |
| 2 | 16 KB | 1.0456 | 4/5 | 26.47 | 27.53 |
| 4 | 32 KB | 1.0413 | 5/5 | 29.45 | 30.57 |
| 8 | 64 KB | 1.0457 | 5/5 | 33.10 | 34.59 |
| 16 | 128 KB | 1.0348 | 5/5 | 43.24 | 44.75 |
| 32 | 256 KB | 1.0372 | 5/5 | 59.01 | 61.08 |
| 64 | 512 KB | 1.0268 | 5/5 | 91.34 | 93.49 |
| 256 | 2 MB | 0.9993 | 2/5 | 310.60 | 310.00 |
| 1024 | 8 MB | 0.9932 | 2/5 | 1203.80 | 1197.90 |
| 4096 | 32 MB | 1.0012 | 4/5 | 4718.91 | 4721.22 |
| 16384 | 128 MB | 1.0043 | 3/5 | 18837.76 | 18884.46 |

    ntok <= 16    median 1.0390   slower 19/25   sign p 0.0146
    ntok >= 256   median 1.0001   slower 11/20   sign p 0.8238

In absolute terms the gap in the affected band is **1.1 to 2.2 µs** — 26.47 →
27.53 at ntok 2, 91.34 → 93.49 at ntok 64. Past 2 MB it is inside the noise in
both directions.

**The ntok=1 reversal is not explained.** It is five cells out of five, at the
one size the letter's headline case uses, and it is the opposite sign from its
neighbours. Two builds differ in more than a flag — code layout, alignment and
the presence of the assert paths all move — so a small size-dependent
difference that is not monotonic is exactly what an unexplained build effect
would look like. It is reported as measured and not smoothed away.

## End to end: no difference, and a confound the design caught

Sixty served requests — gemma-4-12B and Qwen3-8B, 500 / 8 000 / 32 000 tokens,
five repeats, both libraries, 256 generated tokens each, streaming, a random
seed prefix so nothing comes from cache. The request shape is
`benchmarks/campaign-2026-09-03/runner.py`'s, unchanged, so these rows sit
beside that campaign's. Zero errors, and all four sessions verified that the
**serving process** had mapped the arm's library, by md5 out of
`/proc/<pid>/maps` rather than by what the orchestrator believed it installed.

| model | depth | ndebug tok/s | nondebug tok/s | by arm | worst spread |
|---|---|---|---|---|---|
| gemma-4-12B | 500 | 59.66 | 59.81 | 1.0024 | 4.94 % |
| gemma-4-12B | 8 000 | 52.23 | 52.47 | 1.0046 | 2.53 % |
| gemma-4-12B | 32 000 | 41.71 | 41.64 | 0.9983 | 1.39 % |
| Qwen3-8B | 500 | 79.22 | 79.62 | 1.0051 | 1.70 % |
| Qwen3-8B | 8 000 | 73.42 | 73.43 | 1.0002 | 0.23 % |
| Qwen3-8B | 32 000 | 61.82 | 61.87 | 1.0008 | 0.39 % |

**No cell differs by more than 0.6 %, five of six are inside 0.5 %, and five
of six put the unfixed arm nominally ahead.** On Qwen3-8B at 8 000 and 32 000 tokens the five repeats agree to 0.2–0.4 %,
so this is not a measurement too blunt to see a difference; there is no difference to see. That
is the same answer the collective sweep gives at ntok=1, which is the shape a
batch-1 decode step reduces.

### The prefill cell that looked real, and was not

Prefill on Qwen3-8B at 32 000 tokens separates completely — every ndebug run
faster than every nondebug run, 2 134–2 146 against 2 052–2 073 tok/s, a 3.6 %
gap with a 1 % spread and Mann-Whitney p = 0.008. Read as an arm effect it says
the unfixed library costs 3.6 % of prefill.

It is not an arm effect. `librccl` cannot be swapped under a running server, so
each arm needs its own session, and **the arm order was balanced across the two
models** — gemma ran ndebug first, Qwen ran nondebug first. That balance is
what lets the two explanations be told apart:

| | 2nd session / 1st | nondebug / ndebug |
|---|---|---|
| gemma-4-12B, 8 000 | 0.9991 | 0.9991 |
| gemma-4-12B, 32 000 | 1.0026 | 1.0026 |
| Qwen3-8B, 8 000 | **1.0087** | 0.9914 |
| Qwen3-8B, 32 000 | **1.0376** | 0.9637 |

The second session of a pair is faster in three of four cells and the sign by
arm is not consistent, so what the Qwen 32 000 cell measures is **session
order**, not the library. Its ndebug arm simply ran second. A blocked design —
both models with the same arm first — would have produced the same numbers and
published them as a 3.6 % cost.

The order effect itself is unexplained and is not small: 3.8 % on one cell.
Resolving it needs prefill measured with the arms interleaved, which for a
served model means a server restart per repeat. Not done here.

**The 500-token prefill and TTFT cells are uninterpretable** — spreads of 99 %
and 236 %, because at that length TTFT is scheduling jitter. They are recorded
and excluded from every statement above.

## Correctness

`collective_correctness.py` runs the twelve cases
[docs/vfio-atomics.md](../../docs/vfio-atomics.md) has described in prose since
the beginning and which have never had a script behind them: `all_reduce(SUM)`
and `all_gather_into_tensor`, float32 / float16 / bfloat16, 1 024 and 1 048 576
elements, elementwise against ground truth computed on the host from the same
generator that fills the device tensors. A collective that silently does
nothing fails rather than passing on an unchanged buffer.

**12/12 under both libraries**, twice — once in each measurement round. So the
arm that declares six hostcall buffers dispatches, computes correctly, and is
within a few percent on timing, on a platform that can satisfy the requirement.
That is the last link of [root-cause.md](../../docs/root-cause.md)'s chain
closed from the other side: the declaration is only fatal where the platform
cannot honour it.

## The capability matrix

CAL's Table I. Two platform states, three libraries, one sitting, and the
platform state read out of `lspci` and `dmesg` by the script rather than
asserted by the operator.

| | stock 2.30.4<br>13 declarations | 2.27.7 `-NDEBUG`<br>6 declarations | 2.27.7 `+NDEBUG`<br>0 declarations |
|---|---|---|---|
| **atomics present** | 12/12 | 12/12 | 12/12 |
| **atomics absent** | **REFUSED** | **REFUSED** | **12/12** |

Both refusals are the same string, and it is the one the whole repository
started from: `the operation cannot be performed in the present state`.

**The middle column is why this is three columns and not two.** It and the
right-hand column are the same source one line of CMake apart — the B1 pair —
so in the bottom row the only difference between the cell that is refused and
the cell that computes twelve correct collectives is the declaration itself. A
two-column table invites "you compared 2.30.4 with 2.27.7 and the version
changed"; this one does not.

The left column carries the other half: **stock RCCL 2.30.4 is correct when the
platform can satisfy it.** The library is not broken. "Downgrade and it works"
becomes a prediction of the mechanism rather than folklore.

### The toggle, and what it is

One line of VM configuration per card:

    hostpci0: 0000:0b:00.0  ->  0000:0b:00      (and the same for hostpci1)

which passes the card's HDMI audio function alongside the GPU, makes QEMU write
`multifunction=on`, and stops `vfio_pci_enable_rp_atomics()` from advertising
completer support. **Both cards, not one** — `docs/vfio-atomics.md` records that
both have been passed as `.0` since 2026-08-23, and flipping one would make the
row mean "one card lacks atomics", which is a different claim.

What each row measured about itself, from the same script that ran the cells:

| | root ports with `32bit+ 64bit+` | `PCIE atomic ops is not supported` in dmesg |
|---|---|---|
| atomics present | 2 | 0 |
| atomics absent | 0 | 2 |

The guest was shut down, reconfigured and restarted between the rows; the
`present` row was measured before the change and again by inspection after the
revert, and the two agree line for line in `lspci`. Nothing else moved: same
container, same three library files, same twelve cases.

## What this round also corrected

Round 1 recorded `hidden_hostcall_buffer: 0` for **both** arms. The harness read
it with `strings -a … | grep -c`, and both libraries' `.hip_fatbin` is a CCOB —
zstd-compressed — bundle, so `strings` answers 0 whatever the kernels declare.
The same is true of `librccl-final.so`, the library every earlier measurement on
this box used: its recorded 0 is correct, but the method could not have known
it. `allreduce.py` now extracts the device image and reads the metadata note,
records which method answered, and reports `None` rather than 0 when it cannot
read. Round 2's six runs all record 0 and 6 correctly, via `notes`.

Round 1's timing files are kept as `ar-*.jsonl`; they are one sweep per arm with
the old metadata and are superseded by `ar2-*`.

## Reproducing

The builds are CPU-only and need no lease: 8 cores take about 20 minutes over
507 of 510 targets and then a single-threaded `lld` device LTO link holds the
remaining 70, at 18 GB RSS. 91 minutes per arm on this box.

The measurement holds the lease. `logs/MEASURE2.txt` is the full record,
including the restore: the deployed library back to `ab5b50f0`, both services
restarted, both cards back to the 27 971 584-byte VRAM baseline.

    python3 analyze.py        every figure above, recomputed from the JSONL
