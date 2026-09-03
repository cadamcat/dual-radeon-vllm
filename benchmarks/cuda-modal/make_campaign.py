#!/usr/bin/env python3
"""Write a campaign directory for one rented machine.

Seven machines are being measured on the same ladder with the same five or six
models, and the only things that differ are the `gpu=` string Modal is asked
for, the tensor-parallel size, and which models the card can hold. Hand-copying
`run.py` and `app.py` seven times is how a config table ends up disagreeing
with the directory it sits in.

This writes both files from the committed template plus a spec, so what ran is
still beside the data it produced -- the repository's rule -- and the thing
that varies is one dict in one place.

    python3 make_campaign.py h200            # write cuda-h200/campaign-<date>
    python3 make_campaign.py --list

Every generated `run.py` is `harness/runner_cuda.py` with the docstring and
`CFGS` replaced and nothing else, which the generated header records with the
template's md5 so a reader can check that claim.
"""
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
TEMPLATE = os.path.join(BENCH, "harness", "runner_cuda.py")

# Every model on the `llm-ckpt` Volume, with the two facts a config table needs
# and cannot infer: the ladder ceiling the checkpoint's own config.json admits,
# and whether it needs a `max_num_seqs` that is not the default.
#
# `mns` 512 on the hybrid-SSM model rather than the default: it reserves one
# Mamba block per sequence and an 80 GiB card at mml 132 000 has 969, so the
# default 1 024 does not start. 512 is the smallest value that still produces
# the default's CUDA graph capture set (both cap at max_cudagraph_capture_size
# 512), so at batch 1 it replays the same graphs as the default would.
# `mml` is stated here rather than derived from the ladder, and that is the
# point: it must not move when the machine does. Every one of these is the
# value cuda-h100/campaign-2026-09-03 used, so a row from any later machine
# differs from an H100 row in the machine and not in the context budget.
# MG30 is the exception and cannot be otherwise -- its config.json caps at
# 131 072, which is why 132 000 cost it a configuration on 2026-09-03 -- so it
# carries 131 072 on every machine instead.
MODELS = {
    "B8":     dict(model="Qwen3-8B", max_ctx=40960, mml=33000),
    "G12":    dict(model="gemma-4-12B-it-qat-w4a16-ct", max_ctx=262144, mml=132000),
    "G26A4B": dict(model="gemma-4-26B-A4B-AWQ", max_ctx=262144, mml=132000),
    "G31":    dict(model="gemma-4-31B-it-qat-w4a16-ct", max_ctx=262144, mml=132000),
    "Q38":    dict(model="Qwen3.8-27B-AWQ-INT4", max_ctx=262144, mml=132000, mns=512),
    "MG30":   dict(model="Muse-Glimmer-30B-INT4", max_ctx=131072, mml=131072),
}

TARGETS = [500, 1000, 2000, 4000, 6000, 8000, 12000, 16000, 20000, 24000, 32000]
LONG = [48000, 64000, 80000, 96000, 128000]
GEN_PLUS = 512 + 100          # what a rung costs beyond the prompt

SPECS = {
    "h200": dict(gpu="H200", machine="H200-143GB-HBM3e", rate=4.54, tp=1,
                 models=["G31", "G12", "G26A4B", "Q38", "B8", "MG30"],
                 budget_s=3000,
                 why="The H100's compute capability, power cap and SM clock "
                     "ceiling, with 143 771 MiB against 81 559. The one pair "
                     "in this catalogue that moves memory bandwidth alone."),
    "b300": dict(gpu="B300", machine="B300-SXM6", rate=7.10, tp=1,
                 models=["G31", "G12", "G26A4B", "Q38", "B8", "MG30"],
                 budget_s=3000,
                 why="sm_103, 275 040 MiB, PCIe gen 6, a 1 100 W cap."),
    "pro6000": dict(gpu="RTX-PRO-6000", machine="RTX-PRO-6000-Blackwell", rate=3.03,
                    tp=1, models=["G31", "G12", "G26A4B", "Q38", "B8", "MG30"],
                    budget_s=3600,
                    why="sm_120 and GDDR7 -- the workstation Blackwell die, the "
                        "closest thing in a datacentre catalogue to the consumer "
                        "parts this repository is about."),
    # The platform control. Every A100 row this sweep is compared against was
    # measured on Colab; every card tonight was rented from Modal. The L4 is
    # the one GPU that exists on both, and Colab measured G12 on it twice --
    # 2026-08-30 and 2026-09-02, agreeing to 0.2-0.8 % -- so the band that
    # decides this is not something chosen after seeing the answer.
    # The baseline itself, on the platform the new cards were rented from.
    # Every headline ratio tonight divides by an A100 row Colab measured. The
    # L4 control says the two platforms agree on a card that appears in no
    # conclusion; this says whether they agree on the card that appears in all
    # of them. Same mml and same eleven rungs as cuda-a100/campaign-2026-08-30.
    "a100": dict(gpu="A100-80GB", machine="A100-SXM4-80GB", rate=2.50, tp=1,
                 models=["G12", "B8"], budget_s=1500,
                 override={"G12": dict(mml=33000, max_ctx=33000),
                           "B8": dict(mml=33000, max_ctx=33000)},
                 why="The A100 every ratio here divides by, rented rather than "
                     "granted, at the mml and the ladder Colab used."),
    "l4": dict(gpu="L4", machine="L4", rate=0.80, tp=1,
               models=["G12"], budget_s=1800,
               why="The same L4 Colab measured twice, on the same model, to "
                   "put a number on the platform difference every "
                   "machine-to-machine ratio here inherits."),
    "b200": dict(gpu="B200", machine="B200-SXM", rate=6.25, tp=1,
                 models=["B8", "G26A4B"], budget_s=1500,
                 why="sm_100 beside B300's sm_103, on the two models that sit at "
                     "the ends of the mem_busy range: 87 % and 38 %."),
    "h100-tp2": dict(gpu="H100:2", machine="H100-80GB-HBM3-x2", rate=7.90, tp=2,
                     models=["B8", "G12", "G26A4B", "G31", "Q38"], budget_s=3000,
                     why="Two cards with NVLink, against the Radeon pair with no "
                         "P2P at all."),
    # One model, not three, and the choice is the budget's: $15.80/h with $3.51
    # of credit left buys one configuration. B8 is the one worth it -- the TP
    # axis has 1 and 2, the second card was worth 1.484x on this model against
    # 1.029x on the least memory-bound one, and the open question is whether
    # that keeps going or saturates. A third point on the steepest curve
    # answers it; a first point on a flat one would not.
    "h100-tp4": dict(gpu="H100:4", machine="H100-80GB-HBM3-x4", rate=15.80, tp=4,
                     models=["B8", "G26A4B"], budget_s=900,
                     why="Four-way at both ends of the mem_busy range: Qwen3-8B "
                         "at 87 %, which the second card was worth 1.484x to, "
                         "and the 26B MoE at 38 %, which it was worth 1.029x "
                         "to. One point would be an anecdote; two that far "
                         "apart is a test."),
    "pro6000-tp2": dict(gpu="RTX-PRO-6000:2", machine="RTX-PRO-6000-Blackwell-x2",
                        rate=6.06, tp=2,
                        models=["B8", "G12", "G26A4B", "G31", "Q38"], budget_s=3600,
                        why="Two cards with NO NVLink -- nvidia-smi nvlink -s is "
                            "empty on this pair -- against H100:2 which has it."),
}


def cfg_lines(spec):
    """the CFGS table, with each model's ladder cut to what it will accept

    `spec["override"]` lets a control campaign hold `mml` and the ladder where
    the campaign it is a control for held them, rather than where this file's
    defaults put them. The A100 control exists to be compared with rows Colab
    measured at mml 33 000 and eleven rungs; giving it sixteen rungs at
    132 000 would make it a different measurement wearing the same name.
    """
    out = []
    for mid in spec["models"]:
        m = dict(MODELS[mid], **spec.get("override", {}).get(mid, {}))
        want = TARGETS + LONG
        mml = m["mml"]
        rungs = [t for t in want if t + GEN_PLUS <= min(mml, m["max_ctx"])]
        bits = [f'id="{mid}"', f'model="{m["model"]}"', f"mml={mml}"]
        if len(rungs) != len(TARGETS):
            bits.append("targets=" + ("TARGETS + LONG" if rungs == want
                                      else repr(rungs)))
        if m.get("mns"):
            bits.append(f'mns={m["mns"]}')
        if spec["tp"] > 1:
            bits.append(f'tp={spec["tp"]}')
        note = ""
        if max(rungs) < 128000:
            note = (f'\n    # {m["model"]}: its own config.json caps context at '
                    f'{m["max_ctx"]}, so the ladder stops at {max(rungs)}.')
        out.append(f"{note}\n    dict({', '.join(bits)}),")
    return "".join(out)


def write(name, date=None):
    spec = SPECS[name]
    date = date or time.strftime("%Y-%m-%d")
    d = os.path.join(BENCH, f"cuda-{name.split('-')[0]}",
                     f"campaign-{date}" + ("" if spec["tp"] == 1 else f"-tp{spec['tp']}"))
    os.makedirs(d, exist_ok=True)
    src = open(TEMPLATE).read()
    md5 = hashlib.md5(src.encode()).hexdigest()

    q = chr(34) * 3
    doc = (q + f'run.py -- {spec["machine"]}, TP={spec["tp"]}, '
           f'{len(spec["models"])} models.\n\n'
           f'{spec["why"]}\n\n'
           f'Generated by cuda-modal/make_campaign.py from '
           f'harness/runner_cuda.py (md5 {md5}) with the docstring and CFGS\n'
           f'replaced and nothing else. Each model\'s ladder is cut to what its\n'
           f'own config.json admits, not to one number chosen for the set:\n'
           f'asking gemma-4\'s 132 000 of Muse-Glimmer\'s 131 072 cost a\n'
           f'configuration on 2026-09-03 before the runner learned to retry it.\n\n'
           f'    modal run benchmarks/{os.path.relpath(d, BENCH)}/app.py\n' + q)
    i = src.index(q); j = src.index(q, i + 3) + 3
    src = doc + src[j:]
    a = src.index("# BENCH_CFGS picks a subset by id")
    b = src.index("\n]\n", a) + 3
    src = (src[:a] + "# BENCH_CFGS picks a subset by id, as the Radeon runner does.\n\n"
           f"LONG = {LONG}\n\nCFGS = [{cfg_lines(spec)}\n]\n" + src[b:])
    src = src.replace('print("A100_CAMPAIGN_DONE", flush=True)',
                      'print("CAMPAIGN_DONE", flush=True)')
    open(os.path.join(d, "run.py"), "w").write(src)

    app = open(os.path.join(HERE, "app_template.py")).read()
    for k, v in (("@@GPU@@", spec["gpu"]), ("@@MACHINE@@", spec["machine"]),
                 ("@@RATE@@", str(spec["rate"])), ("@@BUDGET@@", str(spec["budget_s"])),
                 ("@@RUN@@", f"{name}-{date}"), ("@@APP@@", f"bench-{name}"),
                 ("@@ORDER@@", json.dumps(spec["models"])),
                 ("@@WHY@@", spec["why"])):
        app = app.replace(k, v)
    open(os.path.join(d, "app.py"), "w").write(app)
    print(f"  {name:<12} -> {os.path.relpath(d, BENCH)}  "
          f"({len(spec['models'])} models, gpu={spec['gpu']}, "
          f"${spec['rate']}/h, cap ${spec['budget_s']/3600*spec['rate']:.2f})")
    return d


if __name__ == "__main__":
    if "--list" in sys.argv or len(sys.argv) < 2:
        for k, s in SPECS.items():
            print(f"  {k:<14} gpu={s['gpu']:<16} tp={s['tp']}  "
                  f"{len(s['models'])} models  ${s['rate']}/h")
        sys.exit(0)
    for n in sys.argv[1:]:
        write(n)
