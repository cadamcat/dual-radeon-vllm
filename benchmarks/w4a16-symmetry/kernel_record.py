"""Record which linear kernel each layer actually gets, from inside TP workers.

Under TP=2 the layers are built in spawned workers, so a monkeypatch applied in
the parent process sees nothing: it has to be written to the module on disk,
which the workers re-import. Three probes here needed that and each carried its
own copy; this is the one copy.

The part worth reading is `read`. It used to be spelled inline as

    chosen = ""
    if pathlib.Path(rec).exists():
        chosen = " | ".join(sorted(set(pathlib.Path(rec).read_text().splitlines())))

which turns "the instrumentation never ran" into an empty string that is then
written to the results file as though it were a finding. A probe whose whole
purpose is to show which kernel ran would report nothing at all and still exit
0, and the kernel-selection comparison would be quietly incomplete. An audit on
2026-08-27 flagged it; every record committed before that fix does carry both
ranks, so no published result depended on the silent path, but it was there.

`read` now refuses: missing file, empty file, or fewer ranks than expected are
all exceptions.
"""

import ast
import pathlib

# the line in choose_mp_linear_kernel that each verdict passes through
ANCHOR = "        can_implement, failure_reason = kernel.can_implement(config)\n"


def install(tag, workdir="/work"):
    """Patch `choose_mp_linear_kernel` on disk. Returns the record path.

    Must be called before the engine starts, so that workers re-import the
    patched module. `tag` names the record file, e.g. "asym-1024".
    """
    import vllm.model_executor.kernels.linear as LK

    kpath = pathlib.Path(LK.__file__)
    ksrc = kpath.read_text()
    n = ksrc.count(ANCHOR)
    if n != 1:
        raise RuntimeError(
            f"expected exactly one anchor in {kpath}, found {n}; the kernel "
            f"selection code changed and this instrumentation needs updating"
        )
    rec = f"{workdir}/kernels-{tag}.txt"
    probe = (
        "        try:\n"
        "            import os as _os\n"
        "            _ok, _why = kernel.can_implement(config)\n"
        "            _k = (kernel.__name__, str(config.weight_type),\n"
        "                  config.group_size, bool(config.zero_points), bool(_ok))\n"
        "            _s = getattr(choose_mp_linear_kernel, '_seen', None)\n"
        "            if _s is None:\n"
        "                _s = set(); choose_mp_linear_kernel._seen = _s\n"
        "            if _k not in _s:\n"
        "                _s.add(_k)\n"
        "                with open('" + rec + "', 'a') as _fh:\n"
        "                    _fh.write('pid=%d %s %s\\n' % (_os.getpid(), _k,\n"
        "                              ('' if _ok else (_why or '')[:70])))\n"
        "        except Exception:\n"
        "            pass\n"
    )
    kpath.write_text(ksrc.replace(ANCHOR, probe + ANCHOR))
    ast.parse(kpath.read_text())
    return rec


def read(rec, expect_ranks=2):
    """The deduplicated records as one line, or an exception.

    An empty or absent record means the instrumentation did not reach the
    workers. That is a failed probe, not a result with no kernels in it.
    """
    p = pathlib.Path(rec)
    if not p.exists():
        raise RuntimeError(
            f"kernel record {rec} was never written: the instrumentation did "
            f"not reach the workers, so kernel selection is unmeasured"
        )
    lines = sorted(set(l for l in p.read_text().splitlines() if l.strip()))
    if not lines:
        raise RuntimeError(f"kernel record {rec} is empty")
    pids = {l.split()[0] for l in lines if l.startswith("pid=")}
    if len(pids) < expect_ranks:
        raise RuntimeError(
            f"kernel record {rec} carries {len(pids)} rank(s), expected "
            f"{expect_ranks}: {sorted(pids)}"
        )
    return " | ".join(lines)
