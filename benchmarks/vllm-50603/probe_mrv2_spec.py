"""Does MRv2 reach the same max_seqlen_q > 1 guard that #53930 warns about?

vllm#53930 adds a warning in TritonAttentionMetadataBuilder.__init__ saying that
speculative decoding forces decode onto the 2D path. @njhill labelled that PR
`mrv1-only` on 2026-08-28, meaning it does not apply to Model Runner V2. Reading
main says otherwise -- both the builder and the `use_3d` guard live in code the
two runners share -- but reading is not running, so this runs it.

Three things are recorded from inside the worker processes, never inferred:

  1. which model runner class was constructed, so the arm is proven rather than
     assumed from the env var;
  2. that TritonAttentionMetadataBuilder.__init__ ran, which is where #53930's
     warning would fire;
  3. every distinct (max_seqlen_q, num_seqs, use_3d) the unified-attention
     launcher saw, with each disqualifying condition separately, so "3D was off"
     can be attributed to the right clause.

All three edits precede any vllm import: 0.27's workers inherit the parent's
already-imported modules, so a recorder installed afterwards never reaches them
(that is what produced an empty record set in the gqa-gate work on this image).

    python3 probe_mrv2_spec.py <v1|v2> <ctx>
"""

import glob
import json
import os
import pathlib
import sys

ARM = sys.argv[1]
CTX = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
SPEC = (sys.argv[3] if len(sys.argv) > 3 else "spec")
MODEL_ARG = sys.argv[4] if len(sys.argv) > 4 else None
OUT = sys.argv[5] if len(sys.argv) > 5 else "/work/mrv2-spec.jsonl"
assert ARM in ("v1", "v2"), ARM
assert SPEC in ("spec", "nospec"), SPEC

# gemma-4 + its MTP assistant does not configure on this image: resolving the
# speculative config reads config.head_dim globally and gemma-4's head dims are
# per-layer, so it raises AmbiguousGlobalPerLayerAttributeError before any
# engine starts. It works on the 0.23.1 image and on 0.28.0 CUDA. The nospec arm
# therefore carries the model, and the spec arm is left recorded as unrunnable
# here rather than worked around by forcing a global read.
MODEL = MODEL_ARG or "/data/incoming/Qwen3-8B"
ASSIST = "/data/incoming/gemma-4-31B-it-assistant"
REC = f"/work/mrv2rec-{ARM}-{SPEC}-{CTX}.txt"

# fail on a mount mistake here rather than three minutes later as an opaque
# "Repo id must be in the form 'namespace/repo_name'" from the HF resolver
for _p in ((MODEL, ASSIST) if SPEC == "spec" else (MODEL,)):
    assert os.path.isdir(_p), f"{_p} is not visible in this container; is /data mounted?"

SP = glob.glob("/opt/python/lib/python*/site-packages")
assert len(SP) == 1, SP
SP = SP[0]
assert "vllm" not in sys.modules, "vllm imported too early; the edits must precede it"


def rec_snippet(tag, expr, ind="    "):
    """Append one line per distinct value, once per process, never raising.

    `ind` must match the body indentation of the block it is spliced into:
    four spaces inside a module-level function, eight inside a method. Getting
    that wrong is an IndentationError at import, which a dry run caught once.
    """
    return (
        f"{ind}try:\n"
        f"{ind}    import os as _os\n"
        f"{ind}    _v = {expr}\n"
        f"{ind}    _g = globals().setdefault('_probe_seen', set())\n"
        f"{ind}    if _v not in _g:\n"
        f"{ind}        _g.add(_v)\n"
        f"{ind}        open('{REC}', 'a').write('pid=%d {tag} %s\\n' % (_os.getpid(), _v,))\n"
        f"{ind}except Exception as _e:\n"
        f"{ind}    open('{REC}.err', 'a').write('{tag} %r\\n' % (_e,))\n"
    )


def inject_into_runner_init(path, tag):
    """Splice a marker into GPUModelRunner.__init__ only.

    Anchoring on the first `def __init__` in the file is wrong: in V1 that
    belongs to AsyncGPUModelRunnerOutput, several classes above the runner. So
    find the class first, then its __init__, then the line that closes the
    signature -- which is one line in MRv2 and several in V1.
    """
    src = pathlib.Path(path).read_text()
    lines = src.split("\n")
    cls = next(i for i, l in enumerate(lines) if l.startswith("class GPUModelRunner"))
    ini = next(i for i in range(cls, len(lines)) if lines[i].strip().startswith("def __init__"))
    close = next(i for i in range(ini, len(lines)) if lines[i].rstrip().endswith("):"))
    body = rec_snippet(tag, "'constructed'", ind="        ").rstrip("\n").split("\n")
    pathlib.Path(path).write_text("\n".join(lines[:close + 1] + body + lines[close + 1:]))
    return cls + 1, ini + 1, close + 1


# ---- 1. which runner constructed, one marker in each --------------------
for rel, tag in (("vllm/v1/worker/gpu_model_runner.py", "RUNNER_V1"),
                 ("vllm/v1/worker/gpu/model_runner.py", "RUNNER_V2")):
    where = inject_into_runner_init(str(pathlib.Path(SP, rel)), tag)
    print(f"injected {tag} into {rel} (class/init/close lines {where})", flush=True)

# ---- 2. the builder #53930 patches ---------------------------------------
tp = pathlib.Path(SP, "vllm/v1/attention/backends/triton_attn.py")
tsrc = tp.read_text()
BANCHOR = "        self.rswa_window = model_config.rswa_window\n"
assert tsrc.count(BANCHOR) == 1, f"builder anchor x{tsrc.count(BANCHOR)}"
tp.write_text(tsrc.replace(
    BANCHOR,
    rec_snippet("BUILDER_INIT", "('spec' if vllm_config.speculative_config is not None else 'nospec')", ind="        ")
    + BANCHOR))

# ---- 3. the guard itself --------------------------------------------------
up = pathlib.Path(SP, "vllm/v1/attention/ops/triton_unified_attention.py")
usrc = up.read_text()
GANCHOR = """    use_3d = not (
        seq_threshold_3D is None
        or num_par_softmax_segments is None
        or softmax_segm_output is None
        or softmax_segm_max is None
        or softmax_segm_expsum is None
        or max_seqlen_q > 1
        or num_seqs > seq_threshold_3D
        or is_batch_invariant
    )
"""
assert usrc.count(GANCHOR) == 1, f"guard anchor x{usrc.count(GANCHOR)}"
guard_expr = (
    "(int(max_seqlen_q), int(num_seqs), bool(use_3d),"
    " 'q>1' if max_seqlen_q > 1 else '',"
    " 'nseq' if (seq_threshold_3D is not None and num_seqs > seq_threshold_3D) else '',"
    " 'nobuf' if (softmax_segm_output is None or seq_threshold_3D is None) else '',"
    " 'binv' if is_batch_invariant else '')"
)
up.write_text(usrc.replace(GANCHOR, GANCHOR + rec_snippet("GUARD", guard_expr)))

for pyc in glob.glob(SP + "/vllm/**/__pycache__/*.pyc", recursive=True):
    os.remove(pyc)

# ---- 4. select the runner, then and only then import vLLM -----------------
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1" if ARM == "v2" else "0"
print(f"ARM={ARM} SPEC={SPEC} MODEL={MODEL} VLLM_USE_V2_MODEL_RUNNER={os.environ['VLLM_USE_V2_MODEL_RUNNER']} ctx={CTX}",
      flush=True)

from vllm import LLM, SamplingParams  # noqa: E402

err = None
try:
    # TRITON_ATTN is forced rather than left to selection: the guard this probe
    # is about lives in that backend's launcher, so a cell that silently landed
    # on ROCM_ATTN would record nothing and read as a clean result.
    kwargs = dict(model=MODEL, tensor_parallel_size=2, max_model_len=CTX + 512,
                  gpu_memory_utilization=0.92,
                  attention_config={"backend": "TRITON_ATTN"})
    if SPEC == "spec":
        kwargs["speculative_config"] = {"method": "mtp", "model": ASSIST,
                                        "num_speculative_tokens": 1}
    llm = LLM(**kwargs)
    tok = llm.get_tokenizer()
    ids = tok("hello " * (CTX + 2000)).input_ids[:CTX]
    llm.generate([tok.decode(ids)],
                 SamplingParams(temperature=0, max_tokens=8, ignore_eos=True))
except Exception as e:  # a failure to start MRv2 here is itself the result
    err = f"{type(e).__name__}: {e}"
    print("ENGINE_FAILED", err[:800], flush=True)

lines = sorted(set(pathlib.Path(REC).read_text().splitlines())) if pathlib.Path(REC).exists() else []
runner = sorted({l.split()[1] for l in lines if "RUNNER_V" in l})
builder = sorted({l.split(maxsplit=2)[2] for l in lines if "BUILDER_INIT" in l})
guards = sorted({l.split(maxsplit=2)[2] for l in lines if "GUARD" in l})
print(f"RUNNER_CONSTRUCTED: {runner}", flush=True)
print(f"BUILDER_INIT: {builder}", flush=True)
for g in guards:
    print("  GUARD", g, flush=True)
errp = pathlib.Path(REC + ".err")
if errp.exists():
    print("RECORDER_ERRORS:", errp.read_text()[:400], flush=True)

with open(OUT, "a") as fh:
    fh.write(json.dumps({"arm": ARM, "spec": SPEC, "model": MODEL, "ctx": CTX, "engine_error": err,
                         "runner_constructed": runner, "builder_init": builder,
                         "guard_rows": guards}) + "\n")
print("MRV2_SPEC_CELL_DONE", flush=True)
