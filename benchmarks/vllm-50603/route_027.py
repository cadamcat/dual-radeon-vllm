"""Worker-side routing proof for the gfx11 gqa gate, on 0.27.

Stage 3's own recorder came back empty on 0.27 and the reason is worth stating,
because it decides whether the timing cells are trustworthy. probe_stage3.py
rewrites chunked_prefill_paged_decode.py from inside main(), i.e. AFTER vLLM has
been imported, and 0.27's workers inherit the parent's already-imported module
rather than re-reading the file: the worker logs the fallback warning at line
419, the pristine line, while the file on disk has it at 433. So the recorder
never reached them.

rocm.py is a different case and does reach them -- probe_stage3.py reloads it
explicitly after the edit, so the parent's module object is the patched one and
that is what the workers inherit. This script exists to prove that rather than
argue it: both edits are applied BEFORE any vllm import, so no inheritance
question arises, and the recorder writes from inside whichever process runs
attention.

    python3 route_027.py <stock|widened> <ctx>
"""

import glob
import os
import pathlib
import sys

ARM, CTX = sys.argv[1], int(sys.argv[2])
OUT = sys.argv[3] if len(sys.argv) > 3 else "/work/route-027.jsonl"
MODEL = "/data/incoming/gemma-3-27b-it-w4a16"

SP = glob.glob("/opt/python/lib/python*/site-packages")
assert len(SP) == 1, SP
SP = SP[0]
assert "vllm" not in sys.modules, "vllm imported too early; the edits must precede it"

# ---- 1. the candidate one-line change, before anything imports it ----------
rocm_py = pathlib.Path(SP, "vllm/platforms/rocm.py")
src = rocm_py.read_text()
OLD, NEW = "gqa_ratio >= 3", "gqa_ratio >= 1"
assert src.count(OLD) == 1, f"gfx11 anchor x{src.count(OLD)}"
assert src.count(NEW) == 1, f"CDNA baseline x{src.count(NEW)}"
if ARM == "widened":
    rocm_py.write_text(src.replace(OLD, NEW))
after = rocm_py.read_text()
if ARM == "widened":
    assert after.count(OLD) == 0 and after.count(NEW) == 2, "patch did not take"
else:
    assert after.count(OLD) == 1 and after.count(NEW) == 1, "tree is not pristine"

# ---- 2. the recorder, likewise before the import ---------------------------
cppd = pathlib.Path(SP, "vllm/v1/attention/ops/chunked_prefill_paged_decode.py")
dsrc = cppd.read_text()
ANCHOR = "    if use_custom:\n"
assert dsrc.count(ANCHOR) == 1
ROUTE = f"/work/route027-{ARM}-{CTX}.txt"
probe = (
    "    try:\n"
    "        import os as _os\n"
    "        _k = (num_queries_per_kv, head_size, block_size, sliding_window,\n"
    "              bool(use_custom))\n"
    "        _s = getattr(chunked_prefill_paged_decode, '_seen', None)\n"
    "        if _s is None:\n"
    "            _s = set(); chunked_prefill_paged_decode._seen = _s\n"
    "        if _k not in _s:\n"
    "            _s.add(_k)\n"
    "            with open('" + ROUTE + "', 'a') as _fh:\n"
    "                _fh.write('pid=%d %s\\n' % (_os.getpid(), _k))\n"
    "    except Exception as _e:\n"
    "        with open('" + ROUTE + ".err', 'a') as _fh:\n"
    "            _fh.write('%r\\n' % (_e,))\n"
)
cppd.write_text(dsrc.replace(ANCHOR, probe + ANCHOR))
# stale bytecode would defeat the whole point
for pyc in glob.glob(SP + "/vllm/**/__pycache__/*.pyc", recursive=True):
    os.remove(pyc)

# ---- 3. only now import vLLM ----------------------------------------------
import torch                                                    # noqa: E402
import vllm.platforms.rocm as rp                                # noqa: E402
from vllm import LLM, SamplingParams                            # noqa: E402

gate = rp.use_rocm_custom_paged_attention(
    torch.bfloat16, 128, 16, 2, 4096, 0, "auto", None, None)
print(f"ARM={ARM} ctx={CTX} gate(gqa=2)={gate} "
      f"multiproc={os.environ.get('VLLM_WORKER_MULTIPROC_METHOD', '(default)')}",
      flush=True)

llm = LLM(model=MODEL, tensor_parallel_size=2, max_model_len=CTX + 512,
          gpu_memory_utilization=0.92)
tok = llm.get_tokenizer()
ids = tok("hello " * (CTX + 4000)).input_ids[:CTX]
llm.generate([tok.decode(ids)],
             SamplingParams(temperature=0, max_tokens=8, ignore_eos=True))

routes = pathlib.Path(ROUTE).read_text().strip() if pathlib.Path(ROUTE).exists() else ""
errs = pathlib.Path(ROUTE + ".err")
print(f"ROUTES({ARM},{CTX}) n={len(routes.splitlines())}", flush=True)
for line in sorted(set(routes.splitlines())):
    print("   ", line, flush=True)
if errs.exists():
    print("RECORDER_ERRORS:", errs.read_text()[:400], flush=True)
import json
with open(OUT, "a") as fh:
    fh.write(json.dumps({"arm": ARM, "ctx": CTX, "gate_gqa2": bool(gate),
                         "routes": sorted(set(routes.splitlines()))}) + "\n")
print("ROUTE_CELL_DONE", flush=True)
