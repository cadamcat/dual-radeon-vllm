import hashlib, os, pathlib, subprocess, sys

PRISTINE = "49fab3b643bf5a88eb65303ce377996b"
PATCHED  = "f1d7a7e3c6656303fa63b6a4c1b8aef5"

def md5(p):
    return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()

def sh(*a, **kw):
    print("+", " ".join(a), flush=True)
    return subprocess.run(a, check=kw.pop("check", True), **kw)

try:
    import vllm
    have = vllm.__version__
except Exception:
    have = None
print("vllm before:", have, flush=True)

if have != "0.28.0":
    sh(sys.executable, "-m", "pip", "-q", "install", "vllm==0.28.0")
    sh(sys.executable, "-m", "pip", "-q", "uninstall", "-y", "torchaudio", check=False)

import importlib
importlib.invalidate_caches()
for m in [k for k in list(sys.modules) if k.startswith("vllm")]:
    del sys.modules[m]
import vllm
print("vllm after:", vllm.__version__, flush=True)
assert vllm.__version__ == "0.28.0", vllm.__version__

sp = os.path.dirname(os.path.dirname(vllm.__file__))
kern = os.path.join(sp, "vllm/v1/attention/ops/triton_unified_attention.py")
print("site-packages:", sp, flush=True)
h = md5(kern)
print("kernel md5 before:", h, flush=True)
if h == PATCHED:
    print("ALREADY PATCHED", flush=True)
else:
    assert h == PRISTINE, (
        f"installed wheel's kernel file is NOT the v0.28.0 tag file: {h} != {PRISTINE}")
    sh("patch", "-p1", "-d", sp, "-i", "/content/52684-kernel.diff")
    h2 = md5(kern)
    print("kernel md5 after:", h2, flush=True)
    assert h2 == PATCHED, f"patched file differs from local reference: {h2} != {PATCHED}"

import torch, triton
print("torch", torch.__version__, "triton", triton.__version__, flush=True)
print("SETUP_DONE", flush=True)
