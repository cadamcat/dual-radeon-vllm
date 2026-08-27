"""Install the same vLLM the gfx1100 container runs, so Stage 2 compiles the
identical kernel_paged_attention_2d source. The container is 0.23.1.dev1, whose
copy of chunked_prefill_paged_decode.py is byte-identical to the v0.23.0 tag
(md5 854daa8f5d878449266519a9206db677); asserted below."""
import hashlib, os, pathlib, subprocess, sys

WANT = "854daa8f5d878449266519a9206db677"
try:
    import vllm; have = vllm.__version__
except Exception:
    have = None
print("vllm before:", have, flush=True)
if have != "0.23.0":
    print("installing vllm==0.23.0", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "vllm==0.23.0"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "-q", "uninstall", "-y", "torchaudio"], check=False)
import importlib
importlib.invalidate_caches()
for m in [k for k in list(sys.modules) if k.startswith("vllm")]:
    del sys.modules[m]
import vllm
print("vllm after:", vllm.__version__, flush=True)
sp = os.path.dirname(os.path.dirname(vllm.__file__))
kern = os.path.join(sp, "vllm/v1/attention/ops/chunked_prefill_paged_decode.py")
got = hashlib.md5(pathlib.Path(kern).read_bytes()).hexdigest()
print("kernel md5:", got, flush=True)
assert got == WANT, f"kernel source differs from the gfx1100 container: {got} != {WANT}"
import torch, triton
print("torch", torch.__version__, "triton", triton.__version__, flush=True)
print("SETUP_DONE", flush=True)
