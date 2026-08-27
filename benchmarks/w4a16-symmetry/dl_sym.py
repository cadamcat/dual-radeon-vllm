"""Fetch RedHatAI/Qwen3.8-27B-INT4 (symmetric w4a16) through hf-mirror.

The guest cannot reach huggingface.co; hf-mirror.com is the working route and
huggingface_hub lives inside the container, not on the guest.
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from huggingface_hub import snapshot_download

p = snapshot_download(
    repo_id="RedHatAI/Qwen3.8-27B-INT4",
    local_dir="/data/incoming/Qwen3.8-27B-INT4-sym",
    max_workers=4,
)
print("DL_DONE", p, flush=True)
