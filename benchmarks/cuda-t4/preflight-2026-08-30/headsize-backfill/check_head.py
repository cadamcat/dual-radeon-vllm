#!/usr/bin/env python3
"""Read every number in this directory's README back out of headsize.jsonl.

Run from this directory:  python3 check_head.py
"""
import json, os, sys

D = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(D, "headsize.jsonl")) if l.strip()]
by = {}
for r in rows:
    by.setdefault(r["kind"], []).append(r)

ok = fail = 0
def chk(label, got, want):
    global ok, fail
    good = got == want
    print(("  ok   " if good else "  FAIL ") + f"{label}: {got!r} vs README {want!r}")
    ok, fail = ok + good, fail + (not good)

print("what vLLM's own ModelConfig resolves")
m = by["model_arch_config"][0]
chk("vllm version", m["vllm"], "0.28.0")
chk("triton_unified_attention.py md5", m["tua_md5"], "49fab3b643bf5a88eb65303ce377996b")
chk("head_size", m["head_size"], 512)
chk("per-layer head sizes", m["per_layer_head_sizes"], [256, 512])
chk("attention heads", m["total_num_attention_heads"], 16)
chk("kv heads", m["total_num_kv_heads"], 8)
chk("hidden layers", m["total_num_hidden_layers"], 48)
chk("model_type", m["model_type"], "gemma4_unified")

print("what the checkpoint's own config.json says")
c = by["raw_config"][0]
chk("head_dim", c["head_dim"], 256)
chk("global_head_dim", c["global_head_dim"], 512)
chk("num_attention_heads", c["num_attention_heads"], 16)
chk("num_key_value_heads", c["num_key_value_heads"], 8)
chk("sliding layers", c["layer_types_counts"]["sliding_attention"], 40)
chk("full layers", c["layer_types_counts"]["full_attention"], 8)

print("what the kernel is actually called with, recorded before the launch")
s = [r for r in by["serve"] if r["recorder_lines"]]
assert s, "no serve row carries recorder lines"
lines = s[-1]["recorder_lines"]
chk("distinct kernel calls", len(lines), 2)
def field(line, key):
    for tok in line.split():
        if tok.startswith(key + "="):
            v = tok.split("=", 1)[1]
            return int(v) if v.lstrip("-").isdigit() else v
    raise KeyError(key)
got = sorted((field(l, "head_size"), field(l, "head_size_padded"),
              field(l, "nq_per_kv"), field(l, "kv_heads"),
              field(l, "BLOCK_M"), field(l, "TILE_PREFILL")) for l in lines)
chk("kernel calls (head_size, padded, nq_per_kv, kv_heads, BLOCK_M, TILE_PREFILL)",
    got, [(256, 256, 2, 8, 16, 32), (512, 512, 16, 1, 16, 32)])
chk("max head_size_padded reaches 512", max(g[1] for g in got), 512)
chk("backend", s[-1]["backend"], "AttentionBackendEnum.TRITON_ATTN")
chk("shared memory required", s[-1]["shmem_required"], "98304")
chk("shared memory limit", s[-1]["shmem_limit"], "65536")
chk("W4A16 kernel", s[-1]["wna16"], "MarlinLinearKernel")

print("with vllm#39018 applied")
p = by["patch39018"][0]
chk("patch applies clean", p["rc"], 0)
chk("head_size_padded gate present", p["has_head_size_padded"], True)
v = by["serve39018"][-1]
chk("engine reaches startup", v["state"], "ready")
chk("backend still TRITON_ATTN", v["backend"], "AttentionBackendEnum.TRITON_ATTN")
chk("no shared-memory failure", v["shmem_required"], None)
chk("KV cache", v["kv_gib"], "5.17")
chk("KV tokens", v["kv_tokens"], "82,383")
g2 = sorted((field(l, "head_size"), field(l, "TILE_PREFILL")) for l in v["recorder_lines"])
chk("the patch halves TILE_PREFILL on the 512 layers only",
    sorted(set(g2)), [(256, 32), (512, 16)])

print("and it generates, which starting does not prove")
inf = {r["target"]: r for r in by["infer39018"] if r["ok"]}
chk("500 rung served", 500 in inf, True)
chk("500 rung prompt tokens", inf[500]["usage"]["prompt_tokens"], 738)
chk("500 rung completion tokens", inf[500]["usage"]["completion_tokens"], 16)
chk("32000 rung served", 32000 in inf, True)
chk("32000 rung prompt tokens", inf[32000]["usage"]["prompt_tokens"], 30018)
chk("32000 rung completion tokens", inf[32000]["usage"]["completion_tokens"], 32)

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
