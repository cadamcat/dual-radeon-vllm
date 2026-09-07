"""Every checkpoint on the guest, and whether vllm#54210's change can reach it.

The gate this PR widens is `use_rocm_custom_paged_attention`: on gfx11 it takes
head_size 128 with gqa_ratio >= 3 today, and >= 1 after the change. So a
checkpoint is reached by the change only at head_size 128 and gqa_ratio 1 or 2 --
anything else is either refused on head_size before gqa_ratio is read, or was
already admitted. This enumerates the box rather than trusting a summary.
"""
import json, os, glob
for d in sorted(glob.glob("/data/incoming/*/")):
    cfg = os.path.join(d, "config.json")
    if not os.path.exists(cfg):
        continue
    c = json.load(open(cfg))
    t = c.get("text_config") or c.get("llm_config") or c
    hd = t.get("head_dim")
    nh, kv = t.get("num_attention_heads"), t.get("num_key_value_heads")
    if hd is None and nh and t.get("hidden_size"):
        hd = t["hidden_size"] // nh
    gqa = (nh // kv) if (nh and kv) else None
    print(json.dumps({
        "checkpoint": os.path.basename(d.rstrip("/")),
        "model_type": c.get("model_type"),
        "head_dim": hd, "num_attention_heads": nh, "num_key_value_heads": kv,
        "gqa_ratio": gqa,
        "reached_by_54210": bool(hd == 128 and gqa in (1, 2)),
        "config_md5": __import__("hashlib").md5(open(cfg, "rb").read()).hexdigest(),
        "source": cfg,
    }))
