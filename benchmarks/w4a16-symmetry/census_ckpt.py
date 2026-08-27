"""Which linear layers does each checkpoint actually quantize?

The A/B compares two independent quantizations of one model, and their
`ignore` lists in config.json are not identical. Rather than assume that
difference is cosmetic, count the packed weights in each checkpoint's
safetensors index: a quantized linear is exactly a `*.weight_packed` entry.

Writes the census to argv[1] so the claim in the write-up is
recomputable without the checkpoints, which are far too large to commit.
"""
import collections
import json
import sys

ARMS = {
    "asym": "/data/incoming/Qwen3.8-27B-AWQ-INT4",
    "sym": "/data/incoming/Qwen3.8-27B-INT4-sym",
}


def bucket(n):
    if ".linear_attn." in n:
        return "linear_attn"
    if ".self_attn." in n:
        return "self_attn"
    if ".mlp." in n:
        return "mlp"
    if n.startswith("model.visual"):
        return "visual"
    if n.startswith("mtp"):
        return "mtp"
    return "other"


def census(d):
    idx = json.load(open(d + "/model.safetensors.index.json"))["weight_map"]
    cfg = json.load(open(d + "/config.json"))
    q = cfg["quantization_config"]
    w = next(iter(q["config_groups"].values()))["weights"]
    packed = sorted(k[: -len(".weight_packed")] for k in idx if k.endswith(".weight_packed"))
    return {
        "path": d,
        "architectures": cfg["architectures"],
        "model_type": cfg.get("model_type"),
        "quant_method": q.get("quant_method"),
        "format": q.get("format"),
        "num_bits": w.get("num_bits"),
        "symmetric": w.get("symmetric"),
        "group_size": w.get("group_size"),
        "strategy": w.get("strategy"),
        "quantized_linear_layers": len(packed),
        "by_module": dict(sorted(collections.Counter(bucket(n) for n in packed).items())),
        "layers": packed,
    }


def main():
    out = {arm: census(d) for arm, d in ARMS.items()}
    a, s = set(out["asym"]["layers"]), set(out["sym"]["layers"])
    out["diff"] = {
        "only_asym": sorted(a - s),
        "only_sym": sorted(s - a),
        "in_both": len(a & s),
    }
    dest = sys.argv[1] if len(sys.argv) > 1 else "ckpt-layer-census.json"
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    for arm in ("asym", "sym"):
        c = out[arm]
        print(f"{arm:<5} symmetric={c['symmetric']} group_size={c['group_size']} "
              f"quantized={c['quantized_linear_layers']} {c['by_module']}")
    print("only_asym:", out["diff"]["only_asym"])
    print("only_sym :", out["diff"]["only_sym"])
    print("in_both  :", out["diff"]["in_both"])
    print("CENSUS_DONE")


if __name__ == "__main__":
    main()
