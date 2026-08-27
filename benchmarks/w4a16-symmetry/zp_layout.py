"""Why admitting `uint4` is not enough: the zero-point tensor is transposed.

Forcing `RDNA3W4A16LinearKernel` onto the asymmetric checkpoint gets all the
way to the kernel call and then dies in the HIP entry check:

    RuntimeError: b_scales must have same group count as qzeros

This reads the shapes straight out of the safetensors headers — no GPU, no
model load, just the JSON header each file carries — and shows why.

`process_weights_after_loading` runs `permute_param_layout_` on `w_q` and on
`w_s`, so scales arrive group-major. It never touches `w_zp`, because on the
symmetric path there is nothing to touch: that path *fabricates* the tensor
itself, as `(groups, out_features)` packed along dim 1, i.e. group-major
already. An asymmetric checkpoint instead ships a real zero-point tensor in the
compressed-tensors layout, which is the transpose of that.

Writes the census to argv[1].
"""

import json
import struct
import sys

ARMS = {
    "asym": "/data/incoming/Qwen3.8-27B-AWQ-INT4",
    "sym": "/data/incoming/Qwen3.8-27B-INT4-sym",
}
PACK_FACTOR = 8  # 4-bit values per int32


def header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


def describe(d, base):
    idx = json.load(open(d + "/model.safetensors.index.json"))["weight_map"]
    h = header(d + "/" + idx[base + ".weight_packed"])
    out = {"layer": base}
    for suf in ("weight_packed", "weight_scale", "weight_zero_point"):
        k = base + "." + suf
        out[suf] = h[k]["shape"] if k in h else None
        if k in h:
            out[suf + "_dtype"] = h[k]["dtype"]
    return out


def main():
    dest = sys.argv[1] if len(sys.argv) > 1 else "zp-layout.json"
    rows = {}
    for arm, d in ARMS.items():
        idx = json.load(open(d + "/model.safetensors.index.json"))["weight_map"]
        base = sorted(k[: -len(".weight_packed")] for k in idx
                      if k.endswith(".weight_packed") and ".mlp." in k)[0]
        rows[arm] = describe(d, base)

    a = rows["asym"]
    n_packed, k_packed = a["weight_packed"]        # (N, K/8)
    N, K = n_packed, k_packed * PACK_FACTOR
    scale_groups = a["weight_scale"][1]
    zp_rows, zp_cols = a["weight_zero_point"]
    rows["derived"] = {
        "N": N,
        "K": K,
        "group_size": K // scale_groups,
        "groups": scale_groups,
        "scale_layout": "(N, groups)",
        "zero_point_layout": "(N/8, groups)",
        # what the symmetric path builds for itself, and therefore what the
        # kernel's entry check expects to see
        "kernel_expects_zp": [scale_groups, N // PACK_FACTOR],
        "checkpoint_provides_zp": [zp_rows, zp_cols],
        "is_transpose": [zp_rows, zp_cols] == [N // PACK_FACTOR, scale_groups],
        "n_over_8": N // PACK_FACTOR,
    }
    with open(dest, "w") as fh:
        json.dump(rows, fh, indent=1)

    d = rows["derived"]
    print(f"layer {a['layer']}")
    print(f"  N={d['N']} K={d['K']} group_size={d['group_size']} groups={d['groups']}")
    print(f"  weight_scale       {a['weight_scale']}   (N, groups)")
    print(f"  weight_zero_point  {a['weight_zero_point']}   (N/8, groups)")
    print(f"  kernel expects zp as {d['kernel_expects_zp']}  (groups, N/8)")
    print(f"  ZP_IS_TRANSPOSED={d['is_transpose']}")
    print(f"  so the entry check compares b_scales.size(0)={d['groups']} "
          f"against qzeros.size(0)={d['n_over_8']}")
    print("LAYOUT_DONE")


if __name__ == "__main__":
    main()
