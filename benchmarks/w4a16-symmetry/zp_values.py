"""What values does the AWQ checkpoint's zero point actually take?

The GPTQ storage convention the RDNA3 kernel implements is `stored = real - 1`
(the kernel adds 1 back), so a real zero point of 0 has no representation. If
this checkpoint uses 0 anywhere, a "subtract one" fix is wrong there and the
patch needs to say so. Pure file reading: unpack the 4-bit values out of the
int32 tensor and look.
"""
import json
import struct
import sys

import numpy as np

D = "/data/incoming/Qwen3.8-27B-AWQ-INT4"
PACK = 8
BITS = 4


def load(path, name):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
        base = 8 + n
        info = hdr[name]
        a, b = info["data_offsets"]
        fh.seek(base + a)
        raw = fh.read(b - a)
    assert info["dtype"] == "I32", info["dtype"]
    return np.frombuffer(raw, dtype="<i4").reshape(info["shape"])


def main():
    idx = json.load(open(D + "/model.safetensors.index.json"))["weight_map"]
    zps = sorted(k for k in idx if k.endswith(".weight_zero_point"))
    print(f"{len(zps)} zero-point tensors in the checkpoint")
    sample = zps[:: max(1, len(zps) // 12)][:12]
    lo, hi = 16, -1
    zeros_total = n_total = 0
    hist = np.zeros(16, dtype=np.int64)
    for name in sample:
        packed = load(D + "/" + idx[name], name)
        vals = np.stack([(packed >> (BITS * i)) & 0xF for i in range(PACK)], -1)
        v = vals.ravel()
        hist += np.bincount(v, minlength=16)
        lo, hi = min(lo, int(v.min())), max(hi, int(v.max()))
        zeros_total += int((v == 0).sum()); n_total += v.size
        print(f"  {name.split('language_model.')[-1]:<46} "
              f"min={v.min():2d} max={v.max():2d} zeros={int((v==0).sum())}")
    print()
    print(f"sampled {n_total} zero-point entries across {len(sample)} tensors")
    print(f"  min={lo}  max={hi}  count_of_zero={zeros_total}")
    print(f"  histogram 0..15: {hist.tolist()}")
    print(f"SUBTRACT_ONE_IS_SAFE={lo >= 1}")
    print("ZPVAL_DONE")


if __name__ == "__main__":
    main()
