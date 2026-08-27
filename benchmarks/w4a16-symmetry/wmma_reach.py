"""Does the WMMA entry point get used, given this checkpoint's real shapes?

gptq_gemm_rdna3 dispatches to gptq_gemm_rdna3_wmma when
  a.size(1) % 16 == 0  (K)   and  b_q_weight.size(1) % 16 == 0  (N)
  and (bf16 and M >= 16) or (half and M >= 64)
b_q_weight is [K/8, N], so size(1) is N. Read N and K per layer off the
safetensors headers and count how many layers can reach it.
"""
import collections, json, struct

D = "/data/incoming/Qwen3.8-27B-AWQ-INT4"
idx = json.load(open(D + "/model.safetensors.index.json"))["weight_map"]
hdrs = {}
def header(shard):
    if shard not in hdrs:
        with open(D + "/" + shard, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdrs[shard] = json.loads(f.read(n))
    return hdrs[shard]

both = k16only = n16only = neither = 0
examples = collections.defaultdict(list)
for key in sorted(k for k in idx if k.endswith(".weight_packed")):
    shape = header(idx[key])[key]["shape"]      # [K/8, N]
    K, N = shape[0] * 8, shape[1]
    ok_k, ok_n = K % 16 == 0, N % 16 == 0
    if ok_k and ok_n:
        both += 1; examples["wmma-eligible"].append((key.split(".")[-2], K, N))
    elif ok_k: k16only += 1; examples["N not mult 16"].append((key.split(".")[-2], K, N))
    elif ok_n: n16only += 1
    else: neither += 1

tot = both + k16only + n16only + neither
print(f"quantised linears: {tot}")
print(f"  K%16==0 and N%16==0  -> WMMA eligible : {both}")
print(f"  K ok, N not mult of 16                : {k16only}")
print(f"  N ok, K not mult of 16                : {n16only}")
print(f"  neither                               : {neither}")
print(f"WMMA_FRACTION={both/tot:.4f}")
for k, v in examples.items():
    print(f"  {k}: e.g. {v[:3]}")
print("WMMA_DONE")
