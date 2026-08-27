"""Does the WMMA entry point get used, given this checkpoint's real shapes?

gptq_gemm_rdna3 dispatches to gptq_gemm_rdna3_wmma when
  a.size(1) % 16 == 0  (K)   and  b_q_weight.size(1) % 16 == 0  (N)
  and (bf16 and M >= 16) or (half and M >= 64)
The kernel sees b_q_weight as [K/8, N], so its size(1) is N; on disk the
checkpoint stores the transpose, (N, K/8), which is what is read here. Count
how many layers can reach the WMMA path.
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
    # weight_packed is (N, K/8) on disk -- dim 0 is N. Confirmed against
    # weight_scale, which is (N, groups): only this reading makes
    # groups == K/group_size come out equal to weight_scale's second dim.
    # (Read the other way round for down_proj it would give K=40960 and
    # groups=1280 against a scale tensor whose second dim is 544.)
    shape = header(idx[key])[key]["shape"]      # (N, K/8)
    N, K = shape[0], shape[1] * 8
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
