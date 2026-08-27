"""Stage 2 for vllm#50603: the same numerics sweep, on CUDA.

Stage 1 measures the ROCm dispatch on gfx1100. This runs the SAME Triton
kernel source (``kernel_paged_attention_2d``) on an A100, launched directly
with the argument list the ROCm dispatch builds, because on CUDA that kernel
is not reachable through ``chunked_prefill_paged_decode`` (the dispatch
unconditionally imports ``vllm.platforms.rocm``).

Purpose: separate "this Triton kernel loses accuracy as context grows,
wherever it runs" from "something about the gfx1100 build of it". build(),
reference() and score() below are byte-identical to the Stage 1 probe, so the
two runs' numbers are directly comparable.

argv: [out.jsonl]
"""

import json
import os
import sys

import torch
import triton

from kernel_lifted import kernel_paged_attention_2d


def set_random_seed(seed):
    import random
    random.seed(seed); torch.manual_seed(seed)

OUT = sys.argv[1] if len(sys.argv) > 1 else "/content/50603-cuda.jsonl"

HEAD_SIZE = 128
BLOCK_SIZE = 16
DTYPE = torch.bfloat16
SHAPES = [(8, 8), (8, 4), (12, 4), (16, 4), (32, 16)]
CTX_LENS = [1024, 2048, 4096, 8192, 16384, 32768]
if os.environ.get("SMOKE"):
    SHAPES = [(8, 4)]
    CTX_LENS = [1024]
WARMUP, ITERS = 3, 10


def build(ctx_len, num_heads, num_kv_heads, bs):
    """One decode step per sequence over a fully cached context of ctx_len."""
    set_random_seed(0)
    blocks_per_seq = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    cache_size = blocks_per_seq * bs + 8

    # N(0,1), as in upstream's other attention kernel tests. The uniform
    # (-1e-3, 1e-3) used by test_prefix_prefill makes the softmax almost flat
    # and the output a near-zero average, which is both an unrealistically
    # benign numerical case and a useless ULP denominator.
    query = torch.randn(bs, num_heads, HEAD_SIZE, dtype=DTYPE)
    # ground-truth KV, laid out per sequence, before any paging
    kv = torch.randn(bs, ctx_len, 2, num_kv_heads, HEAD_SIZE, dtype=DTYPE)
    key, value = kv.unbind(dim=2)

    k_cache = torch.zeros(cache_size, BLOCK_SIZE, num_kv_heads, HEAD_SIZE, dtype=DTYPE)
    v_cache = torch.zeros_like(k_cache)
    perm = torch.randperm(cache_size)[: blocks_per_seq * bs].to(torch.int32)
    block_table = perm.view(bs, blocks_per_seq)

    flat_k = k_cache.view(-1, num_kv_heads, HEAD_SIZE)
    flat_v = v_cache.view(-1, num_kv_heads, HEAD_SIZE)
    for i in range(bs):
        for b in range(blocks_per_seq):
            lo = b * BLOCK_SIZE
            hi = min(lo + BLOCK_SIZE, ctx_len)
            slot = int(block_table[i, b]) * BLOCK_SIZE
            flat_k[slot : slot + (hi - lo)].copy_(key[i, lo:hi])
            flat_v[slot : slot + (hi - lo)].copy_(value[i, lo:hi])

    # the layouts the ROCm paged kernels expect
    k_cache = (
        k_cache.view(-1, BLOCK_SIZE, num_kv_heads, HEAD_SIZE // 8, 8)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    v_cache = (
        v_cache.view(-1, BLOCK_SIZE, num_kv_heads, HEAD_SIZE)
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    seq_lens = torch.full((bs,), ctx_len, dtype=torch.int32)
    query_start_loc = torch.arange(bs + 1, dtype=torch.int32)
    scale = torch.tensor(1.0, dtype=torch.float32)
    return dict(
        query=query, key=key, value=value, k_cache=k_cache, v_cache=v_cache,
        block_table=block_table, seq_lens=seq_lens,
        query_start_loc=query_start_loc, k_scale=scale, v_scale=scale,
        ctx_len=ctx_len, num_heads=num_heads, num_kv_heads=num_kv_heads, bs=bs,
    )


def reference(d):
    """fp32 ground truth, computed from the pre-paging KV, not from the cache."""
    q = d["query"].float()                       # [bs, H, D]
    k = d["key"].float()                         # [bs, L, KV, D]
    v = d["value"].float()
    bs, H, D = q.shape
    KV = k.shape[2]
    rep = H // KV
    out = torch.empty(bs, H, D, dtype=torch.float32)
    sm = D ** -0.5
    for i in range(bs):
        for h in range(H):
            kh = h // rep
            s = (k[i, :, kh, :] @ q[i, h]) * sm   # [L]
            p = torch.softmax(s, dim=0)
            out[i, h] = p @ v[i, :, kh, :]
    return out


def run_arm(d, _unused=None):
    """Direct launch, mirroring the ROCm dispatch's argument list exactly."""
    out = torch.empty_like(d["query"])
    q = d["query"]
    k_cache, v_cache = d["k_cache"], d["v_cache"]
    bt = d["block_table"].to(torch.int32)
    nq, nkv = d["num_heads"], d["num_kv_heads"]
    nqpkv = nq // nkv
    kernel_paged_attention_2d[(d["bs"], nkv)](
        output_ptr=out,
        query_ptr=q,
        key_cache_ptr=k_cache,
        value_cache_ptr=v_cache,
        sink_ptr=None,
        block_tables_ptr=bt,
        seq_lens_ptr=d["seq_lens"],
        alibi_slopes_ptr=None,
        scale=HEAD_SIZE ** -0.5,
        k_scale=d["k_scale"],
        v_scale=d["v_scale"],
        out_scale_inv=1.0,
        num_query_heads=nq,
        num_queries_per_kv=nqpkv,
        num_queries_per_kv_padded=max(triton.next_power_of_2(nqpkv), 16),
        block_table_stride=bt.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
        PHYSICAL_BLOCK_SIZE=v_cache.shape[3],
        HEAD_SIZE=HEAD_SIZE,
        HEAD_SIZE_PADDED=triton.next_power_of_2(HEAD_SIZE),
        USE_ALIBI_SLOPES=False,
        SLIDING_WINDOW=0,
        x=k_cache.shape[4],
        stride_k_cache_0=k_cache.stride(0),
        stride_k_cache_1=k_cache.stride(1),
        stride_k_cache_2=k_cache.stride(2),
        stride_k_cache_3=k_cache.stride(3),
        stride_k_cache_4=k_cache.stride(4),
        stride_v_cache_0=v_cache.stride(0),
        stride_v_cache_1=v_cache.stride(1),
        stride_v_cache_2=v_cache.stride(2),
        stride_v_cache_3=v_cache.stride(3),
        filter_by_query_len=True,
        query_start_len_ptr=d["query_start_loc"],
        USE_SINKS=False,
        USE_FP8=False,
    )
    torch.cuda.synchronize()
    return out


def timed(d):
    for _ in range(WARMUP):
        run_arm(d)
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
          for _ in range(ITERS)]
    for s, e in ev:
        s.record(); run_arm(d); e.record()
    torch.cuda.synchronize()
    ms = sorted(s.elapsed_time(e) for s, e in ev)
    return ms[len(ms) // 2]


def score(out, ref, dtype=DTYPE):
    """Error against the fp32 reference.

    Primary metric is relative L-inf, normalised by the reference's own scale,
    because that is comparable across context lengths. ULP distance is reported
    too but only over elements that are not near-zero: the local ULP of an
    element that happens to land on ~0 is meaningless as a denominator.
    """
    o = out.float()
    diff = (o - ref).abs()
    scale = float(ref.abs().max())
    mant = {torch.bfloat16: 7, torch.float16: 10}[dtype]
    big = ref.abs() > 1e-3 * scale
    mag = torch.maximum(o.abs(), ref.abs()).clamp(min=1e-30)
    ulp = torch.pow(2.0, torch.floor(torch.log2(mag)) - mant)
    d_ulp = (diff / ulp)[big]
    return {
        "ref_absmax": scale,
        "max_abs_err": float(diff.max()),
        "max_rel_err": float(diff.max()) / scale,
        "rms_rel_err": float((diff ** 2).mean().sqrt()) / scale,
        "max_ulp": float(d_ulp.max()) if d_ulp.numel() else 0.0,
        "frac_beyond_1ulp": float((d_ulp > 1.0000001).float().mean()) if d_ulp.numel() else 0.0,
        "n_scored": int(big.sum()),
    }


def main():
    torch.set_default_device("cuda")
    print(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__} "
          f"triton={triton.__version__}", flush=True)
    fh = open(OUT, "w")
    for num_heads, num_kv in SHAPES:
        gqa = num_heads // num_kv
        for ctx in CTX_LENS:
            bs = 4 if ctx <= 8192 else 2
            d = build(ctx, num_heads, num_kv, bs)
            ref = reference(d)
            row = {"gqa_ratio": gqa, "num_heads": num_heads, "num_kv_heads": num_kv,
                   "ctx_len": ctx, "bs": bs, "head_size": HEAD_SIZE,
                   "block_size": BLOCK_SIZE, "dtype": "bf16", "arch": "cuda"}
            try:
                out = run_arm(d)
                row["triton"] = score(out, ref)
                row["triton"]["median_ms"] = timed(d)
            except Exception as exc:
                row["triton"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
            fh.write(json.dumps(row) + "\n"); fh.flush()
            t = row["triton"]
            msg = (f"rel={t['max_rel_err']:.2e} rms={t['rms_rel_err']:.2e} "
                   f"ulp={t['max_ulp']:.1f} {t['median_ms']:.3f}ms"
                   if "error" not in t else f"ERROR {t['error'][:70]}")
            print(f"gqa={gqa} ({num_heads}/{num_kv}) ctx={ctx:<6d} | triton {msg}", flush=True)
            del d, ref
            torch.cuda.empty_cache()
    fh.close()
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
