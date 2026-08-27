"""Stage 1 for vllm#50603: is the gfx11 gqa_ratio>=3 gate costing correctness?

vllm#50603 reports garbled long-context output on gfx1100 from the Triton
paged-attention fallback. The reporter traced the fallback to
``use_rocm_custom_paged_attention``: on gfx11 it requires ``gqa_ratio >= 3``,
their model has 2, so the ROCm CK custom kernel is gated out.

The PR that introduced that bound (vllm#17004, merged 2025-05-21) says why:

    It supports gqa_ratio up to 16, and shows performance gains over the
    existing kernel when gqa_ratio is 3 or higher. Therefore, it is enabled
    for gqa_ratio values between 3 and 16.

So the bound is a performance heuristic, not a capability limit, and the CDNA
branch of the same function runs the same kernel at ``gqa_ratio >= 1``.

This probe drives BOTH paths on identical inputs and scores each against an
fp32 reference, across sequence length and gqa_ratio. It answers three things
at once:

  a) does the Triton path's error grow with context length (the mechanism
     symptom B would need)?
  b) is the CK path correct at gqa_ratio 1 and 2, where the gate excludes it?
  c) is the 2025 performance rationale still true on this hardware today?

Both paths are reached through vLLM's own ``chunked_prefill_paged_decode``
dispatch, with only ``use_rocm_custom_paged_attention`` forced, so nothing
about argument marshalling differs between the arms.

Pure decode: one query token per sequence attending over a fully cached
context, which is what the reported failure is (warmed, deterministic, wrong).

argv: [out.jsonl]
"""

import json
import os
import sys

import torch

import vllm.platforms.rocm as rocm_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
    chunked_prefill_paged_decode,
)

OUT = sys.argv[1] if len(sys.argv) > 1 else "/work/50603.jsonl"

HEAD_SIZE = 128       # the only head size the gfx11 gate admits
BLOCK_SIZE = 16       # likewise the only block size it admits
DTYPE = torch.bfloat16
# (num_heads, num_kv_heads) -> gqa_ratio. 1 and 2 are the excluded ones; 3 is
# the gate's lower edge; (32,16) is gemma-3-27b's real shape, gqa_ratio 2.
SHAPES = [(8, 8), (8, 4), (12, 4), (16, 4), (32, 16)]
CTX_LENS = [1024, 2048, 4096, 8192, 16384, 32768]
if os.environ.get("SMOKE"):        # tiny grid to shake out plumbing first
    SHAPES = [(8, 4)]
    CTX_LENS = [1024]
WARMUP, ITERS = 3, 10

_ORIG_GATE = rocm_platform.use_rocm_custom_paged_attention


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


def run_arm(d, use_custom):
    rocm_platform.use_rocm_custom_paged_attention = lambda *a, **kw: use_custom
    out = torch.empty_like(d["query"])
    # key/value are unused on the pure-decode path (max_query_len == 1 skips
    # context_attention_fwd); the dispatch reads num_kv_heads off key_cache
    # when key is None, which is the documented cross-attention-decode case.
    chunked_prefill_paged_decode(
        query=d["query"], key=None, value=None,
        output=out, kv_cache_dtype="auto",
        key_cache=d["k_cache"], value_cache=d["v_cache"],
        block_table=d["block_table"], query_start_loc=d["query_start_loc"],
        seq_lens=d["seq_lens"], max_seq_len=d["ctx_len"], max_query_len=1,
        k_scale=d["k_scale"], v_scale=d["v_scale"],
    )
    torch.cuda.synchronize()
    return out


def timed(d, use_custom):
    for _ in range(WARMUP):
        run_arm(d, use_custom)
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
          for _ in range(ITERS)]
    for s, e in ev:
        s.record(); run_arm(d, use_custom); e.record()
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
    print(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__}", flush=True)
    print("gate as shipped, for reference:", flush=True)
    for gqa in (1, 2, 3, 4):
        v = _ORIG_GATE(DTYPE, HEAD_SIZE, BLOCK_SIZE, gqa, 4096, 0, "auto", None, None)
        print(f"   use_rocm_custom_paged_attention(gqa_ratio={gqa}) = {v}", flush=True)

    fh = open(OUT, "w")
    for num_heads, num_kv in SHAPES:
        gqa = num_heads // num_kv
        for ctx in CTX_LENS:
            bs = 4 if ctx <= 8192 else 2
            d = build(ctx, num_heads, num_kv, bs)
            ref = reference(d)
            row = {"gqa_ratio": gqa, "num_heads": num_heads, "num_kv_heads": num_kv,
                   "ctx_len": ctx, "bs": bs, "head_size": HEAD_SIZE,
                   "block_size": BLOCK_SIZE, "dtype": "bf16",
                   "gate_as_shipped": bool(_ORIG_GATE(DTYPE, HEAD_SIZE, BLOCK_SIZE,
                                                      gqa, ctx, 0, "auto", None, None))}
            for arm, use_custom in (("triton", False), ("ck", True)):
                try:
                    out = run_arm(d, use_custom)
                    row[arm] = score(out, ref)
                    row[arm]["median_ms"] = timed(d, use_custom)
                except Exception as exc:
                    row[arm] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
            fh.write(json.dumps(row) + "\n"); fh.flush()
            t, c = row["triton"], row["ck"]
            fmt = lambda a: (f"rel={a['max_rel_err']:.2e} rms={a['rms_rel_err']:.2e} "
                             f"ulp={a['max_ulp']:.1f} {a['median_ms']:.3f}ms"
                             if "error" not in a else f"ERROR {a['error'][:60]}")
            print(f"gqa={gqa} ({num_heads}/{num_kv}) ctx={ctx:<6d} gated_in={row['gate_as_shipped']!s:<5s} "
                  f"| triton {fmt(t)} | ck {fmt(c)}", flush=True)
            del d, ref
            torch.cuda.empty_cache()
    fh.close()
    rocm_platform.use_rocm_custom_paged_attention = _ORIG_GATE
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
