"""Stage 1b: a positive control for Stage 1's negative result.

Stage 1 found no length-dependent accuracy loss in either paged-attention path
up to 32K. A negative like that is only worth something if the harness can see
corruption when corruption is there. This makes some.

Two things Stage 1 did not exercise:

  * every ctx_len it swept was a multiple of BLOCK_SIZE, so the final KV tile
    was always exactly full and never straddled seq_len;
  * the cache was zero-filled, so the slots past seq_len held 0, not garbage.

Real caches hold whatever the last sequence left there. vLLM 0.25.0 added a
per-token mask on the K/V loads of the final tile for exactly this reason:

    Slots >= seq_len are unwritten KV cache that may hold NaN/garbage; they
    are score-masked below, but 0 * NaN = NaN would still poison the output.

This container runs 0.23.1.dev, which predates that mask. So: fill the cache
with NaN, use a ctx_len that is NOT a multiple of BLOCK_SIZE, and see whether
the output is poisoned. If it is, the harness detects corruption and Stage 1's
clean sweep means something. If it is not, the mask was not load-bearing here
and that is worth knowing too.

This is NOT the reported bug: vllm#50603 is on 0.25.1, which has the mask.

argv: [out.jsonl]
"""

import json
import sys

import torch

import vllm.platforms.rocm as rocm_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
    chunked_prefill_paged_decode,
)

OUT = sys.argv[1] if len(sys.argv) > 1 else "/work/stage1b.jsonl"
HEAD_SIZE, BLOCK_SIZE, DTYPE = 128, 16, torch.bfloat16
NUM_HEADS, NUM_KV = 8, 4          # gqa_ratio 2, the excluded value
# aligned lengths are the control; the others straddle the final tile
CTX_LENS = [1024, 4096, 1000, 1015, 4095, 4090]
FILLS = ["zeros", "nan", "garbage"]

_ORIG_GATE = rocm_platform.use_rocm_custom_paged_attention


def build(ctx_len, fill, bs=2):
    set_random_seed(0)
    blocks = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    cache_size = blocks * bs + 8
    query = torch.randn(bs, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
    kv = torch.randn(bs, ctx_len, 2, NUM_KV, HEAD_SIZE, dtype=DTYPE)
    key, value = kv.unbind(dim=2)

    shape = (cache_size, BLOCK_SIZE, NUM_KV, HEAD_SIZE)
    if fill == "zeros":
        k_cache, v_cache = torch.zeros(shape, dtype=DTYPE), torch.zeros(shape, dtype=DTYPE)
    elif fill == "nan":
        k_cache = torch.full(shape, float("nan"), dtype=DTYPE)
        v_cache = torch.full(shape, float("nan"), dtype=DTYPE)
    else:  # plausible leftovers from a previous sequence, not a special value
        k_cache = torch.randn(shape, dtype=DTYPE) * 1e4
        v_cache = torch.randn(shape, dtype=DTYPE) * 1e4

    perm = torch.randperm(cache_size)[: blocks * bs].to(torch.int32)
    block_table = perm.view(bs, blocks)
    fk = k_cache.view(-1, NUM_KV, HEAD_SIZE)
    fv = v_cache.view(-1, NUM_KV, HEAD_SIZE)
    for i in range(bs):
        for b in range(blocks):
            lo, hi = b * BLOCK_SIZE, min((b + 1) * BLOCK_SIZE, ctx_len)
            slot = int(block_table[i, b]) * BLOCK_SIZE
            fk[slot : slot + (hi - lo)].copy_(key[i, lo:hi])
            fv[slot : slot + (hi - lo)].copy_(value[i, lo:hi])

    k_cache = (k_cache.view(-1, BLOCK_SIZE, NUM_KV, HEAD_SIZE // 8, 8)
               .permute(0, 2, 3, 1, 4).contiguous())
    v_cache = (v_cache.view(-1, BLOCK_SIZE, NUM_KV, HEAD_SIZE)
               .permute(0, 2, 3, 1).contiguous())
    scale = torch.tensor(1.0, dtype=torch.float32)
    return dict(query=query, key=key, value=value, k_cache=k_cache, v_cache=v_cache,
                block_table=block_table,
                seq_lens=torch.full((bs,), ctx_len, dtype=torch.int32),
                query_start_loc=torch.arange(bs + 1, dtype=torch.int32),
                k_scale=scale, v_scale=scale, ctx_len=ctx_len, bs=bs)


def reference(d):
    q, k, v = d["query"].float(), d["key"].float(), d["value"].float()
    bs, H, D = q.shape
    rep = H // k.shape[2]
    out = torch.empty(bs, H, D, dtype=torch.float32)
    for i in range(bs):
        for h in range(H):
            kh = h // rep
            p = torch.softmax((k[i, :, kh, :] @ q[i, h]) * D ** -0.5, dim=0)
            out[i, h] = p @ v[i, :, kh, :]
    return out


def run(d, use_custom):
    rocm_platform.use_rocm_custom_paged_attention = lambda *a, **kw: use_custom
    out = torch.empty_like(d["query"])
    chunked_prefill_paged_decode(
        query=d["query"], key=None, value=None, output=out, kv_cache_dtype="auto",
        key_cache=d["k_cache"], value_cache=d["v_cache"],
        block_table=d["block_table"], query_start_loc=d["query_start_loc"],
        seq_lens=d["seq_lens"], max_seq_len=d["ctx_len"], max_query_len=1,
        k_scale=d["k_scale"], v_scale=d["v_scale"])
    torch.cuda.synchronize()
    return out


def main():
    torch.set_default_device("cuda")
    import vllm
    print(f"device={torch.cuda.get_device_name(0)} vllm={vllm.__version__}", flush=True)
    fh = open(OUT, "w")
    for fill in FILLS:
        for ctx in CTX_LENS:
            d = build(ctx, fill)
            ref = reference(d)
            row = {"fill": fill, "ctx_len": ctx,
                   "block_aligned": ctx % BLOCK_SIZE == 0,
                   "tail_slots": (-ctx) % BLOCK_SIZE,
                   "gqa_ratio": NUM_HEADS // NUM_KV}
            for arm, uc in (("triton", False), ("ck", True)):
                try:
                    o = run(d, uc).float()
                    finite = bool(torch.isfinite(o).all())
                    scale = float(ref.abs().max())
                    row[arm] = {
                        "all_finite": finite,
                        "n_nan": int(torch.isnan(o).sum()),
                        "n_inf": int(torch.isinf(o).sum()),
                        "max_rel_err": (float((o - ref).abs().max()) / scale)
                        if finite else None,
                    }
                except Exception as exc:
                    row[arm] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
            fh.write(json.dumps(row) + "\n"); fh.flush()
            f = lambda a: ("ERROR" if "error" in a else
                           (f"rel={a['max_rel_err']:.2e}" if a["all_finite"]
                            else f"POISONED nan={a['n_nan']} inf={a['n_inf']}"))
            print(f"fill={fill:<8} ctx={ctx:<6d} aligned={row['block_aligned']!s:<5s} "
                  f"tail={row['tail_slots']:<3d} | triton {f(row['triton']):<28} "
                  f"| ck {f(row['ck'])}", flush=True)
            del d, ref
            torch.cuda.empty_cache()
    fh.close()
    rocm_platform.use_rocm_custom_paged_attention = _ORIG_GATE
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
