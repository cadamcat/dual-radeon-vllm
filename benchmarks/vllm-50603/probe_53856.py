"""gfx1100 runtime evidence for vllm#53856, which had compile validation only.

vllm#53856 masks the ROCm CK paged-attention kernel's vectorized V-cache loads
over the final, incomplete block: logits past seq_len are already masked, but
the V load still consumes the whole block, so 0 * NaN poisons the output. Its
test plan states gfx11/gfx12 got compile validation only, no device.

Stage 1b showed the fault on this box but at gqa_ratio=2, which gfx11 GATES
OUT, so it demonstrated a path stock gfx11 never takes. This closes that and
two other holes:

  1. run at gqa_ratio 4, which the gfx11 gate admits, so the CK path measured
     is the one stock vLLM actually selects here. gqa_ratio 2 is kept as a
     contrast row, clearly marked as force-enabled.
  2. poison K and V separately. #53856 sanitises V only, so "V alone poisons"
     is the claim that has to hold for the fix to be the right shape.
  3. prove which kernel ran, by wrapping ops.paged_attention_rocm rather than
     inferring it from timing.

Not asserted here: how a NaN gets into padding. vLLM zero-fills the backing
allocation (allocate_kv_cache, vllm/v1/worker/utils.py), so that is reuse, and
#53856 already attributes it to sleep mode / allocator behaviour. This probe
injects the NaN deliberately and only measures what the kernel then does.

argv: [out.jsonl]
"""

import json
import sys

import torch

import vllm._custom_ops as ops
import vllm.platforms.rocm as rocm_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
    chunked_prefill_paged_decode,
)

OUT = sys.argv[1] if len(sys.argv) > 1 else "/work/53856.jsonl"
HEAD_SIZE, BLOCK_SIZE, DTYPE = 128, 16, torch.bfloat16
SHAPES = [(16, 4), (8, 4)]            # gqa 4 (gate admits) and 2 (gate excludes)
CTX_LENS = [4096, 4090]               # exactly full final tile, then straddling
POISON = ["none", "k_only", "v_only", "both"]

_ORIG_GATE = rocm_platform.use_rocm_custom_paged_attention
_ORIG_CK = ops.paged_attention_rocm
_ck_calls = {"n": 0}


def _counting_ck(*a, **kw):
    _ck_calls["n"] += 1
    return _ORIG_CK(*a, **kw)


ops.paged_attention_rocm = _counting_ck


def build(ctx_len, num_heads, num_kv, poison, bs=2):
    set_random_seed(0)
    blocks = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    cache_size = blocks * bs + 8
    query = torch.randn(bs, num_heads, HEAD_SIZE, dtype=DTYPE)
    kv = torch.randn(bs, ctx_len, 2, num_kv, HEAD_SIZE, dtype=DTYPE)
    key, value = kv.unbind(dim=2)

    shape = (cache_size, BLOCK_SIZE, num_kv, HEAD_SIZE)
    nan = float("nan")
    k_cache = torch.full(shape, nan, dtype=DTYPE) if poison in ("k_only", "both") \
        else torch.zeros(shape, dtype=DTYPE)
    v_cache = torch.full(shape, nan, dtype=DTYPE) if poison in ("v_only", "both") \
        else torch.zeros(shape, dtype=DTYPE)

    perm = torch.randperm(cache_size)[: blocks * bs].to(torch.int32)
    block_table = perm.view(bs, blocks)
    fk, fv = k_cache.view(-1, num_kv, HEAD_SIZE), v_cache.view(-1, num_kv, HEAD_SIZE)
    for i in range(bs):
        for b in range(blocks):
            lo, hi = b * BLOCK_SIZE, min((b + 1) * BLOCK_SIZE, ctx_len)
            slot = int(block_table[i, b]) * BLOCK_SIZE
            fk[slot : slot + (hi - lo)].copy_(key[i, lo:hi])
            fv[slot : slot + (hi - lo)].copy_(value[i, lo:hi])
    # every real token is written; only slots past seq_len keep the poison
    k_cache = (k_cache.view(-1, BLOCK_SIZE, num_kv, HEAD_SIZE // 8, 8)
               .permute(0, 2, 3, 1, 4).contiguous())
    v_cache = (v_cache.view(-1, BLOCK_SIZE, num_kv, HEAD_SIZE)
               .permute(0, 2, 3, 1).contiguous())
    s = torch.tensor(1.0, dtype=torch.float32)
    return dict(query=query, key=key, value=value, k_cache=k_cache, v_cache=v_cache,
                block_table=block_table,
                seq_lens=torch.full((bs,), ctx_len, dtype=torch.int32),
                query_start_loc=torch.arange(bs + 1, dtype=torch.int32),
                k_scale=s, v_scale=s, ctx_len=ctx_len, bs=bs)


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


def run(d, force):
    """force=None uses the gate as shipped; True/False overrides it."""
    if force is None:
        rocm_platform.use_rocm_custom_paged_attention = _ORIG_GATE
    else:
        rocm_platform.use_rocm_custom_paged_attention = lambda *a, **kw: force
    before = _ck_calls["n"]
    out = torch.empty_like(d["query"])
    chunked_prefill_paged_decode(
        query=d["query"], key=None, value=None, output=out, kv_cache_dtype="auto",
        key_cache=d["k_cache"], value_cache=d["v_cache"],
        block_table=d["block_table"], query_start_loc=d["query_start_loc"],
        seq_lens=d["seq_lens"], max_seq_len=d["ctx_len"], max_query_len=1,
        k_scale=d["k_scale"], v_scale=d["v_scale"])
    torch.cuda.synchronize()
    return out, _ck_calls["n"] > before


def main():
    torch.set_default_device("cuda")
    import vllm
    print(f"device={torch.cuda.get_device_name(0)} vllm={vllm.__version__}", flush=True)
    fh = open(OUT, "w")
    for num_heads, num_kv in SHAPES:
        gqa = num_heads // num_kv
        gated = bool(_ORIG_GATE(DTYPE, HEAD_SIZE, BLOCK_SIZE, gqa, 4096, 0,
                                "auto", None, None))
        print(f"\n--- gqa_ratio={gqa} ({num_heads}/{num_kv}) "
              f"gate_as_shipped={gated} ---", flush=True)
        for poison in POISON:
            for ctx in CTX_LENS:
                d = build(ctx, num_heads, num_kv, poison)
                ref = reference(d)
                row = {"gqa_ratio": gqa, "num_heads": num_heads, "num_kv_heads": num_kv,
                       "gate_as_shipped": gated, "poison": poison, "ctx_len": ctx,
                       "block_aligned": ctx % BLOCK_SIZE == 0,
                       "tail_slots": (-ctx) % BLOCK_SIZE}
                # as shipped, plus an explicit CK arm so the excluded ratio is
                # still measurable and clearly labelled as forced
                for arm, force in (("as_shipped", None), ("ck_forced", True),
                                   ("triton_forced", False)):
                    try:
                        o, used_ck = run(d, force)
                        o = o.float()
                        fin = bool(torch.isfinite(o).all())
                        row[arm] = {"used_ck_kernel": used_ck, "all_finite": fin,
                                    "n_nan": int(torch.isnan(o).sum()),
                                    "max_rel_err": (float((o - ref).abs().max())
                                                    / float(ref.abs().max())) if fin else None}
                    except Exception as exc:
                        row[arm] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
                fh.write(json.dumps(row) + "\n"); fh.flush()
                sh, ck = row["as_shipped"], row["ck_forced"]
                f = lambda a: ("ERR" if "error" in a else
                               (f"ok rel={a['max_rel_err']:.1e}" if a["all_finite"]
                                else f"NaN x{a['n_nan']}"))
                print(f"  poison={poison:<7} ctx={ctx:<5d} tail={row['tail_slots']:<2d} | "
                      f"as_shipped(ck={sh.get('used_ck_kernel')}) {f(sh):<18} | "
                      f"ck_forced {f(ck):<18} | triton {f(row['triton_forced'])}",
                      flush=True)
                del d, ref
                torch.cuda.empty_cache()
    fh.close()
    rocm_platform.use_rocm_custom_paged_attention = _ORIG_GATE
    ops.paged_attention_rocm = _ORIG_CK
    print("\nPROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
