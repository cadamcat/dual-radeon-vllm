"""Does the BLOCK_M=64 crossover move with KV depth on CUDA, as it does on gfx1100?

vllm#52684's author measured, on one RX 7900 GRE, that the crossover between
`BLOCK_M=16` and `BLOCK_M=64` sits at `q_len` 96-128 when `kv_len = q_len`, and
drops to `q_len` 32-64 when `kv_len` is fixed at 4096. Our A100 pass
(`probe_block_m.py`) fixes `kv_len = q_len` on every one of its 140 rows, so it
cannot speak to that axis at all. He said as much in the thread and asked for
this run.

Two things differ from `probe_block_m.py` deliberately:

  * `kv_len` is a parameter rather than `kv_len = q_len`.
  * **Both arms are forced past the PR's own gate.** `select_for()` there
    collapses every arm onto base below `q_len >= 512`, which is correct for
    asking "what does the PR do as written" and useless here, because the
    crossover being tested sits at `q_len` 32-128 -- entirely below that gate.
    The author forced the same way ("a launch proxy"), so this matches his
    method rather than our own earlier one.

Arms and the speedup convention are his, so the two sets land on one axis:

    production  BLOCK_M=16, Triton's default warp count
    bm64        BLOCK_M=64, num_warps=4
    speedup = production_ms / bm64_ms      (>1 means BLOCK_M=64 is faster)

    python3 probe_kv_depth.py /content/kv_depth.jsonl
"""
import json
import sys

import torch
import triton

from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops import triton_unified_attention as tua
from vllm.v1.attention.ops.triton_unified_attention import unified_attention

OUT = sys.argv[1] if len(sys.argv) > 1 else "/content/kv_depth.jsonl"

# his two head patterns, so the medians are taken over the same pair
HEAD_PAIRS = [(32, 8), (28, 4)]
# spans his crossover region (32-64 at kv=4096, 96-128 at kv=q_len) and reaches
# up into the range our own pass already covers
Q_LENS = [16, 32, 64, 96, 128, 256, 512, 1024]
# None means kv_len = q_len, which reproduces his first sweep and ours
KV_LENS = [None, 4096, 16384]
HEAD_SIZE = 128
DTYPE, DNAME = torch.bfloat16, "bf16"
BLOCK_SIZE = 16
NUM_BLOCKS = 8192          # must cover kv_len=16384 at BLOCK_SIZE=16
WARMUP, ITERS = 10, 40

_pow2 = triton.next_power_of_2
_ORIG_SELECT = tua._select_query_block


def want_for(arm, nq):
    """(BLOCK_M, BLOCK_Q, tuned) — forced, with no gate consulted."""
    if arm == "production":
        bm = 16 if nq <= 16 else _pow2(nq)
        return (bm, bm // nq, False)
    return (64, 64 // _pow2(nq), True)      # bm64: num_warps=4


def install_arm(arm, nq):
    want = want_for(arm, nq)
    tua._is_gfx1100 = lambda: True
    tua._select_query_block = lambda msq, n, _w=want: _w
    got = tua._select_query_block(0, nq)
    assert got == want, f"{arm}: seam gave {got}, wanted {want}"
    return got


def build(q_len, kv_len, num_qh, num_kvh):
    set_random_seed(0)
    query = torch.randn(q_len, num_qh, HEAD_SIZE, dtype=DTYPE)
    key_cache = torch.randn(NUM_BLOCKS, BLOCK_SIZE, num_kvh, HEAD_SIZE, dtype=DTYPE)
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor([0, q_len], dtype=torch.int32)
    kv_lens = torch.tensor([kv_len], dtype=torch.int32)
    max_blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    assert max_blocks <= NUM_BLOCKS, (kv_len, max_blocks, NUM_BLOCKS)
    block_tables = torch.randint(0, NUM_BLOCKS, (1, max_blocks), dtype=torch.int32)
    out = torch.empty_like(query)
    return dict(
        q=query, k=key_cache, v=value_cache, out=out,
        cu_seqlens_q=cu_query_lens, seqused_k=kv_lens,
        max_seqlen_q=q_len, max_seqlen_k=kv_len,
        softmax_scale=HEAD_SIZE ** -0.5, causal=True, window_size=(-1, -1),
        block_table=block_tables, softcap=0,
        q_descale=None, k_descale=None, v_descale=None,
    )


def time_call(kwargs):
    for _ in range(WARMUP):
        unified_attention(**kwargs)
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
          for _ in range(ITERS)]
    for s, e in ev:
        s.record()
        unified_attention(**kwargs)
        e.record()
    torch.cuda.synchronize()
    ms = sorted(s.elapsed_time(e) for s, e in ev)
    return ms[len(ms) // 2], ms[0], ms[-1]


def main():
    torch.set_default_device("cuda")
    print(f"device={torch.cuda.get_device_name(0)} "
          f"cap={torch.cuda.get_device_capability(0)} "
          f"torch={torch.__version__} triton={triton.__version__}", flush=True)
    rows = []
    for kv_fixed in KV_LENS:
        for num_qh, num_kvh in HEAD_PAIRS:
            nq = num_qh // num_kvh
            for q_len in Q_LENS:
                kv_len = q_len if kv_fixed is None else kv_fixed
                if kv_len < q_len:
                    continue
                r = {"dtype": DNAME, "head_size": HEAD_SIZE,
                     "heads": f"{num_qh}:{num_kvh}", "num_queries_per_kv": nq,
                     "q_len": q_len, "kv_len": kv_len,
                     "kv_mode": "eq" if kv_fixed is None else str(kv_fixed)}
                for arm in ("production", "bm64"):
                    sel = install_arm(arm, nq)
                    kwargs = build(q_len, kv_len, num_qh, num_kvh)
                    try:
                        med, lo, hi = time_call(kwargs)
                    except Exception as ex:
                        r[f"{arm}_error"] = repr(ex)[:200]
                        continue
                    r[f"{arm}_ms"] = med
                    r[f"{arm}_ms_min"] = lo
                    r[f"{arm}_ms_max"] = hi
                    r[f"{arm}_select"] = list(sel)
                tua._select_query_block = _ORIG_SELECT
                if "production_ms" in r and "bm64_ms" in r:
                    r["bm64_speedup"] = r["production_ms"] / r["bm64_ms"]
                rows.append(r)
                sp = r.get("bm64_speedup")
                print(f"kv={r['kv_mode']:>5s} heads={r['heads']:>5s} q={q_len:>5d} "
                      f"prod={r.get('production_ms', float('nan')):.5f} "
                      f"bm64={r.get('bm64_ms', float('nan')):.5f} "
                      f"speedup={sp:.3f}" if sp else f"kv={r['kv_mode']} q={q_len} ERROR",
                      flush=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"=== KV_DEPTH DONE {len(rows)} rows -> {OUT} ===", flush=True)


if __name__ == "__main__":
    main()
