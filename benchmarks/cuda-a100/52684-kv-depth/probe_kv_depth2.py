"""Pass 2: the same sweep, timed so the short-q rows are not on the floor.

Pass 1 timed one kernel call between one pair of CUDA events. That has a fixed
cost of about 0.22 ms on this VM, which is fine from `q_len=256` upward and
useless below it: at `kv_len=4096`, `q_len=16` the whole measurement came out at
0.233 ms, six per cent above a floor it was mostly made of. `kv_len=4096` is the
depth vllm#52684's author measured, so those are exactly the rows that have to
be right.

This times a **batch** of `REPS` calls between one pair of events and divides,
so the fixed cost is amortised rather than measured. Everything else --
arms, forcing past the gate, head patterns, dtype, head_size, the speedup
convention -- is pass 1's, unchanged.

`kv_len=16384, q_len=1024` is measured by both passes on purpose: it is far
above the floor either way, so agreement there is what says the batched method
measures the same thing before its short-q numbers are believed.

    python3 probe_kv_depth2.py /content/kv_depth2.jsonl
"""
import json
import sys

import torch
import triton

from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops import triton_unified_attention as tua
from vllm.v1.attention.ops.triton_unified_attention import unified_attention

OUT = sys.argv[1] if len(sys.argv) > 1 else "/content/kv_depth2.jsonl"

HEAD_PAIRS = [(32, 8), (28, 4)]
Q_LENS = [16, 32, 64, 96, 128, 256, 1024]
KV_LENS = [4096, 16384]
HEAD_SIZE = 128
DTYPE, DNAME = torch.bfloat16, "bf16"
BLOCK_SIZE = 16
NUM_BLOCKS = 8192
WARMUP = 20
REPS = 50          # calls inside one event pair
BATCHES = 12       # event pairs; the median of these is the row

_pow2 = triton.next_power_of_2
_ORIG_SELECT = tua._select_query_block


def want_for(arm, nq):
    if arm == "production":
        bm = 16 if nq <= 16 else _pow2(nq)
        return (bm, bm // nq, False)
    return (64, 64 // _pow2(nq), True)


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
    max_blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    assert max_blocks <= NUM_BLOCKS
    return dict(
        q=query, k=key_cache, v=value_cache, out=torch.empty_like(query),
        cu_seqlens_q=torch.tensor([0, q_len], dtype=torch.int32),
        seqused_k=torch.tensor([kv_len], dtype=torch.int32),
        max_seqlen_q=q_len, max_seqlen_k=kv_len,
        softmax_scale=HEAD_SIZE ** -0.5, causal=True, window_size=(-1, -1),
        block_table=torch.randint(0, NUM_BLOCKS, (1, max_blocks), dtype=torch.int32),
        softcap=0, q_descale=None, k_descale=None, v_descale=None,
    )


def time_batched(kwargs):
    """ms per call, from batches of REPS calls inside one event pair."""
    for _ in range(WARMUP):
        unified_attention(**kwargs)
    torch.cuda.synchronize()
    per = []
    for _ in range(BATCHES):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(REPS):
            unified_attention(**kwargs)
        e.record()
        torch.cuda.synchronize()
        per.append(s.elapsed_time(e) / REPS)
    per.sort()
    return per[len(per) // 2], per[0], per[-1]


def main():
    torch.set_default_device("cuda")
    print(f"device={torch.cuda.get_device_name(0)} "
          f"torch={torch.__version__} triton={triton.__version__} "
          f"reps={REPS} batches={BATCHES}", flush=True)
    rows = []
    for kv_len in KV_LENS:
        for num_qh, num_kvh in HEAD_PAIRS:
            nq = num_qh // num_kvh
            for q_len in Q_LENS:
                r = {"dtype": DNAME, "head_size": HEAD_SIZE,
                     "heads": f"{num_qh}:{num_kvh}", "num_queries_per_kv": nq,
                     "q_len": q_len, "kv_len": kv_len, "kv_mode": str(kv_len),
                     "timing": "batched", "reps": REPS, "batches": BATCHES}
                for arm in ("production", "bm64"):
                    sel = install_arm(arm, nq)
                    kwargs = build(q_len, kv_len, num_qh, num_kvh)
                    med, lo, hi = time_batched(kwargs)
                    r[f"{arm}_ms"] = med
                    r[f"{arm}_ms_min"] = lo
                    r[f"{arm}_ms_max"] = hi
                    r[f"{arm}_select"] = list(sel)
                tua._select_query_block = _ORIG_SELECT
                r["bm64_speedup"] = r["production_ms"] / r["bm64_ms"]
                rows.append(r)
                print(f"kv={kv_len:>5d} heads={r['heads']:>5s} q={q_len:>5d} "
                      f"prod={r['production_ms']:.5f} bm64={r['bm64_ms']:.5f} "
                      f"speedup={r['bm64_speedup']:.3f}", flush=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"=== KV_DEPTH2 DONE {len(rows)} rows -> {OUT} ===", flush=True)


if __name__ == "__main__":
    main()
